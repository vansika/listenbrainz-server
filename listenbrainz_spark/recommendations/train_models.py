import os
import sys
import json
import uuid
import logging
import itertools
from math import sqrt
from time import time
from operator import add
from datetime import datetime
from collections import namedtuple, defaultdict
from py4j.protocol import Py4JJavaError

import listenbrainz_spark
from listenbrainz_spark import hdfs_connection
from listenbrainz_spark import config, utils, path, schema
from listenbrainz_spark.recommendations.utils import save_html
from listenbrainz_spark.exceptions import SparkSessionNotInitializedException, PathNotFoundException, FileNotFetchedException, \
    HDFSDirectoryNotDeletedException, PathNotFoundException, DataFrameNotCreatedException, DataFrameNotAppendedException

import pyspark.sql.functions as f
from pyspark import RDD
from pyspark.sql import Row
from flask import current_app
from pyspark.sql.utils import AnalysisException
from pyspark.mllib.recommendation import ALS, Rating

Model = namedtuple('Model', 'model validation_rmse rank lmbda iteration model_id training_time rmse_time, alpha')

# training HTML is generated if set to true
SAVE_TRAINING_HTML = True


def parse_dataset(row):
    """ Convert each RDD element to object of class Rating.

        Args:
            row: An RDD row or element.
    """
    return Rating(row['user_id'], row['recording_id'], row['count'])


def compute_rmse(model, data, n, model_id):
    """ Compute RMSE (Root Mean Squared Error).

        Args:
            model: Trained model.
            data (rdd): Rdd used for validation i.e validation_data
            n (int): Number of rows/elements in validation_data.
            model_id (str): Model identification string.
    """
    try:
        predictions = model.predictAll(data.map(lambda x: (x.user, x.product)))
        predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                                           .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
                                           .values()
        return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))
    except Py4JJavaError as err:
        current_app.logger.error('Root Mean Squared Error for model "{}" not computed\n{}'.format(
                                 model_id, str(err.java_exception)), exc_info=True)
        sys.exit(-1)


def preprocess_data(playcounts_df):
    """ Convert and split the dataframe into three RDDs; training data, validation data, test data.

        Args:
            playcounts_df (dataframe): Columns can be depicted as:
                [
                    'user_id', 'recording_id', 'count'
                ]

        Returns:
            training_data (rdd): Used for training.
            validation_data (rdd): Used for validation.
            test_data (rdd): Used for testing.
    """
    current_app.logger.info('Splitting dataframe...')
    training_data, validation_data, test_data = playcounts_df.rdd.map(parse_dataset).randomSplit([4, 1, 1], 45)
    return training_data, validation_data, test_data


def generate_model_id():
    """ Generate model id.
    """
    return '{}-{}'.format(config.MODEL_ID_PREFIX, uuid.uuid4())


def get_best_model_path(model_id):
    """ Get path to save best model

        Args:
            model_id (str): Model identification string of best model.

        Returns:
            path to save best model.
    """

    return config.HDFS_CLUSTER_URI + path.DATA_DIR + '/' + model_id


def get_latest_dataframe_id(dataframe_metadata_df, best_model_metadata):
    """ Get dataframe id of dataframe on which model has been trained.

        Args:
            dataframe_metadata_df (dataframe): Refer to listenbrainz_spark.schema.dataframe_metadata_schema
            best_model_metadata: Dict of best model metadata.
    """
    # get timestamp of recently saved dataframe.
    timestamp = dataframe_metadata_df.select(f.max('dataframe_created').alias('recent_dataframe_timestamp')).take(1)[0]
    # get dataframe id corresponding to most recent timestamp.
    df = dataframe_metadata_df.select('dataframe_id') \
                              .where(f.col('dataframe_created') == timestamp.recent_dataframe_timestamp).take(1)[0]

    best_model_metadata['dataframe_id'] = df.dataframe_id


def get_best_model_metadata(best_model):
    """ Get best model metadata.

        Args:
            best_model (namedtuple): contains best model and related data.

        Returns:
            dict containing best model metadata.
    """

    return {
        'alpha': best_model.alpha,
        'iteration': best_model.iteration,
        'lmbda': best_model.lmbda,
        'model_id': best_model.model_id,
        'rank': best_model.rank,
        'rmse_time': best_model.rmse_time,
        'training_time': best_model.training_time,
        'validation_rmse': best_model.validation_rmse,
    }


def train(training_data, rank, iteration, lmbda, alpha, model_id):
    """ Train model.

        Args:
            training_data (rdd): Used for training.
            rank (int): Number of factors in ALS model.
            iteration (int): Number of iterations to run.
            lmbda (float): Controls regularization.
            alpha (float): Constant for computing confidence.
            model_id (str): Model identification string.

        Returns:
            model: Trained model.

    """
    try:
        model = ALS.trainImplicit(training_data, rank, iterations=iteration, lambda_=lmbda, alpha=alpha)
        return model
    except Py4JJavaError as err:
        current_app.logger.error('Unable to train model "{}"\n{}'.format(model_id, str(err.java_exception)), exc_info=True)
        sys.exit(-1)


def get_best_model(training_data, validation_data, num_validation, ranks, lambdas, iterations, alpha):
    """ Train models and get the best model.

        Args:
            training_data (rdd): Used for training.
            validation_data (rdd): Used for validation.
            num_validation (int): Number of elements/rows in validation_data.
            ranks (list): Number of factors in ALS model.
            lambdas (list): Controls regularization.
            iterations (list): Number of iterations to run.
            alpha (float): Baseline level of confidence weighting applied.

        Returns:
            best_model: Model with least RMSE value.
            model_metadata (dict): Models information such as model id, error etc.
    """
    best_model = None
    best_model_metadata = defaultdict(dict)
    model_metadata = list()

    for rank, lmbda, iteration in itertools.product(ranks, lambdas, iterations):
        model_id = generate_model_id()

        t0 = time()
        model = train(training_data, rank, iteration, lmbda, alpha, model_id)
        mt = '{:.2f}'.format((time() - t0) / 60)

        t0 = time()
        validation_rmse = compute_rmse(model, validation_data, num_validation, model_id)
        vt = '{:.2f}'.format((time() - t0) / 60)

        model_metadata.append((model_id, mt, rank, '{:.1f}'.format(lmbda), iteration, round(validation_rmse, 2), vt))

        if best_model is None or validation_rmse < best_model.validation_rmse:
            best_model = Model(
                model=model,
                validation_rmse=round(validation_rmse, 2),
                rank=rank,
                lmbda=lmbda,
                iteration=iteration,
                model_id=model_id,
                training_time=mt,
                rmse_time=vt,
                alpha=alpha,
            )

    return best_model, model_metadata


def delete_best_model():
    """ Delete best model.
        Note: At any point in time, only one model is in HDFS
    """
    dir_exists = utils.path_exists(path.DATA_DIR)
    if dir_exists:
        utils.delete_dir(path.DATA_DIR, recursive=True)


def save_best_model(model_id, model):
    """ Save best model to HDFS.

        Args:
            model_id (str): Best model identification string.
            model: Trained best model
    """
    delete_best_model()

    dest_path = get_best_model_path(model_id)
    try:
        current_app.logger.info('Saving model...')
        model.save(listenbrainz_spark.context, dest_path)
        current_app.logger.info('Model saved!')
    except Py4JJavaError as err:
        current_app.logger.error('Unable to save best model "{}"\n{}. Aborting...'.format(model_id,
                                 str(err.java_exception)), exc_info=True)
        sys.exit(-1)


def save_model_metadata_to_hdfs(metadata):
    """ Save model metadata.

        Args:
            metadata: dict containing best model metadata.
    """
    metadata_row = schema.convert_model_metadata_to_row(metadata)
    try:
        # Create dataframe from the row object.
        model_metadata_df = utils.create_dataframe(metadata_row, schema.model_metadata_schema)
    except DataFrameNotCreatedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    try:
        current_app.logger.info('Saving model metadata...')
        # Append the dataframe to existing dataframe if already exist or create a new one.
        utils.append(model_metadata_df, path.MODEL_METADATA)
        current_app.logger.info('Model metadata saved...')
    except DataFrameNotAppendedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)


def save_training_html(time_, num_training, num_validation, num_test, model_metadata, best_model_metadata, ti,
                       models_training_time):
    """ Prepare and save taraining HTML.

        Args:
            time_ (dict): Dictionary containing execution time information, can be depicted as:
                {
                    'save_model' : '3.09',
                    ...
                }
            num_training (int): Number of elements/rows in training_data.
            num_validation (int): Number of elements/rows in validation_data.
            num_test (int): Number of elements/rows in test_data.
            model_metadata (dict): Models information such as model id, error etc.
            best_model_metadata (dict): Best Model information such as model id, error etc.
            ti (str): Seconds since epoch when the script was run.
            models_training_data (str): Time taken to train all the models.
    """
    date = datetime.utcnow().strftime('%Y-%m-%d')
    model_html = 'Model-{}-{}.html'.format(uuid.uuid4(), date)
    context = {
        'time' : time_,
        'num_training' : '{:,}'.format(num_training),
        'num_validation' : '{:,}'.format(num_validation),
        'num_test' : '{:,}'.format(num_test),
        'models' : model_metadata,
        'best_model' : best_model_metadata,
        'models_training_time' : models_training_time,
        'total_time' : '{:.2f}'.format((time() - ti) / 3600)
    }
    save_html(model_html, context, 'model.html')



def main(ranks=None, lambdas=None, iterations=None, alpha=None):

    if ranks is None:
        current_app.logger.critical('model param "ranks" missing')
        sys.exit(-1)

    if lambdas is None:
        current_app.logger.critical('model param "lambdas" missing')
        sys.exit(-1)

    if iterations is None:
        current_app.logger.critical('model param "iterations" missing')
        sys.exit(-1)

    if alpha is None:
        current_app.logger.critical('model param "alpha" missing')
        sys.exit(-1)

    ti = time()
    time_ = defaultdict(dict)
    try:
        listenbrainz_spark.init_spark_session('Train Models')
    except SparkSessionNotInitializedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    # Add checkpoint dir to break and save RDD lineage.
    listenbrainz_spark.context.setCheckpointDir(config.HDFS_CLUSTER_URI + path.CHECKPOINT_DIR)

    try:
        playcounts_df = utils.read_files_from_HDFS(path.PLAYCOUNTS_DATAFRAME_PATH)
        dataframe_metadata_df = utils.read_files_from_HDFS(path.DATAFRAME_METADATA)
    except PathNotFoundException as err:
        current_app.logger.error('{}\nConsider running create_dataframes.py'.format(str(err)), exc_info=True)
        sys.exit(-1)
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    time_['load_playcounts'] = '{:.2f}'.format((time() - ti) / 60)

    t0 = time()
    training_data, validation_data, test_data = preprocess_data(playcounts_df)
    time_['preprocessing'] = '{:.2f}'.format((time() - t0) / 60)

    # Rdds that are used in model training iterative process are cached to improve performance.
    # Caching large files may cause Out of Memory exception.
    training_data.persist()
    validation_data.persist()

    # An action must be called for persist to evaluate.
    num_training = training_data.count()
    num_validation = validation_data.count()
    num_test = test_data.count()

    current_app.logger.info('Training models...')
    t0 = time()
    best_model, model_metadata = get_best_model(training_data, validation_data, num_validation, ranks,
                                                lambdas, iterations, alpha)
    models_training_time = '{:.2f}'.format((time() - t0) / 3600)

    best_model_metadata = get_best_model_metadata(best_model)
    best_model_metadata['test_rmse'] = compute_rmse(best_model.model, test_data, num_test, best_model.model_id)

    best_model_metadata['training_data_count'] = num_training
    best_model_metadata['validation_data_count'] = num_validation
    best_model_metadata['test_data_count'] = num_test
    get_latest_dataframe_id(dataframe_metadata_df, best_model_metadata)

    # Cached data must be cleared to avoid OOM.
    training_data.unpersist()
    validation_data.unpersist()

    hdfs_connection.init_hdfs(config.HDFS_HTTP_URI)
    t0 = time()
    save_best_model(best_model.model_id, best_model.model)
    time_['save_model'] = '{:.2f}'.format((time() - t0) / 60)

    save_model_metadata_to_hdfs(best_model_metadata)
    # Delete checkpoint dir as saved lineages would eat up space, we won't be using them anyway.
    try:
        utils.delete_dir(path.CHECKPOINT_DIR, recursive=True)
    except HDFSDirectoryNotDeletedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    if SAVE_TRAINING_HTML:
        save_training_html(time_, num_training, num_validation, num_test, model_metadata, best_model_metadata, ti,
                           models_training_time)

    message = [{
        'type': 'cf_recording_model',
        'model_upload_time': str(datetime.utcnow()),
        'total_time': '{:.2f}'.format((time() - ti) / 3600),
    }]

    return message
