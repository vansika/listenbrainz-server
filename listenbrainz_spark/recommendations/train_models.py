# This script is responsible to train models and save the best model to HDFS. The general
# flow is as follows:
#
# **preprocess_data: playcounts_df is loaded from HDFS and is split into training_data, validation_data
#                    and test_data. Th dataframe is converted to an RDD and each row is converted to a
#                    Rating(row['user_id'], row['recording_id'], row['count']) object.
#
# **get_best_model: Eight models are trained using the training_data RDD. Each model uses a different value of
#                   `rank`, `lambda` and `iteration`. Refer to https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
#                   to know more about these params. The Model with the least validation_rmse is called the best_model.
#                   validation_rmse is Root Mean Squared Error calculated using the validation_data.
#
# **save_model: The best_model which of the previous run of the script is deleted from HDFS and the new best_model is saved to HDFS.
#
# **save_model_metadata_to_hdfs: Since the model is always trained on recently created dataframes, the model_metadata (rank, lambda, training_data_count etc)
#                                is saved corresponding to recently created dataframe_id. The model metadata also contains the unique identification string
#                                for the best model.


import os
import uuid
import logging
import itertools
from math import sqrt
import time
from operator import add
from datetime import datetime
from collections import namedtuple, defaultdict
from py4j.protocol import Py4JJavaError

import listenbrainz_spark
from listenbrainz_spark import hdfs_connection
from listenbrainz_spark import config, utils, path, schema
from listenbrainz_spark.recommendations.utils import save_html
from listenbrainz_spark.exceptions import (SparkSessionNotInitializedException,
                                           PathNotFoundException,
                                           FileNotFetchedException,
                                           HDFSDirectoryNotDeletedException,
                                           PathNotFoundException,
                                           DataFrameNotCreatedException,
                                           DataFrameNotAppendedException)

import pyspark.sql.functions as func
from pyspark import RDD
from pyspark.sql import Row
from flask import current_app
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
        raise


def preprocess_data(playcounts_df):
    """ Convert and split the dataframe into three RDDs; training data, validation data, test data.

        Args:
            playcounts_df: Dataframe containing play(listen) counts of users.

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


def get_model_path(model_id):
    """ Get path to save or load model

        Args:
            model_id (str): Model identification string.

        Returns:
            path to save or load model.
    """

    return config.HDFS_CLUSTER_URI + path.DATA_DIR + '/' + model_id


def get_latest_dataframe_id(dataframe_metadata_df):
    """ Get dataframe id of dataframe on which model has been trained.

        Args:
            dataframe_metadata_df (dataframe): Refer to listenbrainz_spark.schema.dataframe_metadata_schema

        Returns:
            dataframe id
    """
    # get timestamp of recently saved dataframe.
    timestamp = dataframe_metadata_df.select(func.max('dataframe_created').alias('recent_dataframe_timestamp')).take(1)[0]
    # get dataframe id corresponding to most recent timestamp.
    df = dataframe_metadata_df.select('dataframe_id') \
                              .where(func.col('dataframe_created') == timestamp.recent_dataframe_timestamp).take(1)[0]

    return df.dataframe_id


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
        raise


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

        t0 = time.monotonic()
        current_app.logger.info("Training model with model id: {}".format(model_id))
        model = train(training_data, rank, iteration, lmbda, alpha, model_id)
        current_app.logger.info("Model trained!")
        mt = '{:.2f}'.format((time.monotonic() - t0) / 60)

        t0 = time.monotonic()
        current_app.logger.info("Calculating validation RMSE for model with model id : {}".format(model_id))
        validation_rmse = compute_rmse(model, validation_data, num_validation, model_id)
        current_app.logger.info("Validation RMSE calculated!")
        vt = '{:.2f}'.format((time.monotonic() - t0) / 60)

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


def delete_model():
    """ Delete model.
        Note: At any point in time, only one model is in HDFS
    """
    dir_exists = utils.path_exists(path.DATA_DIR)
    if dir_exists:
        utils.delete_dir(path.DATA_DIR, recursive=True)


def save_model(model_id, model):
    """ Save model to HDFS.

        Args:
            model_id (str): Model identification string.
            model: Trained model
    """
    # delete previously saved model before saving a new model
    delete_model()

    dest_path = get_model_path(model_id)
    try:
        current_app.logger.info('Saving model...')
        model.save(listenbrainz_spark.context, dest_path)
        current_app.logger.info('Model saved!')
    except Py4JJavaError as err:
        current_app.logger.error('Unable to save model "{}"\n{}. Aborting...'.format(model_id,
                                 str(err.java_exception)), exc_info=True)
        raise


def save_model_metadata_to_hdfs(metadata):
    """ Save model metadata.

        Args:
            metadata: dict containing model metadata.
    """
    metadata_row = schema.convert_model_metadata_to_row(metadata)
    try:
        # Create dataframe from the row object.
        model_metadata_df = utils.create_dataframe(metadata_row, schema.model_metadata_schema)
    except DataFrameNotCreatedException as err:
        current_app.logger.error(str(err), exc_info=True)
        raise

    try:
        current_app.logger.info('Saving model metadata...')
        # Append the dataframe to existing dataframe if already exist or create a new one.
        utils.append(model_metadata_df, path.MODEL_METADATA)
        current_app.logger.info('Model metadata saved...')
    except DataFrameNotAppendedException as err:
        current_app.logger.error(str(err), exc_info=True)
        raise


def save_training_html(time_, num_training, num_validation, num_test, model_metadata, best_model_metadata, ti,
                       models_training_time):
    """ Prepare and save taraining HTML.

        Args:
            time_ (dict): Dictionary containing execution time information.
            num_training (int): Number of elements/rows in training_data.
            num_validation (int): Number of elements/rows in validation_data.
            num_test (int): Number of elements/rows in test_data.
            model_metadata (dict): Models information such as model id, error etc.
            best_model_metadata (dict): Best Model information such as model id, error etc.
            ti (str): Value of the monotonic clock when the script was run.
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
        'total_time' : '{:.2f}'.format((time.monotonic() - ti) / 3600)
    }
    save_html(model_html, context, 'model.html')



def main(ranks=None, lambdas=None, iterations=None, alpha=None):

    if ranks is None:
        current_app.logger.critical('model param "ranks" missing')


    if lambdas is None:
        current_app.logger.critical('model param "lambdas" missing')
        raise

    if iterations is None:
        current_app.logger.critical('model param "iterations" missing')
        raise

    if alpha is None:
        current_app.logger.critical('model param "alpha" missing')
        raise

    ti = time.monotonic()
    time_ = defaultdict(dict)
    try:
        listenbrainz_spark.init_spark_session('Train Models')
    except SparkSessionNotInitializedException as err:
        current_app.logger.error(str(err), exc_info=True)
        raise

    # Add checkpoint dir to break and save RDD lineage.
    listenbrainz_spark.context.setCheckpointDir(config.HDFS_CLUSTER_URI + path.CHECKPOINT_DIR)

    try:
        playcounts_df = utils.read_files_from_HDFS(path.PLAYCOUNTS_DATAFRAME_PATH)
        dataframe_metadata_df = utils.read_files_from_HDFS(path.DATAFRAME_METADATA)
    except PathNotFoundException as err:
        current_app.logger.error('{}\nConsider running create_dataframes.py'.format(str(err)), exc_info=True)
        raise
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        raise

    time_['load_playcounts'] = '{:.2f}'.format((time.monotonic() - ti) / 60)

    t0 = time.monotonic()
    training_data, validation_data, test_data = preprocess_data(playcounts_df)
    time_['preprocessing'] = '{:.2f}'.format((time.monotonic() - t0) / 60)

    # An action must be called for persist to evaluate.
    num_training = training_data.count()
    num_validation = validation_data.count()
    num_test = test_data.count()

    t0 = time.monotonic()
    best_model, model_metadata = get_best_model(training_data, validation_data, num_validation, ranks,
                                                lambdas, iterations, alpha)
    models_training_time = '{:.2f}'.format((time.monotonic() - t0) / 3600)

    best_model_metadata = get_best_model_metadata(best_model)
    current_app.logger.info("Calculating test RMSE for best model with model id: {}".format(best_model.model_id))
    best_model_metadata['test_rmse'] = compute_rmse(best_model.model, test_data, num_test, best_model.model_id)
    current_app.logger.info("Test RMSE calculated!")

    best_model_metadata['training_data_count'] = num_training
    best_model_metadata['validation_data_count'] = num_validation
    best_model_metadata['test_data_count'] = num_test
    best_model_metadata['dataframe_id'] = get_latest_dataframe_id(dataframe_metadata_df)

    hdfs_connection.init_hdfs(config.HDFS_HTTP_URI)
    t0 = time.monotonic()
    save_model(best_model.model_id, best_model.model)
    time_['save_model'] = '{:.2f}'.format((time.monotonic() - t0) / 60)

    save_model_metadata_to_hdfs(best_model_metadata)
    # Delete checkpoint dir as saved lineages would eat up space, we won't be using them anyway.
    try:
        utils.delete_dir(path.CHECKPOINT_DIR, recursive=True)
    except HDFSDirectoryNotDeletedException as err:
        current_app.logger.error(str(err), exc_info=True)
        raise

    if SAVE_TRAINING_HTML:
        current_app.logger.info('Saving HTML...')
        save_training_html(time_, num_training, num_validation, num_test, model_metadata, best_model_metadata, ti,
                           models_training_time)
        current_app.logger.info('Done!')

    message = [{
        'type': 'cf_recording_model',
        'model_upload_time': str(datetime.utcnow()),
        'total_time': '{:.2f}'.format(time.monotonic() - ti),
    }]

    return message
