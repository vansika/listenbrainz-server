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
from listenbrainz_spark.recommendations.create_dataframes import save_dataframe_metadata_to_HDFS
from listenbrainz_spark.exceptions import SparkSessionNotInitializedException, PathNotFoundException, FileNotFetchedException, \
    HDFSDirectoryNotDeletedException, PathNotFoundException, DataFrameNotCreatedException, DataFrameNotAppendedException

from pyspark.sql import Row
from flask import current_app
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as f
from pyspark.mllib.recommendation import ALS, Rating

Model = namedtuple('Model', 'model validation_rmse rank lmbda iteration model_id training_time rmse_time')

# training HTML is generated if set to true
SAVE_TRAINING_HTML = True

def best_model_id_exists(best_model_id, model_metadata_df):
    """ Check if best_model_id has been previoulsy used.

        Args:
            best_model_id (str): Model indentification string.
            model_metadata_df (str): Dataframe where model metadata is saved.
                Refer to listenbrainz_spark.schema.model_metadata_schema

        Returns:
            None: if best_model_id is not previously used.
            df (dataframe): Dataframe row containing best model id indicating previous use of best_model_id.
    """
    try:
        df = model_metadata_df[model_metadata_df.model_id.isin([best_model_id])].take(1)[0]
    except IndexError:
        return None
    return df

def parse_dataset(row):
    """ Convert each RDD element to object of class Rating.

        Args:
            row: An RDD row or element.
    """
    return Rating(row['user_id'], row['recording_id'], row['count'])

def compute_rmse(model, data, n):
    """ Compute RMSE (Root Mean Squared Error).

        Args:
            model: Trained model.
            data (rdd): Rdd used for validation i.e validation_data
            n (int): Number of rows/elements in validation_data.
    """
    predictions = model.predictAll(data.map(lambda x: (x.user, x.product)))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

def generate_best_model_id():
    """ Generate best model id.
    """
    return '{}-{}'.format(config.MODEL_ID_PREFIX, uuid.uuid4())

def update_dataframe_metadata(best_model_id, new_best_model_id, dataframe_metadata_df):
    """ Insert a row in dataframe_metadata dataframe.

        Args:
            best_model_id (str): Already used model identification string.
            new_best_model_id (str): Newly generated model identification string.
            dataframe_metadata_df (dataframe): Dataframe containing dataframe metadata.
    """
    # get dataframe metadata of recently saved dataframes.
    row = dataframe_metadata_df.select('*').where(f.col('model_id') == best_model_id).collect()[0]
    # Note that this script always uses recently saved dataframes,
    # therefore a new row will be added to dataframe_metadata with metadata of recently
    # saved dataframes replicated except the model_id.
    # model_id in this row == new_best_model_id.
    # this function will be called when this script is invoked without invoking create_datafrmes.py
    metadata = {}
    metadata['dataframe_created'] = row.dataframe_created
    metadata['from_date'] = row.from_date
    metadata['listens_count'] = row.listens_count
    metadata['model_id'] = new_best_model_id
    metadata['playcounts_count'] = row.playcounts_count
    metadata['recordings_count'] = row.recordings_count
    metadata['to_date'] = row.to_date
    metadata['users_count'] = row.users_count
    save_dataframe_metadata_to_HDFS(metadata)

def get_best_model_id(dataframe_metadata_df):
    """ Get best model id from dataframe metadata.

        Args:
            dataframe_metadata_df (dataframe): Refer to listenbrainz_spark.schema.dataframe_metadata_schema

        Returns:
            best_model_id (str): Model identification string fetched from dataframe_metadata_df which is
                not yet assigned to any model.
            new_best_model_id (str): Model identification string generated when model_id in dataframe_metadata_df
                has been assigned to a model.
    """
    # get timestamp of most recent saved dataframes.
    timestamp = dataframe_metadata_df.select(f.max('dataframe_created') \
        .alias('recent_dataframe_timestamp')).take(1)[0]
    # get model id corresponding to most recent timestamp.
    model_id = dataframe_metadata_df.select('model_id') \
        .where(f.col('dataframe_created') == timestamp.recent_dataframe_timestamp).take(1)[0]
    best_model_id = model_id.model_id

    try:
        model_metadata_df = utils.read_files_from_HDFS(path.MODEL_METADATA)
    except PathNotFoundException:
        # will only be invoked on first run of this script.
        return best_model_id
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    # if best_model_id has been previoulsy used generate a new best model id
    if best_model_id_exists(best_model_id, model_metadata_df):
        new_best_model_id = generate_best_model_id()
        # update dataframe_metadata with new_best_model_id
        update_dataframe_metadata(best_model_id, new_best_model_id, dataframe_metadata_df)
        return new_best_model_id

    return best_model_id

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

def train(model_id, training_data, validation_data, num_validation, ranks, lambdas, iterations):
    """ Train the data and get models as per given parameters i.e. ranks, lambdas and iterations.

        Args:
            model_id (str): Best model identification string.
            training_data (rdd): Used for training.
            validation_data (rdd): Used for validation.
            num_validation (int): Number of elements/rows in validation_data.
            ranks (list): Number of factors in ALS model.
            lambdas (list): Controls regularization.
            iterations (list): Number of iterations to run.

        Returns:
            best_model: Model with least RMSE value.
            model_metadata (dict): Models information such as model id, error etc.
            best_model_metadata (dict): Best Model information such as model id, error etc.
    """
    best_model = None
    best_model_metadata = defaultdict(dict)
    model_metadata = list()
    alpha = 3.0
    for rank, lmbda, iteration in itertools.product(ranks, lambdas, iterations):
        t0 = time()
        try:
            model = ALS.trainImplicit(training_data, rank, iterations=iteration, lambda_=lmbda, alpha=alpha)
        except Py4JJavaError as err:
            current_app.logger.error('Unable to train model "{}"\n{}'.format(model_id, str(err.java_exception)), exc_info=True)
            sys.exit(-1)
        mt = '{:.2f}'.format((time() - t0) / 60)
        t0 = time()
        try:
            validation_rmse = compute_rmse(model, validation_data, num_validation)
        except Py4JJavaError as err:
            current_app.logger.error('Root Mean Squared Error for model "{}" for validation data not computed\n{}'.format(
                model_id, str(err.java_exception)), exc_info=True)
            sys.exit(-1)
        vt = '{:.2f}'.format((time() - t0) / 60)
        model_metadata.append((model_id, mt, rank, '{:.1f}'.format(lmbda), iteration, round(validation_rmse, 2), vt))
        if best_model is None or validation_rmse < best_model.validation_rmse:
            best_model = Model(model=model, validation_rmse=validation_rmse, rank=rank, lmbda=lmbda, iteration=iteration,
                model_id=model_id, training_time=mt, rmse_time=vt)

    best_model_metadata = {'validation_rmse': best_model.validation_rmse, 'rank': best_model.rank, 'lmbda':
            best_model.lmbda, 'iteration': best_model.iteration, 'model_id': best_model.model_id, 'training_time':
                best_model.training_time, 'rmse_time': best_model.rmse_time, 'alpha': alpha}
    return best_model, model_metadata, best_model_metadata

def save_model(model_id, model):
    """ Save best model to HDFS.

        Args:
            model_id (str): Model identification string of best model.
            model: Best model.
    """
    try:
        model.save(listenbrainz_spark.context, config.HDFS_CLUSTER_URI + path.DATA_DIR + '/' + model_id)
    except Py4JJavaError as err:
        current_app.logger.error('Unable to save best model "{}"\n{}. Aborting...'.format(model_id,
            str(err.java_exception)), exc_info=True)
        sys.exit(-1)

def save_model_index(test_rmse, model_id):
    """ Overwrite model_index dataframe.
        Refer to listenbrainz_spark.schema.model_index_schema

        Args:
            test_rmse (float): test rmse of best model.
            model_id (str): Model identification string of best model.
    """
    metadata_row = schema.convert_model_index_to_row(test_rmse, model_id)
    try:
        # Create dataframe from the row object.
        model_index_df = utils.create_dataframe(metadata_row, schema.model_index_schema)
    except DataFrameNotCreatedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)
    try:
        # The dataframe is overwritten since we wish to store model id of recently trained model.
        utils.save_parquet(model_index_df, path.INDEX)
    except DataFrameNotAppendedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

def delete_model(model_id):
    """ Delete a model.

        Args:
            model_id (str): Model identification string of model to be deleted.
    """
    utils.delete_dir(path.DATA_DIR + model_id, recursive=True)

def save_model_delete_metadata(model_id, timestamp):
    """ Save model indexes that have been deleted from HDFS.

        Args:
            model__id (str): Model identification string of model to be deleted.
    """
    # delete the previous model from HDFS.
    delete_model(model_id)

    metadata_row = schema.convert_model_deleted_to_row(model_id, timestamp)
    try:
        # Create dataframe from the row object.
        model_deleted_df = utils.create_dataframe(metadata_row, schema.deleted_models_schema)
    except DataFrameNotCreatedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)
    try:
        # Append the dataframe to existing dataframe if already exist or create a new one.
        utils.append(model_deleted_df, path.DELETED_MODELS)
    except DataFrameNotAppendedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

def save_model_metadata_to_HDFS(metadata):
    """ Save model metadata.

        Args:
            metadata (dict): Model metadata.
    """
    metadata_row = schema.convert_model_metadata_to_row(metadata)
    try:
        # Create dataframe from the row object.
        model_metadata_df = utils.create_dataframe(metadata_row, schema.model_metadata_schema)
    except DataFrameNotCreatedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)
    try:
        # Append the dataframe to existing dataframe if already exist or create a new one.
        utils.append(model_metadata_df, path.MODEL_METADATA)
    except DataFrameNotAppendedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

def compare_model_indexes(model, metadata):
    """ Decide if the best model generated should be saved or not.

        Args:
            model: Best model.
            metadata (dict): Best model metadata.
    """
    # current timestamp is locked here to use same timestamp in different dataframes.
    curr_timestamp = datetime.utcnow()
    metadata['model_created'] = curr_timestamp
    try:
        model_index_df = utils.read_files_from_HDFS(path.INDEX)
    except PathNotFoundException:
        # will be invoked on first run of this script i.e model index dataframe does not exist.
        save_model_index(metadata['test_rmse'], metadata['model_id'])
        save_model(metadata['model_id'], model)
        save_model_metadata_to_HDFS(metadata)
        return
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    # Since the model_index dataframe will always contain one row
    model_id_to_delete = model_index_df.select('*').take(1)[0].model_id
    # Save model index of recently trained model.
    save_model_index(metadata['test_rmse'], metadata['model_id'])
    # save model metadata of recently trained model.
    save_model_metadata_to_HDFS(metadata)
    # save recently trained model.
    save_model(metadata['model_id'], model)
    # save model index of deleted model i.e the previous model.
    save_model_delete_metadata(model_id_to_delete, curr_timestamp)

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

def main():
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
    model_id = get_best_model_id(dataframe_metadata_df)

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

    model, model_metadata, best_model_metadata = train(model_id, training_data, validation_data, num_validation,
        config.RANKS, config.LAMBDAS, config.ITERATIONS)
    models_training_time = '{:.2f}'.format((time() - t0) / 3600)

    try:
        best_model_metadata['test_rmse'] = compute_rmse(model.model, test_data, num_test)
    except Py4JJavaError as err:
        current_app.logger.error('Root mean squared error for best model for test data not computed\n{}\nAborting...'\
            .format(str(err.java_exception)), exc_info=True)
        sys.exit(-1)

    best_model_metadata['training_data_count'] = num_training
    best_model_metadata['validation_data_count'] = num_validation
    best_model_metadata['test_data_count'] = num_test
    # Cached data must be cleared to avoid OOM.
    training_data.unpersist()
    validation_data.unpersist()

    hdfs_connection.init_hdfs(config.HDFS_HTTP_URI)
    t0 = time()
    compare_model_indexes(model.model, best_model_metadata)
    time_['save_model'] = '{:.2f}'.format((time() - t0) / 60)

    # Delete checkpoint dir as saved lineages would eat up space, we won't be using them anyway.
    try:
        utils.delete_dir(path.CHECKPOINT_DIR, recursive=True)
    except HDFSDirectoryNotDeletedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    if SAVE_TRAINING_HTML:
        save_training_html(time_, num_training, num_validation, num_test, model_metadata, best_model_metadata, ti,
            models_training_time)




