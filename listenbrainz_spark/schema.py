from datetime import datetime
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, ArrayType, StringType, TimestampType, FloatType, IntegerType, BooleanType


# NOTE: please keep this schema definition alphabetized
listen_schema = [
    StructField('artist_mbids', ArrayType(StringType()), nullable=True),
    StructField('artist_msid', StringType(), nullable=False),
    StructField('artist_name', StringType(), nullable=False),
    StructField('listened_at', TimestampType(), nullable=False),
    StructField('recording_mbid', StringType(), nullable=True),
    StructField('recording_msid', StringType(), nullable=False),
    StructField('release_mbid', StringType(), nullable=True),
    StructField('release_msid', StringType(), nullable=True),
    StructField('release_name', StringType(), nullable=True),
    StructField('tags', ArrayType(StringType()), nullable=True),
    StructField('track_name', StringType(), nullable=False),
    StructField('user_name', StringType(), nullable=False),
]

# As of now the dataframe will only contain one row i.e. model id of model currently saved in HDFS.
model_index_schema = [
    StructField('model_id', StringType(), nullable=False), # Model id or identification string of best model.
    StructField('test_rmse', FloatType(), nullable=False), # Root mean squared error for test data.
]

# schema to contain model parameters.
model_param_schema = [
    StructField('alpha', FloatType(), nullable=False), # Baseline level of confidence weighting applied.
    StructField('lmbda', FloatType(), nullable=False), # Controls over fitting.
    StructField('iteration', IntegerType(), nullable=False), # Number of iterations to run.
    StructField('rank', IntegerType(), nullable=False), # Number of hidden features in our low-rank approximation matrices.
]

model_param_schema = StructType(sorted(model_param_schema, key=lambda field: field.name))

dataframe_metadata_schema = [
    StructField('dataframe_created', TimestampType(), nullable=False), # Timestamp when dataframes are created and saved in HDFS.
    StructField('dataframe_id', StringType(), nullable=False), # dataframe id or identification string of dataframe.
    # Timestamp from when listens have been used to train, validate and test the model.
    StructField('from_date', TimestampType(), nullable=False),
    # Number of listens recorded in a given time frame (between from_date and to_date, both inclusive).
    StructField('listens_count', IntegerType(), nullable=False),
    StructField('playcounts_count', IntegerType(), nullable=False), # Summation of training data, validation data and test data.
    StructField('recordings_count', IntegerType(), nullable=False), # Number of distinct recordings heard in a given time frame.
    # Timestamp till when listens have been used to train, validate and test the model.
    StructField('to_date', TimestampType(), nullable=False),
    StructField('users_count', IntegerType(), nullable=False), # Number of users active in a given time frame.
]

model_metadata_schema = [
    StructField('dataframe_id', StringType(), nullable=False), # dataframe id or identification string of dataframe.
    StructField('model_created', TimestampType(), nullable=False), # Timestamp when the model is saved in HDFS.
    StructField('model_param', model_param_schema, nullable=False), # Parameters used to train the model.
    StructField('model_id', StringType(), nullable=False), # Model id or identification string of best model.
    StructField('test_data_count', IntegerType(), nullable=False), # Number of listens used to test the model.
    StructField('test_rmse', FloatType(), nullable=False), # Root mean squared error for test data.
    StructField('training_data_count', IntegerType(), nullable=False), # Number of listens used to train the model.
    StructField('validation_data_count', IntegerType(), nullable=False), # Number of listens used to validate the model.
    StructField('validation_rmse', FloatType(), nullable=False), # Root mean squared error for validation data.
]

# The field names of the schema need to be sorted, otherwise we get weird
# errors due to type mismatches when creating DataFrames using the schema
# Although, we try to keep it sorted in the actual definition itself, we
# also sort it programmatically just in case
dataframe_metadata_schema = StructType(sorted(dataframe_metadata_schema, key=lambda field: field.name))
listen_schema = StructType(sorted(listen_schema, key=lambda field: field.name))
model_metadata_schema = StructType(sorted(model_metadata_schema, key=lambda field: field.name))
model_index_schema = StructType(sorted(model_index_schema, key=lambda field:field.name))

def convert_listen_to_row(listen):
    """ Convert a listen to a pyspark.sql.Row object.

        Args:
            listen (dict): a single dictionary representing a listen

        Returns:
            pyspark.sql.Row object - a Spark SQL Row based on the defined listen schema
    """
    meta = listen['track_metadata']
    return Row(
        listened_at=datetime.fromtimestamp(listen['listened_at']),
        user_name=listen['user_name'],
        artist_msid=meta['additional_info']['artist_msid'],
        artist_name=meta['artist_name'],
        artist_mbids=meta['additional_info'].get('artist_mbids', []),
        release_msid=meta['additional_info'].get('release_msid', ''),
        release_name=meta.get('release_name', ''),
        release_mbid=meta['additional_info'].get('release_mbid', ''),
        track_name=meta['track_name'],
        recording_msid=listen['recording_msid'],
        recording_mbid=meta['additional_info'].get('recording_mbid', ''),
        tags=meta['additional_info'].get('tags', []),
    )

def convert_dataframe_metadata_to_row(meta):
    """ Convert dataframe metadata to a pyspark.sql.Row object.

        Args:
            meta (dict): a single dictionary representing model metadata.

        Returns:
            pyspark.sql.Row object - a Spark SQL Row based on the defined dataframe metadata schema.
    """
    return Row(
        dataframe_created=datetime.utcnow(),
        dataframe_id=meta.get('dataframe_id'),
        from_date=meta.get('from_date'),
        listens_count=meta.get('listens_count'),
        playcounts_count=meta.get('playcounts_count'),
        recordings_count=meta.get('recordings_count'),
        to_date=meta.get('to_date'),
        users_count=meta.get('users_count'),
    )

def convert_model_index_to_row(test_rmse, model_id):
    """ Convert function args to a pyspark.sql.Row object.

        Args:
            test_rmse (float): test rmse of best model.
            model_id (str): Model id or identification string of best model.

        Returns:
            pyspark.sql.Row object - a Spark SQL Row.
    """
    return Row(
        test_rmse=test_rmse,
        model_id=model_id,
    )

def convert_model_metadata_to_row(meta):
    """ Convert model metadata to row object.

    Args:
        meta (dict): A dictionary containing model metadata.

    Returns:
        pyspark.sql.Row object - A Spark SQL row.
    """
    return Row(
        dataframe_id=meta.get('dataframe_id'),
        model_created=datetime.utcnow(),
        model_id=meta.get('model_id'),
        model_param=Row(
            alpha=meta.get('alpha'),
            lmbda=meta.get('lmbda'),
            iteration=meta.get('iteration'),
            rank=meta.get('rank'),
        ),
        test_data_count=meta.get('test_data_count'),
        test_rmse=meta.get('test_rmse'),
        training_data_count=meta.get('training_data_count'),
        validation_data_count=meta.get('validation_data_count'),
        validation_rmse=meta.get('validation_rmse'),
    )

def convert_to_spark_json(listen):
    meta = listen['track_metadata']
    return {
        'listened_at': str(datetime.fromtimestamp(listen['listened_at'])),
        'user_name': listen['user_name'],
        'artist_msid': meta['additional_info']['artist_msid'],
        'artist_name': meta['artist_name'],
        'artist_mbids': meta['additional_info'].get('artist_mbids', []),
        'release_msid': meta['additional_info'].get('release_msid', ''),
        'release_name': meta.get('release_name', ''),
        'release_mbid': meta['additional_info'].get('release_mbid', ''),
        'track_name': meta['track_name'],
        'recording_msid': listen['recording_msid'],
        'recording_mbid': meta['additional_info'].get('recording_mbid', ''),
        'tags': meta['additional_info'].get('tags', []),
    }
