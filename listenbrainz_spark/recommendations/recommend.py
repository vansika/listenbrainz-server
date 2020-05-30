import sys
import os
import json
import logging
from time import time
from datetime import datetime
from collections import defaultdict
from py4j.protocol import Py4JJavaError

import listenbrainz_spark
from listenbrainz_spark import config, utils, path
from listenbrainz_spark.recommendations.train_models import get_path_to_save_best_model
from listenbrainz_spark.exceptions import SQLException, SparkSessionNotInitializedException, PathNotFoundException, \
    FileNotFetchedException, ViewNotRegisteredException

from flask import current_app
import pyspark.sql.functions as func
from pyspark.sql.functions import col
from pyspark.sql.utils import AnalysisException
from pyspark.mllib.recommendation import MatrixFactorizationModel


def get_best_model_id():
    """ Get model id of recently created model.

        Returns:
            best_model_id (str): Model identification string.
    """
    try:
        model_metadata = utils.read_files_from_HDFS(path.MODEL_METADATA)
    except PathNotFoundException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    latest_ts = model_metadata.select(func.max('model_created').alias('model_created')).take(1)[0].model_created
    best_model_id = model_metadata.select('model_id') \
                                  .where(col('model_created') == latest_ts).take(1)[0].model_id

    return best_model_id

def load_model():
    """ Load best model from given path in HDFS.
    """
    best_model_id = get_best_model_id()
    dest_path = get_path_to_save_best_model(best_model_id)
    return MatrixFactorizationModel.load(listenbrainz_spark.context, dest_path)


def generate_recommendations(candidate_set, limit, recordings_df, model):
    """ Generate recommendations from the candidate set.

        Args:
            candidate_set (rdd): RDD with elements as:
                [
                    'user_id', 'recording_id'
                ]
            limit (int): Number of recommendations to be generated.
            recordings_df: Dataframe containing distinct recordings and corresponding
                           mbids and names.
            model (parquet): Best model after training.

        Returns:
            recommended_recordings_mbids: list of recommended recording mbids.
    """
    recommendations = model.predictAll(candidate_set).takeOrdered(limit, lambda product: -product.rating)
    recommended_recording_ids = [(recommendations[i].product) for i in range(len(recommendations))]

    df = None

    df = recordings_df.select('mb_recording_mbid')\
                      .where(recordings_df.recording_id.isin(recommended_recording_ids))

    recommended_recording_mbids = [row.mb_recording_mbid for row in df.collect()]

    return recommended_recording_mbids


def get_recommendations_for_user(model, user_id, user_name, recordings_df, top_artists_candidate_set,
                                 similar_artists_candidate_set):
    """ Get recommended recordings which belong to top artists and artists similar to top
        artists listened to by the user.

        Args:
            model: Best model after training.
            user_id (int): user id of the user.
            user_name (str): User name of the user.
            recordings_df: Dataframe containing distinct recordings and corresponding
                           mbids and names.
            top_artists_candidate_set: Dataframe containing recording ids that belong to top artists.
            similar_artists_candidate_set: Dataframe containing recording ids that belong to similar artists.

        Returns:
            user_recommendations_top_artist: list of recommended recordings of top artist.
            user_recommendations_similar_artist: list of recommended recordings of similar artist.
    """
    top_artists_recordings = top_artists_candidate_set.select('user_id', 'recording_id') \
                                                      .where(col('user_id') == user_id)

    top_artists_recordings_rdd = top_artists_recordings.rdd.map(lambda r: (r['user_id'], r['recording_id']))

    user_recommendations_top_artist = generate_recommendations(top_artists_recordings_rdd,
                                                               config.RECOMMENDATION_TOP_ARTIST_LIMIT,
                                                               recordings_df, model)

    if len(user_recommendations_top_artist) == 0:
        current_app.logger.info('Top artists recommendations not generated for "{}"'.format(user_name))

    similar_artists_recordings = similar_artists_candidate_set.select('user_id', 'recording_id') \
                                                              .where(col('user_id') == user_id)
    try:
        similar_artists_recordings.take(1)[0]
        similar_artists_recordings_rdd = similar_artists_recordings.rdd.map(lambda r: (r['user_id'], r['recording_id']))
        user_recommendations_similar_artist = generate_recommendations(similar_artists_recordings_rdd,
                                                                       config.RECOMMENDATION_SIMILAR_ARTIST_LIMIT,
                                                                       recordings_df, model)
    except IndexError:
        user_recommendations_similar_artist = []
        current_app.logger.info('Similar artist recordings not found for "{}"'.format(user_name))
        current_app.logger.info('Similar artist recommendations not generated for "{}"'.format(user_name))

    return user_recommendations_top_artist, user_recommendations_similar_artist


def get_recommendations_for_all(recordings_df, model, top_artists_candidate_set, similar_artists_candidate_set):
    """ Get recommendations for all active users.

        Args:
            recordings_df: Dataframe containing distinct recordings and corresponding
                           mbids and names.
            model: Best model after training.
            top_artists_candidate_set: Dataframe containing recording ids that belong to top artists.
            similar_artists_candidate_set: Dataframe containing recording ids that belong to similar artists.

        Returns:
            messages (list): user recommendations.
    """
    messages = []
    current_app.logger.info('Generating recommendations...')
    # active users in the last week/month.
    # users for whom recommendations will be generated.
    users_df = top_artists_candidate_set.select('user_id', 'user_name').distinct()

    for row in users_df.collect():
        user_name = row.user_name
        user_id = row.user_id

        user_recommendations_top_artist, user_recommendations_similar_artist = get_recommendations_for_user(
            model, user_id, user_name, recordings_df,
            top_artists_candidate_set, similar_artists_candidate_set
        )

        messages.append({
            'musicbrainz_id': user_name,
            'type': 'cf_recording_recommendations',
            'top_artist': user_recommendations_top_artist,
            'similar_artist': user_recommendations_similar_artist,
        })

    current_app.logger.info('Recommendations Generated!')
    return messages


def main():
    try:
        listenbrainz_spark.init_spark_session('Recommendations')
    except SparkSessionNotInitializedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    try:
        recordings_df = utils.read_files_from_HDFS(path.RECORDINGS_DATAFRAME_PATH)
        top_artists_candidate_set = utils.read_files_from_HDFS(path.TOP_ARTIST_CANDIDATE_SET)
        similar_artists_candidate_set = utils.read_files_from_HDFS(path.SIMILAR_ARTIST_CANDIDATE_SET)
    except PathNotFoundException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)
    except FileNotFetchedException as err:
        current_app.logger.error(str(err), exc_info=True)
        sys.exit(-1)

    current_app.logger.info('Loading model...')
    try:
        model = load_model()
    except Py4JJavaError as err:
        current_app.logger.error('Unable to load model "{}"\n{}\nAborting...'.format(best_model_id, str(err.java_exception)),
                                 exc_info=True)
        sys.exit(-1)

    # an action must be called to persist data in memory
    recordings_df.count()
    recordings_df.persist()

    messages = get_recommendations_for_all(recordings_df, model, top_artists_candidate_set,
                                           similar_artists_candidate_set)
    # persisted data must be cleared from memory after usage to avoid OOM
    recordings_df.unpersist()

    return messages
