""" This file contains handler functions for rabbitmq messages we
    receive from the Spark cluster.
"""
import json

import listenbrainz.db.user as db_user
import listenbrainz.db.stats as db_stats
import listenbrainz.db.recommendations_cf_recording as db_recommendations_cf_recording
import listenbrainz.db.missing_releases_musicbrainz as db_missing_releases_musicbrainz

from flask import current_app, render_template
from pydantic import ValidationError
from brainzutils.mail import send_mail
from datetime import datetime, timezone, timedelta
from data.model.user_daily_activity import UserDailyActivityStatJson
from data.model.user_listening_activity import UserListeningActivityStatJson
from data.model.user_artist_stat import UserArtistStatJson
from data.model.user_release_stat import UserReleaseStatJson
from data.model.user_recording_stat import UserRecordingStatJson

TIME_TO_CONSIDER_STATS_AS_OLD = 20  # minutes
TIME_TO_CONSIDER_RECOMMENDATIONS_AS_OLD = 7  # days


def is_new_user_stats_batch():
    """ Returns True if this batch of user stats is new, False otherwise

    User stats come in as multiple rabbitmq messages. We only wish to send an email once per batch.
    So, we check the database and see if the difference between the last time stats were updated
    and right now is greater than 12 hours.
    """
    last_update_ts = db_stats.get_timestamp_for_last_user_stats_update()
    if last_update_ts is None:
        last_update_ts = datetime.min.replace(tzinfo=timezone.utc)  # use min datetime value if last_update_ts is None

    return datetime.now(timezone.utc) - last_update_ts > timedelta(minutes=TIME_TO_CONSIDER_STATS_AS_OLD)


def is_new_cf_recording_recommendation_batch():
    """ Returns True if this batch of recommendations is new, False otherwise
    """
    create_ts = db_recommendations_cf_recording.get_timestamp_for_last_recording_recommended()
    if create_ts is None:
        return True

    return datetime.now(timezone.utc) - create_ts > timedelta(days=TIME_TO_CONSIDER_RECOMMENDATIONS_AS_OLD)


def notify_cf_recording_recommendations_update():
    """ Send an email to notify recommendations are being written into db.
    """
    if current_app.config['TESTING']:
        return

    send_mail(
        subject="Recommendations being written into the DB - ListenBrainz",
        text=render_template('emails/cf_recording_recommendation_notification.txt', now=str(datetime.utcnow())),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN']
    )


def notify_user_stats_update(stat_type):
    if not current_app.config['TESTING']:
        send_mail(
            subject="New user stats are being written into the DB - ListenBrainz",
            text=render_template('emails/user_stats_notification.txt', now=str(datetime.utcnow()), stat_type=stat_type),
            recipients=['listenbrainz-observability@metabrainz.org'],
            from_name='ListenBrainz',
            from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN']
        )


def _get_entity_model(entity):
    if entity == 'artists':
        return UserArtistStatJson
    elif entity == 'releases':
        return UserReleaseStatJson
    elif entity == 'recordings':
        return UserRecordingStatJson
    raise ValueError("Unknown entity type: %s" % entity)


def handle_user_entity(data):
    """ Take entity stats for a user and save it in the database. """
    musicbrainz_id = data['musicbrainz_id']
    user = db_user.get_by_mb_id(musicbrainz_id)
    if not user:
        current_app.logger.critical("Calculated stats for a user that doesn't exist in the Postgres database: %s", musicbrainz_id)
        return

    # send a notification if this is a new batch of stats
    if is_new_user_stats_batch():
        notify_user_stats_update(stat_type=data.get('type', ''))
    current_app.logger.debug("inserting stats for user %s", musicbrainz_id)

    stats_range = data['stats_range']
    entity = data['entity']
    data[entity] = data['data']

    # Strip extra data
    to_remove = {'musicbrainz_id', 'type', 'entity', 'data', 'stats_range'}
    data_mod = {key: data[key] for key in data if key not in to_remove}

    entity_model = _get_entity_model(entity)

    # Get function to insert statistics
    db_handler = getattr(db_stats, 'insert_user_{}'.format(entity))

    try:
        db_handler(user['id'], entity_model(**{stats_range: data_mod}))
    except ValidationError:
        current_app.logger.error("""ValidationError while inserting {stats_range} top {entity} for user with user_id: {user_id}.
                                 Data: {data}""".format(stats_range=stats_range, entity=entity,
                                                        user_id=user['id'], data=json.dumps({stats_range: data_mod}, indent=3)),
                                 exc_info=True)


def handle_user_listening_activity(data):
    """ Take listening activity stats for user and save it in database. """
    musicbrainz_id = data['musicbrainz_id']
    user = db_user.get_by_mb_id(musicbrainz_id)
    if not user:
        current_app.logger.critical("Calculated stats for a user that doesn't exist in the Postgres database: %s", musicbrainz_id)
        return

    # send a notification if this is a new batch of stats
    if is_new_user_stats_batch():
        notify_user_stats_update(stat_type=data.get('type', ''))
    current_app.logger.debug("inserting stats for user %s", musicbrainz_id)

    stats_range = data['stats_range']

    # Strip extra data
    to_remove = {'musicbrainz_id', 'type', 'stats_range'}
    data_mod = {key: data[key] for key in data if key not in to_remove}

    try:
        db_stats.insert_user_listening_activity(user['id'], UserListeningActivityStatJson(**{stats_range: data_mod}))
    except ValidationError:
        current_app.logger.error("""ValidationError while inserting {stats_range} listening_activity for user with
                                    user_id: {user_id}. Data: {data}""".format(stats_range=stats_range, user_id=user['id'],
                                                                               data=json.dumps({stats_range: data_mod},
                                                                                               indent=3)),
                                 exc_info=True)


def handle_user_daily_activity(data):
    """ Take daily activity stats for user and save it in database. """
    musicbrainz_id = data['musicbrainz_id']
    user = db_user.get_by_mb_id(musicbrainz_id)
    if not user:
        current_app.logger.critical("Calculated stats for a user that doesn't exist in the Postgres database: %s", musicbrainz_id)
        return

    # send a notification if this is a new batch of stats
    if is_new_user_stats_batch():
        notify_user_stats_update(stat_type=data.get('type', ''))
    current_app.logger.debug("inserting stats for user %s", musicbrainz_id)

    stats_range = data['stats_range']

    # Strip extra data
    to_remove = {'musicbrainz_id', 'type', 'stats_range'}
    data_mod = {key: data[key] for key in data if key not in to_remove}

    try:
        db_stats.insert_user_daily_activity(user['id'], UserDailyActivityStatJson(**{stats_range: data_mod}))
    except ValidationError:
        current_app.logger.error("""ValidationError while inserting {stats_range} daily_activity for user with
                                    user_id: {user_id}. Data: {data}""".format(user_id=user['id'],
                                                                               data=json.dumps(data_mod, indent=3)),
                                 exc_info=True)


def handle_dump_imported(data):
    """ Process the response that the cluster sends after importing a new full dump

    We don't really need to _do_ anything, just send an email over for observability.
    """
    if current_app.config['TESTING']:
        return

    dump_name = data['imported_dump']
    import_completion_time = data['time']
    send_mail(
        subject='A full data dump has been imported into the Spark cluster',
        text=render_template('emails/dump_import_notification.txt', dump_name=dump_name, time=import_completion_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )


def handle_dataframes(data):
    """ Send an email after dataframes have been successfully created and uploaded to HDFS.
    """
    if current_app.config['TESTING']:
        return

    dataframe_upload_time = data['dataframe_upload_time']
    dataframe_creation_time = data['total_time']
    from_date = data['from_date']
    to_date = data['to_date']
    send_mail(
        subject='Dataframes have been uploaded to HDFS',
        text=render_template('emails/cf_recording_dataframes_upload_notification.txt', time_to_upload=dataframe_upload_time,
                             from_date=from_date, to_date=to_date, total_time=dataframe_creation_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )


def handle_missing_releases_musicbrainz(data):
    musicbrainz_id = data['musicbrainz_id']
    user = db_user.get_by_mb_id(musicbrainz_id)
    if not user:
        current_app.logger.critical("Calculted releases missing from MusicBrainz for a user that doesn't exist"
                                    " in the Postgres database: {}".format(musicbrainz_id))
        return

    current_app.logger.debug("Inserting missing releases data for {}".format(musicbrainz_id))

    missing_releases_data = data['missing_releases']
    db_missing_releases_musicbrainz.insert_user_missing_releases_data(user['id'], missing_releases_data)

    current_app.logger.debug("Missing releases data for {} inserted".format(musicbrainz_id))


def handle_model(data):
    """ Send an email after trained data (model) has been successfully uploaded to HDFS.
    """
    if current_app.config['TESTING']:
        return

    model_upload_time = data['model_upload_time']
    model_creation_time = data['total_time']
    send_mail(
        subject='Model created and successfully uploaded to HDFS',
        text=render_template('emails/cf_recording_model_upload_notification.txt', time_to_upload=model_upload_time,
                             total_time=model_creation_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )


def handle_candidate_sets(data):
    """ Send an email after candidate sets have been successfully uploaded to HDFS.
    """
    if current_app.config['TESTING']:
        return

    candidate_sets_upload_time = data['candidate_sets_upload_time']
    candidate_set_creation_time = data['total_time']
    from_date = data['from_date']
    to_date = data['to_date']
    send_mail(
        subject='Candidate sets created and successfully uploaded to HDFS',
        text=render_template('emails/cf_candidate_sets_upload_notification.txt', time_to_upload=candidate_sets_upload_time,
                             from_date=from_date, to_date=to_date, total_time=candidate_set_creation_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )


def handle_recommendations(data):
    """ Take recommended recordings for a user and save it in the db.
    """
    musicbrainz_id = data['musicbrainz_id']
    user = db_user.get_by_mb_id(musicbrainz_id)
    if not user:
        current_app.logger.critical("Generated recommendations for a user that doesn't exist in the Postgres database: {}"
                                    .format(musicbrainz_id))
        return

    current_app.logger.debug("inserting recommendation for {}".format(musicbrainz_id))
    top_artist_recording_mbids = data['top_artist']
    similar_artist_recording_mbids = data['similar_artist']

    db_recommendations_cf_recording.insert_user_recommendation(user['id'], top_artist_recording_mbids,
                                                               similar_artist_recording_mbids)

    current_app.logger.debug("recommendation for {} inserted".format(musicbrainz_id))


def notify_mapping_import(data):
    """ Send an email after msid mbid mapping has been successfully imported into the cluster.
    """
    if current_app.config['TESTING']:
        return

    mapping_name = data['imported_mapping']
    import_completion_time = data['time']
    send_mail(
        subject='MSID MBID mapping has been imported into the Spark cluster',
        text=render_template('emails/mapping_import_notification.txt', mapping_name=mapping_name, time=import_completion_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )


def notify_artist_relation_import(data):
    """ Send an email after artist relation has been sucessfully imported into the cluster.
    """
    if current_app.config['TESTING']:
        return

    artist_relation_name = data['import_artist_relation']
    import_completion_time = data['time']
    send_mail(
        subject='Artist relation has been imported into the Spark cluster',
        text=render_template('emails/artist_relation_import_notification.txt',
                             artist_relation_name=artist_relation_name, time=import_completion_time),
        recipients=['listenbrainz-observability@metabrainz.org'],
        from_name='ListenBrainz',
        from_addr='noreply@'+current_app.config['MAIL_FROM_DOMAIN'],
    )
