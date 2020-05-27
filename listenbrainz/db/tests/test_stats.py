import json
import os
import listenbrainz.db.stats as db_stats
import listenbrainz.db.user as db_user

from datetime import datetime, timezone
from listenbrainz.db.testing import DatabaseTestCase


class StatsDatabaseTestCase(DatabaseTestCase):

    def setUp(self):
        DatabaseTestCase.setUp(self)
        self.user = db_user.get_or_create(1, 'stats_user')

    def test_insert_user_artists(self):
        """ Test if artist stats are inserted correctly """
        with open(self.path_to_data_file('user_top_artists_db.json')) as f:
            artists_data = json.load(f)

        db_stats.insert_user_artists(user_id=self.user['id'], artists={'all_time': artists_data})

        result = db_stats.get_all_user_stats(user_id=self.user['id'])
        self.assertDictEqual(result['artist']['all_time'], artists_data)

    def test_insert_user_releases(self):
        """ Test if release stats are inserted correctly """
        with open(self.path_to_data_file('user_top_releases_db.json')) as f:
            releases_data = json.load(f)

        db_stats.insert_user_releases(user_id=self.user['id'], releases={'all_time': releases_data})

        result = db_stats.get_all_user_stats(user_id=self.user['id'])
        self.assertDictEqual(result['release']['all_time'], releases_data)

    def test_insert_user_stats_mult_ranges(self):
        """ Test if multiple time range data is inserted correctly """
        with open(self.path_to_data_file('user_top_artists_db.json')) as f:
            artists_data = json.load(f)

        db_stats.insert_user_artists(user_id=self.user['id'], artists={'all_time': artists_data})
        db_stats.insert_user_artists(user_id=self.user['id'], artists={'year': artists_data})

        result = db_stats.get_user_artists(1)

        self.assertDictEqual(result['artist']['all_time'], artists_data)
        self.assertDictEqual(result['artist']['year'], artists_data)

    def test_insert_user_stats_mult_ranges(self):
        """ Test if multiple time range data is inserted correctly """
        with open(self.path_to_data_file('user_top_releases_db.json')) as f:
            releases_data = json.load(f)

        db_stats.insert_user_releases(user_id=self.user['id'], releases={'all_time': releases_data})
        db_stats.insert_user_releases(user_id=self.user['id'], releases={'year': releases_data})

        result = db_stats.get_all_user_stats(1)

        self.assertDictEqual(result['release']['all_time'], releases_data)
        self.assertDictEqual(result['release']['year'], releases_data)

    def insert_test_data(self):
        """ Insert test data into the database """

        with open(self.path_to_data_file('user_top_artists_db.json')) as f:
            artists = json.load(f)
        with open(self.path_to_data_file('user_top_releases_db.json')) as f:
            releases = json.load(f)

        db_stats.insert_user_artists(self.user['id'], {'all_time': artists})
        db_stats.insert_user_releases(self.user['id'], {'all_time': releases})

        return {
            'user_artists': artists,
            'user_releases': releases,
        }

    def test_get_timestamp_for_last_user_stats_update(self):
        ts = datetime.now(timezone.utc)
        self.insert_test_data()
        received_ts = db_stats.get_timestamp_for_last_user_stats_update()
        self.assertGreaterEqual(received_ts, ts)

    def test_get_user_stats(self):
        data_inserted = self.insert_test_data()

        data = db_stats.get_user_stats(self.user['id'], 'artist')
        self.assertDictEqual(data['artist']['all_time'], data_inserted['user_artists'])

        data = db_stats.get_user_stats(self.user['id'], 'release')
        self.assertDictEqual(data['release']['all_time'], data_inserted['user_releases'])

    def test_get_user_artists(self):
        data_inserted = self.insert_test_data()
        result = db_stats.get_all_user_stats(self.user['id'])

        self.assertDictEqual(result['artist']['all_time'], data_inserted['user_artists'])

    def test_get_all_user_stats(self):
        data_inserted = self.insert_test_data()
        result = db_stats.get_all_user_stats(self.user['id'])
        self.assertDictEqual(result['artist']['all_time'], data_inserted['user_artists'])
        self.assertDictEqual(result['release']['all_time'], data_inserted['user_releases'])
        self.assertGreater(int(result['last_updated'].strftime('%s')), 0)

    def test_valid_stats_exist(self):
        self.assertFalse(db_stats.valid_stats_exist(self.user['id'], 7))
        self.insert_test_data()
        self.assertTrue(db_stats.valid_stats_exist(self.user['id'], 7))

    def test_delete_user_stats(self):
        self.assertFalse(db_stats.valid_stats_exist(self.user['id'], 7))
        self.insert_test_data()
        data = db_stats.delete_user_stats(self.user['id'])
        self.assertFalse(db_stats.valid_stats_exist(self.user['id'], 7))
