import importlib.util
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'tests'))
from kkbox import (FEATURES, feature_rows, feature_sql, import_dataset, predict,
                   save_prediction, split_for, status, connect)
from kkbox_fixture import make_fixture


class KKBOXDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = make_fixture(Path(self.temp.name) / 'input')
        self.db = Path(self.temp.name) / 'test.sqlite3'

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return import_dataset(self.folder, self.db, limit=1500,
                              source_kind='synthetic_test_fixture', progress=lambda _: None)

    def test_sql_features_exclude_future_outcomes_and_keep_original_labels(self):
        manifest = self.load()
        self.assertEqual(manifest['snapshot'], '2017-03-01')
        self.assertEqual(manifest['post_cutoff_transactions_excluded'], 1500)
        self.assertEqual(manifest['stored_transactions'], 3000)
        rows = feature_rows(self.db, manifest['id'])
        self.assertEqual(len(rows), 1500)
        for row in rows:
            self.assertEqual(set(row['values']), set(FEATURES))
            self.assertEqual(row['values']['transactions_90d'], 2)
            self.assertEqual(row['values']['last_list_price'], 149)
            self.assertEqual(row['values']['last_cancel_flag'], 0)
            self.assertEqual(row['values']['days_since_transaction'], 14)
            self.assertEqual(row['values']['days_to_expiry'], -1)
            self.assertEqual(row['split_name'], split_for(row['msno']))
        self.assertEqual({r['is_churn'] for r in rows}, {0, 1})

    def test_reimport_is_idempotent_and_exact_duplicates_are_removed(self):
        # Duplicate records in the second source must not double feature counts.
        (self.folder / 'transactions_v2.csv').write_bytes((self.folder / 'transactions.csv').read_bytes())
        first = self.load()
        second = self.load()
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(first['exact_duplicates_removed'], 3000)
        self.assertEqual(len(status(self.db)['datasets']), 1)
        with connect(self.db) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM kkbox_transactions').fetchone()[0], 3000)

    def test_invalid_input_does_not_replace_existing_dataset(self):
        first = self.load()
        with (self.folder / 'transactions.csv').open('a') as handle:
            handle.write('SYNTHETIC-TEST-ONLY-000001,20170220,20170320,30,nan,129,1,0\n')
        # All rows are sampled when the limit exceeds the fixture size.
        with self.assertRaisesRegex(ValueError, 'Invalid numeric'):
            import_dataset(self.folder, self.db, limit=2500, source_kind='synthetic_test_fixture', progress=lambda _: None)
        self.assertEqual([x['id'] for x in status(self.db)['datasets']], [first['id']])

    def test_engagement_benchmark_is_not_accepted_as_original_churn_data(self):
        (self.folder / 'train_v2.csv').rename(self.folder / 'kkbox_engagement_labeled_dataset.csv')
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            self.load()

    def test_prepared_sql_matches_executed_feature_definition(self):
        normalise = lambda sql: ' '.join(sql.split())
        self.assertEqual(normalise((ROOT / 'sql/kkbox_features.sql').read_text()), normalise(feature_sql() + ';'))
        self.assertEqual(normalise((ROOT / 'sql/kkbox_snowflake_features.sql').read_text()),
                         normalise('CREATE OR REPLACE VIEW KKBOX_FEATURES AS\n' + feature_sql('snowflake') + ';'))


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install requirements-kkbox.txt for model tests')
class KKBOXTrainingTests(unittest.TestCase):
    def test_train_export_and_persist_prediction(self):
        from train_kkbox import train
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'test.sqlite3'
            source = make_fixture(Path(folder) / 'input')
            manifest = import_dataset(source, db, limit=2000, source_kind='synthetic_test_fixture', progress=lambda _: None)
            result = train(db, manifest['id'], tracking=False, progress=lambda _: None)
            artifact = result['artifact']
            self.assertEqual(artifact['source_kind'], 'synthetic_test_fixture')
            self.assertGreater(artifact['metrics']['auc'], .6)
            self.assertLess(artifact['metrics']['logloss'], artifact['constant_baseline']['logloss'])
            self.assertGreater(artifact['export_verified_rows'], 20)
            self.assertEqual(artifact['metrics']['rows'], artifact['partitions']['test']['n'])
            records = feature_rows(db, manifest['id'])
            record = next(r for r in records if r['split_name'] == 'test')
            stored = save_prediction(db, result['model_id'], record['msno'])
            self.assertAlmostEqual(stored['probability'], predict(record['values'], artifact), places=12)
            with self.assertRaisesRegex(ValueError, 'finite'):
                predict({**record['values'], 'last_list_price': float('nan')}, artifact)
            self.assertEqual(status(db)['model']['id'], result['model_id'])

    def test_snowflake_parity_verifier_checks_values_labels_and_identities(self):
        from kkbox_warehouse import verify
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'test.sqlite3'
            source = make_fixture(Path(folder) / 'input')
            manifest = import_dataset(source, db, limit=500, source_kind='synthetic_test_fixture', progress=lambda _: None)
            records = feature_rows(db, manifest['id'])
            class Cursor:
                description = [(x.upper(),) for x in ['dataset_id', 'msno', 'split_name', 'is_churn'] + FEATURES]
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def execute(self, sql, params): self.params = params
                def __iter__(self):
                    return iter([(manifest['id'], r['msno'], r['split_name'], r['is_churn'], *[Decimal(str(r['values'][k])) for k in FEATURES]) for r in records])
            class Connection:
                def cursor(self): return Cursor()
            self.assertEqual(verify(Connection(), db, manifest['id'])['rows_verified'], 500)
            records[0]['values']['last_list_price'] += 1
            with self.assertRaisesRegex(ValueError, 'feature parity'):
                verify(Connection(), db, manifest['id'])
