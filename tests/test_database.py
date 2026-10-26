import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from database import Store

class DatabaseTests(unittest.TestCase):
    def test_restart_import_idempotency_and_prediction_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'edition.sqlite3';store=Store(path)
            profile={'id':'row-1','label':1,'date':'2017-01-01','values':{'feature':3}}
            store.seed([profile],'downloaded_benchmark');store.seed([profile],'downloaded_benchmark')
            result={'model_version':'v1','probability':0.4}
            self.assertEqual(store.record(profile['values'],result),1)
            restarted=Store(path)
            self.assertEqual(restarted.summary()['profiles'],1)
            self.assertEqual(restarted.summary()['predictions'],1)
            self.assertEqual(restarted.profiles()[0]['values'],profile['values'])
            with restarted.connect() as db:
                row=db.execute('SELECT model_version,probability,features FROM predictions').fetchone()
            self.assertEqual(row[:2],('v1',0.4));self.assertEqual(json.loads(row[2]),profile['values'])

    def test_dashboard_remains_readable_during_background_import(self):
        from commercial import Commercial
        from kkbox import connect, status
        import sqlite3
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'edition.sqlite3'
            Commercial(Store(path))
            with connect(path):
                pass
            original_connect=sqlite3.connect
            def short_timeout(*args, **kwargs):
                kwargs['timeout']=0.05
                return original_connect(*args, **kwargs)
            with original_connect(path) as writer:
                writer.execute('BEGIN IMMEDIATE')
                writer.execute("INSERT INTO commercial_events(kind,payload) VALUES ('pending','{}')")
                with patch('sqlite3.connect',side_effect=short_timeout):
                    app=Commercial(Store(path))
                    self.assertGreater(app.summary()['readers'],0)
                    self.assertIsNone(status(path)['model'])
                writer.rollback()
