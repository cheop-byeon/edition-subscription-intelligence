import importlib.util,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(importlib.util.find_spec('streamlit'),'Optional analytics dependencies are not installed')
class StreamlitTests(unittest.TestCase):
    def test_all_analysis_pages_render(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'EDITION_DB':str(Path(folder)/'test.sqlite3')}):
            app=AppTest.from_file(str(ROOT/'apps/edition.py')).run(timeout=20)
            self.assertFalse(app.exception)
            for page in ['Subscribers & retention','Readers & campaigns','Paywall','Experiments','Assistant','Models & data','Research']:
                app.sidebar.radio[0].set_value(page).run(timeout=20)
                self.assertFalse(app.exception,page)

    def test_campaign_and_reader_response_share_one_store(self):
        from streamlit.testing.v1 import AppTest
        import sys
        sys.path.insert(0,str(ROOT/'backend'))
        from commercial import Commercial
        from database import Store
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'EDITION_DB':str(Path(folder)/'test.sqlite3')}):
            app=AppTest.from_file(str(ROOT/'apps/edition.py')).run(timeout=20)
            app.sidebar.radio[0].set_value('Readers & campaigns').run(timeout=20)
            next(x for x in app.text_input if x.label=='Campaign name').set_value('Unified workflow')
            next(x for x in app.button if x.label=='Save campaign draft').click().run(timeout=20)
            self.assertFalse(app.exception)
            store=Commercial(Store(Path(folder)/'test.sqlite3'))
            self.assertEqual(store.summary()['campaigns'][0]['name'],'Unified workflow')
            app.sidebar.radio[0].set_value('Paywall').run(timeout=20)
            next(x for x in app.selectbox if x.label=='Reader').set_value('reader-0009')
            next(x for x in app.button if x.label=='Evaluate decision').click().run(timeout=20)
            self.assertFalse(app.exception)
            next(x for x in app.button if x.label=='Simulate accepting offer').click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(sum(r['conversions'] for r in store.summary()['experiment']),1)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'Optional KKBOX training dependencies are not installed')
    def test_kkbox_fixture_evidence_and_prediction_in_same_app(self):
        from streamlit.testing.v1 import AppTest
        import sys
        sys.path.insert(0,str(ROOT/'backend'))
        sys.path.insert(0,str(ROOT/'tests'))
        from kkbox import import_dataset,connect
        from train_kkbox import train
        from kkbox_fixture import make_fixture
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'test.sqlite3'
            manifest=import_dataset(make_fixture(Path(folder)/'input'),db,limit=2000,source_kind='synthetic_test_fixture',progress=lambda _:None)
            train(db,manifest['id'],tracking=False,progress=lambda _:None)
            with patch.dict(os.environ,{'EDITION_DB':str(db)}):
                app=AppTest.from_file(str(ROOT/'apps/edition.py')).run(timeout=20)
                app.sidebar.radio[0].set_value('Models & data').run(timeout=20)
                self.assertFalse(app.exception)
                self.assertTrue(any('synthetic test records' in w.value for w in app.warning))
                next(x for x in app.button if x.label=='Save this churn prediction').click().run(timeout=20)
                self.assertFalse(app.exception)
                app.sidebar.radio[0].set_value('Subscribers & retention').run(timeout=20)
                self.assertFalse(app.exception)
                next(x for x in app.text_input if x.label=='Retention draft name').set_value('Research audience')
                next(x for x in app.button if x.label=='Save retention draft').click().run(timeout=20)
                self.assertFalse(app.exception)
            with connect(db) as local:
                import json
                draft=local.execute('SELECT model_id,dataset_id,members FROM retention_drafts').fetchone()
                self.assertEqual(draft['dataset_id'],manifest['id'])
                self.assertGreater(len(json.loads(draft['members'])),0)
                self.assertTrue(all(r['risk'] >= .3 for r in json.loads(draft['members'])))
                self.assertEqual(local.execute('SELECT COUNT(*) FROM kkbox_predictions').fetchone()[0],1)

    def test_simulated_experiment_analysis_renders_results(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'EDITION_DB':str(Path(folder)/'test.sqlite3')}):
            app=AppTest.from_file(str(ROOT/'apps/edition.py')).run(timeout=20)
            app.sidebar.radio[0].set_value('Experiments').run(timeout=20)
            next(x for x in app.button if x.label=='Run simulated experiment').click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any('Pearson' in x.value for x in app.markdown))
            self.assertTrue(any('Newcombe' in x.value for x in app.markdown))
            self.assertTrue(any('Synthetic experiment only' in x.value for x in app.warning))
            self.assertTrue(any('does not demonstrate real paywall uplift' in x.value for x in app.markdown))
