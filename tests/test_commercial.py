import copy, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from commercial import Commercial, ARTIFACT, score
from database import Store
class CommercialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.app=Commercial(Store(Path(self.temp.name)/'db.sqlite3'))
    def tearDown(self):self.temp.cleanup()
    def test_brand_isolation_and_campaign_consent(self):
        rows=self.app.readers('VG','sales',0)
        self.assertTrue(rows);self.assertTrue(all(r['brand']=='VG' and r['eligible'] and not r['subscribed'] for r in rows))
        c=self.app.campaign({'name':'Draft','brand':'VG','kind':'sales','threshold':0})
        self.assertEqual(c['members'],len(rows));self.assertEqual(len(self.app.export(c['id']).splitlines()),len(rows)+1)
    def test_paywall_guards_and_assignment(self):
        subscriber=next(r for r in self.app.readers() if r['subscribed'])
        self.assertEqual(self.app.paywall({'reader_id':subscriber['id']})['action'],'open')
        reader=next(r for r in self.app.readers() if r['eligible'] and not r['subscribed'])
        a=self.app.paywall({'reader_id':reader['id'],'public_interest':True});self.assertEqual(a['action'],'open')
        b=self.app.paywall({'reader_id':reader['id'],'dismissals':3});self.assertEqual(b['action'],'no_offer');self.assertEqual(b['content_access'],'teaser')
    def test_outcomes_are_validated_and_not_duplicated(self):
        result=self.app.paywall({'reader_id':'reader-0002'})
        payload={'decision_id':result['decision_id'],'converted':True,'revenue':129}
        self.app.outcome(payload)
        with self.assertRaises(ValueError):self.app.outcome(payload)
        with self.assertRaises(ValueError):self.app.outcome({**payload,'revenue':float('nan')})
    def test_registry_activation_and_rollback(self):
        candidate=copy.deepcopy(ARTIFACT);candidate['version']='candidate-v2'
        id=self.app.register({'artifact':candidate})['id'];self.assertNotEqual(self.app.artifact()['version'],'candidate-v2')
        self.app.activate(id);self.assertEqual(self.app.artifact()['version'],'candidate-v2')
        self.app.activate(1);self.assertEqual(self.app.artifact()['version'],ARTIFACT['version'])
        candidate['models']['sales']['metrics']['auc']=.5
        with self.assertRaises(ValueError):self.app.register({'artifact':candidate})
    def test_reproducible_training(self):
        from train_commercial import train
        a=train(42)
        self.assertEqual(a['dataset_sha256'],ARTIFACT['dataset_sha256'])
        self.assertAlmostEqual(a['models']['sales']['metrics']['auc'],ARTIFACT['models']['sales']['metrics']['auc'],places=10)
        self.assertEqual(a['profiles'],ARTIFACT['profiles'])
    def test_scoring_and_monitoring(self):
        rows=self.app.readers();self.assertEqual(len(rows),240)
        self.assertTrue(all((r['sales'] is None and 0<=r['churn']<=1) if r['subscribed'] else (r['churn'] is None and 0<=r['sales']<=1) for r in rows))
        with self.assertRaises(ValueError):score({},ARTIFACT['models']['sales'])
        self.assertEqual(len(self.app.summary()['monitoring']['features']),6)
