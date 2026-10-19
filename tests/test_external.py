import json, math, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from external_engine import predict_export, score
PATH=ROOT/'artifacts/external_model.json'
@unittest.skipUnless(PATH.exists(),'External training artifact not available')
class ExternalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.artifact=json.loads(PATH.read_text())
    def test_record_inference_matches_training_predictions(self):
        for p in self.artifact['profiles']:
            self.assertAlmostEqual(predict_export(p['values'],self.artifact['model']),p['prediction'],places=10)
    def test_evaluation_partitions(self):
        a=self.artifact;s=a['splits'];self.assertEqual(sum(p['n'] for p in s.values()),a['rows_after_deduplication'])
        self.assertEqual(a['original_rows']-a['duplicate_rows_removed'],a['rows_after_deduplication'])
        for l,r in [('train','selection'),('selection','calibration'),('calibration','test')]:self.assertLess(s[l]['end'],s[r]['start'])
        self.assertNotIn('date_converted',a['features']);self.assertNotIn('subscription',a['features'])
    def test_external_improves_over_constant(self):
        a=self.artifact;self.assertLess(a['metrics']['logloss'],a['baselines']['constant']['logloss']);self.assertGreater(a['metrics']['average_precision'],a['metrics']['prevalence'])
    def test_schema_and_nonfinite_rejected(self):
        a=self.artifact;p=dict(a['profiles'][0]['values']);k=a['features'][0]
        for bad in [[],{},dict(p,unknown=1),dict(p,**{k:True}),dict(p,**{k:float('inf')}),dict(p,**{k:'123'})]:
            with self.assertRaises(ValueError):predict_export(bad,a['model'])
    def test_missing_and_negative_values_imputed(self):
        a=self.artifact;p=dict(a['profiles'][0]['values']);k=a['features'][0]
        neg=predict_export(dict(p,**{k:-99}),a['model']);missing=predict_export(dict(p,**{k:None}),a['model'])
        self.assertEqual(neg,missing);self.assertTrue(0<=neg<=1)
    def test_provenance_and_scope(self):
        a=self.artifact;self.assertEqual(len(a['archive_sha256']),64);self.assertIn('synthetic',a['target']);self.assertGreaterEqual(len(a['limitations']),4)
        r=score(a['profiles'][0]['values'],a);self.assertNotIn('offer',r);self.assertNotIn('incremental_value',r)
if __name__=='__main__':unittest.main(verbosity=2)
