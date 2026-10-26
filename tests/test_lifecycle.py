import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from lifecycle import features,label,generate,materialize,fingerprint
class LifecycleTests(unittest.TestCase):
    def test_future_and_cutoff_events_cannot_leak(self):
        events=[{'date':'2024-01-30','type':'visit'},{'date':'2024-02-01','type':'visit'},{'date':'2024-02-04','type':'premium_read'}]
        v=features(events,'2024-02-01','2023-01-01');self.assertEqual(v['visits_30d'],1);self.assertEqual(v['premium_reads_30d'],0)
    def test_labels_and_censoring(self):
        events=[{'date':'2024-02-01','type':'subscribe'},{'date':'2024-03-02','type':'cancel'}]
        self.assertEqual(label(events,'2024-02-01','sales','2024-03-02'),1)
        self.assertEqual(label(events,'2024-02-01','churn','2024-03-02'),0)
        with self.assertRaises(ValueError):label(events,'2024-02-01','sales','2024-02-20')
    def test_temporal_training_and_exported_inference(self):
        from train_lifecycle import train
        from commercial import score
        a=train(42)
        for model in a['models'].values():
            self.assertGreater(model['metrics']['auc'],.5)
            self.assertLess(model['metrics']['brier'],model['metrics']['constant_brier'])
            self.assertTrue(0<=score(a['profiles'][0]['values'],model)<=1)
        self.assertIn('data_contract',a)
    def test_temporal_embargo_and_reproducible_data(self):
        raw=generate(42);rows=materialize(raw)
        self.assertEqual(len(rows),12000);self.assertEqual(len({r['id'] for r in rows}),12000)
        self.assertLess(max(r['observed_until'] for r in rows if r['partition']=='train'),min(r['snapshot'] for r in rows if r['partition']=='validation'))
        self.assertEqual(fingerprint(raw),fingerprint(generate(42)))
