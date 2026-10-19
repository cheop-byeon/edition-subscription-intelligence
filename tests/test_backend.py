import json, math, random, subprocess, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from engine import decision, validate, DEFAULT, BRANDS, blocked
from train import dr_scores, generate, arrays, auc
MODEL=json.loads((ROOT/'artifacts/model.json').read_text())
class BackendTests(unittest.TestCase):
    def test_guards_override_high_propensity(self):
        for field,value in [('consent',False),('subscribed',True),('public_interest',True),('dismissals',3)]:
            d=decision(dict(DEFAULT,active_days=28,articles=120,premium_reads=40,**{field:value}),MODEL)
            self.assertEqual(d['action']['id'],'open');self.assertEqual(d['incremental_value'],0);self.assertTrue(d['guardrails']);self.assertIsNone(d['action']['probability']);self.assertEqual(d['feature_contributions'],[])
    def test_schema_rejects_bad_inputs(self):
        for bad in [[],None,{'active_days':float('nan')},{'articles':-1},{'premium_reads':30,'articles':2},{'active_days':True},{'consent':'false'},{'brand':'unknown'},{'active_days':1.5},{'unknown':1},{'brand':{}},{'brand':[]}]:
            with self.assertRaises(ValueError):validate(bad)
    def test_value_not_conversion_is_objective(self):
        d=decision(dict(DEFAULT,active_days=20,articles=55,premium_reads=20,recency_hours=4),MODEL)
        self.assertEqual(d['action']['value'],max(a['value'] for a in d['alternatives']))
        self.assertNotEqual(d['action']['id'],max(d['alternatives'],key=lambda x:x['probability'])['id'])
    def test_python_worker_parity(self):
        rng=random.Random(42);profiles=[]
        for _ in range(120):
            a=rng.randint(0,120);profiles.append(dict(DEFAULT,brand=rng.choice(BRANDS),active_days=rng.randint(0,28),articles=a,premium_reads=rng.randint(0,min(a,40)),recency_hours=rng.randint(0,168),dismissals=rng.randint(0,5),consent=rng.choice([True,False]),public_interest=rng.choice([True,False]),subscribed=rng.choice([True,False])))
        proc=subprocess.run(['node',str(ROOT/'tests/parity.mjs')],input=json.dumps(profiles),capture_output=True,text=True,check=True)
        js=json.loads(proc.stdout)
        for p,j in zip(profiles,js):
            py=decision(p,MODEL);self.assertEqual(py['action']['id'],j['action']['id']);self.assertEqual(py['guardrails'],j['guardrails'])
            for a,b in zip(py['alternatives'],j['alternatives']):
                if a['probability'] is None:self.assertIsNone(b['probability']);self.assertIsNone(b['value'])
                else:self.assertAlmostEqual(a['probability'],b['probability'],places=12);self.assertAlmostEqual(a['value'],b['value'],places=10)
    def test_dr_ips_on_balanced_complete_log(self):
        # Each arm is logged once for each context; fractional outcomes represent
        # exact expectation, making the policy value known without sampling noise.
        prob=np.tile([[.1,.3,.5,.4]],(8,1));arm=np.tile(np.arange(4),2);y=prob[np.arange(8),arm];target=np.full(8,2);ps=np.full(8,.25)
        dr,ips=dr_scores(prob,arm,y,ps,target);expected=.5*39-.8
        self.assertAlmostEqual(dr.mean(),expected);self.assertAlmostEqual(ips.mean(),expected)
        # DR remains unbiased with a deliberately wrong reward model if logging
        # probabilities are correct and the complete balanced log is retained.
        wrong=np.tile([[.2,.2,.2,.2]],(8,1));dr,_=dr_scores(wrong,arm,y,ps,target);self.assertAlmostEqual(dr.mean(),expected)
    def test_synthetic_assignment_and_isolation(self):
        rows=generate(4200);self.assertEqual(len({r['reader_id'] for r in rows}),len(rows))
        for r in rows:
            self.assertEqual(r['propensity'],1 if blocked(r) else .25)
            if blocked(r):self.assertEqual(r['action'],0)
        self.assertLess(MODEL['train_window'][1],MODEL['validation_window'][0]);self.assertLess(MODEL['validation_window'][1],MODEL['test_window'][0])
    def test_metrics_sane(self):
        r=MODEL['reports']['all'];self.assertGreater(r['auc'],.65);self.assertLess(r['brier'],.2)
        self.assertEqual(sum(a['n'] for a in r['policy_mix']),r['n']);self.assertEqual(sum(a['n'] for a in r['randomised_arms']),r['eligible_n']);self.assertGreaterEqual(r['policy_mix'][0]['n'],r['blocked_n'])
    def test_auc_ties(self):self.assertEqual(auc(np.array([0,1,0,1]),np.array([.5,.5,.5,.5])),.5)
if __name__=='__main__':unittest.main(verbosity=2)
