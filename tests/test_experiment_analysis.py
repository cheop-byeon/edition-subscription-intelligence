import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from experiment_analysis import analyze, sample_size, simulate


class ExperimentTests(unittest.TestCase):
    def test_known_chi_square_and_newcombe_interval(self):
        # Independent reference: Pearson 2x2 table [[40,85],[24,96]].
        result = analyze(24,120,40,125)
        self.assertAlmostEqual(result['chi_square'],4.568370165745858,places=7)
        self.assertAlmostEqual(result['pvalue'],.03256753591515619,places=5)
        self.assertAlmostEqual(result['difference_ci'][0],.0098,places=4)
        self.assertAlmostEqual(result['difference_ci'][1],.2260,places=4)
        self.assertTrue(result['significant'])

    def test_null_symmetry_sparse_and_invalid_counts(self):
        self.assertEqual(analyze(10,100,10,100)['pvalue'],1)
        a,b=analyze(10,100,20,100),analyze(20,100,10,100)
        self.assertAlmostEqual(a['pvalue'],b['pvalue'])
        self.assertAlmostEqual(a['difference_ci'][0],-b['difference_ci'][1])
        self.assertIsNone(analyze(0,100,0,100)['pvalue'])
        with self.assertRaises(ValueError):analyze(101,100,2,100)

    def test_power_and_randomization_contract(self):
        n=sample_size(.1,.015)
        self.assertGreater(n,6000);self.assertLess(n,7000)
        self.assertGreater(sample_size(.1,.01),n)
        self.assertGreater(sample_size(.1,.015,power=.9),n)
        rows=simulate(200,.1,.02,42)
        self.assertEqual(rows,simulate(200,.1,.02,42))
        self.assertEqual(len({r['reader_id'] for r in rows}),400)
        self.assertEqual(sum(r['arm']=='control' for r in rows),200)
        self.assertTrue(all(r['outcome_complete'] for r in rows))
