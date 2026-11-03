"""Focused tests for the NB201 fast-tier adapter; no training data writes."""
import importlib.util
import unittest
from pathlib import Path

class AdapterTest(unittest.TestCase):
    def test_panel_pairing_rejects_missing_or_duplicate_predictions(self):
        import evaluate as module
        self.assertTrue(hasattr(module, 'analyze_panel'), 'Panel evaluation is not implemented')
        records = [dict(arch_index=i, id=str(i), best_accuracy=30. + i, arch_family=str(i // 2)) for i in range(6)]
        rows = [dict(idx=r['arch_index'], gt_best_accuracy=r['best_accuracy'],
                     pred_best_accuracy=r['best_accuracy'] + error, condition=cond, seed=seed)
                for r in records for seed in range(42, 48)
                for cond, error in [('no-layer', 2.), ('taylor', 1.)]]
        metrics, results, draws = module.analyze_panel(rows, records, 100)
        self.assertEqual(results['all']['improvement'], 1.)
        self.assertEqual(len(metrics), 24)
        with self.assertRaises(AssertionError):
            module.analyze_panel(rows[:-1], records, 100)
        with self.assertRaises(AssertionError):
            module.analyze_panel(rows + rows[:1], records, 100)

    def test_paired_mae_sign_and_reproducibility(self):
        import evaluate as module
        self.assertTrue(hasattr(module, 'paired_mae'), 'Paired uncertainty is not implemented')
        y = module.np.array([30., 40., 50., 60.])
        control = module.np.tile(y + 2, (6, 1))
        taylor = module.np.tile(y + 1, (6, 1))
        result, draws = module.paired_mae(y, control, taylor, ['a', 'a', 'b', 'b'], 100)
        self.assertEqual(result['improvement'], 1.)
        self.assertEqual(result['record_ci95'], [1., 1.])
        self.assertEqual(result['family_ci95'], [1., 1.])
        again, repeated = module.paired_mae(y, control, taylor, ['a', 'a', 'b', 'b'], 100)
        self.assertEqual(result, again)
        module.np.testing.assert_array_equal(draws, repeated)

    def test_frozen_cohorts_and_matched_inputs(self):
        path = Path(__file__).with_name('evaluate.py')
        self.assertTrue(path.exists(), 'NB201 evaluation adapter is not implemented')
        spec = importlib.util.spec_from_file_location('evaluate', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = path.parent.parent
        for dataset in ('cifar10', 'cifar100'):
            records, cohort, sources = module.load_records(root, dataset)
            self.assertEqual(len(records), 1100)
            train, val, test = module.panel.split_by_family(records, 2026, frac=module.panel.fracs(.25))
            for name, rows in [('train', train), ('val', val), ('test', test)]:
                self.assertEqual({r['arch_index'] for r in rows}, set(cohort['splits'][name]))
            plain, _ = module.panel.featurize(records, ())
            taylor, _ = module.panel.featurize(records, ('taylor_imp',))
            self.assertEqual(plain.shape, (1100, 5))
            self.assertEqual(taylor.shape, (1100, 8))
            module.np.testing.assert_array_equal(plain, taylor[:, :5])
            self.assertTrue(module.np.isfinite(taylor).all())
            self.assertTrue(sources)

if __name__ == '__main__':
    unittest.main()
