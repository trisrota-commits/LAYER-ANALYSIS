"""Focused coverage for prediction-only descriptive tables."""
from pathlib import Path
import importlib
import unittest


class DescriptiveTests(unittest.TestCase):
    def test_seed_intervals_preserve_each_seed(self):
        import numpy as np
        from scipy.stats import t
        m = importlib.import_module('analysis.revision_descriptive')
        repo = Path(__file__).resolve().parents[1]
        p = np.array([[1., 3., 2., 5.], [2., 3., 4., 4.], [1., 2., 3., 4.]])
        y = np.array([1., 2., 3., 4.])
        rows = m.seed_rows(repo, p, y, [42, 42, 42], 'pilot', ['a', 'b', 'c'])
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row['seeds'], [42, 42, 42])
            self.assertEqual(row['interval_kind'], 'student_t95_repetitions')
            values = np.array(row['seed_values'])
            self.assertAlmostEqual(row['mean'], values.mean())
            self.assertAlmostEqual(row['high']-row['mean'], t.ppf(.975, 2)*values.std(ddof=1)/np.sqrt(3))

    def test_real_inputs_complete_tables_and_original_baseline_protocol(self):
        import json
        import tempfile
        import numpy as np
        from analysis import revision_descriptive as m
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'descriptive'
            receipt = m.run(repo, out)
            expected = {'S2': 4, 'S3': 4, 'S4': 4, 'S10': 9, 'S11': 9,
                        'S12': 9, 'S13': 9, 'S14': 2, 'S15': 3, 'S17': 5,
                        'S18': 2, 'S20': 4, 'S22': 6, 'MainIV': 9, 'MainVII': 4}
            self.assertEqual(set(receipt['tables']), set(expected) | {'S16'})
            for table, groups in expected.items():
                rows = json.loads((out / f'table_{table}.json').read_text())
                self.assertEqual(len(rows), groups*5, table)
                self.assertEqual(len({r['group'] for r in rows}), groups, table)
                for group in {r['group'] for r in rows}:
                    self.assertEqual({r['metric'] for r in rows if r['group'] == group}, set(m.lemur.METRICS))
                for row in rows:
                    if row['interval_kind'] == 'student_t95_seeds':
                        self.assertEqual(row['seeds'], list(range(42, 48)))
                        self.assertEqual(len(row['sources']), 6)
                    self.assertTrue(np.isfinite([row['mean'], row['low'], row['high']]).all())
                self.assertTrue((out / f'table_{table}.csv').exists())
                self.assertTrue((out / f'table_{table}.md').exists())
            timing = json.loads((out / 'table_S16.json').read_text())
            self.assertEqual(len(timing), 13)  # eleven table rows plus two caption costs
            zero = next(r for r in timing if r['group'] == 'no-layer')
            self.assertEqual(zero['seconds'], 0.)
            self.assertEqual(zero['interval_kind'], 'defined_zero_additional_cost')
            s4 = json.loads((out / 'table_S4.json').read_text())
            # Independent frozen report oracle, not a rounded PDF comparison.
            pools = m.lemur.pools(repo)
            _, blob = m.lemur.get(pools['baseline'], '/xgboost_report.json')
            xgb = json.loads(blob)['methods']
            for obs in [2, 3]:
                for row in [r for r in s4 if r['group'] == f'XGBoost [{obs}]']:
                    ref = xgb[f'xgboost_{obs}acc']['metrics'][row['metric']]
                    for actual, key in [('mean', 'estimate'), ('low', 'lower'), ('high', 'upper')]:
                        self.assertAlmostEqual(row[actual], ref[key], places=12)
            for proxy in ['synflow', 'nwot']:
                _, blob = m.lemur.get(pools['baseline'], f'/{proxy}_report.json')
                archived = json.loads(blob)['metrics']
                for row in [r for r in s4 if r['group'] == proxy]:
                    ref = archived[row['metric']]
                    for actual, key in [('mean', 'estimate'), ('low', 'lower'), ('high', 'upper')]:
                        self.assertAlmostEqual(row[actual], ref[key], places=12)
                    self.assertEqual(row['valid_resamples'], ref['valid_resamples'])
            self.assertTrue(receipt['input_preservation_verified'])
            for name, digest in receipt['output_sha256'].items():
                self.assertEqual(m.sha(out / name), digest)
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                m.run(repo, out)
            print('VERIFIED TABLE GROUP COUNTS', {k: v['groups'] for k, v in receipt['tables'].items()})

    def test_changed_frozen_input_refused_before_output(self):
        import json
        import tempfile
        from analysis import revision_descriptive as m
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            names = [m.ARCHIVE, 'evidence/reproduce_main_paired_tables.py']
            names += [f'evidence/two-experiments-final-v1/{model}-s{s}-predictions.csv'
                      for model in ['epoch1', 'ridge', 'random_forest', 'hist_gradient_boosting'] for s in m.SEEDS]
            for name in names:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'changed')
            (repo / 'INPUT_MANIFEST.json').write_text(json.dumps({'files': dict.fromkeys(names, '0'*64)}))
            with self.assertRaisesRegex(ValueError, 'Frozen descriptive input changed'):
                m.run(repo, repo / 'output')
            self.assertFalse((repo / 'output').exists())


if __name__ == '__main__':
    unittest.main()
