from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LemurTests(unittest.TestCase):
    def test_main_matrix_uses_six_seeds_and_exact_targets(self):
        path = ROOT / 'analysis/thesis_lemur.py'
        self.assertTrue(path.exists(), 'LEMUR adapter missing')
        spec = importlib.util.spec_from_file_location('lemur_test', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        data, target, sources = m.load_main(ROOT)
        self.assertEqual(len(data), 36)
        self.assertEqual(len(target), 115)
        self.assertTrue(all(p.shape == (6, 115) for p in data.values()))
        values = m.metric_function(ROOT)(data[(3, 'off', 'taylor')], target).mean(axis=0)
        self.assertAlmostEqual(values[0], 8.474931819445873, places=10)
        self.assertAlmostEqual(values[1], 5.8986503304347835, places=10)

    def test_descriptive_table_coverage(self):
        from analysis import thesis_lemur as m
        self.assertTrue(hasattr(m, 'build_tables'), 'table adapter is missing')
        tables, sources = m.build_tables(ROOT)
        self.assertEqual(set(tables), {'6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.9', '6.12', 'A.1'})
        self.assertEqual(len(tables['6.4']), 36)
        self.assertEqual(len(tables['6.2']), 8)
        self.assertEqual(len(tables['6.12']), 8)
        self.assertAlmostEqual(tables['6.5'][0]['low'], 0.12508218045280545, places=9)


if __name__ == '__main__':
    unittest.main()
