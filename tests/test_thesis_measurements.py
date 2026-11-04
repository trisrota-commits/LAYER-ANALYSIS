from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MeasurementTests(unittest.TestCase):
    def test_dependence_and_timing_export_contract(self):
        self.assertTrue((ROOT / 'analysis/thesis_measurements.py').exists(), 'measurement adapter missing')
        from analysis.thesis_measurements import build_tables
        tables, cells = build_tables(ROOT)
        self.assertEqual(set(tables), {'6.1', '6.13'})
        self.assertEqual(len(cells), 196)
        timing = {r['operation']: r for r in tables['6.13']}
        self.assertEqual(timing['taylor_importance']['n'], 96)
        self.assertAlmostEqual(timing['taylor_importance']['seconds'], .49419544586756575, places=12)
        self.assertEqual(timing['zcp_compute']['shared_layer_candidates'], 93)
        early = next(r for r in tables['6.1'] if r['stream'] == 'early_to_epoch50' and r['feature'] == 'Sensitivity')
        self.assertEqual(early['significant'], 18)
        self.assertAlmostEqual(early['MIC'], .231, places=3)


if __name__ == '__main__':
    unittest.main()
