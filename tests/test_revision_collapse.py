"""S25 exact fixed-score paired bootstrap reproduction."""
from pathlib import Path
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CollapseTests(unittest.TestCase):
    def test_published_s25_results_and_safety(self):
        self.assertTrue((ROOT / 'analysis/revision_collapse.py').is_file())
        from analysis.revision_collapse import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            result = run(ROOT, out)
            self.assertEqual(result['verification'], 'all original bootstrap arrays and intervals reproduced')
            self.assertEqual(result['n'], 115)
            self.assertEqual(result['collapsed'], 9)
            self.assertAlmostEqual(result['scores']['epoch3_accuracy']['auc'], .9737945492662474)
            before = (out / 'complete.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run(ROOT, out)
            self.assertEqual((out / 'complete.json').read_bytes(), before)
            self.assertEqual(json.loads(before)['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
