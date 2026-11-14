"""Saved-prediction initialization table coverage; no calibration refitting."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class InitializationTests(unittest.TestCase):
    def test_complete_paired_and_descriptive_exports(self):
        self.assertTrue((ROOT / 'analysis/revision_initialization.py').is_file())
        from analysis import revision_initialization as module
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            with patch.object(module, 'DRAWS', 64):
                module.run(ROOT, out)
            data = json.loads((out / 'results.json').read_text())
            self.assertEqual(len(data['paired']), 35)
            self.assertEqual(len(data['descriptive']), 50)
            self.assertEqual({r['table'] for r in data['paired']}, {'S41', 'S43', 'S45'})
            self.assertEqual({r['metric'] for r in data['paired']}, set(module.METRICS))
            self.assertTrue(all(r['n'] == 86 and r['draws'] == 64 for r in data['paired']))
            self.assertTrue(all(r['low'] <= r['high'] for r in data['paired']))
            receipt = json.loads((out / 'complete.json').read_text())
            self.assertEqual(receipt['status'], 'complete')
            from analysis.thesis_common import sha
            for name, digest in receipt['files_sha256'].items():
                self.assertEqual(sha(out / name), digest)
            before = (out / 'complete.json').read_bytes()
            with self.assertRaises(FileExistsError):
                module.run(ROOT, out)
            self.assertEqual((out / 'complete.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
