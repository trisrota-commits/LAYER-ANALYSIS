"""Focused tests for the exported-data thesis runner (no GPU or database)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_inventory_covers_thesis_lists(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), '--list'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads(result.stdout)
        self.assertEqual(len(inventory['tables']), 25)
        self.assertEqual(len(inventory['figures']), 9)
        self.assertEqual(len(set(inventory['tables'])), 25)
        self.assertEqual(len(set(inventory['figures'])), 9)

    def test_existing_output_is_refused_without_touching_it(self):
        with tempfile.TemporaryDirectory(prefix='hermes-verify-runner-') as tmp:
            marker = Path(tmp) / 'keep.txt'
            marker.write_text('preserve me')
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), '--out', tmp],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Existing output preserved', result.stderr)
            self.assertEqual(marker.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
