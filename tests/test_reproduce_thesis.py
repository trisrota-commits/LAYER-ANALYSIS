"""Focused tests for the exported-data thesis runner (no GPU or database)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import hashlib
import importlib.util
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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

    def test_preflight_uses_only_the_minimal_input_manifest(self):
        spec = importlib.util.spec_from_file_location('minimal_runner_test', ROOT / 'run_all.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory(prefix='hermes-verify-minimal-') as tmp:
            root = Path(tmp)
            data = b'value\n1\n'
            (root / 'sample.csv').write_bytes(data)
            files = {'sample.csv': hashlib.sha256(data).hexdigest()}
            (root / 'INPUT_MANIFEST.json').write_text(json.dumps({'schema': 1, 'files': files}))
            with patch.object(runner, 'ROOT', root):
                try:
                    result = runner.preflight()
                except Exception as error:
                    self.fail('Minimal preflight failed: ' + str(error))
            self.assertEqual(result['input_sha256'], files)


if __name__ == '__main__':
    unittest.main()
