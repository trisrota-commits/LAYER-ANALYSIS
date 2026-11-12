"""CLI safety tests for explicit CPU-only revision commands."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RevisionRunnerTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), *args],
                              capture_output=True, text=True)

    def test_revision_commands_are_discoverable(self):
        result = self.call('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--revision', result.stdout)
        self.assertIn('loss-confidence', result.stdout)
        self.assertIn('classical-no-taylor', result.stdout)
        self.assertIn('report', result.stdout)

    def test_existing_revision_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'keep'; marker.write_text('preserve')
            result = self.call('--revision', 'loss-confidence', '--archive', 'missing.tar.gz', '--out', temp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Existing output preserved', result.stderr)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_missing_inputs_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = str(Path(temp) / 'new')
            for mode in ['loss-confidence', 'lemur-paired', 'classical-no-taylor']:
                result = self.call('--revision', mode, '--out', out)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--archive', result.stderr)
                self.assertFalse(Path(out).exists())
            result = self.call('--revision', 'report', '--out', out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--inputs', result.stderr)

    def test_revision_cannot_silently_override_legacy_mode(self):
        for flag in ['--list', '--check']:
            result = self.call('--revision', 'nb201-qwen', flag, '--out', 'unused')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cannot be combined', result.stderr)


if __name__ == '__main__':
    unittest.main()
