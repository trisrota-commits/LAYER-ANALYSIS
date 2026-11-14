"""Fail-closed checks for the loss/confidence export analysis."""
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import importlib.util
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RevisionLossConfidenceTests(unittest.TestCase):
    def test_explicit_bonferroni40_cli_routes_to_original_analysis(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('loss-confidence-bonferroni40', result.stdout)
        self.assertNotIn('loss-confidence-all-pairs', result.stdout)
        spec = importlib.util.spec_from_file_location('runner_bonf40_test', ROOT / 'run_all.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        archive = ROOT / 'evidence/loss-confidence-all-metrics-inputs-v1.tar.gz'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            argv = ['run_all.py', '--revision', 'loss-confidence-bonferroni40',
                    '--archive', str(archive), '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch('analysis.revision_loss_confidence.run') as run:
                runner.main()
                run.assert_called_once_with(archive.resolve(), out.resolve())
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'),
                                     '--revision', 'loss-confidence-bonferroni40', '--out', str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires --archive', result.stderr)
            self.assertFalse(out.exists())

    def test_existing_output_is_preserved_before_reading_inputs(self):
        self.assertTrue((ROOT / 'analysis/revision_loss_confidence.py').is_file())
        from analysis import revision_loss_confidence as lc
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'keep'; out.mkdir()
            marker = out / 'sentinel'; marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'Existing output'):
                lc.run(Path(temp) / 'missing.tar.gz', out)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_bad_archive_checksum_rejected_without_output(self):
        self.assertTrue((ROOT / 'analysis/revision_loss_confidence.py').is_file())
        from analysis import revision_loss_confidence as lc
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'bad.tar.gz'; archive.write_bytes(b'not the frozen archive')
            out = Path(temp) / 'output'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                lc.run(archive, out)
            self.assertFalse(out.exists())

    def test_metrics_match_shared_formulas(self):
        self.assertTrue((ROOT / 'analysis/revision_loss_confidence.py').is_file())
        from analysis import revision_loss_confidence as lc
        from analysis.revision_common import metrics
        y = np.array([20., 25., 25., 50., 80.])
        pred = np.array([[21., 24., 30., 45., 75.], [22., 22., 22., 44., 70.]])
        np.testing.assert_allclose(lc.metrics(y, pred), metrics(y, pred)[:, [1, 0, 2, 3, 4]], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
