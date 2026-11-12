"""Fail-closed checks for the loss/confidence export analysis."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RevisionLossConfidenceTests(unittest.TestCase):
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
