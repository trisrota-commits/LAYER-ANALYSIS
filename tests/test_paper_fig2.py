"""Exercise the real plot generator and inspect its PDF text."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pymupdf

ROOT = Path(__file__).resolve().parents[1]


class PaperFigure2Tests(unittest.TestCase):
    def test_requested_labels_and_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'figures/paper_fig2/make_fig2.py')],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            pdf = Path(tmp) / 'images/fig_zcp.pdf'
            self.assertTrue(pdf.is_file(), 'Expected images/fig_zcp.pdf')
            with pymupdf.open(pdf) as document:
                self.assertEqual(len(document), 1)
                text = document[0].get_text()
            self.assertIn('Dead-act. fraction', text)
            self.assertNotIn('Activation', text)
            self.assertNotIn('r =', text)
            self.assertNotIn('p =', text)
