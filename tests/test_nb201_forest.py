"""Exercise the standalone forest generator using bundled numerical evidence."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'figures/nb201_forest/make_nb201_forest_only.py'


class NB201ForestTests(unittest.TestCase):
    def test_portable_generation_preserves_every_plotted_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--out', str(out)],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((out / 'plot_data.csv').read_bytes(),
                             (ROOT / 'figures/nb201_forest/plot_data.csv').read_bytes())
            with pymupdf.open(out / 'fig_nb201.pdf') as doc:
                self.assertEqual(len(doc), 1)
                text = doc[0].get_text()
                self.assertIn('CIFAR-10-valid', text)
                self.assertNotIn('(a)', text)
                self.assertNotIn('(b)', text)
            repeated = subprocess.run([sys.executable, '-B', str(SCRIPT), '--out', str(out)],
                                      capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('Existing output preserved', repeated.stderr)


if __name__ == '__main__':
    unittest.main()
