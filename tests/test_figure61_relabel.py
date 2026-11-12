"""Portable label-only PDF regression tests (no TeX installation required)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

SCRIPT = Path(__file__).resolve().parents[1] / 'figures/figure6_1/relabel_summary.py'


def load_module():
    if not SCRIPT.is_file():
        raise AssertionError('Label-only publication module is not implemented')
    spec = importlib.util.spec_from_file_location('figure61_relabel', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelabelTests(unittest.TestCase):
    def test_only_label_changes_and_vectors_survive(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.pdf'
            with pymupdf.open() as doc:
                page = doc.new_page(width=300, height=160)
                page.insert_text((180, 80), 'CIFAR-10', fontsize=9)
                page.insert_text((40, 120), 'Unchanged plot annotation', fontsize=9)
                page.draw_line((30, 100), (270, 100), color=(0, 0, 0))
                doc.save(source)
            before = source.read_bytes()
            out = root / 'result'
            # Only font discovery is substituted: use a real built-in vector font.
            with patch.object(module, 'load_font', return_value=pymupdf.Font('helv'), create=True):
                module.relabel_summary(source, out, font_path=root / 'test-font.pfb')
            self.assertTrue((out / 'taylor_summary.pdf').is_file())
            self.assertTrue((out / 'taylor_summary.png').is_file())
            self.assertEqual(source.read_bytes(), before)
            with pymupdf.open(source) as original, pymupdf.open(out / 'taylor_summary.pdf') as saved:
                page = saved[0]
                self.assertEqual(page.get_text().splitlines().count('CIFAR-10-valid'), 1)
                self.assertNotIn('CIFAR-10', page.get_text().splitlines())
                self.assertIn('Unchanged plot annotation', page.get_text())
                self.assertEqual(page.rect, original[0].rect)
                # Redaction adds a white rectangle and reorders paint operations;
                # the original vector line's geometry and styling must survive.
                original_line = original[0].get_drawings()[0]
                saved_line = next(d for d in page.get_drawings() if d['items'][0][0] == 'l')
                for key in ('items', 'color', 'width', 'type', 'dashes'):
                    self.assertEqual(saved_line[key], original_line[key])
                self.assertEqual(page.get_images(), [])
                old_rect = original[0].search_for('CIFAR-10')[0]
                new_rect = page.search_for('CIFAR-10-valid')[0]
                self.assertAlmostEqual(old_rect.x1, new_rect.x1, places=3)
                arrays = []
                for p in (original[0], page):
                    pix = p.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
                    arrays.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n))
                changed = np.any(arrays[0] != arrays[1], axis=2)
                region = (old_rect | new_rect) + (-2, -3, 2, 3)
                mask = np.zeros(changed.shape, dtype=bool)
                mask[int(region.y0*3):int(region.y1*3)+1, int(region.x0*3):int(region.x1*3)+1] = True
                self.assertTrue(np.any(changed))
                self.assertFalse(np.any(changed & ~mask))

    def test_existing_output_is_preserved(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.pdf'
            source.write_bytes(b'input must not be touched')
            out = root / 'existing'
            out.mkdir()
            sentinel = out / 'taylor_summary.pdf'
            sentinel.write_bytes(b'precious existing output')
            with self.assertRaises(FileExistsError):
                module.relabel_summary(source, out)
            self.assertEqual(sentinel.read_bytes(), b'precious existing output')
            self.assertEqual(source.read_bytes(), b'input must not be touched')
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
