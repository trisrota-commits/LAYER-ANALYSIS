"""Fast revision regression checks; never run the 10,000-draw analyses here."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]


class RevisionAnalysisTests(unittest.TestCase):
    def modules(self):
        result = []
        for name in ('revision_nb201', 'revision_lemur'):
            try:
                result.append(importlib.import_module('analysis.' + name))
            except ModuleNotFoundError as exc:
                self.fail(f'Required import-safe revision API missing: {exc}')
        return result

    def test_existing_output_is_preserved_before_input_access(self):
        for module in self.modules():
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'output'
                out.mkdir()
                sentinel = out / 'keep.txt'
                sentinel.write_text('preserve me')
                args = [Path(tmp) / 'missing-repo', out]
                if module.__name__.endswith('lemur'):
                    args.append(Path(tmp) / 'missing.zip')
                with self.assertRaisesRegex(RuntimeError, 'Existing output preserved'):
                    module.run(*args)
                self.assertEqual(sentinel.read_text(), 'preserve me')
                self.assertEqual(list(out.iterdir()), [sentinel])

    def test_missing_manifest_fails_without_creating_output(self):
        for module in self.modules():
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'output'
                args = [Path(tmp), out]
                if module.__name__.endswith('lemur'):
                    args.append(Path(tmp) / 'missing.zip')
                with self.assertRaises(FileNotFoundError):
                    module.run(*args)
                self.assertFalse(out.exists())

    def test_nb201_frozen_dataset_schema(self):
        nb, _ = self.modules()
        self.assertEqual(list(nb.SPEC), ['CIFAR-10-valid', 'CIFAR-100', 'ImageNet16-120'])
        frozen = json.loads((REPO / 'INPUT_MANIFEST.json').read_text())['files']
        for dataset, relative in nb.SPEC.items():
            with self.subTest(dataset=dataset):
                y, pred, families, hashes = nb.load_dataset(REPO, dataset, relative, frozen)
                self.assertEqual(y.shape, (275,))
                self.assertEqual(pred.shape, (2, 6, 275))
                self.assertEqual(families.shape, (275,))
                self.assertGreater(len(np.unique(families)), 1)
                self.assertEqual(set(hashes), {relative + '/panel.csv', relative + '/registry.json'})
                self.assertTrue(np.isfinite(nb.metrics(y, pred)).all())

    def test_nb201_manifest_mismatch_fails_closed(self):
        nb, _ = self.modules()
        dataset, relative = next(iter(nb.SPEC.items()))
        frozen = json.loads((REPO / 'INPUT_MANIFEST.json').read_text())['files']
        frozen[relative + '/panel.csv'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Frozen input mismatch'):
            nb.load_dataset(REPO, dataset, relative, frozen)

    def test_metric_and_reference_order(self):
        nb, lemur = self.modules()
        expected = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']
        self.assertEqual(list(nb.METRICS), expected)
        self.assertEqual(list(lemur.METRICS), expected)
        self.assertIs(nb.metrics, lemur.metrics)
        self.assertEqual(lemur.REFERENCES, ['All seven / ZCP-OFF', 'XGBoost / 3 accuracies',
                                           'Calibrated SynFlow', 'Calibrated NASWOT'])
        y = np.array([1., 2., 4., 8.])
        pred = np.broadcast_to(y, (2, 6, 4)).copy()
        values = nb.metrics(y, pred)
        self.assertEqual(values.shape, (2, 6, 5))
        np.testing.assert_allclose(values, np.broadcast_to([0, 0, 1, 1, 1], values.shape))

    def test_lemur_bad_archive_fails_before_loader(self):
        _, lemur = self.modules()
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'wrong.zip'
            archive.write_bytes(b'not the frozen archive')
            out = Path(tmp) / 'output'
            with self.assertRaisesRegex(ValueError, 'Archive SHA256 mismatch'):
                lemur.run(REPO, out, archive)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
