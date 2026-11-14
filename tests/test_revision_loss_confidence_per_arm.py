"""S48 per-arm intervals: frozen evidence, shared family draws, no refitting."""
import csv
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from analysis import revision_loss_confidence as source

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'evidence/loss-confidence-all-metrics-inputs-v1.tar.gz'


class PerArmTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT / 'analysis/revision_loss_confidence_per_arm.py').is_file(),
                        'S48 per-arm exporter is missing')
        return importlib.import_module('analysis.revision_loss_confidence_per_arm')

    def test_actual_archive_shared_draws_and_full_precision_exports(self):
        export = self.module()
        self.assertEqual(export.DRAWS, 10000)
        loaded = source.load(ARCHIVE)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'export'
            with patch.object(export, 'DRAWS', 12):
                export.run(ARCHIVE, out)
            result = json.loads((out / 'results.json').read_text())
            rows = result['results']
            self.assertEqual(len(rows), 60)
            self.assertEqual({r['population'] for r in rows}, {'all'})
            self.assertEqual({r['arm'] for r in rows}, set(source.ARMS))
            self.assertEqual({r['metric'] for r in rows}, set(source.METRICS))
            with (out / 'per_arm_metrics.csv').open(newline='') as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 60)
            with np.load(out / 'bootstrap_per_arm.npz') as arrays:
                self.assertEqual(set(arrays.files), set(loaded))
                for dataset, (y, p, families, _, _) in loaded.items():
                    blocks = [np.flatnonzero(families == f) for f in np.unique(families)]
                    draws = np.random.default_rng(0).integers(len(blocks), size=(12, len(blocks)))
                    expected = np.array([source.metrics(y[idx], p[:, idx]) for idx in
                                         [np.concatenate([blocks[g] for g in draw]) for draw in draws]])
                    np.testing.assert_array_equal(arrays[dataset], expected)
                    values = source.metrics(y, p)
                    for i, row in enumerate(rows):
                        if row['dataset'] != dataset:
                            continue
                        ai = source.ARMS.index(row['arm']); mi = source.METRICS.index(row['metric'])
                        self.assertEqual(row['estimate'], values[ai, mi])
                        lo, hi = np.quantile(expected[:, ai, mi], [.025, .975])
                        self.assertEqual((row['ci95_low'], row['ci95_high']), (lo, hi))
                        for key in ['estimate', 'ci95_low', 'ci95_high']:
                            self.assertEqual(float(csv_rows[i][key]), row[key])
            receipt = json.loads((out / 'complete.json').read_text())
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(receipt['source_archive_sha256'], source.EXPECTED)
            self.assertEqual(set(receipt['versions']), {'python', 'numpy', 'scipy'})
            for filename, digest in receipt['source_files_sha256'].items():
                self.assertEqual(source.sha((ROOT / filename).read_bytes()), digest)
            for filename, digest in receipt['files_sha256'].items():
                self.assertEqual(source.sha((out / filename).read_bytes()), digest)
            self.assertEqual(set(receipt['files_sha256']),
                             {'per_arm_metrics.csv', 'results.json', 'per_arm_metrics.md',
                              'paper_three_metrics.md', 'bootstrap_per_arm.npz'})
            full = (out / 'per_arm_metrics.md').read_text()
            paper = (out / 'paper_three_metrics.md').read_text()
            self.assertIn('Pearson', full)
            self.assertIn('R2', full)
            self.assertNotIn('Pearson', paper)
            self.assertNotIn('R2', paper)
            for text in [full, paper]:
                self.assertIn('pointwise', text)
                self.assertIn('unadjusted', text)
                self.assertIn('Spearman', text)

    @unittest.skipUnless(os.environ.get('S48_FULL_BOOTSTRAP') == '1', 'opt-in 10000-draw reproduction')
    def test_full_bootstrap_reproduces_known_paper_interval(self):
        export = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'export'
            export.run(ARCHIVE, out)
            result = json.loads((out / 'results.json').read_text())
            row = next(r for r in result['results'] if r['dataset'] == 'CIFAR-10-valid'
                       and r['arm'] == 'A' and r['metric'] == 'MAE')
            np.testing.assert_allclose([row['estimate'], row['ci95_low'], row['ci95_high']],
                                       [2.953093115659266, 2.376178594173472, 3.7177298353337496],
                                       atol=1e-12, rtol=0)
            self.assertEqual(len(result['results']), 60)
            self.assertTrue(all(r['draws'] == 10000 for r in result['results']))
            with np.load(out / 'bootstrap_per_arm.npz') as arrays:
                for dataset in source.BUNDLES:
                    self.assertEqual(arrays[dataset].shape, (10000, 4, 5))
                    self.assertTrue(np.isfinite(arrays[dataset]).all())
            print('FULL S48 REPRODUCTION:', row)

    def test_existing_output_and_bad_archive_fail_closed(self):
        export = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'keep'; out.mkdir()
            marker = out / 'sentinel'; marker.write_text('preserve')
            bad = Path(tmp) / 'bad.tar.gz'; bad.write_bytes(b'invalid archive')
            with self.assertRaisesRegex(ValueError, 'Existing output'):
                export.run(bad, out)
            self.assertEqual(marker.read_text(), 'preserve')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                export.run(bad, Path(tmp) / 'new')
            self.assertFalse((Path(tmp) / 'new').exists())

    def test_missing_dataset_is_rejected_before_output(self):
        export = self.module()
        loaded = source.load(ARCHIVE)
        loaded.pop(next(iter(loaded)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'export'
            with patch.object(source, 'load', return_value=loaded), patch.object(export, 'DRAWS', 2):
                with self.assertRaisesRegex(ValueError, 'datasets'):
                    export.run(ARCHIVE, out)
            self.assertFalse(out.exists())

    def test_nonfinite_draw_and_descriptive_mismatch_fail_closed(self):
        export = self.module()
        loaded = source.load(ARCHIVE)
        dataset = next(iter(loaded))
        y, p, families, saved, comparisons = loaded[dataset]
        saved['all']['A']['mean_metrics']['MAE'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'export'
            with patch.object(source, 'load', return_value=loaded):
                with self.assertRaises(AssertionError):
                    export.run(ARCHIVE, out)
            self.assertFalse(out.exists())
            with patch.object(export, 'DRAWS', 1), patch.object(
                    export, 'metrics', side_effect=[source.metrics(y, p), np.full((4, 5), np.nan)]):
                with self.assertRaisesRegex(ValueError, 'Undefined bootstrap'):
                    export.run(ARCHIVE, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
