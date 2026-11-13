"""All-pairs saved-prediction analysis; fast real-input verification."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class AllPairsTests(unittest.TestCase):
    def test_real_input_coverage_and_receipt(self):
        self.assertTrue((ROOT / 'analysis/revision_loss_confidence_all_pairs.py').is_file(),
                        'All-pairs adapter is missing')
        from analysis import revision_loss_confidence_all_pairs as module
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'results'
            with patch.object(module, 'DRAWS', 32):
                module.run(ROOT / 'evidence/loss-confidence-all-metrics-inputs-v1.tar.gz', out)
            with (out / 'all_comparisons.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 90)
            self.assertEqual(len({(r['dataset'], r['contrast'], r['metric']) for r in rows}), 90)
            self.assertEqual({r['metric'] for r in rows}, set(module.METRICS))
            self.assertEqual({r['population'] for r in rows}, {'all'})
            self.assertTrue(all(int(r['draws']) == 32 for r in rows))
            with np.load(out / 'bootstrap_differences.npz', allow_pickle=False) as arrays:
                for row in rows:
                    a, b = row['contrast'].split(' -> ')
                    pair = (module.ARMS.index(a), module.ARMS.index(b))
                    draws = arrays[row['dataset']][:, module.PAIRS.index(pair), module.METRICS.index(row['metric'])]
                    expected = np.quantile(draws, [.025, .975, .025 / 30, 1 - .025 / 30])
                    actual = [float(row[k]) for k in ('ci95_low', 'ci95_high', 'bonferroni30_low', 'bonferroni30_high')]
                    np.testing.assert_allclose(actual, expected)
                    lo, hi = actual[2:]
                    conclusion = 'Candidate better' if lo > 0 else 'Candidate worse' if hi < 0 else 'Inconclusive'
                    self.assertEqual(row['conclusion'], conclusion)
            receipt = json.loads((out / 'complete.json').read_text())
            self.assertEqual(receipt['status'], 'complete')
            for name, digest in receipt['files_sha256'].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            self.assertIn('Pearson', (out / 'results.md').read_text())
            self.assertIn('R2', (out / 'results.md').read_text())

    def test_cli_mode_and_guards(self):
        command = [sys.executable, '-B', str(ROOT / 'run_all.py')]
        result = subprocess.run(command + ['--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('loss-confidence-all-pairs', result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            result = subprocess.run(command + ['--revision', 'loss-confidence-all-pairs', '--out', str(out)],
                                    capture_output=True, text=True)
            self.assertIn('requires --archive', result.stderr)
            self.assertFalse(out.exists())

    def test_existing_output_and_bad_archive(self):
        from analysis.revision_loss_confidence_all_pairs import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'keep'; out.mkdir()
            sentinel = out / 'sentinel'; sentinel.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                run(Path(tmp) / 'missing', out)
            self.assertEqual(sentinel.read_bytes(), b'preserve')
            archive = Path(tmp) / 'bad.tar.gz'; archive.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'Archive checksum mismatch'):
                run(archive, Path(tmp) / 'new')
            self.assertFalse((Path(tmp) / 'new').exists())


if __name__ == '__main__':
    unittest.main()
