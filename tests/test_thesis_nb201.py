"""Focused exported-data NB201 checks; no training or canonical writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def adapter():
    path = REPO / 'analysis/thesis_nb201.py'
    assert path.is_file(), 'NB201 adapter is not implemented'
    spec = importlib.util.spec_from_file_location('_test_thesis_nb201_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NB201Tests(unittest.TestCase):
    def test_classical_recomputes_four_adjusted_intervals_from_saved_panels(self):
        module = adapter()
        root = REPO / 'evidence/nb201-classical-matched-v1'
        sources = [p for p in root.rglob('*') if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with tempfile.TemporaryDirectory() as tmp:
            metrics, rows = module.build_classical_tables(REPO, Path(tmp))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(metrics), 8)
            for row in rows:
                saved = json.loads((root / row['dataset'] / row['model'] / 'results.json').read_text())['all']
                self.assertEqual(row['n'], 275)
                self.assertAlmostEqual(row['improvement'], saved['improvement'], places=12)
                self.assertAlmostEqual(row['family_bonferroni8_low'], saved['family_bonferroni8'][0], places=12)
                self.assertAlmostEqual(row['family_bonferroni8_high'], saved['family_bonferroni8'][1], places=12)
                self.assertEqual(row['verification_status'], 'verified')
                self.assertTrue(Path(row['report_path']).is_file())
            self.assertTrue(all(all(k in row for k in ('MAE','RMSE','Pearson','Spearman','R2')) for row in metrics))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def test_primary_projection_uses_recomputed_values_and_labels_mixed_mismatch(self):
        module = adapter()
        self.assertTrue(callable(getattr(module, 'build_tables', None)), 'Missing unified entry point')
        source = REPO / 'evidence/nb201-qwen-primary-v1/cifar10/analysis/results.json'
        saved = json.loads(source.read_text())
        report = json.loads(source.read_text())
        report['robustness']['crossed_mixed_model']['p'] = 0.987654321
        metric, primary, appendix, verification = module.primary_rows('cifar10', report, saved, source)
        self.assertEqual(len(metric), 2)
        self.assertEqual(len(primary), 1)
        self.assertEqual(verification['core_status'], 'verified')
        self.assertEqual(verification['mixed_status'], 'mismatch')
        self.assertEqual(appendix[0]['mixed_p'], 0.987654321)
        self.assertEqual(primary[0]['record_holm_p'], report['primary']['record']['holm_p'])
        self.assertEqual(primary[0]['seed_holm_p'], report['primary']['seed']['holm_p'])
        self.assertEqual(appendix[0]['cell_signflip_p'], report['robustness']['cell_signflip']['p'])
        self.assertTrue(all(not isinstance(v, (dict, list)) for rows in (metric, primary, appendix) for row in rows for v in row.values()))
        report['counts']['all'] += 1
        self.assertEqual(module.primary_rows('cifar10', report, saved, source)[3]['core_status'], 'mismatch')
        self.assertEqual(module.primary_rows('imagenet16', report, saved, source)[2], [])


if __name__ == '__main__':
    unittest.main()
