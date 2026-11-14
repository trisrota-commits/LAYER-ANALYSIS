"""Synthetic contract tests; these fixtures are not experiment results."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    bundles = []
    for dataset, suffix in [('cifar-10', 'cifar10'), ('cifar-100', 'cifar100'), ('imagenette', 'imagenette')]:
        name = 'lemur-robustness-all3-v1-' + suffix
        bundles.append(dict(dataset=dataset, name=name))
        folder = root / name / 'results/analysis'
        folder.mkdir(parents=True)
        (folder / 'study_report.json').write_text(json.dumps(dict(completed_architectures=1, selected_architectures=1,
            measurement_failures=0, scaling_checks=1, scaling_disagreements=0)))
        (folder / 'rank_agreement.json').write_text('[]')
        rows = [dict(candidate=0, epoch=1, batch_size=32, mode='eval', augmentation='off', loss_reduction=reduction,
            candidate_status='complete', n_successful_measurements=6, coverage_variants_at_checkpoint=1,
            coverage_id='synthetic', raw_mean=value, rescaled_mean=64, raw_cv=0.25) for reduction, value in [('mean', 2), ('sum', 64)]]
        with (folder / 'measurement_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (folder / 'complete.json').write_text(json.dumps(dict(status='complete', files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})))
    (root / 'manifest.json').write_text(json.dumps(dict(bundles=bundles)))


class Point4RunnerTests(unittest.TestCase):
    def test_coverage_filter_precedes_duplicate_check(self):
        for eligible in (False, True):
            with self.subTest(eligible=eligible), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); fixture(root)
                folder = root / 'lemur-robustness-all3-v1-cifar100/results/analysis'
                path = folder / 'measurement_summary.csv'
                with path.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                extra = dict(rows[0], candidate='37', coverage_id='coverage-a',
                             coverage_variants_at_checkpoint='1' if eligible else '3',
                             n_successful_measurements='6' if eligible else '4')
                rows.extend([extra, dict(extra, coverage_id='coverage-b')])
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                receipt = json.loads((folder / 'complete.json').read_text())
                receipt['files_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (folder / 'complete.json').write_text(json.dumps(receipt))
                before = path.read_bytes()
                result = self.run_report(root)
                if eligible:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Ambiguous coverage', result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('=== imagenette ===', result.stdout)
                    self.assertIn('1 32 eval off 1 2 64 64 25', result.stdout)
                self.assertEqual(path.read_bytes(), before)

    def test_rejects_incomplete_changed_or_unsafe_exports_before_reporting(self):
        for failure in ['missing', 'changed', 'unlisted', 'traversal', 'duplicate', 'empty']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); fixture(root)
                folder = root / 'lemur-robustness-all3-v1-imagenette/results/analysis'
                if failure == 'missing':
                    (folder / 'complete.json').unlink()
                elif failure == 'changed':
                    (folder / 'study_report.json').write_text('{}')
                elif failure == 'unlisted':
                    receipt = json.loads((folder / 'complete.json').read_text())
                    del receipt['files_sha256']['measurement_summary.csv']
                    (folder / 'complete.json').write_text(json.dumps(receipt))
                else:
                    manifest = json.loads((root / 'manifest.json').read_text())
                    if failure == 'traversal':
                        manifest['bundles'][0]['name'] = '../outside'
                    elif failure == 'duplicate':
                        manifest['bundles'][2] = manifest['bundles'][0]
                    else:
                        manifest['bundles'] = []
                    (root / 'manifest.json').write_text(json.dumps(manifest))
                result = self.run_report(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Candidates:', result.stdout)

    def run_report(self, root):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), '--point4-report', str(root)],
                              capture_output=True, text=True)

    def test_reports_export_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = self.run_report(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('=== imagenette ===', result.stdout)
            self.assertIn('1 32 eval off 1 2 64 64 25', result.stdout)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
