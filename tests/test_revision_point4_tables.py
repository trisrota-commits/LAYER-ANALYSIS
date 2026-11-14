"""Focused integrity and recovered-evidence regression tests (no training)."""
import importlib
from pathlib import Path
import tempfile
import unittest


def adapter():
    return importlib.import_module('analysis.revision_point4_tables')


class Point4Tests(unittest.TestCase):
    def test_bad_archive_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'bad.tar.gz'
            archive.write_bytes(b'not the pinned archive')
            out = Path(tmp) / 'out'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                adapter().run(archive, out)
            self.assertFalse(out.exists())

    def test_safe_paths_and_member_integrity(self):
        import io
        import json
        import tarfile
        import hashlib
        from unittest.mock import patch
        for bad in ('../escape', '/absolute', 'C:/drive', 'a\\b', 'a/./b', 'a//b'):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                adapter()._safe(bad)
        cases = [
            [('x', b'payload'), ('EXPORT_MANIFEST.json', json.dumps({'files_sha256': {'x': 'wrong'}}).encode())],
            [('x', b'payload'), ('EXPORT_MANIFEST.json', b'{"files_sha256": {}}')],
            [('../escape', b'payload')],
            [('x', b'1'), ('x', b'2')],
        ]
        receipt_base = f'{adapter().ROOT}/lemur-robustness-all3-v1-cifar10/results/analysis/'
        for receipt in ({'status': 'incomplete', 'files_sha256': {}},
                        {'status': 'complete', 'files_sha256': {n: 'bad' for n in adapter().REQUIRED}}):
            payload = json.dumps(receipt).encode()
            name = receipt_base + 'complete.json'
            manifest = {'files_sha256': {name: hashlib.sha256(payload).hexdigest()}}
            cases.append([(name, payload), ('EXPORT_MANIFEST.json', json.dumps(manifest).encode())])
        for members in cases:
            with self.subTest(members=members), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'fixture.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    for name, data in members:
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        tar.addfile(member, io.BytesIO(data))
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                with patch.object(adapter(), 'ARCHIVE_SHA256', digest), self.assertRaises(ValueError):
                    adapter().run(archive, Path(tmp) / 'out')
                self.assertFalse((Path(tmp) / 'out').exists())

    def test_eligibility_and_matched_coverage(self):
        base = dict(candidate='1', epoch='1', batch_size='32', mode='eval', augmentation='off',
                    loss_reduction='mean', candidate_status='complete', n_successful_measurements='6',
                    coverage_variants_at_checkpoint='1', coverage_id='a', raw_cv='.2', raw_mean='4', rescaled_mean='128')
        large = dict(base, batch_size='128', raw_mean='1')
        cv, discrepancy, details = adapter()._numeric_tables([base, large], 'fixture')
        self.assertEqual(cv[0]['median'], .2)
        self.assertEqual(cv[0]['unit'], 'ratio')
        self.assertEqual(discrepancy[0]['raw_median'], 2)
        self.assertEqual(discrepancy[0]['rescaled_median'], 0)
        self.assertEqual(discrepancy[0]['closer'], 1)
        self.assertEqual(len(details), 1)
        for change in ({'coverage_id': 'b'}, {'n_successful_measurements': '5'},
                       {'candidate_status': 'failed'}, {'coverage_variants_at_checkpoint': '2'}, {'raw_mean': '0'}):
            _, rows, _ = adapter()._numeric_tables([base, dict(large, **change)], 'fixture')
            self.assertEqual(rows[0]['n'], 0)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            adapter()._numeric_tables([base, base], 'fixture')

    def test_real_archive_tables_and_receipts(self):
        import json
        import hashlib
        archive = Path(__file__).resolve().parents[1] / 'evidence/paper-missing-evidence-v1.tar.gz'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report'
            report = adapter().run(archive, out)
            self.assertEqual([r['n'] for r in report['tables']['S57']], [14, 13, 162, 155, 148, 140])
            self.assertEqual([round(r['median'], 3) for r in report['tables']['S57']], [.129, .199, .022, .041, .033, .065])
            row = next(r for r in report['tables']['S56'] if r['dataset'] == 'cifar100' and r['epoch'] == 2)
            self.assertAlmostEqual(row['raw_median'], 1.9784729676583055)
            self.assertTrue(report['flags'])
            self.assertEqual(len(report['tables']['S55']), 9)
            self.assertEqual([round(r['median'], 3) for r in report['tables']['S55']],
                             [.092, .111, .131, .101, .087, .093, .094, .095, .094])
            self.assertEqual([r['closer'] for r in report['tables']['S56']],
                             [12, 12, 12, 152, 152, 151, 136, 136, 136])
            self.assertEqual([r['oom_batch32'] for r in report['tables']['S53']], [0, 0, 432])
            self.assertEqual([r['oom_batch128'] for r in report['tables']['S53']], [224, 1337, 1851])
            self.assertEqual([r['output_contract_failures'] for r in report['tables']['S53']], [0, 144, 288])
            self.assertEqual([r['completed_missing_modules'] for r in report['tables']['S53']], [3, 21, 13])
            receipt = json.loads((out / 'complete.json').read_text())
            for name, digest in receipt['files_sha256'].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            before = (out / 'complete.json').read_bytes()
            with self.assertRaises(FileExistsError):
                adapter().run(archive, out)
            self.assertEqual(before, (out / 'complete.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
