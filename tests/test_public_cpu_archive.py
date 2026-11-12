from pathlib import Path
import hashlib
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class PublicCPUArchiveTests(unittest.TestCase):
    def test_cpu_archive_has_no_deferred_framework_source_and_loads(self):
        from analysis import revision_classical, revision_lemur
        archive = ROOT / 'evidence/two-experiments-cpu-inputs-v1.zip'
        with zipfile.ZipFile(archive) as z:
            self.assertFalse(any(n.startswith('files/nn-gpt/') for n in z.namelist()))
            self.assertIn('PUBLICATION_MANIFEST.json', z.namelist())
        data, records, receipt, predictions, source, versions = revision_classical.load(archive)
        self.assertEqual([len(data[s]['ids']) for s in ['train', 'val', 'test']], [978, 58, 115])
        self.assertEqual(receipt['source_archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertTrue(all(len(r) == 5 for split in data.values() for r in split['x']))
        pred, target, registry, sources, hashes, packet_hash = revision_lemur.load_inputs(ROOT, archive)
        self.assertEqual(pred.shape, (5, 6, 115))
        self.assertEqual(len(target), 115)


if __name__ == '__main__':
    unittest.main()
