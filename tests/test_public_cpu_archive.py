from pathlib import Path
import hashlib
import csv
import io
import json
from collections import Counter
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class PublicCPUArchiveTests(unittest.TestCase):
    def test_cpu_archive_has_no_deferred_framework_source_and_loads(self):
        from analysis import revision_lemur
        archive = ROOT / 'evidence/two-experiments-cpu-inputs-v1.zip'
        with zipfile.ZipFile(archive) as z:
            self.assertFalse(any(n.startswith('files/nn-gpt/') for n in z.namelist()))
            self.assertIn('PUBLICATION_MANIFEST.json', z.namelist())
            prefix = 'files/classical-eight-features-v1/'
            records = z.read(prefix+'records.csv')
            receipt = json.loads(z.read(prefix+'complete.json'))
            self.assertEqual(hashlib.sha256(records).hexdigest(), receipt['files_sha256']['records.csv'])
            counts = Counter(r['split'] for r in csv.DictReader(io.StringIO(records.decode())))
            self.assertEqual([counts[s] for s in ['train','val','test']], [978,58,115])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), revision_lemur.PUBLIC_ARCHIVE_SHA256)
        pred, target, registry, sources, hashes, packet_hash = revision_lemur.load_inputs(ROOT, archive)
        self.assertEqual(pred.shape, (5, 6, 115))
        self.assertEqual(len(target), 115)


if __name__ == '__main__':
    unittest.main()
