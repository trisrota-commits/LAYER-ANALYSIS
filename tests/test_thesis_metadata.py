from pathlib import Path
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_exported_snapshot_counts(self):
        path = ROOT / 'analysis/thesis_metadata.py'
        self.assertTrue(path.exists(), 'metadata adapter missing')
        spec = importlib.util.spec_from_file_location('metadata_test', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        tables = m.build_tables(ROOT)
        self.assertEqual(set(tables), {'4.1', '4.2', '4.3', '4.4'})
        counts = {r['dataset']: r['records'] for r in tables['4.1']}
        self.assertEqual(counts['places365'], 5)
        self.assertEqual(counts['Total'], 1139)


if __name__ == '__main__':
    unittest.main()
