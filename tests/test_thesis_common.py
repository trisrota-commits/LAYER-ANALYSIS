import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class ArchiveSafetyTests(unittest.TestCase):
    def test_traversal_archive_is_rejected(self):
        path = ROOT / 'analysis/thesis_common.py'
        self.assertTrue(path.exists(), 'thesis archive helper is missing')
        spec = importlib.util.spec_from_file_location('thesis_common_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='hermes-verify-archive-') as tmp:
            archive = Path(tmp) / 'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside.txt', 'must not write')
            with self.assertRaises(ValueError):
                module.unpack_zip(archive, Path(tmp) / 'dest')
            self.assertFalse((Path(tmp) / 'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()
