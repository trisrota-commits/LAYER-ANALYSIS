"""Checks for the archived-data CPU classical fit adapter."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RevisionClassicalTests(unittest.TestCase):
    def test_preserves_existing_output(self):
        self.assertTrue((ROOT / 'analysis/revision_classical.py').is_file())
        from analysis import revision_classical as classical
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'keep'; out.mkdir()
            marker = out / 'sentinel'; marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'Existing output'):
                classical.run(Path(tmp) / 'missing.zip', out)
            self.assertEqual(marker.read_text(), 'preserve')

    def test_rejects_changed_archive_before_creating_output(self):
        self.assertTrue((ROOT / 'analysis/revision_classical.py').is_file())
        from analysis import revision_classical as classical
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'changed.zip'; archive.write_bytes(b'bad archive')
            out = Path(tmp) / 'new'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                classical.run(archive, out)
            self.assertFalse(out.exists())

    def test_package_version_gate(self):
        self.assertTrue((ROOT / 'analysis/revision_classical.py').is_file())
        from analysis import revision_classical as classical
        expected = {'numpy': '2.2.6', 'scipy': '1.15.3', 'scikit-learn': '1.7.2'}
        with patch.object(classical.importlib.metadata, 'version', side_effect=lambda name: expected[name]):
            classical.check_versions(expected)
        with patch.object(classical.importlib.metadata, 'version', return_value='unexpected'):
            with self.assertRaisesRegex(ValueError, 'requirements-classical'):
                classical.check_versions(expected)


if __name__ == '__main__':
    unittest.main()
