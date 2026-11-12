from pathlib import Path
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RevisionReportTests(unittest.TestCase):
    def test_existing_output_and_input_receipt_safety(self):
        self.assertTrue((ROOT / 'analysis/revision_report.py').is_file())
        from analysis import revision_report as report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'keep'; out.mkdir()
            marker = out / 'sentinel'; marker.write_text('preserve')
            with self.assertRaises((ValueError, AssertionError)):
                report.run(root / 'missing', out)
            self.assertEqual(marker.read_text(), 'preserve')
            bad = root / 'bad'; bad.mkdir()
            (bad / 'complete.json').write_text(json.dumps({'status':'complete','files_sha256':{'../sentinel':'invalid'}}))
            with self.assertRaises((ValueError, AssertionError)):
                report.verified(bad)


if __name__ == '__main__':
    unittest.main()
