from pathlib import Path
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PairedTests(unittest.TestCase):
    def test_original_report_parsers_keep_all_metrics_and_groups(self):
        self.assertTrue((ROOT / 'analysis/thesis_paired.py').exists(), 'paired adapter missing')
        from analysis.thesis_paired import parse_main, parse_lodo, parse_lofo
        with zipfile.ZipFile(ROOT / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip') as packet:
            reports = {name: packet.read('experiment_logs/' + name).decode('utf-8') for name in
                       ['paired_main_tables.log', 'lodo_tables_reproduced.log', 'lofo_tables_reproduced.log']}
        main = parse_main(reports['paired_main_tables.log'])
        self.assertEqual(len(main['A.2']), 35)
        self.assertEqual(len(main['A.3']), 10)
        self.assertEqual(len(main['6.8']), 5)
        desc, paired = parse_lodo(reports['lodo_tables_reproduced.log'])
        self.assertEqual(len(desc), 70)
        self.assertEqual(len(paired), 35)
        desc = parse_lofo(reports['lofo_tables_reproduced.log'])
        self.assertEqual(len(desc), 50)


if __name__ == '__main__':
    unittest.main()
