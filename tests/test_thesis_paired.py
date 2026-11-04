from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PairedTests(unittest.TestCase):
    def test_original_report_parsers_keep_all_metrics_and_groups(self):
        self.assertTrue((ROOT / 'analysis/thesis_paired.py').exists(), 'paired adapter missing')
        from analysis.thesis_paired import parse_main, parse_lodo, parse_lofo
        base = ROOT / 'evidence/thesis-reports'
        main = parse_main((base / 'paired_main_tables.log').read_text(encoding='utf-8'))
        self.assertEqual(len(main['A.2']), 35)
        self.assertEqual(len(main['A.3']), 10)
        self.assertEqual(len(main['6.8']), 5)
        desc, paired = parse_lodo((base / 'lodo_tables_reproduced.log').read_text(encoding='utf-8'))
        self.assertEqual(len(desc), 70)
        self.assertEqual(len(paired), 35)
        desc = parse_lofo((base / 'lofo_tables_reproduced.log').read_text(encoding='utf-8'))
        self.assertEqual(len(desc), 50)


if __name__ == '__main__':
    unittest.main()
