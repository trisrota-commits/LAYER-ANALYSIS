"""Coverage contract for the current seven main and 61 supplementary tables."""
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PaperTableTests(unittest.TestCase):
    def test_export_requires_complete_workflows_and_preserves_tables(self):
        from analysis.revision_paper_tables import export_tables
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);work=out/'workflows';work.mkdir()
            with self.assertRaisesRegex(ValueError,'Incomplete workflow inventory'):
                export_tables(ROOT,work,out,{}, {})
            self.assertFalse((out/'tables').exists())
            (out/'tables').mkdir();marker=out/'tables/keep';marker.write_text('preserve')
            with self.assertRaises(FileExistsError):export_tables(ROOT,work,out,{}, {})
            self.assertEqual(marker.read_text(),'preserve')

    def test_numpy_values_export_without_rounding_or_hiding_nonfinite(self):
        import numpy as np
        import json
        from analysis import revision_paper_tables as module
        self.assertTrue(hasattr(module,'native'))
        value=module.native({'count':np.int64(3),'flag':np.bool_(True),'values':np.array([.1234567890123])})
        self.assertEqual(json.loads(json.dumps(value,allow_nan=False)),{'count':3,'flag':True,'values':[.1234567890123]})
        with self.assertRaises(ValueError):json.dumps(module.native(np.float64('nan')),allow_nan=False)

    def test_inventory_has_every_table_and_only_two_contextual_tables(self):
        self.assertTrue((ROOT / 'analysis/revision_paper_tables.py').is_file())
        from analysis.revision_paper_tables import inventory
        result = inventory()
        expected = {f'S{i}' for i in range(1,62)} | {f'main_{i}' for i in ('I','II','III','IV','V','VI','VII')}
        self.assertEqual(set(result), expected)
        self.assertEqual({k for k,v in result.items() if v=='context'}, {'S1','S46'})
        self.assertEqual(result['S6'],'classical-saved')
        self.assertEqual(result['S39'],'paired-tables')
        self.assertEqual(result['S48'],'loss-confidence-per-arm')
        from analysis.revision_paper_tables import EXPECTED_ROWS, validate_table
        self.assertEqual(set(EXPECTED_ROWS),{k for k,v in result.items() if v!='context'})
        with self.assertRaisesRegex(ValueError,'Incomplete main_VI'):
            validate_table('main_VI',[{}]*5)
        with self.assertRaisesRegex(ValueError,'Missing fields'):
            validate_table('S6',[{}]*15)
        with self.assertRaisesRegex(ValueError,'Missing intervals'):
            validate_table('S2',[{'metric':'RMSE','mean':1.0}]*20)

    def test_existing_output_preserved_before_loading(self):
        self.assertTrue((ROOT / 'analysis/revision_paper_tables.py').is_file())
        from analysis.revision_paper_tables import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); marker=out/'keep';marker.write_text('keep')
            with self.assertRaises(FileExistsError):run(ROOT,out)
            self.assertEqual(marker.read_text(),'keep')


if __name__ == '__main__':unittest.main()
