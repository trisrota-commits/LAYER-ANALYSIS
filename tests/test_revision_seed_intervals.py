"""Uploaded patch functionality must be a real runner command, not a .patch file."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SeedIntervalTests(unittest.TestCase):
    def test_runner_command_and_preservation(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'run_all.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('seed-intervals',result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'run_all.py'),'--revision','seed-intervals','--out',tmp],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Existing output preserved',result.stderr)

    def test_all_44_rows_and_full_precision(self):
        from analysis.revision_seed_intervals import run
        import json
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';rows=run(ROOT,out)
            self.assertEqual(len(rows),44)
            self.assertEqual({r['table'] for r in rows},{'S2','S3','S10','S11','S12','S13'})
            r=next(r for r in rows if r['table']=='S2' and r['row']=='2 accuracy + no layer')
            self.assertAlmostEqual(r['RMSE'],11.471549990343567,places=12)
            self.assertNotEqual(r['RMSE'],round(r['RMSE'],6))
            self.assertEqual(json.loads((out/'complete.json').read_text())['rows'],44)
            with self.assertRaises(FileExistsError):run(ROOT,out)


if __name__=='__main__':unittest.main()
