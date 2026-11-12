from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class NestedRevisionTests(unittest.TestCase):
    def test_cli_advertises_nested_mode(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'run_all.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('nested-outer', result.stdout)

    def test_existing_output_is_preserved(self):
        self.assertTrue((ROOT/'analysis/revision_nested.py').is_file())
        from analysis.revision_nested import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'keep';out.mkdir();(out/'sentinel').write_text('preserve')
            with self.assertRaises(FileExistsError):run(ROOT,Path(tmp)/'missing',out)
            self.assertEqual((out/'sentinel').read_text(),'preserve')

    def test_published_analyzer_uses_bundled_archive(self):
        self.assertTrue((ROOT/'analysis/revision_nested.py').is_file())
        from analysis.revision_nested import load_analyzer
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            module=load_analyzer(ROOT,ROOT/'evidence/nested-outer-six-seeds-v1.tar.gz',out)
            y,pred,selected,folds,groups,registry,selections,per_fold=module.load()
            self.assertEqual(pred.shape,(2,6,1139))
            self.assertEqual([len(b) for b in module.group_blocks(folds,groups)],[212,215,212])
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
