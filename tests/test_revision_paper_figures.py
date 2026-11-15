"""Current-paper figure inventory and fail-closed entry-point contracts."""
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class PaperFigureDriverTests(unittest.TestCase):
    def test_one_runner_combines_tables_and_figures(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'run_all.py'),'--help'],capture_output=True,text=True)
        self.assertIn('paper-figures',result.stdout)
        from analysis.revision_paper import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            with patch('analysis.revision_paper_tables.run',return_value={'numerical_tables':66,'contextual_tables':2}) as tables:
                with patch('analysis.revision_paper_figures.run',return_value={'figure_count':19}) as figures:
                    report=run(ROOT,out)
                    tables.assert_called_once_with(ROOT.resolve(),out.resolve()/'tables')
                    figures.assert_called_once_with(ROOT.resolve(),out.resolve()/'tables',out.resolve()/'figures')
                    self.assertEqual(report['figures'],19)
            with self.assertRaises(FileExistsError):run(ROOT,out)

    def test_inventory_has_all_current_figures(self):
        self.assertTrue((ROOT/'analysis/revision_paper_figures.py').is_file())
        from analysis.revision_paper_figures import FIGURES
        self.assertEqual(set(FIGURES),{f'main_{i}' for i in range(1,8)}|{f'S{i}' for i in range(1,13)})
        self.assertEqual(len(set(FIGURES.values())),19)
        self.assertEqual(FIGURES['main_3'],'fig_nb201_improve')
        self.assertEqual(FIGURES['S12'],'fig_robust')

    def test_existing_output_and_incomplete_tables_refused(self):
        self.assertTrue((ROOT/'analysis/revision_paper_figures.py').is_file())
        from analysis.revision_paper_figures import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'keep';out.mkdir();marker=out/'sentinel';marker.write_text('preserve')
            with self.assertRaises(FileExistsError):run(ROOT,Path(tmp)/'missing',out)
            self.assertEqual(marker.read_text(),'preserve')
            with self.assertRaises((FileNotFoundError,ValueError)):run(ROOT,Path(tmp)/'missing',Path(tmp)/'new')
            self.assertFalse((Path(tmp)/'new').exists())


if __name__=='__main__':unittest.main()
