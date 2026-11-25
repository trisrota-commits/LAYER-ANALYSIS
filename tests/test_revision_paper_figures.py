"""Current-paper figure inventory and fail-closed entry-point contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class PaperFigureDriverTests(unittest.TestCase):
    def test_public_image_runner_does_not_recalculate_tables(self):
        import run_images
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Path(tmp)/'tables';inputs.mkdir();out=Path(tmp)/'images'
            with patch('analysis.revision_paper_tables.run',side_effect=AssertionError('No table recalculation')):
                with patch('analysis.public_runners.images') as render:
                    run_images.main(['--inputs',str(inputs),'--out',str(out)])
                    render.assert_called_once_with(ROOT,inputs.resolve(),out.resolve())

    def test_inventory_has_all_current_figures(self):
        self.assertTrue((ROOT/'analysis/revision_paper_figures.py').is_file())
        from analysis.revision_paper_figures import FIGURES, normalized_id
        self.assertEqual(set(FIGURES),{f'main_{i}' for i in range(1,8)}|{f'S{i}' for i in range(1,11)})
        self.assertEqual(len(set(FIGURES.values())),17)
        self.assertEqual(FIGURES['main_3'],'fig_nb201_improve')
        self.assertEqual(FIGURES['S10'],'fig_robust')
        for old in range(3,13):self.assertEqual(normalized_id(f'S{old}'),f'S{old-2}')
        for old in ('S1','S2'):
            with self.assertRaises(ValueError):normalized_id(old)

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
