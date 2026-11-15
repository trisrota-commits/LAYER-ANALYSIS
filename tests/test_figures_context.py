"""Precommit checks for the rendered, source-grounded contextual figures."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pymupdf
from analysis import figures_context as context
from analysis import thesis_lemur, thesis_measurements

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    tables=folder/'tables';tables.mkdir()
    data,y,_=thesis_lemur.load_main(ROOT)
    metric=thesis_lemur.metric_function(ROOT)
    rows=[dict(feature=f,metric='RMSE',observations=3,zcp='off',seeds=list(range(42,48)),n_test=115,
               mean=float(metric(data[3,'off',f],y).mean(axis=0)[0])) for f in ('no-layer','taylor')]
    (tables/'main_IV.json').write_text(json.dumps(rows))
    primary=[]
    for name in ['nb201-qwen-primary-v1/cifar10','nb201-qwen-primary-v1/cifar100','nb201-imagenet16-primary-v1']:
        report=json.loads((ROOT/'evidence'/name/'analysis/results.json').read_text())
        ref=report['descriptive']['noncollapsed']['no-layer']['mean']['MAE']
        primary.append(dict(relative_reduction_percent=100*report['primary']['record']['improvement']/ref))
    (tables/'main_III.json').write_text(json.dumps(primary))
    measured,_=thesis_measurements.build_tables(ROOT)
    (tables/'S16.json').write_text(json.dumps(measured['6.13']))


class ContextFigureTests(unittest.TestCase):
    def test_actual_data_render_bounds_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);fixture(base);out=base/'figures'
            figures=context.render(ROOT,base,out)
            self.assertEqual(set(figures),{'main_1','main_7'})
            d=json.loads((out/'overview_data.json').read_text())
            self.assertAlmostEqual(d['lemur_rmse']['taylor'],8.474931819445873)
            self.assertAlmostEqual(d['taylor_seconds_per_checkpoint'],.49419544586756575)
            tally=json.loads((out/'tally_data.json').read_text())
            self.assertEqual([r['better'] for r in tally['rows']],[5,8,2,2,17])
            self.assertEqual([r['total'] for r in tally['rows']],[6,11,3,6,26])
            for item in figures.values():
                for ext in ('pdf','png'):
                    file=out/item['files'][ext]['path']
                    self.assertEqual(context.sha(file),item['files'][ext]['sha256'])
                with pymupdf.open(out/item['files']['pdf']['path']) as doc:
                    self.assertEqual(len(doc),1)
                    for block in doc[0].get_text('dict')['blocks']:
                        for line in block.get('lines',[]):
                            for span in line['spans']:
                                x0,y0,x1,y1=span['bbox']
                                self.assertGreaterEqual(x0,-.5);self.assertGreaterEqual(y0,-.5)
                                self.assertLessEqual(x1,doc[0].rect.width+.5);self.assertLessEqual(y1,doc[0].rect.height+.5)
            with pymupdf.open(out/'fig_overview.pdf') as doc:
                text=doc[0].get_text()
                for label in ['Candidate','Predictor','Target accuracy','LEMUR: full-run best accuracy',
                              'NB201: dataset-specific benchmark target','0.49','9.09','8.47']:self.assertIn(label,text)
                self.assertNotIn('Final accuracy',text)
            before=(out/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):context.render(ROOT,base,out)
            self.assertEqual((out/'manifest.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
