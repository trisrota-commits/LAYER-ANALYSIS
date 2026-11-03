import unittest
import numpy as np
from pathlib import Path
import importlib.util
import tempfile
import csv
import json
import hashlib
from unittest.mock import patch
import types
import warnings


def load():
    path = Path(__file__).with_name('primary_endpoint.py')
    assert path.exists(), 'Independent primary-endpoint implementation missing'
    spec = importlib.util.spec_from_file_location('primary_endpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrimaryTests(unittest.TestCase):
    def test_mixed_model_rejects_invalid_hessian(self):
        import pandas as pd
        m=load()
        fit=types.SimpleNamespace(converged=True,pvalues={'Intercept':.04},bse={'Intercept':.2},params={'Intercept':.5},
            conf_int=lambda: pd.DataFrame([[.1,.9]],index=['Intercept']))
        def bad_fit(**kwargs):
            warnings.warn('The Hessian matrix at the estimated parameter values is not positive definite.')
            return fit
        with patch('statsmodels.formula.api.mixedlm',return_value=types.SimpleNamespace(fit=bad_fit)):
            result=m.crossed_mixed(np.arange(24.).reshape(4,6))
        self.assertIsNone(result['p'])
        self.assertNotEqual(result['status'],'ok')

    def test_qwen_reader_requires_complete_matrix_and_receipts(self):
        m=load()
        self.assertTrue(hasattr(m,'qwen_inputs'),'Qwen result reader missing')
        thesis=Path(__file__).resolve().parent.parent
        with self.assertRaisesRegex(ValueError,'Missing Qwen artifacts'):
            m.qwen_inputs(thesis/'evidence/nb201-v1-qwen-matched-v1','cifar10')
        with tempfile.TemporaryDirectory(prefix='hermes-verify-qwen-reader-') as tmp:
            root=Path(tmp)
            # Disposable schema fixture only, not a trained Qwen result.
            reg=[dict(record_id='a',architecture_id=3,dataset='cifar10-valid',target=50.,family='x'),dict(record_id='b',architecture_id=32,dataset='cifar10-valid',target=60.,family='y')]
            contract={'runs':{}};manifest={}
            for cond in ('no-layer','taylor'):
                rp=root/'prompts/cifar10'/cond/'test_registry.json';rp.parent.mkdir(parents=True);rp.write_text(json.dumps(reg))
                rh=hashlib.sha256(rp.read_bytes()).hexdigest()
                for seed in m.SEEDS:
                    key=f'cifar10-{cond}-s{seed}';p=root/'results'/key;p.mkdir(parents=True)
                    text='record_id,prediction,target,seed\n'+f'a,49,50,{seed}\nb,59,60,{seed}\n'
                    (p/'per_record_predictions.csv').write_text(text)
                    digest=hashlib.sha256((p/'per_record_predictions.csv').read_bytes()).hexdigest()
                    (p/'test_complete.json').write_text(json.dumps(dict(run=key,n_samples=2,n_valid=2,parse_failures=0)))
                    (p/'prediction_contract.json').write_text(json.dumps(dict(run=key,n=2,finite_accuracy_predictions=2,prediction_sha256=digest,target_kind='peak_mean_validation_curve')))
                    (p/'runtime_preflight.json').write_text(json.dumps(dict(run=key,effective_seed=seed,adapter_seed=seed,over_cap_records=0)))
                    contract['runs'][key]=dict(dataset='cifar10',condition=cond,seed=seed,target_kind='peak_mean_validation_curve',input_sha256={'test_registry.json':rh})
            contract['counts']={'test':2}
            (root/'contract.json').write_text(json.dumps(contract))
            manifest['contract.json']=hashlib.sha256((root/'contract.json').read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            text,registry,sources=m.qwen_inputs(root,'cifar10')
            self.assertEqual(len(list(csv.DictReader(text.splitlines()))),24)
            self.assertEqual(registry,reg)
            (root/'results/cifar10-taylor-s47/per_record_predictions.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'hash'):
                m.qwen_inputs(root,'cifar10')

    def test_saved_analysis_and_no_overwrite(self):
        m = load()
        self.assertTrue(hasattr(m, 'run_panel'), 'Analysis persistence missing')
        base=Path(__file__).resolve().parent.parent/'evidence/nb201-classical-matched-v1/cifar100/gbm'
        with tempfile.TemporaryDirectory(prefix='hermes-verify-primary-report-') as tmp:
            out=Path(tmp)/'report'
            result=m.run_panel(base/'panel.csv',base/'test_registry.json',out,draws=100,mixed=False)
            self.assertEqual(result['counts']['all'],275)
            self.assertIsNone(result['primary']['seed']['p'])
            self.assertTrue((out/'complete.json').exists())
            self.assertTrue((out/'resamples.npz').exists())
            with self.assertRaises(ValueError):
                m.run_panel(base/'panel.csv',base/'test_registry.json',out,draws=100,mixed=False)

    def test_published_lemur_primary_numbers_from_actual_predictions(self):
        m = load()
        self.assertTrue(hasattr(m, 'analyze'), 'Primary and robustness analysis missing')
        base = Path(__file__).resolve().parent.parent/'evidence/two-experiments-final-v1'
        registry=json.loads((base/'test_registry.json').read_text())
        y=np.array([r['target'] for r in registry])
        pred=np.empty((2,len(y),6))
        for c,prefix in enumerate(('no_layer','qwen_reference')):
            for j,seed in enumerate(m.SEEDS):
                with (base/f'{prefix}-s{seed}-predictions.csv').open() as f:
                    rows={r['record_id']:r for r in csv.DictReader(f)}
                pred[c,:,j]=[float(rows[r['record_id']]['prediction']) for r in registry]
        result,draws=m.analyze(y,pred,registry,draws=100,mixed=False)
        self.assertEqual(result['primary']['record']['n'],106)
        self.assertAlmostEqual(result['primary']['record']['improvement'],.4956,delta=.0001)
        self.assertAlmostEqual(result['primary']['record']['p'],.0044,delta=.0001)
        self.assertAlmostEqual(result['primary']['seed']['p'],.0006,delta=.0001)
        self.assertAlmostEqual(result['collapse_auc']['taylor']['auc'],.991,delta=.001)
        self.assertEqual(draws['record_bootstrap'].shape,(100,))

    def test_loader_pairs_exact_registry_and_rejects_missing_rows(self):
        m = load()
        self.assertTrue(hasattr(m, 'load_panel'), 'Panel loader missing')
        root = Path(__file__).resolve().parent.parent
        base = root / 'evidence/nb201-classical-matched-v1/cifar10/gbm'
        y, pred, registry = m.load_panel(base/'panel.csv', base/'test_registry.json')
        self.assertEqual(pred.shape, (2,275,6))
        self.assertEqual(len(registry),275)
        with (base/'panel.csv').open() as f:
            rows=list(csv.DictReader(f))
        with tempfile.TemporaryDirectory(prefix='hermes-verify-primary-') as tmp:
            p=Path(tmp)/'bad.csv'
            for bad in (rows[:-1], rows+rows[:1], [dict(rows[0], gt_best_accuracy='0')]+rows[1:], [dict(rows[0],pred_best_accuracy='nan')]+rows[1:]):
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(bad)
                with self.assertRaises(ValueError):
                    m.load_panel(p, base/'test_registry.json')

    def test_t_test_and_holm_match_library(self):
        from scipy import stats
        m = load()
        values = np.array([.2, .5, -.1, .9, .3, .7])
        result = m.paired_t(values)
        expected = stats.ttest_1samp(values, 0.)
        self.assertAlmostEqual(result['p'], float(expected.pvalue))
        self.assertAlmostEqual(result['t'], float(expected.statistic))
        np.testing.assert_allclose(result['ci95'], stats.t.interval(.95, df=5, loc=values.mean(), scale=stats.sem(values)))
        np.testing.assert_allclose(m.holm([.01, .04]), [.02, .04])
        self.assertIsNone(m.paired_t(np.ones(6))['p'])


if __name__ == '__main__':
    unittest.main()
