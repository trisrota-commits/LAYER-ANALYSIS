"""Stratified BCa must retain three independent fold-level jackknife samples."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import bootstrap
from analysis.revision_common import metrics


def test_nested_matches_multisample_scipy():
    path=Path(__file__).resolve().parents[1]/'analysis/revision_nested_bca.py'
    assert path.is_file(), 'Missing nested BCa implementation'
    spec=importlib.util.spec_from_file_location('nested_bca_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(9)
    y=np.arange(36.)
    p=y+rng.normal(size=(2,6,36))*np.array([3,2])[:,None,None]
    blocks=[[np.array([i]) for i in range(0,9)],
            [np.array([i]) for i in range(9,21)],
            [np.array([i]) for i in range(21,36)]]
    rows,draws=module.calculate(y,p,blocks,n_resamples=1000)
    def stat(*sample):
        idx=np.concatenate([b[int(g)] for b,s in zip(blocks,sample) for g in s])
        m=metrics(y[idx],p[:,:,idx]).mean(axis=1)
        return (m[1]-m[0])*[-1,-1,1,1,1]
    ref=bootstrap(tuple(np.arange(len(b)) for b in blocks),stat,vectorized=False,
                  paired=False,n_resamples=0,batch=50,method='BCa',confidence_level=.95,
                  bootstrap_result=SimpleNamespace(bootstrap_distribution=draws.T))
    np.testing.assert_allclose([r['ci95_low'] for r in rows],ref.confidence_interval.low)
    np.testing.assert_allclose([r['ci95_high'] for r in rows],ref.confidence_interval.high)
    rng=np.random.default_rng(0)
    first=tuple(rng.integers(len(b),size=len(b)) for b in blocks)
    np.testing.assert_allclose(draws[0],stat(*first))
