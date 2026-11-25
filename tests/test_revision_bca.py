"""BCa arithmetic checked against SciPy using the same resamples/jackknife."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy import stats


def test_bca_matches_scipy():
    path = Path(__file__).resolve().parents[1] / 'analysis/revision_bca.py'
    assert path.is_file(), 'Missing portable BCa implementation'
    spec = importlib.util.spec_from_file_location('bca_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    y = np.array([1., 2., 3., 4., 8., 13., 21.])
    jack = np.array([np.delete(y, i).mean() for i in range(len(y))])
    result = stats.bootstrap((y,), np.mean, method='BCa', n_resamples=2000,
                             rng=np.random.default_rng(0))
    got = module.interval(result.bootstrap_distribution, y.mean(), jack, .95)
    np.testing.assert_allclose([got['low'], got['high']], result.confidence_interval,
                               rtol=1e-12, atol=1e-12)


def test_xgboost_paired_seed_metric_not_ensemble():
    path = Path(__file__).resolve().parents[1] / 'analysis/revision_xgboost_bca.py'
    assert path.is_file(), 'Missing XGBoost BCa calculation'
    spec = importlib.util.spec_from_file_location('xgb_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from analysis.revision_common import metrics
    rng = np.random.default_rng(9)
    y = np.linspace(0, 100, 30)
    p = y + rng.normal(size=(2, 6, 30))*np.array([4, 6])[:, None, None]
    result, draws, jack = module.calculate(p, y, n_resamples=1000)
    scores = metrics(y, p).mean(axis=1)
    expected = (scores[0]-scores[1])*[-1, -1, 1, 1, 1]
    np.testing.assert_allclose([r['improvement'] for r in result], expected)
    idx = np.random.default_rng(0).integers(len(y), size=(250, len(y)))
    score = metrics(y[idx], p[:, :, idx]).mean(axis=1)
    np.testing.assert_allclose(draws[:250], (score[0]-score[1])*[-1, -1, 1, 1, 1])
    score = metrics(y[1:], p[:, :, 1:]).mean(axis=1)
    np.testing.assert_allclose(jack[0], (score[0]-score[1])*[-1, -1, 1, 1, 1])


def test_bca_rejects_nonfinite_and_degenerate():
    import pytest
    from analysis.revision_bca import interval
    for draws, jack in [(np.array([np.nan, 1]), np.arange(5.)),
                        (np.arange(10.), np.ones(5))]:
        with pytest.raises((ValueError, RuntimeError)):
            interval(draws, 4., jack)
