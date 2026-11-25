"""Post-hoc BCa intervals; midrank bias correction and paired-unit jackknife."""
import numpy as np
from scipy.stats import norm
from analysis.thesis_common import require


def interval(draws, observed, jackknife, confidence=.95):
    draws, jackknife = np.asarray(draws), np.asarray(jackknife)
    require(draws.ndim == jackknife.ndim == 1 and len(draws) > 1 and len(jackknife) > 2,
            'BCa needs scalar resamples and leave-one-unit-out estimates')
    require(np.isfinite(draws).all() and np.isfinite(jackknife).all()
            and np.isfinite(observed) and 0 < confidence < 1, 'Invalid BCa inputs')
    probability = ((draws < observed).sum() + .5 * (draws == observed).sum()) / len(draws)
    require(0 < probability < 1, 'Degenerate BCa bias correction')
    z0 = norm.ppf(probability)
    influence = jackknife.mean() - jackknife
    denominator = 6 * np.sum(influence**2)**1.5
    require(denominator > 0, 'Degenerate BCa jackknife acceleration')
    acceleration = np.sum(influence**3) / denominator
    alpha = (1-confidence)/2
    z = norm.ppf([alpha, 1-alpha])
    divisor = 1 - acceleration*(z0+z)
    require(np.all(divisor > 0), 'BCa transformation crosses a pole')
    q = norm.cdf(z0 + (z0+z)/divisor)
    require(np.isfinite(q).all() and 0 < q[0] < q[1] < 1, 'Invalid BCa quantiles')
    low, high = np.quantile(draws, q, method='linear')
    tail = len(draws)*min(q[0], 1-q[1])
    return dict(low=float(low), high=float(high), confidence=confidence,
                bias_z0=float(z0), acceleration=float(acceleration),
                quantile_low=float(q[0]), quantile_high=float(q[1]),
                effective_tail_draws=float(tail), sparse_tail=bool(tail < 20))
