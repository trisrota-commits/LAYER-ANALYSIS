"""Metrics shared by saved-prediction revision analyses.

Preserves the metric order and calculations of the historical
``two-experiments-final-v1/analyze.py`` implementation. The last axis is
records; preceding axes may be predictors, seeds or bootstrap draws.
"""
import numpy as np
from scipy.stats import rankdata

METRICS = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']


def correlation(y, p):
    yy = y - y.mean(axis=-1, keepdims=True)
    pp = p - p.mean(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (yy * pp).sum(axis=-1) / np.sqrt((yy * yy).sum(axis=-1) * (pp * pp).sum(axis=-1))


def metrics(y, p):
    error = p - y
    variance = ((y - y.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - (error * error).mean(axis=-1) / variance
    return np.stack([
        np.sqrt((error * error).mean(axis=-1)), abs(error).mean(axis=-1),
        correlation(y, p), correlation(rankdata(y, axis=-1), rankdata(p, axis=-1)), r2,
    ], axis=-1)
