"""LEMUR calculations from archived predictions, preserving experiment identities."""
from functools import lru_cache
import io
import json
import re
import tarfile
import zipfile

import numpy as np
from scipy.stats import t

from analysis.thesis_common import csv_rows, module, require

METRICS = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']
FEATURES = ['no-layer', 'weightwatcher', 'cka', 'effective-rank', 'taylor', 'gradient-flow', 'sensitivity', 'activation', 'all-seven']


@lru_cache(maxsize=1)
def metric_function(repo):
    return module(repo / 'evidence/reproduce_main_paired_tables.py', 'existing_main_metrics').metrics


@lru_cache(maxsize=1)
def pools(repo):
    result = {}
    with zipfile.ZipFile(repo / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip') as packet:
        for key, name in [('main', 'article-tables-only-20261014-111938.tar.gz'),
                          ('baseline', 'article-timing-baselines-20261014-113919.tar.gz')]:
            with tarfile.open(fileobj=io.BytesIO(packet.read(name))) as archive:
                result[key] = {m.name.removeprefix('./'): archive.extractfile(m).read()
                               for m in archive if m.isfile() and m.name.endswith(('.csv', '.json'))}
        with zipfile.ZipFile(io.BytesIO(packet.read('overleaf-experiment-logs-20261015-100424.zip'))) as archive:
            result['extra'] = {n: archive.read(n) for n in archive.namelist() if n.endswith(('.csv', '.json'))}
    return result


def get(pool, suffix):
    names = [n for n in pool if n.endswith(suffix)]
    require(len(names) == 1, f'Expected one member for {suffix}; found {names}')
    return names[0], pool[names[0]]


def matrix(blobs, n):
    canonical, predictions = None, []
    for data in blobs:
        rows = csv_rows(data)
        require(len(rows) == n, 'Prediction population changed')
        keys = [(row.get('run_id') or row.get('idx') or row.get('record_id'), row.get('dataset', ''),
                 float(row.get('gt_best_accuracy', row.get('target')))) for row in rows]
        require(len({k[0] for k in keys}) == n, 'Duplicate/missing prediction identities')
        if canonical is None:
            canonical = keys
        require(keys == canonical, 'Ordered prediction identities/datasets/targets differ')
        predictions.append([float(row.get('pred_best_accuracy', row.get('prediction'))) for row in rows])
    p, y = np.array(predictions), np.array([k[2] for k in canonical])
    require(np.isfinite(p).all() and np.isfinite(y).all(), 'Nonfinite predictions or targets')
    return p, y, canonical


def load_main(repo):
    pool = pools(repo)
    data, sources, canonical = {}, {}, None
    for obs in [2, 3]:
        for zcp in ['on', 'off']:
            for feature in FEATURES:
                blobs, paths = [], []
                for seed in range(42, 48):
                    which = 'main'
                    if seed >= 45:
                        batch = 'tables9-12-seeds45-47-v1'
                        run = f'{obs}acc-zcp-{zcp}-{feature}-s{seed}'
                        if (obs, zcp, feature) == (3, 'off', 'taylor'):
                            batch, run = 'taylor-3acc-zcp-off-seeds45-47-v1', f'3acc-taylor-s{seed}'
                        if (obs, zcp, feature) == (3, 'on', 'no-layer'):
                            which = 'extra'
                            batch, run = 'zcp-on-no-layer-3acc-seeds45-47-v1', f'3acc-no-layer-s{seed}'
                        suffix = f'{batch}/results/{run}/test_predictions.csv'
                    elif zcp == 'off':
                        suffix = f'zcp-off-toggle-refit7-v1/results/{obs}acc-{feature}-s{seed}/test_predictions.csv'
                    elif feature in ['no-layer', 'all-seven']:
                        which = 'extra'
                        arm = 'layer' if feature == 'all-seven' else 'no_layer'
                        third = 'on' if obs == 3 else 'off'
                        suffix = f'out/accuracy_predictor_85_5_10/final_refit_7ep_v2/third-{third}-{arm}-s{seed}/test_predictions.csv'
                    elif obs == 3:
                        suffix = f'layer-ablation-refit7-v1/results/{feature}-s{seed}/test_predictions.csv'
                    else:
                        which = 'extra'
                        suffix = f'layer-ablation-2acc-refit7-v1/results/2acc-{feature}-s{seed}/test_predictions.csv'
                    name, blob = get(pool[which], suffix)
                    paths.append(which + ':' + name)
                    blobs.append(blob)
                p, y, keys = matrix(blobs, 115)
                if canonical is None:
                    canonical = keys
                require(keys == canonical, 'Main conditions are not identically paired')
                data[(obs, zcp, feature)] = p
                sources[f'{obs}/{zcp}/{feature}'] = paths
    return data, y, sources


def summary(repo, pred, target, **labels):
    values = metric_function(repo)(pred, target)
    means = values.mean(axis=0)
    result = dict(labels)
    result.update(dict(zip(METRICS, map(float, means))))
    return result


def seed_intervals(repo, pred, target, **labels):
    values = metric_function(repo)(pred, target)
    means = values.mean(axis=0)
    half = t.ppf(.975, len(values) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return [dict(labels, metric=m, mean=float(means[i]), low=float(means[i]-half[i]), high=float(means[i]+half[i])) for i, m in enumerate(METRICS)]


def direct_pair(repo, candidate, reference, target):
    require(candidate.shape == reference.shape == (6, len(target)), 'Unpaired direct comparison')
    metric = metric_function(repo)
    indices = np.random.default_rng(0).integers(0, len(target), size=(10000, len(target)))
    sign = np.array([-1, -1, 1, 1, 1])
    draws = []
    for start in range(0, 10000, 200):
        ix = indices[start:start+200]
        draws.append(((metric(candidate[:, ix], target[ix]) - metric(reference[:, ix], target[ix])) * sign).mean(axis=0))
    draws = np.concatenate(draws)
    c, r = metric(candidate, target).mean(axis=0), metric(reference, target).mean(axis=0)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return [dict(metric=m, no_layer=float(r[i]), taylor=float(c[i]), improvement=float((c[i]-r[i])*sign[i]),
                 low=float(low[i]), high=float(high[i]), valid_draws=10000) for i, m in enumerate(METRICS)]


def build_tables(repo):
    data, target, sources = load_main(repo)
    pool = pools(repo)
    tables = {key: [] for key in ['6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.9', '6.12', 'A.1']}
    for obs in [2, 3]:
        for arm in ['no-layer', 'layer']:
            blobs = []
            for repetition in [1, 2, 3]:
                if obs == 3 or (arm == 'no-layer' and repetition == 3):
                    suffix = f'historical-22test-additions-v1/results/{obs}acc-{arm}-r{repetition}/test_predictions.csv'
                else:
                    suffix = f'historical-2acc-reproduction-v2/results/{arm}-r{repetition}/test_predictions.csv'
                _, blob = get(pool['extra'], suffix)
                blobs.append(blob)
            p, y, _ = matrix(blobs, 22)
            tables['6.2'].append(summary(repo, p, y, split='initial', observations=obs, layer=arm, repetitions=3))
            feature = 'all-seven' if arm == 'layer' else 'no-layer'
            tables['6.2'].append(summary(repo, data[obs, 'on', feature], target, split='grouped', observations=obs, layer=arm, repetitions=6))
    for (obs, zcp, feature), pred in data.items():
        tables['6.4'].append(summary(repo, pred, target, observations=obs, zcp=zcp, feature=feature))
        if obs == 3 and zcp == 'off':
            tables['A.1'].extend(seed_intervals(repo, pred, target, feature=feature))
    tables['6.5'] = direct_pair(repo, data[3, 'off', 'taylor'], data[3, 'off', 'no-layer'], target)
    for obs in [2, 3]:
        blobs = [get(pool['baseline'], f'xgboost_{obs}acc/predictions-s{s}.csv')[1] for s in range(42, 48)]
        p, y, _ = matrix(blobs, 115)
        tables['6.3'].append(summary(repo, p, y, model=f'XGBoost {obs} accuracies'))
    for name in ['synflow', 'nwot']:
        p, y, _ = matrix([get(pool['baseline'], f'/{name}_predictions.csv')[1]], 115)
        tables['6.3'].append(summary(repo, p, y, model=name))
    combos = {'T': data[3, 'off', 'taylor']}
    for label, batch, run in [
        ('T+S', 'round2-taylor-pairs-3acc-off-v1', 'taylor-sensitivity'),
        ('T+A', 'round2-taylor-pairs-3acc-off-v1', 'taylor-activation'),
        ('T+C', 'round2-taylor-pairs-3acc-off-v1', 'taylor-cka'),
        ('T+G', 'readmission-3acc-off-v1', 'taylor-gradient-flow'),
        ('T+E', 'readmission-3acc-off-v1', 'taylor-effective-rank'),
        ('T+S+A', 'round3-triples-3acc-off-v1', 'taylor-sensitivity-activation'),
        ('T+S+C', 'round3-triples-3acc-off-v1', 'taylor-sensitivity-cka'),
        ('T+S+G', 'readmission-3acc-off-v1', 'taylor-sensitivity-gradient-flow'),
        ('T+S+E', 'readmission-3acc-off-v1', 'taylor-sensitivity-effective-rank'),
    ]:
        blobs = [get(pool['main'], f'{batch}/results/{run}-s{s}/test_predictions.csv')[1] for s in range(42, 48)]
        p, y, _ = matrix(blobs, 115)
        require(np.array_equal(y, target), 'Combination target mismatch')
        combos[label] = p
    tables['6.6'] = [summary(repo, pred, target, features=name) for name, pred in combos.items()]
    p, y, _ = matrix([get(pool['main'], f'taylor-sensitivity-zcp-on-v1/results/taylor-sensitivity-s{s}/test_predictions.csv')[1] for s in range(42, 48)], 115)
    require(np.array_equal(y, target), 'T+S ON target mismatch')
    for name, zcp, pred in [('Taylor epochs1-3', 'OFF', combos['T']), ('Taylor epochs1-3', 'ON', data[3, 'on', 'taylor']),
                             ('Taylor+sensitivity', 'OFF', combos['T+S']), ('Taylor+sensitivity', 'ON', p)]:
        tables['6.7'].append(summary(repo, pred, target, condition=name, zcp=zcp))
    base = repo / 'evidence/two-experiments-final-v1'
    for model in ['epoch1', 'no_layer', 'ridge', 'random_forest', 'hist_gradient_boosting', 'qwen_reference']:
        p, y, keys = matrix([(base / f'{model}-s{s}-predictions.csv').read_bytes() for s in range(42, 48)], 115)
        if model in ['epoch1', 'no_layer']:
            tables['6.7'].append(summary(repo, p, y, condition=model, zcp='OFF'))
        else:
            tables['6.9'].append(summary(repo, p, y, model=model))
    init = csv_rows(get(pool['main'], 'init-taylor-comparison-v1/results-successful-only-v1/test_predictions.csv')[1])
    proxy = {name: {row['run_id']: row for row in csv_rows(get(pool['baseline'], f'/{name}_predictions.csv')[1])}
             for name in ['synflow', 'nwot']}
    ids = [row['run_id'] for row in init]
    require(len(ids) == len(set(ids)) == 86, 'Initialization cohort mismatch')
    y = np.array([float(proxy['synflow'][rid]['gt_best_accuracy']) for rid in ids])
    require(all(round(y[i], 2) == float(row['gt_best_accuracy']) for i, row in enumerate(init)), 'Rounded initialization target mismatch')
    for name in ['synflow', 'nwot']:
        require(all(float(proxy[name][rid]['gt_best_accuracy']) == y[i] and
                    proxy[name][rid]['architecture_id'] == init[i]['architecture_id'] and
                    proxy[name][rid]['dataset'] == init[i]['dataset'] for i, rid in enumerate(ids)), 'Initialization identity mismatch')
    tables['6.12'].append(summary(repo, np.array([[float(row['pred_best_accuracy']) for row in init]]), y,
                                   calibration='existing', model='Taylor'))
    for name in ['synflow', 'nwot']:
        pred = np.array([[float(proxy[name][rid]['pred_best_accuracy']) for rid in ids]])
        tables['6.12'].append(summary(repo, pred, y, calibration='existing', model=name))
    common = csv_rows(repo / 'evidence/initialization-proxy-combinations-reproduced-v1/predictions.csv')
    require([row['record_id'] for row in common] == ids, 'Common calibration test identity mismatch')
    require(np.array_equal(np.array([float(row['target']) for row in common]), y), 'Common calibration target mismatch')
    for name in ['Taylor', 'SynFlow', 'NASWOT', 'Taylor+SynFlow', 'Taylor+NASWOT']:
        pred = np.array([[float(row[name]) for row in common]])
        tables['6.12'].append(summary(repo, pred, y, calibration='common852', model=name))
    return tables, sources
