"""Read-only verification of the published numerical subset; no candidate training.

Default: reproduce 10,000-draw NB201 analyses except mixed models, and verify
LEMUR descriptive metrics. --quick uses 100 draws and does not compare intervals.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent


def check_equal(actual, expected, context):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check_equal(actual[key], value, context + '/' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(context + ': length mismatch')
        for i, (a, e) in enumerate(zip(actual, expected)):
            check_equal(a, e, context + '/' + str(i))
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-11, equal_nan=True):
            raise AssertionError(f'{context}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(f'{context}: {actual!r} != {expected!r}')


def verify_manifest():
    manifest = json.loads((ROOT / 'LOCAL_SOURCE_MANIFEST.json').read_text())['files']
    for name, record in manifest.items():
        data = (ROOT / name).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise AssertionError('Source artifact changed: ' + name)
    print(f'PASS: {len(manifest)} original artifact hashes')


def verify_lemur():
    base = ROOT / 'evidence/two-experiments-final-v1'
    registry = json.loads((base / 'test_registry.json').read_text())
    ids = [r['record_id'] for r in registry]
    if len(ids) != len(set(ids)):
        raise AssertionError('Duplicate LEMUR registry IDs')
    y = np.array([r['target'] for r in registry])
    expected = list(csv.DictReader((base / 'descriptive.csv').open(newline='')))
    models = sorted({r['model'] for r in expected})
    for model in models:
        per_seed = []
        for seed in range(42, 48):
            with (base / f'{model}-s{seed}-predictions.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            by_id = {r['record_id']: r for r in rows}
            if len(by_id) != len(rows) or set(by_id) != set(ids):
                raise AssertionError('LEMUR identity mismatch: ' + model)
            check_equal([float(by_id[i]['target']) for i in ids], y.tolist(), model + '/targets')
            if any(int(r['seed']) != seed for r in rows):
                raise AssertionError('LEMUR seed mismatch')
            p = np.array([float(by_id[i]['prediction']) for i in ids])
            if not np.isfinite(p).all():
                raise AssertionError('Nonfinite LEMUR prediction')
            error = p - y
            per_seed.append({'RMSE': float(np.sqrt(np.mean(error ** 2))),
                             'MAE': float(np.mean(np.abs(error))),
                             'Pearson': float(stats.pearsonr(y, p).statistic),
                             'Spearman': float(stats.spearmanr(y, p).statistic),
                             'R2': float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2))})
        for row in [r for r in expected if r['model'] == model]:
            check_equal(np.mean([v[row['metric']] for v in per_seed]), float(row['mean']), model + '/' + row['metric'])
    print(f'PASS: LEMUR {len(models)} conditions, six seeds, all five descriptive means')


def verify_nb201(quick):
    path = ROOT / 'nb201_v1/primary_endpoint.py'
    spec = importlib.util.spec_from_file_location('published_primary_endpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roots = [('cifar10', ROOT / 'evidence/nb201-qwen-primary-v1/cifar10'),
             ('cifar100', ROOT / 'evidence/nb201-qwen-primary-v1/cifar100'),
             ('imagenet16', ROOT / 'evidence/nb201-imagenet16-primary-v1')]
    for name, base in roots:
        y, predictions, registry = module.load_panel(base / 'inputs/panel.csv', base / 'inputs/registry.json')
        saved = json.loads((base / 'analysis/results.json').read_text())
        actual, _ = module.analyze(y, predictions, registry, draws=100 if quick else 10000, mixed=False)
        for key in ['counts', 'primary', 'seed_improvements', 'descriptive']:
            check_equal(actual[key], saved[key], name + '/' + key)
        if not quick:
            for key in saved['robustness']:
                if key != 'crossed_mixed_model':
                    check_equal(actual['robustness'][key], saved['robustness'][key], name + '/robustness/' + key)
            check_equal(actual['collapse_auc'], saved['collapse_auc'], name + '/collapse_auc')
        print(f'PASS: NB201 {name}, {len(y)} records, primary tests and all five metrics'
              + ('' if quick else ', bootstrap and collapse intervals'))
    # Negative control: the comparison must reject altered results.
    try:
        check_equal(1.0, 2.0, 'negative control')
    except AssertionError:
        pass
    else:
        raise AssertionError('Verifier accepted known-bad value')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    verify_manifest()
    verify_lemur()
    verify_nb201(args.quick)
    print('VERIFIED SUBSET ONLY: no GPU retraining, mixed-model refit, classical NB201 bootstrap, or complete thesis-table reproduction.')


if __name__ == '__main__':
    main()
