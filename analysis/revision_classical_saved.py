"""Saved predictions only: complete S6, S7 and main VII classical tables.

No estimator imports, fitting, tuning, or ensemble predictions. S6 differences
are with minus without Taylor (not improvement-signed); S7 is Qwen-favouring.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from analysis import revision_common as m

METHODS = ['qwen_reference', 'ridge', 'random_forest', 'hist_gradient_boosting']
SEEDS = list(range(42, 48))
ARCHIVE = 'evidence/thesis-new-logs-20261112-084434.zip'
ARCHIVE_SHA256 = '2b233d441517d5fca8c4d6d564abce824706c8d2828aae0dbf48d867c895e249'
NPZ_MEMBER = 'recent_results/statistical-test-results-full-v1/classical_no_taylor_bootstrap.npz'
PREFIX = 'evidence/two-experiments-final-v1/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_csv(data, seed, ids, targets):
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')))
    require([r['record_id'] for r in rows] == ids, 'Prediction record identity/order mismatch')
    require(len(ids) == 115 and len(set(ids)) == 115, 'Expected 115 unique test records')
    require(all(int(r['seed']) == seed for r in rows), 'Prediction seed mismatch')
    require(np.array_equal([float(r['target']) for r in rows], targets), 'Prediction target mismatch')
    values = np.array([float(r['prediction']) for r in rows])
    require(np.isfinite(values).all(), 'Nonfinite predictions')
    return values


def load_inputs(repo):
    repo = Path(repo)
    manifest = json.loads((repo / 'INPUT_MANIFEST.json').read_text())['files']
    sources = {}

    def frozen(name):
        data = (repo / name).read_bytes()
        require(name in manifest and sha(data) == manifest[name], 'Frozen input hash mismatch: ' + name)
        sources[name] = sha(data)
        return data

    registry = json.loads(frozen(PREFIX + 'test_registry.json'))
    ids = [r['record_id'] for r in registry]
    y = np.array([r['target'] for r in registry], dtype=float)
    require(len(ids) == 115 and len(set(ids)) == 115 and np.isfinite(y).all(), 'Invalid registry')
    with_taylor = np.array([[prediction_csv(frozen(PREFIX + f'{method}-s{seed}-predictions.csv'),
                                           seed, ids, y) for seed in SEEDS] for method in METHODS])
    archive = (repo / ARCHIVE).read_bytes()
    require(sha(archive) == ARCHIVE_SHA256, 'Frozen archive hash mismatch')
    sources[ARCHIVE] = sha(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        data = z.read(NPZ_MEMBER)
        sources[ARCHIVE + '!' + NPZ_MEMBER] = sha(data)
        with np.load(io.BytesIO(data), allow_pickle=False) as a:
            pred = a['predictions'].copy()
            old_draws = a['differences'].copy()
            require(np.array_equal(a['targets'], y), 'NPZ target mismatch')
        require(pred.shape == (4, 6, 115) and np.isfinite(pred).all(), 'Invalid saved prediction tensor')
        require(old_draws.shape == (3, 10000, 5) and np.isfinite(old_draws).all(), 'Invalid archived draws')
        require(np.array_equal(pred[0], with_taylor[0]), 'NPZ Qwen identity/order mismatch')
        # Classical CSVs carry the IDs absent from the NPZ; require every arm/seed.
        for i, method in enumerate(METHODS[1:], 1):
            for j, seed in enumerate(SEEDS):
                member = f'recent_results/classical-no-taylor-v1/{method}-s{seed}-predictions.csv'
                raw = z.read(member)
                values = prediction_csv(raw, seed, ids, y)
                require(np.array_equal(values, pred[i,j]), 'NPZ/CSV predictions mismatch: ' + member)
                sources[ARCHIVE + '!' + member] = sha(raw)
        summary = list(csv.DictReader(io.StringIO(z.read('recent_results/classical-no-taylor-v1/summary.csv').decode(), newline='')))
        point = m.metrics(y, pred).mean(axis=1)
        for r in summary:
            value = point[METHODS.index(r['model']), m.METRICS.index(r['metric'])]
            require(abs(value - float(r['mean'])) < 2e-12, 'Archived point estimate mismatch')
    return ids, y, with_taylor, pred, old_draws, sources


def recompute(y, pred):
    """Original revision_report.py RNG, batches, paired metrics and seed mean."""
    point = m.metrics(y, pred).mean(axis=1)
    sign = np.array([-1, -1, 1, 1, 1])
    rng = np.random.default_rng(0)
    draws = np.empty((3, 10000, 5))
    indices = np.empty((10000, 115), dtype=np.int64)
    for start in range(0, 10000, 250):
        idx = rng.integers(115, size=(250, 115))
        indices[start:start+250] = idx
        values = m.metrics(y[idx], pred[:, :, idx]).mean(axis=1)
        draws[:, start:start+250] = (values[0][None, :, :] - values[1:]) * sign
    require(np.isfinite(draws).all(), 'Nonfinite bootstrap draws')
    bounds = np.quantile(draws, [.025, .975, .025/15, 1-.025/15], axis=1)
    return point, (point[0]-point[1:])*sign, draws, indices, bounds


def run(repo, out):
    """Export to a new directory only; write completion receipt last. Return report."""
    repo, out = Path(repo), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('Existing output preserved: ' + str(out))
    ids, y, with_taylor, pred, old_draws, sources = load_inputs(repo)
    point, gain, draws, indices, bounds = recompute(y, pred)
    discrepancy = float(np.max(abs(draws-old_draws)))
    require(discrepancy < 2e-12, 'Recomputed bootstrap differs from frozen archive')
    old_bounds = np.quantile(old_draws, [.025, .975, .025/15, 1-.025/15], axis=1)
    require(np.allclose(bounds, old_bounds, rtol=0, atol=2e-12), 'Archived interval mismatch')
    with_seed_metrics = m.metrics(y, with_taylor)
    with_point = with_seed_metrics.mean(axis=1)
    s6, vii, s7, per_seed = [], [], [], []
    for i, method in enumerate(METHODS):
        for k, metric in enumerate(m.METRICS):
            vii.append(dict(model=method, metric=metric, value=float(with_point[i,k])))
            for j, seed in enumerate(SEEDS):
                per_seed.append(dict(model=method, seed=seed, metric=metric, arm='with_taylor', value=float(with_seed_metrics[i,j,k])))
            if i == 0:
                continue
            s6.append(dict(model=method, metric=metric, without_taylor=float(point[i,k]),
                           with_taylor=float(with_point[i,k]), change=float(with_point[i,k]-point[i,k])))
            s7.append(dict(model=method, metric=metric, reference=float(point[i,k]), candidate=float(point[0,k]),
                           improvement=float(gain[i-1,k]), ci95_low=float(bounds[0,i-1,k]),
                           ci95_high=float(bounds[1,i-1,k]), bonferroni15_low=float(bounds[2,i-1,k]),
                           bonferroni15_high=float(bounds[3,i-1,k])))
    without_seed_metrics = m.metrics(y, pred)
    for i, method in enumerate(METHODS[1:], 1):
        for j, seed in enumerate(SEEDS):
            for k, metric in enumerate(m.METRICS):
                per_seed.append(dict(model=method, seed=seed, metric=metric, arm='without_taylor', value=float(without_seed_metrics[i,j,k])))
    notes = [
        'S6 paper discrepancy: HistGB RMSE change is printed +0.3507; full-precision with-minus-without is +0.3506212262803121 (rounds +0.3506). The printed change matches subtraction of rounded arms.',
        'S6: all five metrics, with-minus-without Taylor; no within-model paired intervals are claimed. The paper prints RMSE/MAE only.',
        'Main VII: means of six seed-labelled predictor metrics, never metrics of ensemble predictions.',
        'S7: positive improvement favours Qwen with Taylor; errors are classical minus Qwen, other metrics Qwen minus classical.',
        '10000 paired test-record bootstrap draws, NumPy default_rng seed 0; fitted predictors fixed, metrics computed per seed then averaged.',
        'Percentile intervals: .025/.975 and .025/15, 1-.025/15. Bonferroni family is all 15 classical contrasts/metrics.',
        'No fitting or selection rerun. NPZ predictions cross-checked against identity-bearing classical CSVs and the frozen Qwen CSVs.',
        'These are fixed-test-record post-hoc comparisons, not uncertainty over independent fits; seed labels can share identical predictions.',
    ]
    report = dict(schema=1, n=115, seeds=SEEDS, metrics=m.METRICS, methods=METHODS,
                  paper_discrepancies=[dict(table='S6', model='hist_gradient_boosting', metric='RMSE',
                                           column='change', printed=.3507, recomputed=s6[10]['change'],
                                           explanation='Printed change matches subtraction of rounded arms.')],
                  s6=s6, main_vii=vii, s7=s7, per_seed=per_seed, notes=notes,
                  source_sha256=sources, script_sha256=sha(Path(__file__).read_bytes()),
                  metric_helper_sha256=sha(Path(m.__file__).read_bytes()),
                  archived_draws_max_abs_difference=discrepancy,
                  archived_bounds_max_abs_difference=float(np.max(abs(bounds-old_bounds))))
    # Exclusive directory creation also guards against a concurrent writer.
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('s6',s6), ('main_vii',vii), ('s7',s7), ('per_seed',per_seed)]:
        with (out / (name+'.csv')).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez_compressed(out / 'bootstrap.npz', differences=draws, indices=indices,
                        predictions=pred, with_taylor_predictions=with_taylor,
                        targets=y, record_ids=np.array(ids), methods=np.array(METHODS),
                        seeds=np.array(SEEDS), metrics=np.array(m.METRICS))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# Saved classical paper tables', ''] + ['- '+s for s in notes]
    for name, rows in [('S6',s6), ('Main VII',vii), ('S7',s7)]:
        fields = list(rows[0])
        lines += ['', '## '+name, '', '| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']
        lines += ['| '+' | '.join(str(r[k]) for k in fields)+' |' for r in rows]
    (out / 'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt = dict(status='complete', s6_metric_rows=15, s6_numeric_cells=45,
                   main_vii_numeric_cells=20, s7_metric_rows=15, bootstrap_draws=10000,
                   source_sha256=sources,
                   files_sha256={p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    (out / 'complete.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.out)
    print('COMPLETE: S6 45 values; Main VII 20 values; S7 15 paired intervals; 10000 saved draws:', args.out)
