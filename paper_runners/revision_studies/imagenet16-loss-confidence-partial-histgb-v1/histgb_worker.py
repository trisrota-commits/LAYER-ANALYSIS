from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
ROOT = Path('/a/project')
BUNDLE = ROOT / 'imagenet16-loss-confidence-partial-histgb-v1'
OUT = Path('/a/output')
SEEDS = (42, 43, 44, 45, 46, 47)
LC = ('train_loss', 'probe_loss', 'true_confidence', 'max_confidence', 'entropy')
CONTRASTS = (('A_LC', 'A_LC_T'), ('A', 'A_LC'), ('A', 'A_T'), ('A_T', 'A_LC_T'))
PARAMS = {'loss': 'squared_error', 'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 20, 'l2_regularization': 0.0, 'max_bins': 255, 'early_stopping': False}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def js(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda : f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def put(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def write_csv(p, rows):
    with Path(p).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def verify():
    c = js(BUNDLE / 'results/histgb_contract.json')
    require(sha(BUNDLE / 'histgb_worker.py') == c['histgb_worker_sha256'], 'Worker changed')
    for (relative, digest) in c['sources_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Input changed: ' + relative)
    for (package, version) in c['packages'].items():
        require(importlib.metadata.version(package) == version, 'Package version mismatch: ' + package)
    return c

def metric_row(y, prediction):
    error = prediction - y
    variable = len(y) > 1 and np.ptp(y) > 0
    pred_variable = np.ptp(prediction) > 0
    return {'MAE': float(np.abs(error).mean()), 'RMSE': float(np.sqrt(np.square(error).mean())), 'Pearson': float(pearsonr(y, prediction).statistic) if variable and pred_variable else None, 'Spearman': float(spearmanr(y, prediction).statistic) if variable and pred_variable else None, 'R2': float(r2_score(y, prediction)) if variable else None}

def paired_mae(y, reference, candidate, families, draws=10000):
    require(reference.shape == candidate.shape == (len(SEEDS), len(y)), 'Unpaired prediction shapes')
    delta = (np.abs(reference - y) - np.abs(candidate - y)).mean(axis=0)
    groups = np.unique(families)
    require(len(groups) > 1, 'Insufficient test families')
    sums = np.array([delta[families == g].sum() for g in groups])
    sizes = np.array([(families == g).sum() for g in groups])
    rng = np.random.default_rng(0)
    sampled = rng.integers(len(groups), size=(draws, len(groups)))
    bootstrap = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return ({'n': len(y), 'families': len(groups), 'MAE_improvement': float(delta.mean()), 'family_ci95': np.quantile(bootstrap, [0.025, 0.975]).tolist(), 'family_bonferroni8': np.quantile(bootstrap, [0.025 / 8, 1 - 0.025 / 8]).tolist(), 'positive_means': 'candidate has lower MAE'}, bootstrap)

def main():
    start = time.monotonic()
    contract = verify()
    require(OUT.is_dir() and (not any(OUT.iterdir())), 'Output directory must be empty')
    acquisition = BUNDLE / 'results'
    rows = [json.loads(line) for line in (acquisition / 'snapshot.jsonl').read_text().splitlines() if line.strip()]
    cohort = js(ROOT / 'nb201-v1-imagenet16-full/cohort.json')
    with (ROOT / 'nb201-v1-imagenet16-full/targets.csv').open(newline='') as f:
        target_rows = list(csv.DictReader(f))
    members = {int(r['arch_index']): r for r in cohort['architectures']}
    targets = {int(r['architecture_id']): r for r in target_rows}
    by_id = {int(r['architecture_id']): r for r in rows}
    require(len(rows) == len(by_id) == 799 and len(members) == len(target_rows) == len(targets) == 1100, 'Partial/full cohort count mismatch')
    selection = js(BUNDLE / 'selection.json')
    require(set(by_id) == set(selection['included_ids']) and set(by_id).issubset(members) and set(members) == set(targets), 'Frozen partial-cohort identity mismatch')
    records = [by_id[i] for i in sorted(by_id)]
    for r in records:
        index = r['architecture_id']
        (member, target) = (members[index], targets[index])
        require(r['record_id'] == member['id'] == target['record_id'] and r['architecture_string'] == member['arch_str'] and (r['family'] == member['arch_family']) and (r['dataset'] == 'ImageNet16-120') and (r['candidate_seed'] == 777) and (r['split'] == target['split']) and (index in cohort['splits'][r['split']]) and (r['target_kind'] == 'final_validation_accuracy') and (r['target_epoch'] == 200) and (r['benchmark_target_seeds'] == [777, 888]) and (r['target'] == float(target['target_final_validation_accuracy'])), 'Frozen record/target mismatch')
        require([e['epoch'] for e in r['epochs']] == [1, 2, 3], 'Early checkpoint mismatch')
        for e in r['epochs']:
            require(all((math.isfinite(e[k]) for k in ('validation_accuracy', 'taylor', *LC))), 'Nonfinite feature')
            require(0 <= e['validation_accuracy'] <= 100, 'Invalid accuracy')
            require(e['taylor'] >= 0 and e['train_loss'] >= 0 and (e['probe_loss'] >= 0), 'Invalid loss/Taylor')
            require(0 <= e['true_confidence'] <= e['max_confidence'] <= 1, 'Invalid confidence')
            require(0 <= e['entropy'] <= math.log(120) + 1e-12, 'Invalid entropy')
    require(len({r['record_id'] for r in records}) == 799, 'Duplicate record IDs')
    group_sets = {split: {r['family'] for r in records if r['split'] == split} for split in ('train', 'val', 'test')}
    require(not group_sets['train'] & group_sets['val'] and (not group_sets['train'] & group_sets['test']) and (not group_sets['val'] & group_sets['test']), 'Family leakage')
    feature_names = ['acc1', 'acc2', 'acc3', 'd32', 'd21']
    feature_names += [f'taylor_e{e}' for e in (1, 2, 3)]
    feature_names += [f'{key}_e{e}' for key in LC for e in (1, 2, 3)]
    matrix = []
    for r in records:
        epochs = r['epochs']
        acc = [e['validation_accuracy'] for e in epochs]
        matrix.append(acc + [acc[2] - acc[1], acc[1] - acc[0]] + [e['taylor'] for e in epochs] + [e[key] for key in LC for e in epochs])
    X = np.asarray(matrix, dtype=float)
    y = np.array([r['target'] for r in records], dtype=float)
    require(X.shape == (799, len(feature_names)) and np.isfinite(X).all(), 'Feature matrix invalid')
    require(np.isfinite(y).all() and ((0 <= y) & (y <= 100)).all(), 'Targets invalid')
    fit_indices = np.array([i for (i, r) in enumerate(records) if r['split'] in ('train', 'val')])
    test_indices = np.array([i for (i, r) in enumerate(records) if r['split'] == 'test'])
    require(len(fit_indices) == 601 and len(test_indices) == 198 and {s: sum(r['split'] == s for r in records) for s in selection['counts']} == selection['counts'], 'Frozen partial split changed')
    arms = {'A': list(range(5)), 'A_LC': list(range(5)) + list(range(8, len(feature_names))), 'A_T': list(range(8)), 'A_LC_T': list(range(len(feature_names)))}
    yt = y[test_indices]
    families = np.array([records[i]['family'] for i in test_indices])
    populations = {'all': np.ones(len(yt), dtype=bool), 'target_ge20': yt >= 20.0}
    put(OUT / 'fit_contract.json', {'seeds': list(SEEDS), 'candidate_training_seed': 777, 'fit_records': len(fit_indices), 'test_records': len(test_indices), 'arms': {name: [feature_names[i] for i in cols] for (name, cols) in arms.items()}, 'histgb_parameters': HistGradientBoostingRegressor(random_state=42, **PARAMS).get_params(), 'imputer': 'Mean imputer fitted on train+validation only; all inputs finite', 'target': 'Unrounded mean epoch200 x-valid accuracy over seeds777/888', 'primary_contrast': 'A_LC -> A_LC_T', 'primary_population': 'target_ge20', 'supporting_population': 'all', 'uncertainty': '10000 operation-family cluster bootstrap draws; seed0; seedwise errors averaged before resampling; no seed resampling', 'multiplicity': 'Bonferroni8 intervals for four MAE contrasts x two populations. Other metrics are descriptive.', 'analysis_scope': 'partial_cohort', 'limitations': 'PARTIAL-COHORT supporting analysis: validated completed acquisitions only. Storage interruption truncated ordered shards; inclusion is not random. Original split assignments retained; identical records across all four arms. Intervals condition on this subset, split and fitted predictors and do not account for acquisition-selection bias. Target>=20 is an operational threshold.', 'packages': contract['packages'], 'sources_sha256': contract['sources_sha256']})
    feature_rows = []
    for (r, values) in zip(records, X):
        feature_rows.append({'record_id': r['record_id'], 'architecture_id': r['architecture_id'], 'family': r['family'], 'split': r['split'], 'target': r['target'], **{name: float(v) for (name, v) in zip(feature_names, values)}})
    write_csv(OUT / 'features.csv', feature_rows)
    predictions = {}
    panel = []
    metric_rows = []
    with threadpool_limits(limits=1):
        for (arm, columns) in arms.items():
            predictions[arm] = []
            for seed in SEEDS:
                model = make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(random_state=seed, **PARAMS))
                model.fit(X[fit_indices][:, columns], y[fit_indices])
                prediction = model.predict(X[test_indices][:, columns])
                require(np.isfinite(prediction).all(), 'Nonfinite prediction')
                predictions[arm].append(prediction)
                for (i, value) in zip(test_indices, prediction):
                    r = records[i]
                    panel.append({'record_id': r['record_id'], 'architecture_id': r['architecture_id'], 'family': r['family'], 'condition': arm, 'seed': seed, 'target': float(y[i]), 'prediction': float(value)})
                for (population, mask) in populations.items():
                    metric_rows.append({'population': population, 'condition': arm, 'seed': seed, 'n': int(mask.sum()), **metric_row(yt[mask], prediction[mask])})
                print('FIT_COMPLETE', arm, seed, flush=True)
            predictions[arm] = np.array(predictions[arm])
    write_csv(OUT / 'predictions.csv', panel)
    write_csv(OUT / 'metrics_by_seed.csv', metric_rows)
    results = {}
    comparisons = {}
    bootstrap_arrays = {}
    for (population, mask) in populations.items():
        results[population] = {}
        comparisons[population] = {}
        for arm in arms:
            selected = [r for r in metric_rows if r['population'] == population and r['condition'] == arm]
            means = {}
            for metric in ('MAE', 'RMSE', 'Pearson', 'Spearman', 'R2'):
                values = [r[metric] for r in selected]
                means[metric] = float(np.mean(values)) if all((v is not None for v in values)) else None
            results[population][arm] = {'n': int(mask.sum()), 'mean_metrics': means, 'unique_prediction_vectors_across_seeds': len(np.unique(predictions[arm][:, mask], axis=0))}
        for (reference, candidate) in CONTRASTS:
            name = reference + ' -> ' + candidate
            (comparison, draws) = paired_mae(yt[mask], predictions[reference][:, mask], predictions[candidate][:, mask], families[mask])
            comparisons[population][name] = comparison
            bootstrap_arrays[population + '__' + reference + '__' + candidate] = draws
    put(OUT / 'results.json', results)
    put(OUT / 'comparisons.json', comparisons)
    np.savez_compressed(OUT / 'bootstrap_draws.npz', **bootstrap_arrays)
    with (OUT / 'predictions.csv').open(newline='') as f:
        saved = list(csv.DictReader(f))
    require(len(saved) == len(panel), 'Prediction readback count mismatch')
    require(len({(r['condition'], r['seed'], r['record_id']) for r in saved}) == len(saved), 'Duplicate predictions')
    for (a, b) in zip(saved, panel):
        require(a['record_id'] == b['record_id'] and a['condition'] == b['condition'] and (int(a['seed']) == b['seed']) and (float(a['target']) == b['target']) and (float(a['prediction']) == b['prediction']), 'Prediction readback mismatch')
    verify()
    put(OUT / 'complete.json', {'status': 'complete', 'fit_records': len(fit_indices), 'test_records': len(test_indices), 'conditions': list(arms), 'seeds': list(SEEDS), 'elapsed_seconds': time.monotonic() - start, 'files_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    print('\nPRIMARY COMPARISON: A_LC -> A_LC_T')
    print(json.dumps(comparisons['target_ge20']['A_LC -> A_LC_T'], indent=2))
    print('\nMEAN METRICS')
    print(json.dumps(results, indent=2))
    print('IMAGENET16_PARTIAL_HISTGB_COMPLETE', flush=True)
if __name__ == '__main__':
    main()
