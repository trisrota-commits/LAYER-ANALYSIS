"""Six-seed nested outer metrics and paired within-fold architecture-group bootstrap.

No refitting. All classification records; fixed trained predictors and selections.
"""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from scipy.stats import rankdata

ARCHIVE = Path('C:/Users/trisr/Downloads/nested-outer-six-seeds-v1.tar.gz')
OUTPUT = Path('C:/Users/trisr/Downloads/nested-outer-six-seed-analysis-v1')
EXPECTED = 'c183e6640d634ed61a78b4b96f5ccecf4d1f84dcf5ff45ee0b42d92768ca9e0f'
SEEDS = list(range(42, 48))
ARMS = ['no-layer', 'taylor']
METRICS = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']
DRAWS = 10000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def correlation(y, pred):
    a = y - y.mean(axis=-1, keepdims=True)
    b = pred - pred.mean(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (a*b).sum(axis=-1) / np.sqrt((a*a).sum(axis=-1) * (b*b).sum(axis=-1))


def metrics(y, pred):
    error = pred - y
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - (error*error).sum(axis=-1) / ((y-y.mean(axis=-1, keepdims=True))**2).sum(axis=-1)
    return np.stack([np.sqrt((error*error).mean(axis=-1)), abs(error).mean(axis=-1),
                     correlation(y, pred), correlation(rankdata(y, axis=-1), rankdata(pred, axis=-1)), r2], axis=-1)


def group_blocks(folds, groups):
    require(set(folds) == {0, 1, 2}, 'Expected three outer folds')
    ownership = {}
    blocks = []
    for fold in range(3):
        family = []
        for group in sorted(set(groups[folds == fold])):
            require(group not in ownership, 'Architecture group occurs in multiple outer folds')
            ownership[group] = fold
            family.append(np.flatnonzero((folds == fold) & (groups == group)))
        require(len(family) > 1, 'Insufficient groups in fold')
        blocks.append(family)
    return blocks


def sample_indices(blocks, rng):
    return np.concatenate([block[i] for block in blocks for i in rng.integers(len(block), size=len(block))])


def load():
    require(sha(ARCHIVE.read_bytes()) == EXPECTED, 'Archive checksum mismatch')
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        members = tar.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate members')
        require(all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members), 'Unsafe archive member')
        payload = {m.name: tar.extractfile(m).read() for m in members}
    manifest = json.loads(payload['manifest.json'])
    require(manifest['seeds'] == SEEDS and manifest['conditions'] == ARMS, 'Unexpected seeds/conditions')
    require(set(payload) == set(manifest['files_sha256']) | {'manifest.json'}, 'Archive inventory mismatch')
    for name, digest in manifest['files_sha256'].items():
        require(sha(payload[name]) == digest, 'Member checksum mismatch: '+name)
    def js(name): return json.loads(payload[name])
    def csv_rows(name): return list(csv.DictReader(io.StringIO(payload[name].decode())))
    canonical = None
    prediction = np.empty((2, 6, 1139))
    selected_predictions = np.empty((6, 1139))
    selections = {}
    fold_metrics = []
    for si, seed in enumerate(SEEDS):
        prefix = f'seed{seed}/'
        contract = js(prefix+'outer_contract.json')
        selection = js(prefix+'selections.json')
        require(sha(payload[prefix+'selections.json']) == contract['selection_lock_sha256'], 'Selection-lock mismatch')
        require(selection.get('outer_results_used') is False, 'Selection must exclude outer results')
        expected = {f'outer-f{f}-{a}' for f in range(3) for a in ARMS}
        require(set(contract['runs']) == expected and contract['epochs'] == 7, 'Incomplete outer contract')
        selections[str(seed)] = selection['folds']
        for fold in range(3):
            left, right = [contract['runs'][f'outer-f{fold}-{arm}'] for arm in ARMS]
            chosen = selection['folds'][str(fold)]['trial']
            require(left['learning_rate'] == right['learning_rate'] == chosen['learning_rate'], 'Unmatched outer learning rates')
            require(left['trial_id'] == right['trial_id'] == chosen['trial_id'], 'Unmatched selection identity')
        for ai, arm in enumerate(ARMS):
            raw = {}
            for fold in range(3):
                key = f'outer-f{fold}-{arm}'; run = contract['runs'][key]
                require(run['seed'] == seed and run['phase'] == 'outer' and run['fold'] == fold and run['feature'] == arm, 'Run identity mismatch')
                source = prefix+'evidence/'+key+'/'
                registry = js(source+'registry.json')
                require(sha(payload[source+'registry.json']) == run['input_sha256']['registry.json'], 'Registry input hash mismatch')
                rows = csv_rows(source+'test_predictions.csv')
                done, fit = js(source+'test_complete.json'), js(source+'refit_complete.json')
                require(done['run'] == fit['run'] == key and done['epochs'] == fit['epochs'] == 7, 'Run/epoch receipt mismatch')
                require(len(rows) == len(registry) == run['eval_n'] == done['n_samples'], 'Prediction coverage mismatch')
                require(fit['training_records'] == run['fit_n'], 'Fitting-count mismatch')
                require(js(source+'nested_runtime.json')['packages'] == js(prefix+'environment_lock.json'), 'Runtime version mismatch')
                fold_target, fold_pred = [], []
                for index, (item, row) in enumerate(zip(registry, rows)):
                    require(int(row['idx']) == index and row['dataset'] == item['dataset'] and float(row['gt_best_accuracy']) == item['target'], 'Raw prediction/registry mismatch')
                    if item['task'] != 'img-classification':
                        continue
                    value = float(row['pred_best_accuracy'])
                    require(np.isfinite(value) and 0 <= value <= 100, 'Invalid prediction requires an explicit correlation policy')
                    rid = item['record_id']
                    require(rid not in raw, 'Duplicate out-of-fold record')
                    raw[rid] = (item['architecture_id'], item['dataset'], fold, float(item['target']), value)
                    fold_target.append(float(item['target'])); fold_pred.append(value)
                values = metrics(np.array(fold_target), np.array(fold_pred))
                require(np.isfinite(values).all(), 'Undefined fold metric')
                fold_metrics.append(dict(seed=seed, fold=fold, method=arm, n=len(fold_target), **dict(zip(METRICS, map(float, values)))))
            exported = csv_rows(prefix+arm+'.csv')
            ids = [r['record_id'] for r in exported]
            require(len(ids) == len(set(ids)) == len(raw) == 1139 and set(ids) == set(raw), 'Normalized export coverage mismatch')
            identity = []
            for row in exported:
                actual = (row['architecture_id'], row['dataset'], int(row['fold']), float(row['target']), float(row['prediction']))
                require(actual == raw[row['record_id']], 'Normalized export differs from original evidence')
                identity.append((row['record_id'], *actual[:-1]))
            if canonical is None: canonical = identity
            require(identity == canonical, 'Record/group/fold/target order differs across seeds or arms')
            prediction[ai, si] = [float(r['prediction']) for r in exported]
    y = np.array([r[4] for r in canonical]); folds = np.array([r[3] for r in canonical]); groups = np.array([r[1] for r in canonical])
    for si, seed in enumerate(SEEDS):
        for fold in range(3):
            arm = selections[str(seed)][str(fold)]['trial']['feature']
            selected_predictions[si, folds == fold] = prediction[ARMS.index(arm), si, folds == fold]
    require(np.isfinite(prediction).all() and np.isfinite(y).all(), 'Nonfinite data')
    return y, prediction, selected_predictions, folds, groups, canonical, selections, fold_metrics


def save_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    require(not OUTPUT.exists(), 'Existing output preserved')
    y, pred, selected, folds, groups, registry, selections, fold_metrics = load()
    blocks = group_blocks(folds, groups)
    scores = metrics(y, pred)
    selected_scores = metrics(y, selected)
    require(np.isfinite(scores).all() and np.isfinite(selected_scores).all(), 'Undefined metric')
    sign = np.array([-1, -1, 1, 1, 1])
    per_seed = []
    for si, seed in enumerate(SEEDS):
        for ai, arm in enumerate(ARMS):
            per_seed.append(dict(seed=seed, method=arm, n=len(y), **dict(zip(METRICS, map(float, scores[ai, si])))))
        per_seed.append(dict(seed=seed, method='selected-procedure', n=len(y), **dict(zip(METRICS, map(float, selected_scores[si])))))
    means = scores.mean(axis=1)
    gain = (means[1]-means[0])*sign
    seed_gains = (scores[1]-scores[0])*sign
    rng = np.random.default_rng(0)
    bootstrap = np.empty((DRAWS, 5))
    for draw in range(DRAWS):
        indices = sample_indices(blocks, rng)
        value = metrics(y[indices], pred[:, :, indices]).mean(axis=1)
        bootstrap[draw] = (value[1]-value[0])*sign
        if (draw+1) % 2000 == 0: print('BOOTSTRAP',draw+1,'/',DRAWS,flush=True)
    require(np.isfinite(bootstrap).all(), 'Undefined bootstrap draw; no silent draw exclusion')
    low, high = np.quantile(bootstrap,[.025,.975],axis=0)
    adjlow, adjhigh = np.quantile(bootstrap,[.025/5,1-.025/5],axis=0)
    paired = [dict(metric=metric, no_layer=float(means[0,k]), taylor=float(means[1,k]), improvement=float(gain[k]),
                   ci95_low=float(low[k]), ci95_high=float(high[k]), bonferroni5_low=float(adjlow[k]), bonferroni5_high=float(adjhigh[k]),
                   seeds_favoring_taylor=int((seed_gains[:,k]>0).sum()),
                   adjusted_result='Taylor improvement supported' if adjlow[k]>0 else 'Taylor worsening supported' if adjhigh[k]<0 else 'Inconclusive')
              for k,metric in enumerate(METRICS)]
    average = [dict(method=arm, **dict(zip(METRICS,map(float,means[ai])))) for ai,arm in enumerate(ARMS)]
    average.append(dict(method='selected-procedure',**dict(zip(METRICS,map(float,selected_scores.mean(axis=0))))))
    require(sha(ARCHIVE.read_bytes()) == EXPECTED, 'Archive changed during analysis')
    OUTPUT.mkdir()
    save_csv(OUTPUT/'metrics_by_seed.csv',per_seed)
    save_csv(OUTPUT/'metrics_by_fold.csv',fold_metrics)
    save_csv(OUTPUT/'mean_metrics.csv',average)
    save_csv(OUTPUT/'paired_bootstrap.csv',paired)
    np.savez_compressed(OUTPUT/'bootstrap_differences.npz',differences=bootstrap,seed_improvements=seed_gains,predictions=pred,target=y,fold=folds,group=groups)
    report = dict(status='complete',seeds=SEEDS,population='All1139 classification records; no target threshold',
                  records=len(y),groups_per_fold=[len(b) for b in blocks],records_per_fold=[int((folds==f).sum()) for f in range(3)],
                  mean_metrics=average,per_seed=per_seed,paired=paired,selections=selections,
                  method='10000 paired architecture-group bootstrap draws stratified by original outer fold; sample the observed number of classification-bearing groups within each fold with replacement; keep all classification records of sampled groups; pool folds, compute each metric per seed then average; identical resamples for arms and seeds; no seed resampling',
                  correction='Bonferroni5 across MAE,RMSE,Pearson,Spearman,R2 for Taylor vs no-layer only',
                  limitations=['Post-hoc supporting analysis conditional on the observed outer folds, fitted models and inner-selected settings',
                               'Not uncertainty for rerunning the full nested selection/training procedure',
                               'Seeds share records/folds and are not independent datasets',
                               'Selected-procedure metrics are descriptive; no additional inference is claimed'],
                  archive_sha256=EXPECTED,script_sha256=sha(Path(__file__).read_bytes()))
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUTPUT/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256={p.name:sha(p.read_bytes()) for p in OUTPUT.iterdir() if p.is_file()}),indent=2)+'\n')
    print('COMPLETE',OUTPUT,flush=True)
    print('GROUPS PER FOLD',report['groups_per_fold'])
    for r in average:print('MEAN',r)
    for r in paired:print('PAIRED',r)

if __name__=='__main__':main()
