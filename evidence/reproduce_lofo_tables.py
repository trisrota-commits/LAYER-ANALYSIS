#!/usr/bin/env python
"""Read-only frozen-archive LOFO reproduction. Requires numpy and scipy; no training.
Example: python reproduce_lofo_tables.py --archive inputs.zip --output new.log
Existing output is always refused. --verify-output checks an existing report and
exercises the refusal path without changing it. No archive extraction is needed.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, t

SHA = '311fc68d41b988163b12a2af359280eea6433bd98a27ef7a274dbdf77e47cb69'
FAMILIES = ('cnn', 'detector', 'segmentation', 'other', 'transformer')
SEEDS = tuple(range(42, 48))
METRICS = ('RMSE', 'MAE', 'Pearson', 'Spearman', 'R2')
HERE = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def member(path):
    return path.removeprefix('/a/mm/')

def corr(y, p):
    yc = y - y.mean()
    pc = p - p.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(yc * yc) * np.sum(pc * pc, axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (pc @ yc) / denom
    # Explicit constant checks avoid floating-point nonzero centered residuals.
    out[(np.ptp(p, axis=1) == 0) | (np.ptp(y) == 0)] = np.nan
    return np.clip(out, -1, 1)

def metrics(y, p):
    """One row per fitted seed; average ranks for ties; undefined is NaN."""
    err = p - y
    mse = np.mean(err ** 2, axis=1)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(err ** 2, axis=1) / sst if np.ptp(y) > 0 else np.full(len(p), np.nan)
    return np.column_stack((np.sqrt(mse), np.mean(np.abs(err), axis=1),
                            corr(y, p), corr(rankdata(y), rankdata(p, axis=1)), r2))

def prompts(z, path):
    rows = [json.loads(x) for x in z.read(path).decode().splitlines() if x.strip()]
    result = []
    for row in rows:
        users = '\n'.join(x['content'] for x in row['messages'] if x['role'] == 'user')
        targets = [json.loads(x['content']) for x in row['messages'] if x['role'] == 'assistant']
        datasets = re.findall(r'^dataset:\s*(\S+)\s*$', users, re.M)
        require(len(targets) == len(datasets) == 1, 'Ambiguous prompt ' + path)
        result.append((row['architecture_id'], datasets[0], float(targets[0]['best_accuracy'])))
    return result

def verify_report(args):
    before = args.output.read_bytes()
    text = before.decode()
    require(text.count('DESCRIPTIVE ') == 15, 'Expected 15 descriptives')
    require(text.count('BOOTSTRAP ') == 25, 'Expected 25 bootstrap rows')
    require(text.count('PER_SEED ') == 90, 'Expected 90 per-seed rows')
    require('manifest_files_verified=677' in text, 'Manifest verification missing')
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--archive', str(args.archive),
                           '--output', str(args.output)], capture_output=True, text=True)
    require(proc.returncode != 0 and 'Refusing to overwrite' in proc.stderr, 'Overwrite was not refused')
    require(args.output.read_bytes() == before, 'Existing output changed')
    # Independent scalar SciPy checks plus constant-metric and duplicated-group tests.
    from scipy.stats import pearsonr, spearmanr
    y = np.array([1., 2., 2., 5.]); p = np.array([[1., 3., 2., 4.], [2., 2., 2., 2.]])
    m = metrics(y, p)
    require(np.isclose(m[0, 2], pearsonr(y, p[0]).statistic), 'Pearson check')
    require(np.isclose(m[0, 3], spearmanr(y, p[0]).statistic), 'Spearman check')
    require(np.isnan(m[1, 2:4]).all(), 'Constant predictions must invalidate correlations')
    require(np.isnan(metrics(np.ones(4), p)[:, 2:]).all(), 'Constant targets must invalidate correlations/R2')
    print('PASS: report 15 descriptives / 90 per-seed / 25 bootstrap rows; metric checks; overwrite refused; unchanged SHA256=' + sha(before))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=HERE / 'holdout-reproduction-inputs-728444c5.zip')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'experiment_logs/lofo_tables_reproduced.log')
    parser.add_argument('--reference', type=Path, default=HERE / 'overleaf-logs-only-20261015-100424/experiment_logs/paired_holdout_tables.log')
    parser.add_argument('--verify-output', action='store_true')
    args = parser.parse_args()
    if args.verify_output:
        verify_report(args)
        return
    if args.output.exists():
        parser.error('Refusing to overwrite: ' + str(args.output))
    require(args.output.resolve() not in (args.archive.resolve(), args.reference.resolve()), 'Output must be new')
    archive_hash = sha(args.archive.read_bytes())
    require(archive_hash == SHA, 'Archive SHA256 mismatch: ' + archive_hash)
    reference_bytes = args.reference.read_bytes()
    ref = reference_bytes.decode('utf-8-sig')
    expected = {}
    for line in ref.split('## NEW six-seed architecture-grouped Taylor LOFO bootstrap')[-1].splitlines():
        cells = line.strip('|').split('|')
        if len(cells) == 5 and cells[1] in METRICS:
            family = re.match('[A-Za-z]+', cells[0]).group().lower()
            expected[family, cells[1]] = (float(cells[2]), *map(float, cells[3].strip('[]').split(',')), int(cells[4]))
    require(len(expected) == 25, 'Reference lacks exactly 25 six-seed rows')
    lines = ['LOFO TABLES: INDEPENDENT PREDICTION RECOMPUTATION',
             'archive=' + str(args.archive), 'archive_sha256=' + archive_hash,
             'reference=' + str(args.reference), 'reference_sha256=' + sha(reference_bytes),
             'script_sha256=' + sha(Path(__file__).read_bytes()),
             f'Python={platform.python_version()} NumPy={np.__version__} SciPy={scipy.__version__}',
             'Seeds=42,43,44,45,46,47; all five metrics computed separately per seed then averaged.',
             'Descriptive CI=mean +/- scipy.stats.t.ppf(.975,5)*sample_std(ddof=1)/sqrt(6); unclipped.',
             'Bootstrap=10000; numpy.random.default_rng(0) reset each fold; PCG64; first-appearance architecture groups;',
             'rng.integers(0,group_count,size=group_count), concatenate every row of each sampled group in draw order.',
             'No seed resampling. Record-weighted metrics within each seed. Positive favors TaylorOFF: baseline-candidate errors, candidate-baseline correlations/R2.',
             'A draw is invalid per metric if ANY of six seed differences is nonfinite. CI=np.quantile(valid,[.025,.975],method=linear).',
             'CSV run_id is empty: ordered prompt architecture_id establishes grouping; CSV index/dataset/target alignment is checked, not an independent CSV architecture identifier.',
             'LIMITATION: original42-44 directories are the supplied seed cohort; absent seed receipts/training arguments cannot independently prove their fitting seeds.',
             'LIMITATION: source hashes recorded by historical contracts for files absent from archive cannot be revalidated. No training or inference executed.']
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP entries')
        manifest = json.loads(z.read('reproduction_input_manifest.json'))['files']
        require(set(names) == set(manifest) | {'reproduction_input_manifest.json'}, 'Manifest/archive membership differs')
        for path, entry in manifest.items():
            data = z.read(path)
            require(sha(data) == entry['sha256'] and len(data) == entry['bytes'], 'Manifest mismatch ' + path)
        lines.append(f'manifest_files_verified={len(manifest)} manifest_sha256={sha(z.read("reproduction_input_manifest.json"))}')
        print(lines[-1], flush=True)
        plan = json.loads(z.read('lofo-taylor-off-v1/plan.json'))
        contract = json.loads(z.read('lofo-original-seeds45-47-v1/contract.json'))
        require(contract['seeds'] == [45,46,47] and contract['zcp_enabled'] is True, 'Original contract cohort/toggle')
        require(len(plan['runs']) == 30 and len(contract['runs']) == 30, 'Run cohort mismatch')
        checked, absent = set(), set()
        def pinned(mapping):
            for path, value in mapping.items():
                p = member(path)
                if p in manifest:
                    require(manifest[p]['sha256'] == value, 'Pinned hash mismatch ' + p)
                    checked.add(p)
                else:
                    absent.add(p)
        pinned(plan['pinned_sha256'])
        pinned(contract['source_sha256'])
        for run in contract['runs'].values():
            pinned(run['input_and_source_sha256'])
        lines.append(f'Historical pinned hashes verified when archived={len(checked)}; unavailable unique files={len(absent)}')
        lines.extend('UNAVAILABLE_PINNED_SOURCE ' + x for x in sorted(absent))
        comparisons = []
        for family in FAMILIES:
            print('Recomputing ' + family, flush=True)
            fold = plan['folds'][family]
            fm = json.loads(z.read(member(fold['manifest'])))
            require(fm['counts'] == fold['counts'] and fm['holdout'] == family, 'Fold manifest mismatch')
            basepath = f'out/generalization_prompts/lofo_{family}_3acc_trainonly_v1'
            frozen = prompts(z, basepath + '/no_layer/test_llm_dataset.jsonl')
            require(len(frozen) == fold['counts']['test'], 'Test count mismatch')
            y = np.array([r[2] for r in frozen])
            require(np.isfinite(y).all(), 'Nonfinite targets')
            by_group = {}
            for i, row in enumerate(frozen):
                require(bool(row[0]), 'Missing architecture ID')
                by_group.setdefault(row[0], []).append(i)
            groups = [np.array(ids) for ids in by_group.values()]
            lines.append(f'FOLD {family} records={len(y)} groups={len(groups)} split_counts={json.dumps(fold["counts"], sort_keys=True)}')
            predictions, values = {}, {}
            for arm, mode in [('layerON','layer'), ('no-layerON','no-layer'), ('TaylorOFF','taylor')]:
                pp = []
                for seed in SEEDS:
                    if mode == 'taylor':
                        run = plan['runs'][f'{family}-taylor-s{seed}']
                        require(run['family'] == family and run['seed'] == seed and run['zcp_enabled'] is False and run['layer_features'] == ['taylor_imp'], 'Taylor identity mismatch')
                        root, prompt = member(run['output']), member(run['data'])
                    else:
                        prompt = basepath + '/' + ('no_layer' if mode == 'no-layer' else 'layer')
                        root = (f'out/accuracy_predictor_generalization_lofo_v1/lofo_{family}_3acc/{mode}-s{seed}' if seed < 45 else f'lofo-original-seeds45-47-v1/results/{family}-{mode}-s{seed}')
                        if seed >= 45:
                            matches = [r for r in contract['runs'].values() if member(r['output']) == root]
                            require(len(matches) == 1, 'Original contract missing run')
                            run = matches[0]
                            require(run['family'] == family and run['mode'] == mode.replace('-', '_') and run['seed'] == seed and member(run['prompt']) == prompt and run['counts'] == fold['counts'], f'Original contract run mismatch: {run} versus {(family,mode,seed,prompt,fold["counts"])}')
                    require(prompts(z, prompt + '/test_llm_dataset.jsonl') == frozen, 'Ordered prompt architecture/dataset/target mismatch')
                    csvpath = root + '/test_predictions.csv'
                    rows = list(csv.DictReader(io.StringIO(z.read(csvpath).decode())))
                    require(len(rows) == len(frozen), 'CSV count mismatch ' + root)
                    pred = []
                    for i, (row, truth) in enumerate(zip(rows, frozen)):
                        require(int(row['idx']) == i and row['dataset'] == truth[1] and float(row['gt_best_accuracy']) == truth[2], 'CSV alignment mismatch ' + root)
                        pred.append(float(row['pred_best_accuracy']))
                    require(np.isfinite(pred).all(), 'Nonfinite prediction ' + root)
                    for receipt in ('seed_verified.json', 'run_complete.json'):
                        path = root + '/' + receipt
                        if path in names:
                            rr = json.loads(z.read(path))
                            require(rr['seed'] == seed, 'Receipt seed mismatch')
                            if receipt == 'run_complete.json':
                                for key in ('family','arm','data','output','counts','zcp_enabled'):
                                    require(rr[key] == run[key], 'Completion identity mismatch ' + key)
                            if 'test_counts' in rr:
                                require(rr['test_counts']['total_samples'] == len(y) and rr['test_counts']['valid_predictions'] == len(y) and rr['test_counts']['parse_failures'] == 0, 'Seed receipt counts')
                            lines.append('RECEIPT_VERIFIED ' + path)
                        else:
                            lines.append('RECEIPT_UNAVAILABLE ' + path)
                    pp.append(pred)
                    lines.append('RUN_ALIGNMENT_VERIFIED ' + csvpath)
                predictions[arm] = np.array(pp)
                vals = metrics(y, predictions[arm])
                require(np.isfinite(vals).all(), 'Undefined full-test metric')
                values[arm] = vals
                for seed, row in zip(SEEDS, vals):
                    lines.append(f'PER_SEED {family} {arm} s{seed} ' + ' '.join(f'{m}={v:.15g}' for m,v in zip(METRICS,row)))
                means = vals.mean(axis=0)
                half = t.ppf(.975,5) * vals.std(axis=0,ddof=1) / np.sqrt(6)
                lines.append(f'DESCRIPTIVE {family} {arm} ' + ' '.join(f'{m}={v:.10f}[{v-h:.10f},{v+h:.10f}]' for m,v,h in zip(METRICS,means,half)))
            sign = np.array([-1,-1,1,1,1])
            point = ((values['TaylorOFF'] - values['no-layerON']) * sign).mean(axis=0)
            draws = np.full((10000,5), np.nan)
            rng = np.random.default_rng(0)
            for b in range(10000):
                idx = np.concatenate([groups[g] for g in rng.integers(0,len(groups),size=len(groups))])
                diff = (metrics(y[idx],predictions['TaylorOFF'][:,idx]) - metrics(y[idx],predictions['no-layerON'][:,idx])) * sign
                valid = np.isfinite(diff).all(axis=0)
                draws[b,valid] = diff[:,valid].mean(axis=0)
            for j, metric in enumerate(METRICS):
                finite = draws[np.isfinite(draws[:,j]),j]
                low, high = np.quantile(finite,[.025,.975],method='linear')
                observed = (point[j],low,high,len(finite))
                old = expected[family,metric]
                matches = [abs(a-b) <= .00000050001 for a,b in zip(observed[:3],old[:3])] + [observed[3] == old[3]]
                status = 'MATCH' if all(matches) else 'MISMATCH'
                comparisons.append(status)
                lines.append(f'BOOTSTRAP {family} {metric} delta={point[j]:.12f} CI=[{low:.12f},{high:.12f}] valid={len(finite)}/10000 reference={old} comparison={status} fields_match={matches}')
                print(lines[-1], flush=True)
        lines.append(f'COMPARISON rows=25 matches={comparisons.count("MATCH")} mismatches={comparisons.count("MISMATCH")} tolerance=half-unit-at-six-decimals; counts exact')
        lines.append('SOURCE_HASHES: every manifest entry verified before analysis; original artifacts remain unchanged.')
        lines.extend('SOURCE_SHA256 ' + e['sha256'] + ' ' + p for p,e in sorted(manifest.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(lines[-len(manifest)-2], flush=True)
    print('WROTE ' + str(args.output), flush=True)

if __name__ == '__main__':
    main()
