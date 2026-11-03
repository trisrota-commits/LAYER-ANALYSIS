#!/usr/bin/env python3
"""Full initialization-Taylor comparison; no Qwen or candidate training.

Depends on the existing, hash-pinned init-taylor-pilot-v2 bundle, read-only.
prepare/submit/show run on w0-mi; worker/score-one run inside the GPU Job.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback

HOST = Path('/shared/ssd/home/b-t-bose/lmur')
POD = Path('/a/mm')
OUTPUT = Path('/a/zcp-results')
NAME = 'init-taylor-comparison-v1'
PILOT = 'init-taylor-pilot-v2'
PILOT_SHA = '630004ddf077e3557c6f6f1c9d1b995627a2da2b0245cec293a82c437ec58577'
RAW = 'out/acc_predict_rebuilt_probe/llm_finetuning_data.jsonl'
MANIFEST = 'out/generalization_manifests_85_5_10/acc_predict_grouped_85_5_10_split_manifest.json'
NAMESPACE = 'b-t-bose'
SEEDS = (42, 43, 44, 45, 46, 47)
METRICS = ('RMSE', 'MAE', 'Pearson', 'Spearman', 'R2')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def from_pod(root, path):
    return root / str(PurePosixPath(path).relative_to('/a/mm'))


def pilot_module(root):
    path = root / PILOT / 'runner.py'
    require(sha(path) == PILOT_SHA, 'Pilot runner differs from validated v2; stop')
    spec = importlib.util.spec_from_file_location('verified_taylor_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kube(*args):
    return subprocess.run(['kubectl', '-n', NAMESPACE, *args], text=True,
                          capture_output=True, check=True).stdout


def collect(root):
    """Resolve exact raw records through prompt architecture IDs and CSV idx."""
    raw = jsonl(root / RAW)
    manifest = load(root / MANIFEST)
    require(manifest['source_sha256'] == sha(root / RAW), 'Manifest snapshot hash mismatch')
    assignments = manifest['record_to_split']
    require(len(raw) == len({r['id'] for r in raw}) == 1151, 'Raw count/ID mismatch')
    require(set(assignments) == {r['id'] for r in raw}, 'Manifest ID coverage mismatch')
    require(dict(Counter(assignments.values())) == {'train': 978, 'validation': 58, 'test': 115},
            'Frozen split counts changed')
    records, inputs = [], []
    groups = {s: set() for s in ('train', 'validation', 'test')}
    for index, raw_row in enumerate(raw):
        split = assignments[raw_row['id']]
        groups[split].add(raw_row['nn'])
        target = round(float(raw_row['best_accuracy']), 2)
        require(math.isfinite(target), 'Nonfinite raw target')
        record = {k: raw_row[k] for k in ('id', 'nn', 'dataset', 'task')}
        record.update(index=index, split=split, target=target,
                      eligible=raw_row['task'] == 'img-classification')
        records.append(record)
        item = {k: raw_row[k] for k in ('id', 'nn', 'dataset', 'task', 'nn_code', 'prm')}
        if isinstance(item['prm'], str):
            item['prm'] = json.loads(item['prm'])
        require(isinstance(item['prm'], dict), f'Invalid parameters: {item["id"]}')
        inputs.append(item)
    require(not (groups['train'] & groups['validation'] or groups['train'] & groups['test']
                 or groups['validation'] & groups['test']), 'Architecture split overlap')
    test = [r for r in records if r['split'] == 'test']
    by_arch = {r['nn']: r for r in test}
    require(len(by_arch) == 115 and all(r['eligible'] for r in test), 'Unexpected test population')
    require(dict(Counter(r['split'] for r in records if r['eligible'])) ==
            {'train': 966, 'validation': 58, 'test': 115}, 'Classification counts changed')
    files = [root / RAW, root / MANIFEST]
    baseline, reference = {}, None
    for seed in SEEDS:
        owner = ('zcp-off-toggle-refit7-v1' if seed <= 44 else
                 'taylor-3acc-zcp-off-seeds45-47-v1')
        contract_path = root / owner / 'contract.json'
        run = load(contract_path)['runs'][f'3acc-taylor-s{seed}']
        prompt_path = from_pod(root, run['data']) / 'test_llm_dataset.jsonl'
        csv_path = from_pod(root, run['output']) / 'test_predictions.csv'
        require(sha(prompt_path) == run['input_sha256']['test_llm_dataset.jsonl'],
                f'Test prompt hash drift: seed {seed}')
        prompts = jsonl(prompt_path)
        with csv_path.open(newline='', encoding='utf-8') as f:
            predictions = list(csv.DictReader(f))
        require(len(prompts) == len(predictions) == 115, f'Wrong test count: {seed}')
        arches = [p['architecture_id'] for p in prompts]
        require(len(set(arches)) == 115 and set(arches) == set(by_arch), 'Prompt membership mismatch')
        require(sorted(int(r['idx']) for r in predictions) == list(range(115)), 'CSV index mismatch')
        values, identity = {}, []
        for row in sorted(predictions, key=lambda r: int(r['idx'])):
            idx = int(row['idx'])
            prompt = prompts[idx]
            record = by_arch[prompt['architecture_id']]
            answers = [m['content'] for m in prompt['messages'] if m['role'] == 'assistant']
            require(len(answers) == 1, 'Ambiguous prompt target')
            answer = json.loads(answers[0])
            target, pred = float(row['gt_best_accuracy']), float(row['pred_best_accuracy'])
            require(math.isfinite(pred) and math.isfinite(target), 'Nonfinite saved prediction/target')
            require(row['dataset'] == record['dataset'], 'CSV dataset differs')
            require(abs(target - record['target']) < 1e-8 and
                    abs(target - float(answer['best_accuracy'])) < 1e-8, 'Target mismatch')
            require(float(row['gt_best_epoch']) == float(answer['best_epoch']), 'Best epoch mismatch')
            values[record['id']] = pred
            identity.append((record['id'], record['nn'], row['dataset'], target, float(answer['best_epoch'])))
        if reference is None:
            reference = identity
        require(identity == reference, 'Cross-seed test identities/targets differ')
        baseline[str(seed)] = values
        files.extend((contract_path, prompt_path, csv_path))
    return records, inputs, baseline, files


def build_job(template, root):
    require(template['metadata']['name'] == PILOT, 'Expected saved v2 pilot Job')
    job = copy.deepcopy(template)
    job['metadata'] = {'name': NAME, 'namespace': NAMESPACE, 'labels': {'experiment-batch': NAME}}
    spec = job['spec']
    require('selector' not in spec and 'manualSelector' not in spec, 'Unexpected live Job selectors')
    spec['backoffLimit'] = 0
    spec['activeDeadlineSeconds'] = 86400
    spec['template']['metadata'] = {'labels': {'experiment-batch': NAME}}
    pod = spec['template']['spec']
    require(len(pod['containers']) == 1, 'Unexpected containers')
    volumes = {v['name']: v for v in pod['volumes']}
    require(volumes['source']['hostPath']['path'] == str(root), 'Wrong source mount')
    require(volumes['results']['hostPath']['path'] == str(root / PILOT / 'results'), 'Wrong old output mount')
    volumes['results']['hostPath']['path'] = str(root / NAME / 'results')
    container = pod['containers'][0]
    require(any(m['name'] == 'source' and m.get('readOnly') is True
                for m in container['volumeMounts']), 'Writable source refused')
    before = f'/a/mm/{PILOT}/runner.py worker'
    after = f'/a/mm/{NAME}/runner.py worker'
    require(len(container['args']) == 1 and container['args'][0].count(before) == 1,
            'Unexpected pilot entrypoint')
    container['args'][0] = container['args'][0].replace(before, after)
    return job


def prepare(root=HOST):
    bundle = root / NAME
    require(not bundle.exists(), f'Preserve existing bundle: {bundle}')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Job already exists')
    pilot = pilot_module(root)
    pilot_contract = pilot.check_bundle(root)
    pilot.verify(root)
    require(pilot_contract['seed'] == 0 and pilot_contract['batch_size'] == 8
            and pilot_contract['repeats'] == 2 and pilot_contract['deterministic_warn_only'], 'Pilot settings changed')
    records, inputs, baseline, files = collect(root)
    input_by_id = {r['id']: r for r in inputs}
    pilot_inputs = jsonl(root / PILOT / 'results/results.jsonl')
    original_inputs = load(root / PILOT / 'pilot_inputs.json')
    for item in original_inputs:
        require({k: item[k] for k in input_by_id[item['id']]} == input_by_id[item['id']],
                'Pilot candidate no longer matches raw configuration')
    reused = {r['id']: r for r in pilot_inputs}
    require(len(reused) == 3 and all(r['status'] == 'ok' for r in reused.values()), 'Pilot reuse incomplete')
    template = root / PILOT / 'job.json'
    job = build_job(load(template), root)
    subprocess.run(['kubectl', '-n', NAMESPACE, 'create', '--dry-run=server', '-f', '-'],
                   input=json.dumps(job), text=True, check=True)
    files.extend([root / PILOT / n for n in ('runner.py', 'contract.json', 'pilot_inputs.json',
                 'job.json', 'results/results.jsonl', 'results/summary.json', 'results/runtime.json')])
    files.extend(root / p for p in pilot_contract['sources'])
    contract = {'experiment': NAME, 'seed': 0, 'batch_size': 8, 'repeats': 2,
                'per_candidate_timeout_seconds': 120, 'bootstrap_replicates': 10000,
                'bootstrap_seed': 0, 'baseline_seeds': list(SEEDS),
                'calibration': 'fixed affine least squares, raw score, finite classification training scores only',
                'target': 'best_accuracy rounded to 2 decimals, matched to saved LLM targets',
                'validation': 'scored for coverage only; not used for fitting or selection',
                'sources': {p.relative_to(root).as_posix(): sha(p) for p in set(files)},
                'prepared': {}, 'test_order': [r['id'] for r in records if r['split'] == 'test']}
    bundle.mkdir()
    (bundle / 'results').mkdir()
    (bundle / 'inputs').mkdir()
    with (bundle / 'runner.py').open('xb') as f:
        f.write(Path(__file__).read_bytes())
    for record, item in zip(records, inputs):
        if record['eligible']:
            save(bundle / 'inputs' / f'{record["index"]:04d}.json', item)
    save(bundle / 'records.json', records)
    save(bundle / 'baseline_predictions.json', baseline)
    save(bundle / 'pilot_reuse.json', reused)
    save(bundle / 'job.json', job)
    for path in [bundle / n for n in ('runner.py', 'records.json', 'baseline_predictions.json',
                                      'pilot_reuse.json', 'job.json')] + list((bundle / 'inputs').glob('*.json')):
        contract['prepared'][path.relative_to(bundle).as_posix()] = sha(path)
    save(bundle / 'contract.json', contract)
    check(root)
    print('PREPARED; NOT SUBMITTED:', bundle)
    print('Classification scope: 966 train / 58 validation / 115 test; 12 non-classification training records excluded.')
    print('Three verified pilot records reused; seed=0; two same-seed checks per newly scored record.')


def check(root):
    bundle = root / NAME
    contract = load(bundle / 'contract.json')
    require(contract['experiment'] == NAME, 'Wrong experiment contract')
    for rel, expected in contract['sources'].items():
        require(sha(root / rel) == expected, f'Source/artifact drift: {rel}')
    for rel, expected in contract['prepared'].items():
        require(sha(bundle / rel) == expected, f'Prepared input drift: {rel}')
    return contract


def submit(root=HOST):
    check(root)
    bundle = root / NAME
    require(not any((bundle / 'results').iterdir()), 'Outputs exist; preserve and inspect')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Job exists; do not resubmit')
    print(kube('create', '--dry-run=server', '-f', str(bundle / 'job.json')).strip())
    print(kube('create', '-f', str(bundle / 'job.json')).strip())


def score_one(index, root=POD, output=OUTPUT):
    import torch
    bundle = root / NAME
    contract = load(bundle / 'contract.json')
    path = bundle / 'inputs' / f'{index:04d}.json'
    require(sha(path) == contract['prepared'][f'inputs/{index:04d}.json'], 'Candidate input drift')
    require(sha(Path(__file__)) == contract['prepared']['runner.py'], 'Worker source drift')
    pilot = pilot_module(root)
    for rel in pilot.SOURCES:
        require(sha(root / rel) == contract['sources'][rel], 'Extractor source drift')
    require(torch.cuda.is_available(), 'CUDA required')
    proxy = importlib.import_module('ab.gpt.util.method.zero_cost_proxies')
    layer = importlib.import_module('ab.nn.util.LayerAnalysis')
    require(Path(proxy.__file__).resolve() == (root / pilot.SOURCES[1]).resolve(), 'Wrong proxy import')
    require(Path(layer.__file__).resolve() == (root / pilot.SOURCES[0]).resolve(), 'Wrong layer import')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    row = load(path)
    result = {'id': row['id'], 'index': index, 'dataset': row['dataset'], 'nn': row['nn'],
              'status': 'failed', 'seed': contract['seed'], 'attempts': [],
              'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(0), 'deterministic_warn_only': True}}
    try:
        require(row['dataset'].lower() in proxy._DATASET_SHAPES, 'Unknown dataset shape')
        for shape, out_shape in proxy._shape_candidates(row['dataset']):
            attempt = {'shape': list(shape)}
            result['attempts'].append(attempt)
            try:
                for name in ('first', 'second'):
                    attempt[name] = pilot.measure(row, contract['seed'], contract['batch_size'],
                                                  shape, out_shape, proxy, layer.LayerAnalyzer)
                pilot.check_repeats(attempt['first'], attempt['second'])
                attempt['status'] = 'ok'
                result.update(status='ok', score=attempt['first']['score'])
                break
            except pilot.PilotInvariantError:
                attempt.update(status='failed', traceback=traceback.format_exc())
                break
            except Exception:
                attempt.update(status='failed', traceback=traceback.format_exc())
                if 'first' in attempt:
                    break
    except Exception:
        result['traceback'] = traceback.format_exc()
    save(output / 'candidates' / f'{index:04d}.json', result)
    print('SCORE_RESULT', row['id'], result['status'], flush=True)


def rank(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    counts = np.diff(np.r_[starts, len(values)])
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return ranks


def metrics(y, predictions):
    """One row per fitted predictor; never ensemble predictions across seeds."""
    import numpy as np
    y = np.asarray(y, dtype=float)
    p = np.atleast_2d(np.asarray(predictions, dtype=float))
    require(y.ndim == 1 and len(y) >= 2 and p.shape[1] == len(y), 'Metric shape mismatch')
    require(np.isfinite(y).all() and np.isfinite(p).all(), 'Nonfinite metric inputs')
    error = p - y
    yc = y - y.mean()
    pc = p - p.mean(axis=1, keepdims=True)
    yr = rank(y)
    yr -= yr.mean()
    pr = np.stack([rank(row) for row in p])
    pr -= pr.mean(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.column_stack((np.sqrt(np.mean(error ** 2, axis=1)), np.mean(abs(error), axis=1),
            np.sum(pc * yc, axis=1) / np.sqrt(np.sum(pc ** 2, axis=1) * np.sum(yc ** 2)),
            np.sum(pr * yr, axis=1) / np.sqrt(np.sum(pr ** 2, axis=1) * np.sum(yr ** 2)),
            1 - np.sum(error ** 2, axis=1) / np.sum(yc ** 2)))


def fit_affine(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.ndim == 1 and x.shape == y.shape and len(x) >= 2, 'Calibration coverage insufficient')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite calibration values')
    xc, yc = x - x.mean(), y - y.mean()
    require(float(xc @ xc) > 0, 'Constant training scores; cannot calibrate')
    slope = float((xc @ yc) / (xc @ xc))
    return slope, float(y.mean() - slope * x.mean())


def nullable(value):
    return float(value) if math.isfinite(float(value)) else None


def evaluate(records, scores, baseline, contract, output):
    import numpy as np
    train = [r for r in records if r['split'] == 'train' and r['eligible'] and r['id'] in scores]
    by_id = {r['id']: r for r in records}
    test = [by_id[rid] for rid in contract['test_order']]
    missing = [r['id'] for r in test if r['id'] not in scores]
    require(not missing, f'{len(missing)} test scores missing; no reduced-test comparison allowed. See extraction_summary.json.')
    require(len({r['nn'] for r in test}) == len(test), 'Test architecture duplication requires grouped bootstrap')
    slope, intercept = fit_affine([scores[r['id']] for r in train], [r['target'] for r in train])
    y = np.array([r['target'] for r in test])
    pred = slope * np.array([scores[r['id']] for r in test]) + intercept
    reference = np.array([[baseline[str(s)][r['id']] for r in test] for s in SEEDS])
    points = metrics(y, np.vstack((pred, reference)))
    direction = np.array([-1, -1, 1, 1, 1])
    improvement = (points[0] - points[1:].mean(axis=0)) * direction
    draws = np.empty((contract['bootstrap_replicates'], len(METRICS)))
    rng = np.random.default_rng(contract['bootstrap_seed'])
    all_predictions = np.vstack((pred, reference))
    for b in range(len(draws)):
        idx = rng.integers(0, len(y), len(y))
        sampled = metrics(y[idx], all_predictions[:, idx])
        draws[b] = (sampled[0] - sampled[1:].mean(axis=0)) * direction
    comparisons = {}
    for j, name in enumerate(METRICS):
        valid = draws[:, j][np.isfinite(draws[:, j])]
        interval = [float(v) for v in np.quantile(valid, [0.025, 0.975])] if len(valid) else [None, None]
        comparisons[name] = {'initialization_taylor': nullable(points[0, j]),
            'taylor_off_mean_s42_s47': nullable(points[1:, j].mean()),
            'improvement_positive_favors_initialization': nullable(improvement[j]),
            'paired_95ci': interval, 'valid_resamples': len(valid),
            'invalid_resamples': len(draws) - len(valid)}
    path = output / 'test_predictions.csv'
    with path.open('x', newline='', encoding='utf-8') as f:
        fields = ['run_id', 'architecture_id', 'dataset', 'gt_best_accuracy', 'init_taylor_score',
                  'pred_best_accuracy'] + [f'taylor_off_s{s}' for s in SEEDS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(test):
            writer.writerow({'run_id': record['id'], 'architecture_id': record['nn'],
                'dataset': record['dataset'], 'gt_best_accuracy': y[i],
                'init_taylor_score': scores[record['id']], 'pred_best_accuracy': pred[i],
                **{f'taylor_off_s{s}': reference[j, i] for j, s in enumerate(SEEDS)}})
    report = {'experiment': NAME, 'calibration': {'slope': slope, 'intercept': intercept,
        'training_finite': len(train), 'training_eligible': sum(r['split'] == 'train' and r['eligible'] for r in records),
        'input': 'raw initialization Taylor score only', 'fit': 'affine least squares; training partition only',
        'clipping': False, 'train_record_ids': [r['id'] for r in train]},
        'test_records': len(test), 'initialization_seed': contract['seed'],
        'baseline_seeds': list(SEEDS), 'metrics': comparisons,
        'baseline_per_seed': {str(s): {k: nullable(points[i+1, j]) for j, k in enumerate(METRICS)}
                              for i, s in enumerate(SEEDS)},
        'uncertainty': '10000 paired test-record percentile bootstrap draws; fitted proxy and six LLM models held fixed; no seed resampling',
        'limitations': ['Initialization seed 0 only; two identical-seed repeats are not independent seeds.',
                       'Only classification training records with successful scores enter calibration; coverage reported.',
                       'No validation/test tuning of the affine mapping; no claim of optimal proxy/calibrator.',
                       'Unadjusted exploratory intervals; earlier feature selection on this test set limits confirmation.',
                       'No claim that three-record pilot established full architecture compatibility.'],
        'test_predictions_sha256': sha(path)}
    save(output / 'report.json', report)
    verify_report(output)
    return report


def verify_report(output):
    import numpy as np
    report = load(output / 'report.json')
    require(sha(output / 'test_predictions.csv') == report['test_predictions_sha256'], 'Prediction hash mismatch')
    with (output / 'test_predictions.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == report['test_records'] and len({r['run_id'] for r in rows}) == len(rows), 'CSV identity/count mismatch')
    y = [float(r['gt_best_accuracy']) for r in rows]
    p = [[float(r['pred_best_accuracy']) for r in rows]] + [
         [float(r[f'taylor_off_s{s}']) for r in rows] for s in SEEDS]
    points = metrics(y, p)
    for j, name in enumerate(METRICS):
        for expected, actual in ((report['metrics'][name]['initialization_taylor'], points[0, j]),
                                 (report['metrics'][name]['taylor_off_mean_s42_s47'], points[1:, j].mean())):
            require((expected is None and not np.isfinite(actual)) or
                    (expected is not None and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)),
                    f'CSV metric mismatch: {name}')
    return report


def worker(root=POD, output=OUTPUT):
    import numpy as np
    contract = check(root)
    require(not any(output.iterdir()), 'Output directory not empty; preserve it')
    (output / 'candidates').mkdir()
    (output / 'logs').mkdir()
    save(output / 'runtime.json', {'numpy': np.__version__, 'python': sys.version,
                                   'contract_sha256': sha(root / NAME / 'contract.json')})
    records = load(root / NAME / 'records.json')
    reused = load(root / NAME / 'pilot_reuse.json')
    scores, outcomes = {}, []
    for record in records:
        index, rid = record['index'], record['id']
        result_path = output / 'candidates' / f'{index:04d}.json'
        outcome = {'id': rid, 'split': record['split'], 'task': record['task'], 'dataset': record['dataset']}
        if not record['eligible']:
            outcome['status'] = 'unsupported_nonclassification'
        elif rid in reused:
            result = {**reused[rid], 'index': index, 'reused_from': PILOT}
            save(result_path, result)
            scores[rid] = float(result['score'])
            outcome.update(status='ok', reused=True)
        else:
            try:
                with (output / 'logs' / f'{index:04d}.log').open('x', encoding='utf-8') as log:
                    proc = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                           'score-one', '--index', str(index)], stdout=log,
                                          stderr=subprocess.STDOUT, timeout=contract['per_candidate_timeout_seconds'])
                require(proc.returncode == 0, f'Child exit {proc.returncode}; see logs/{index:04d}.log')
                result = load(result_path)
                require(result['id'] == rid and result['index'] == index, 'Child result identity mismatch')
                outcome['status'] = result['status']
                if result['status'] == 'ok':
                    score = float(result['score'])
                    require(math.isfinite(score) and score >= 0, 'Invalid successful score')
                    scores[rid] = score
            except Exception:
                outcome.update(status='failed', traceback=traceback.format_exc())
        outcomes.append(outcome)
        with (output / 'coverage.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(outcome) + '\n')
            f.flush()
        print('PROGRESS', index + 1, '/', len(records), rid, outcome['status'], flush=True)
    check(root)
    coverage = {s: dict(Counter(o['status'] for o in outcomes if o['split'] == s))
                for s in ('train', 'validation', 'test')}
    save(output / 'scores.json', scores)
    save(output / 'extraction_summary.json', {'coverage': coverage, 'records': len(records),
        'eligible': sum(r['eligible'] for r in records), 'finite_scores': len(scores),
        'scope': 'seed=0, synthetic inputs/labels, two same-seed repeats; three verified pilot records reused'})
    print('EXTRACTION_COVERAGE', json.dumps(coverage), flush=True)
    report = evaluate(records, scores, load(root / NAME / 'baseline_predictions.json'), contract, output)
    check(root)
    save(output / 'complete.json', {'experiment': NAME, 'report_sha256': sha(output / 'report.json'),
        'contract_sha256': sha(root / NAME / 'contract.json')})
    print('INIT_TAYLOR_COMPARISON_VERIFIED', json.dumps(report['metrics']), flush=True)


def show(root=HOST):
    check(root)
    output = root / NAME / 'results'
    if (output / 'extraction_summary.json').is_file():
        print(json.dumps(load(output / 'extraction_summary.json'), indent=2))
    complete = load(output / 'complete.json')
    require(complete['contract_sha256'] == sha(root / NAME / 'contract.json'), 'Completion contract mismatch')
    require(complete['report_sha256'] == sha(output / 'report.json'), 'Report hash mismatch')
    report = load(output / 'report.json')
    csv_path = output / 'test_predictions.csv'
    require(sha(csv_path) == report['test_predictions_sha256'], 'Prediction hash mismatch')
    with csv_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    contract = load(root / NAME / 'contract.json')
    records = {r['id']: r for r in load(root / NAME / 'records.json')}
    require([r['run_id'] for r in rows] == contract['test_order'], 'Saved test identity/order mismatch')
    for row in rows:
        record = records[row['run_id']]
        require(row['architecture_id'] == record['nn'] and row['dataset'] == record['dataset']
                and float(row['gt_best_accuracy']) == record['target'], 'Saved test metadata mismatch')
    print('Calibration training records:', report['calibration']['training_finite'], '/', report['calibration']['training_eligible'])
    print('Metric | Init Taylor (seed 0) | Taylor/OFF (mean seeds 42-47) | Improvement | Paired 95% CI')
    def fmt(value):
        return 'undefined' if value is None else f'{value:.6f}'
    for name in METRICS:
        row = report['metrics'][name]
        print(name, '|', fmt(row['initialization_taylor']), '|', fmt(row['taylor_off_mean_s42_s47']),
              '|', fmt(row['improvement_positive_favors_initialization']), '|', row['paired_95ci'],
              '| invalid resamples:', row['invalid_resamples'])
    print('Positive improvement favors initialization Taylor; unadjusted intervals conditional on the fitted predictors.')
    print('ARTIFACT_VERIFIED: hashes and test identities; CSV metrics were recomputed inside the Job. No Qwen retraining.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'worker', 'score-one', 'show'))
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.action == 'score-one':
        require(args.index is not None, '--index is required')
        score_one(args.index)
    else:
        {'prepare': prepare, 'submit': submit, 'worker': worker, 'show': show}[args.action]()
