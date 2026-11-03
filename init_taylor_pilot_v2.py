#!/usr/bin/env python3
"""Initialization-only Taylor engineering pilot; not an accuracy experiment.

On w0-mi: python3 init_taylor_pilot_v2.py prepare
Then:     python3 init_taylor_pilot_v2.py submit
After Job completion: python3 init_taylor_pilot_v2.py verify

Only a new external bundle is written. Existing sources/results are read-only.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import warnings
from unittest.mock import patch

HOST = Path('/shared/ssd/home/b-t-bose/lmur')
POD = Path('/a/mm')
NAME = 'init-taylor-pilot-v2'
NAMESPACE = 'b-t-bose'
SNAPSHOT = 'out/acc_predict_rebuilt_probe/llm_finetuning_data.jsonl'
SNAPSHOT_SHA = '093375e5f94fdbf100bffd82e44638f1fe45b670e440f8adb8d03569d9a39f01'
SOURCES = ('nn-dataset/ab/nn/util/LayerAnalysis.py',
           'nn-gpt/ab/gpt/util/method/zero_cost_proxies.py')


class PilotInvariantError(RuntimeError):
    """A correctness failure must not be bypassed by another input shape."""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def kubectl(*args):
    return subprocess.run(['kubectl', '-n', NAMESPACE, *args], check=True,
                          text=True, capture_output=True).stdout


def select_rows(path):
    """First classification record per distinct dataset; never inspect targets."""
    selected, datasets, ids = [], set(), set()
    count = 0
    with path.open(encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            identity = row.get('id')
            require(isinstance(identity, str) and identity not in ids,
                    f'Missing/duplicate snapshot ID at line {index + 1}')
            ids.add(identity)
            if (len(selected) == 3 or row.get('task') != 'img-classification'
                    or row.get('dataset') in datasets):
                continue
            keys = ('id', 'nn', 'dataset', 'task', 'nn_code', 'prm')
            require(all(row.get(k) is not None for k in keys),
                    f'Missing model fields: {identity}')
            item = {k: row[k] for k in keys}
            item['snapshot_line'] = index + 1
            if isinstance(item['prm'], str):
                item['prm'] = json.loads(item['prm'])
            require(isinstance(item['prm'], dict), f'Invalid prm: {identity}')
            require(isinstance(item['nn_code'], str), f'Invalid source: {identity}')
            selected.append(item)
            datasets.add(item['dataset'])
    require(count == 1151, f'Snapshot count changed: {count}')
    require(len(selected) == 3, 'Need three distinct classification datasets')
    return selected


def make_job(template, root):
    require(template.get('kind') == 'Job', 'Expected saved Job template')
    spec = copy.deepcopy(template['spec']['template']['spec'])
    require(len(spec['containers']) == 1 and not spec.get('initContainers'),
            'Unexpected sidecars/init containers; inspect before reuse')
    require(spec.get('nodeSelector') == {'kubernetes.io/hostname': 'w2-mi'},
            'Template node selector changed')
    volumes = {v['name']: v for v in spec['volumes']}
    require(set(volumes) == {'source', 'results', 'dshm'}, 'Mount contract changed')
    require(volumes['source']['hostPath']['path'] == str(root), 'Wrong source root')
    require(volumes['results']['hostPath']['path'] == str(root / 'zcp-five-cost-v1/results'),
            'Original output mount changed')
    volumes['results']['hostPath']['path'] = str(root / NAME / 'results')
    c = spec['containers'][0]
    require(c['name'] == 'benchmark', 'Unexpected container name')
    require(c['image'] == 'abrainone/ai-linux:cu12.6.3-latest', 'Template image changed')
    require(c['resources']['limits'].get('nvidia.com/gpu') == '1', 'Expected one GPU')
    mounts = {m['name']: m for m in c['volumeMounts']}
    require(mounts['source'].get('readOnly') is True
            and mounts['source']['mountPath'] == '/a/mm', 'Source must be read-only')
    require(mounts['results']['mountPath'] == '/a/zcp-results', 'Result mount changed')
    c['workingDir'] = '/tmp'
    c['command'] = ['/bin/bash', '-lc']
    c['args'] = [
        'set -euo pipefail\n'
        'export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\n'
        'export PYTHONDONTWRITEBYTECODE=1\n'
        'export TORCH_HOME=/tmp/torch-cache\n'
        'export HF_HOME=/tmp/hf-cache\n'
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
        'export CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
        'cd /tmp\n'
        f'python -u /a/mm/{NAME}/runner.py worker\n'
    ]
    spec['restartPolicy'] = 'Never'
    spec['automountServiceAccountToken'] = False
    return {'apiVersion': 'batch/v1', 'kind': 'Job',
            'metadata': {'name': NAME, 'namespace': NAMESPACE,
                         'labels': {'experiment-batch': NAME}},
            'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': 1800,
                     'template': {'metadata': {'labels': {'experiment-batch': NAME}},
                                  'spec': spec}}}


def prepare(root=HOST):
    bundle = root / NAME
    require(not bundle.exists(), f'Preserve existing bundle; refusing overwrite: {bundle}')
    require(not kubectl('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(),
            'Job already exists; inspect rather than rerun')
    require(sha(root / SNAPSHOT) == SNAPSHOT_SHA, 'Frozen snapshot hash mismatch')
    rows = select_rows(root / SNAPSHOT)
    template_path = root / 'zcp-five-cost-v1/pilot.json'
    job = make_job(load(template_path), root)
    contract = {
        'experiment': NAME, 'purpose': 'engineering pilot, no accuracy evaluation',
        'namespace': NAMESPACE, 'seed': 0, 'batch_size': 8, 'repeats': 2,
        'deterministic_algorithms': True, 'deterministic_warn_only': True,
        'inputs': 'synthetic standard-normal images and uniform random class labels',
        'aggregation': 'unweighted mean of finite per-layer taylor_imp; require every selected weight layer',
        'selection': 'first classification record per distinct dataset in frozen file, three records',
        'sources': {p: sha(root / p) for p in SOURCES},
        'snapshot_sha256': SNAPSHOT_SHA, 'template_sha256': sha(template_path),
        'runner_sha256': sha(Path(__file__)),
        'candidate_ids': [r['id'] for r in rows],
    }
    # Validate server acceptance before creating the bundle. No live Job yet.
    checked = subprocess.run(['kubectl', '-n', NAMESPACE, 'create', '--dry-run=server',
                              '-f', '-', '-o', 'name'], input=json.dumps(job),
                             text=True, capture_output=True, check=True)
    print(checked.stdout.strip())
    bundle.mkdir()
    (bundle / 'results').mkdir()
    with (bundle / 'runner.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    save(bundle / 'pilot_inputs.json', rows)
    save(bundle / 'job.json', job)
    contract['input_sha256'] = sha(bundle / 'pilot_inputs.json')
    contract['job_sha256'] = sha(bundle / 'job.json')
    save(bundle / 'contract.json', contract)
    require(load(bundle / 'pilot_inputs.json') == rows, 'Input readback failed')
    check_bundle(root)
    print('PREPARED; NOT SUBMITTED:', bundle)
    for row in rows:
        print('CANDIDATE:', row['id'], row['nn'], row['dataset'])


def check_bundle(root):
    bundle = root / NAME
    contract = load(bundle / 'contract.json')
    require(contract['experiment'] == NAME, 'Wrong experiment')
    for relative, expected in contract['sources'].items():
        require(sha(root / relative) == expected, f'Source drift: {relative}')
    require(sha(root / SNAPSHOT) == contract['snapshot_sha256'], 'Snapshot drift')
    for filename, key in [('runner.py', 'runner_sha256'),
                          ('pilot_inputs.json', 'input_sha256'), ('job.json', 'job_sha256')]:
        require(sha(bundle / filename) == contract[key], f'Bundle drift: {filename}')
    return contract


def submit(root=HOST):
    check_bundle(root)
    bundle = root / NAME
    require(not any((bundle / 'results').iterdir()), 'Outputs exist; do not rerun')
    require(not kubectl('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(),
            'Job exists; do not resubmit')
    path = str(bundle / 'job.json')
    print(kubectl('create', '--dry-run=server', '-f', path).strip())
    print(kubectl('create', '-f', path).strip())


def parameter_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.reshape(-1).view(dtype=importlib.import_module('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def deny_weight_load(*args, **kwargs):
    raise RuntimeError('Pilot forbids checkpoint/pretrained-weight loading')


def measure(row, seed, batch_size, shape, out_shape, proxy, analyzer_class):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda:0')
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = None
    try:
        with patch.object(torch, 'load', deny_weight_load), \
             patch.object(torch.nn.Module, 'load_state_dict', deny_weight_load), \
             patch.object(torch.hub, 'load_state_dict_from_url', deny_weight_load):
            model = proxy._instantiate_model(row['nn_code'], shape, out_shape, row['prm'], device)
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter() - started
            before = parameter_hash(model)
            buffer_before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
            classes = out_shape[0]
            require(isinstance(classes, int) and classes >= 2, 'Unsupported class count')
            x = torch.randn((batch_size, *shape[1:]), device=device)
            y = torch.randint(0, classes, (batch_size,), device=device)
            analyzer = analyzer_class(model, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                try:
                    raw = analyzer.taylor_importance([(x, y)], torch.nn.CrossEntropyLoss(), num_batches=1)
                finally:
                    for warning in captured:
                        print('TAYLOR_WARNING', str(warning.message), flush=True)
            torch.cuda.synchronize()
            extraction_seconds = time.perf_counter() - t0
            table = analyzer.build_layer_table({'taylor_importance': raw})
            require(table, 'No weight layers returned')
            values = {k: v['taylor_imp'] for k, v in table.items()}
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                        for v in values.values()), 'Incomplete/nonfinite Taylor layer coverage')
            after = parameter_hash(model)
            if before != after:
                raise PilotInvariantError('Parameters changed during scoring')
            buffer_after = dict(model.named_buffers())
            changed_buffers = [k for k, v in buffer_before.items()
                               if k not in buffer_after or not torch.equal(v, buffer_after[k].detach().cpu())]
            return {'score': math.fsum(values.values()) / len(values), 'layers': values,
                    'raw': raw, 'shape': list(shape), 'output_shape': list(out_shape),
                    'warnings': [{'category': w.category.__name__, 'message': str(w.message)}
                                 for w in captured],
                    'parameter_sha256_before': before, 'parameter_sha256_after': after,
                    'changed_buffers': changed_buffers, 'setup_seconds': setup_seconds,
                    'extraction_seconds': extraction_seconds,
                    'wall_seconds_including_verification': time.perf_counter() - started}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def check_repeats(first, second):
    if first['parameter_sha256_before'] != second['parameter_sha256_before']:
        raise PilotInvariantError('Initialization differs on repeat')
    if first['layers'].keys() != second['layers'].keys():
        raise PilotInvariantError('Layer identities changed')
    if not all(math.isclose(value, second['layers'][name], rel_tol=1e-6, abs_tol=1e-10)
               for name, value in first['layers'].items()):
        raise PilotInvariantError('Taylor repeat differs')


def worker(root=POD, output=Path('/a/zcp-results')):
    contract = check_bundle(root)
    require(not any(output.iterdir()), 'Output directory is not empty; refusing rerun')
    import torch
    require(torch.cuda.is_available(), 'An allocated CUDA GPU is required')
    proxy = importlib.import_module('ab.gpt.util.method.zero_cost_proxies')
    layer_module = importlib.import_module('ab.nn.util.LayerAnalysis')
    require(Path(proxy.__file__).resolve() == (root / SOURCES[1]).resolve(), 'Wrong proxy import')
    require(Path(layer_module.__file__).resolve() == (root / SOURCES[0]).resolve(), 'Wrong layer import')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = load(root / NAME / 'pilot_inputs.json')
    require([r['id'] for r in rows] == contract['candidate_ids'], 'Candidate identity mismatch')
    save(output / 'runtime.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
         'gpu': torch.cuda.get_device_name(0), 'node': os.environ.get('NODE_NAME'),
         'contract_sha256': sha(root / NAME / 'contract.json'),
         'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
         'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()})
    results = []
    for row in rows:
        result = {'id': row['id'], 'nn': row['nn'], 'dataset': row['dataset'],
                  'seed': contract['seed'], 'attempts': [], 'status': 'failed'}
        try:
            require(row['dataset'].lower() in proxy._DATASET_SHAPES,
                    'Unknown dataset: refusing generic fallback shape')
            for shape, out_shape in proxy._shape_candidates(row['dataset']):
                attempt = {'shape': list(shape)}
                result['attempts'].append(attempt)
                try:
                    first = measure(row, contract['seed'], contract['batch_size'], shape,
                                    out_shape, proxy, layer_module.LayerAnalyzer)
                    attempt['first'] = first
                    second = measure(row, contract['seed'], contract['batch_size'], shape,
                                     out_shape, proxy, layer_module.LayerAnalyzer)
                    attempt['second'] = second
                    check_repeats(first, second)
                    attempt['status'] = 'ok'
                    result.update(status='ok', score=first['score'],
                                  layer_count=len(first['layers']), shape=list(shape))
                    break
                except PilotInvariantError:
                    attempt.update(status='failed', traceback=traceback.format_exc())
                    raise
                except Exception:
                    attempt.update(status='failed', traceback=traceback.format_exc())
                    if 'first' in attempt:
                        raise
        except Exception:
            result['traceback'] = traceback.format_exc()
        results.append(result)
        with (output / 'results.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print('CANDIDATE_RESULT', json.dumps(result), flush=True)
    check_bundle(root)
    passed = sum(r['status'] == 'ok' for r in results)
    summary = {'experiment': NAME, 'passed': passed, 'attempted': len(rows),
               'status': 'pass' if passed == len(rows) else 'failed',
               'scope': 'extraction-only pilot; no calibration, accuracy metrics or full-cohort claims',
               'contract_sha256': sha(root / NAME / 'contract.json'),
               'results_sha256': sha(output / 'results.jsonl')}
    save(output / 'summary.json', summary)
    require(load(output / 'summary.json') == summary, 'Summary readback failed')
    require(passed == len(rows), f'Pilot incomplete: {passed}/{len(rows)}; inspect preserved failures')
    print('INIT_TAYLOR_PILOT_VERIFIED', json.dumps(summary), flush=True)


def verify(root=HOST):
    contract = check_bundle(root)
    output = root / NAME / 'results'
    summary = load(output / 'summary.json')
    require(summary['contract_sha256'] == sha(root / NAME / 'contract.json'), 'Contract mismatch')
    require(summary['results_sha256'] == sha(output / 'results.jsonl'), 'Result hash mismatch')
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    require([r['id'] for r in rows] == contract['candidate_ids'], 'Result coverage mismatch')
    require(summary['status'] == 'pass' and summary['passed'] == summary['attempted'] == 3,
            'Pilot failed; preserve and inspect outputs')
    for row in rows:
        require(row['status'] == 'ok', 'Failed candidate')
        attempt = row['attempts'][-1]
        for key in ('first', 'second'):
            run = attempt[key]
            require(run['parameter_sha256_before'] == run['parameter_sha256_after'], 'Weight mutation')
        check_repeats(attempt['first'], attempt['second'])
        print(row['id'], row['dataset'], 'score=', row['score'], 'layers=', row['layer_count'])
    print('ARTIFACT_VERIFIED: 3/3 extraction and repeat checks; no predictive comparison yet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'worker', 'verify'))
    args = parser.parse_args()
    {'prepare': prepare, 'submit': submit, 'worker': worker, 'verify': verify}[args.action]()
