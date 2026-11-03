#!/usr/bin/env python3
"""One epoch, distinct snapshot architecture IDs, isolated feature costs.
Normalized-source duplicate guard; not a structural graph equivalence test.
Execute only trusted snapshot code, in a pod with read-only repository/data mounts.
"""
import argparse
import ast
import copy
import csv
import hashlib
import importlib
import importlib.abc
import json
import math
import os
import random
import socket
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

DATASETS = dict(zip(
    ('celeba-gender', 'cifar-10', 'cifar-100', 'imagenette', 'mnist', 'places365', 'svhn'),
    ('CelebA', 'CIFAR10', 'CIFAR100', 'Imagenette', 'MNIST', 'Places365', 'SVHN'),
))
FEATURES = {
    'weightwatcher': ('weightwatcher_analysis', 'ww_alpha'),
    'gradient_flow': ('gradient_flow', 'grad_norm'),
    'activation_statistics': ('activation_statistics', 'dead_frac'),
    'taylor_importance': ('taylor_importance', 'taylor_imp'),
    'effective_rank': ('effective_rank', 'rank_ratio'),
    'cka_similarity': ('cka_similarity', 'cka_redund'),
    'sensitivity': ('sensitivity_analysis', 'sensitivity'),
}
ARMS = {name: (name,) for name in FEATURES}
ARMS['all_seven'] = tuple(FEATURES)


def build_arms(extra_arms=()):
    arms = dict(ARMS)
    for specification in extra_arms:
        names = [name.strip() for name in specification.split(',')]
        if len(names) < 2 or len(set(names)) != len(names) or any(name not in FEATURES for name in names):
            raise ValueError('Each --extra-arm needs two or more distinct names from: ' + ','.join(FEATURES))
        features = tuple(name for name in FEATURES if name in names)
        if features in arms.values():
            raise ValueError(f'Duplicate extraction arm: {specification}')
        arms['+'.join(features)] = features
    return arms


class NoDatabase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ab.nn.util.Train' or fullname.startswith('ab.nn.util.db'):
            raise ImportError(f'Benchmark forbids database/training-writer import: {fullname}')


def source_fingerprint(code):
    normalized = ast.dump(ast.parse(code), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def select_candidates(snapshot, count, seed):
    rng, pools, fingerprints, ids = random.Random(seed), {}, {}, set()
    for index, line in enumerate(snapshot.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('task') != 'img-classification' or row.get('dataset') not in DATASETS:
            continue
        for key in ('id', 'nn', 'nn_code', 'transform_code', 'prm', 'metric'):
            if row.get(key) in (None, ''):
                raise ValueError(f'Snapshot row {index}: missing {key}')
        if row['id'] in ids:
            raise ValueError(f'Duplicate snapshot record id: {row["id"]}')
        ids.add(row['id'])
        code = row['nn_code']
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest not in fingerprints:
            fingerprints[digest] = source_fingerprint(code)
        row.update(row_index=index, source_ast_sha256=fingerprints[digest], code_sha256=digest)
        pools.setdefault(row['dataset'], []).append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, used_arch, used_nn = [], set(), set()
    order = sorted(pools)
    rng.shuffle(order)
    while len(selected) < count:
        progressed = False
        for dataset in order:
            pool = pools[dataset]
            while pool:
                row = pool.pop()
                if row['source_ast_sha256'] not in used_arch and row['nn'] not in used_nn:
                    selected.append(row)
                    used_arch.add(row['source_ast_sha256'])
                    used_nn.add(row['nn'])
                    progressed = True
                    break
            if len(selected) == count:
                return selected
        if not progressed:
            raise ValueError(f'Only {len(selected)} distinct snapshot architecture IDs after source deduplication; requested {count}')
    return selected


def code_namespace(code, name):
    namespace = {'__name__': name, '__file__': f'<snapshot:{name}>'}
    exec(compile(code, namespace['__file__'], 'exec'), namespace)
    return namespace


def load_data(row, data_root):
    import torchvision
    module = importlib.import_module('ab.nn.loader.' + row['dataset'])
    module.data_dir = data_root
    transform = code_namespace(row['transform_code'], 'benchmark_transform')['transform']
    cls = getattr(torchvision.datasets, DATASETS[row['dataset']])
    original = cls.__init__

    def offline_init(self, *args, **kwargs):
        kwargs['download'] = False
        return original(self, *args, **kwargs)

    with patch.object(cls, '__init__', offline_init):
        return module.loader(transform, row['task'])


def timed(function, device):
    import torch
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    sync()
    start = time.perf_counter()
    try:
        value = function()
        sync()
        return {'seconds': time.perf_counter() - start, 'status': 'ok'}, value
    except Exception as exc:
        sync()
        return {'seconds': time.perf_counter() - start, 'status': 'failed', 'error': repr(exc)}, None


def checkpoint_reset(model, device):
    import numpy as np
    import torch
    state = model.state_dict()
    for key, value in state.items():
        state[key] = value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    gradients = [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None

    def restore():
        model.load_state_dict(state)
        for p, g in zip(model.parameters(), gradients):
            p.grad = None if g is None else g.to(p.device).clone()
        for module, training in modes:
            module.training = training
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    return restore


def benchmark(row, data_root, seed, device, arms=None, epoch_limit_minutes=None):
    import numpy as np
    import torch
    from ab.nn.util.Classes import DataRoll
    from ab.nn.util.LayerAnalysis import LayerAnalyzer
    from ab.nn.util.Util import train_loader_f, test_loader_f, get_in_shape, default_epoch_limit_minutes
    limit = default_epoch_limit_minutes if epoch_limit_minutes is None else epoch_limit_minutes
    result = {k: row[k] for k in ('id', 'row_index', 'nn', 'source_ast_sha256', 'dataset')}
    result.update(status='failed', epoch=1, seed=seed, epoch_limit_minutes=limit, arms={})
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = code_namespace(row['nn_code'], 'benchmark_model')
        source_prm = row['prm'] if isinstance(row['prm'], dict) else json.loads(row['prm'])
        required = set(net['supported_hyperparameters']()) | {'batch', 'transform'}
        missing = [k for k in required if source_prm.get(k) in (None, '')]
        if missing:
            raise ValueError(f'Missing historical parameters: {missing}')
        prm = {k: copy.deepcopy(source_prm[k]) for k in sorted(required)}
        prm['epoch_max'] = 1
        result['effective_parameters'] = copy.deepcopy(prm)
        shape, _, train_set, test_set = load_data(row, data_root)
        train = train_loader_f(train_set, int(prm['batch']), 1)
        test = test_loader_f(test_set, int(prm['batch']), 1)
        model = net['Net'](get_in_shape(train_set, 1), shape, prm, device).to(device)
        model.train_setup(prm)
        model.train()
        result['training'], _ = timed(lambda: model.learn(DataRoll(train, limit)), device)
        if result['training']['status'] != 'ok':
            return result
        metric = importlib.import_module('ab.nn.metric.' + row['metric']).create_metric(shape)

        def evaluate():
            model.eval()
            metric.reset()
            with torch.no_grad():
                for inputs, labels in test:
                    metric(model(inputs.to(device)), labels.to(device))
            return float(metric.result())

        result['evaluation'], accuracy = timed(evaluate, device)
        result['accuracy'] = accuracy if accuracy is not None and math.isfinite(accuracy) else None
        if result['evaluation']['status'] != 'ok':
            return result
        loss = getattr(model, 'criterion', None) or getattr(model, 'loss_fn', None) or torch.nn.CrossEntropyLoss()
        restore = checkpoint_reset(model, device)

        def extract(arm, features):
            analyzer = LayerAnalyzer(model, device)
            raw = {}
            if arm == 'all_seven':
                raw = analyzer.full_analysis(train, loss, num_batches=1)
            else:
                for name in features:
                    method = getattr(analyzer, FEATURES[name][0])
                    if name in ('weightwatcher', 'effective_rank'):
                        raw[name] = method()
                    elif name in ('activation_statistics', 'cka_similarity'):
                        raw[name] = method(train, num_batches=1)
                    else:
                        raw[name] = method(train, loss, num_batches=5 if name == 'sensitivity' else 1)
            table = analyzer.build_layer_table(raw)
            analyzer.summarize(raw)
            return raw, table

        for arm, features in (ARMS if arms is None else arms).items():
            restore()
            measurement, output = timed(lambda: extract(arm, features), device)
            if output is not None:
                raw, table = output
                coverage = {name: sum(isinstance(v.get(FEATURES[name][1]), (int, float))
                            and math.isfinite(v[FEATURES[name][1]]) for v in table.values()) for name in features}
                errors = {name: str(raw.get(name, {}).get('error', 'No finite layer values'))
                          for name in features if not coverage[name] or 'error' in raw.get(name, {})}
                measurement.update(finite_layer_counts=coverage, layer_count=len(table), feature_errors=errors)
                if errors:
                    measurement['status'] = 'failed'
            result['arms'][arm] = measurement
        result['status'] = 'ok' if all(m['status'] == 'ok' for m in result['arms'].values()) else 'partial'
    except Exception as exc:
        result['error'] = repr(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'repo', 'data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260928, help='Candidate selection seed')
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--epoch-limit-minutes', type=float, help='Override the one-epoch time guard; not the epoch count')
    parser.add_argument('--only-id', action='append', default=[], help='Retry this exact snapshot record from the selected cohort; repeat for more IDs')
    parser.add_argument('--extra-arm', action='append', default=[], metavar='FEATURE1,FEATURE2',
                        help='Optional comma-separated combination; repeat for additional combinations')
    parser.add_argument('--pilot', action='store_true', help='Execute first selected candidate only')
    parser.add_argument('--select-only', action='store_true', help='Write manifest without loading datasets or training')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.epoch_limit_minutes is not None and (not math.isfinite(args.epoch_limit_minutes) or args.epoch_limit_minutes <= 0):
        parser.error('--epoch-limit-minutes must be finite and positive')
    if len(set(args.only_id)) != len(args.only_id):
        parser.error('Duplicate --only-id value')
    try:
        arms = build_arms(args.extra_arm)
    except ValueError as exc:
        parser.error(str(exc))
    args.snapshot, args.repo, args.data, args.out = [p.resolve() for p in (args.snapshot, args.repo, args.data, args.out)]
    if args.out.exists():
        parser.error('Output already exists; use a new directory')
    if not (args.repo / 'ab/nn/util/LayerAnalysis.py').is_file() or not args.data.is_dir():
        parser.error('Check --repo (requires LayerAnalysis.py) and --data (actual pod data mount)')
    sys.meta_path.insert(0, NoDatabase())
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    selected = select_candidates(args.snapshot, args.count, args.seed)
    if args.only_id:
        wanted = set(args.only_id)
        missing = wanted - {str(row['id']) for row in selected}
        if missing:
            parser.error('Requested record IDs are not in the selected cohort: ' + ', '.join(sorted(missing)))
        selected = [row for row in selected if str(row['id']) in wanted]
    args.out.mkdir(parents=True)
    manifest = {'snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                'selection_seed': args.seed, 'training_seed': args.training_seed, 'epochs': 1,
                'source_cohort_count': args.count, 'only_record_ids': args.only_id,
                'epoch_limit_minutes_override': args.epoch_limit_minutes,
                'extraction_arms': arms,
                'node_environment': os.environ.get('NODE_NAME'), 'hostname': socket.gethostname(),
                'data_root': str(args.data), 'repo': str(args.repo),
                'identity': 'unique snapshot nn; normalized-source AST duplicates excluded; not structural graph equivalence',
                'selection_python_version': sys.version,
                'counts': dict(Counter(r['dataset'] for r in selected)),
                'records': [{k: r[k] for k in ('id', 'row_index', 'nn', 'source_ast_sha256', 'code_sha256', 'dataset', 'metric', 'prm')} for r in selected]}
    (args.out / 'candidate_selection.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Selected distinct snapshot architecture IDs:', len(selected), manifest['counts'], flush=True)
    if args.select_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run training inside a CUDA pod, not the login node')
    checks = {}
    for row in selected:
        dataset = row['dataset']
        if dataset in checks:
            continue
        try:
            _, _, train, test = load_data(row, args.data)
            train[0], test[0]  # Exercise decoding and the selected historical transform.
            checks[dataset] = {'status': 'ok', 'train': len(train), 'test': len(test)}
        except Exception as exc:
            checks[dataset] = {'status': 'failed', 'error': repr(exc)}
    (args.out / 'dataset_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    if any(v['status'] != 'ok' for v in checks.values()):
        raise RuntimeError('Dataset preflight failed; inspect dataset_checks.json. No training started.')
    device = torch.device('cuda:0')
    fields = ['id', 'nn', 'source_ast_sha256', 'dataset', 'arm', 'training_seconds', 'evaluation_seconds', 'extraction_seconds', 'status', 'error']
    with (args.out / 'timings.jsonl').open('x', encoding='utf-8') as log, (args.out / 'timings.csv').open('x', newline='', encoding='utf-8') as table:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for row in selected[:1] if args.pilot else selected:
            result = benchmark(row, args.data, args.training_seed, device, arms=arms,
                               epoch_limit_minutes=args.epoch_limit_minutes)
            result.update(gpu=torch.cuda.get_device_name(device), torch_version=str(torch.__version__))
            log.write(json.dumps(result, allow_nan=False) + '\n')
            log.flush()
            for arm, measure in result['arms'].items() or [('', {'status': 'failed', 'error': result.get('error', 'Training/evaluation failed')})]:
                writer.writerow({**{k: result[k] for k in fields[:4]}, 'arm': arm,
                                 'training_seconds': result.get('training', {}).get('seconds'),
                                 'evaluation_seconds': result.get('evaluation', {}).get('seconds'),
                                 'extraction_seconds': measure.get('seconds'), 'status': measure['status'],
                                 'error': measure.get('error') or json.dumps(measure.get('feature_errors', {}))})
            table.flush()
            print(row['id'], row['dataset'], result['status'], flush=True)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    print('Results:', args.out, flush=True)


if __name__ == '__main__':
    main()
