#!/usr/bin/env python3
"""Enrich the frozen dataset with individual Taylor values; CPU token audit.

Frozen parameter values + model-declared keys + batch/transform. Read the first
matching record WITH direct layer data. No 50.json reselection, no SQLite,
no canonical writes, no training, no Git actions.
Requires the existing six-decimal frozen-prompt helper, used read-only.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import types

HOST = Path('/shared/ssd/home/b-t-bose/lmur')
POD = Path('/a/mm')
NAME = 'per-layer-taylor-frozen-v6'
EXCLUDED_TRAIN_RECORDS = {
    '2b2f2b5d2a9b869664339ddd4d8d9294':
        'User-authorized exclusion: matching MaxVit/CIFAR-10 records lack individual layer statistics.'
}
PKG = 'nn-gpt/ab/gpt/act/acc_predictor'
HELPER = 'prepare_per_layer_taylor_v1.py'
HELPER_SHA = '74f9d036b7ee894432ad049a535d2645b56922f31be4c77a74753303913ce0fa'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    require(jsonl(path) == rows, f'Readback differs: {path}')


def from_pod(root, path):
    return root / str(PurePosixPath(path).relative_to('/a/mm'))


def helper(root):
    path = root / HELPER
    require(sha(path) == HELPER_SHA, 'Corrected frozen-prompt helper missing or changed')
    spec = importlib.util.spec_from_file_location('frozen_prompt_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal(node, values):
    """Evaluate declarations only; never import or execute neural-network code."""
    if isinstance(node, ast.Name):
        require(node.id in values, f'Unresolved declaration name: {node.id}')
        return values[node.id]
    if isinstance(node, ast.Call):
        constructors = {'set': set, 'frozenset': frozenset, 'list': list, 'tuple': tuple}
        require(isinstance(node.func, ast.Name) and node.func.id in constructors
                and not node.keywords and len(node.args) <= 1, 'Nonliteral supported-hyperparameter declaration')
        return constructors[node.func.id](*(literal(a, values) for a in node.args))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.Add)):
        left, right = literal(node.left, values), literal(node.right, values)
        return left | right if isinstance(node.op, ast.BitOr) else left + right
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        constructor = list if isinstance(node, ast.List) else tuple if isinstance(node, ast.Tuple) else set
        return constructor(literal(n, values) for n in node.elts)
    return ast.literal_eval(node)


def supported_keys(code, function_name='supported_hyperparameters'):
    """Read the frozen model's actual declaration, not a fixed global key list.

    Literal declarations, local/module constants and set unions are supported.
    Dynamic metadata stops for inspection rather than executing model code or
    inventing a fallback key set.
    """
    tree = ast.parse(code)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]
    require(functions, f'Missing metadata declaration: {function_name}')
    function = functions[-1]  # Python module semantics: the last definition wins.
    require(not function.decorator_list and not function.args.args and not function.args.kwonlyargs
            and function.args.vararg is None and function.args.kwarg is None, 'Unsupported metadata function signature')
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            try:
                value = literal(node.value, values)
            except Exception:
                for name in names:
                    values.pop(name, None)
            else:
                for name in names:
                    values[name] = value
    for node in function.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            value = literal(node.value, values)
            for target in node.targets:
                values[target.id] = value
            continue
        require(isinstance(node, ast.Return), 'Dynamic supported_hyperparameters() body requires inspection')
        result = literal(node.value, values)
        require(isinstance(result, (set, frozenset, list, tuple, dict)), 'Invalid supported-hyperparameter declaration')
        keys = set(result)
        require(all(isinstance(k, str) and k for k in keys), 'Invalid supported parameter name')
        return keys | {'batch', 'transform'}
    raise RuntimeError('supported_hyperparameters() has no literal return')


class MissingMetadata(RuntimeError):
    pass


def read_metadata_module(root, module, hashes):
    require(module and all(part and part not in ('.', '..') and '/' not in part and '\\' not in part
                           for part in module.split('.')), 'Unsafe metadata module path')
    relative = Path(*module.split('.')).with_suffix('.py')
    paths = [base / relative for base in (root / 'nn-dataset', root / 'nn-gpt')]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise MissingMetadata(f'Metadata module not available locally: {module}')
    require(len({sha(p) for p in paths}) == 1, f'Ambiguous metadata module sources: {module}')
    path = paths[0]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(str(path) not in hashes or hashes[str(path)] == digest, f'Metadata source changed: {path}')
    hashes[str(path)] = digest
    return data.decode('utf-8'), str(path)


def resolve_metadata(code, module, symbol, root, hashes, origin, seen):
    identity = (origin, symbol)
    require(identity not in seen, f'Cyclic supported-hyperparameter import: {module}:{symbol}')
    seen = seen | {identity}
    tree = ast.parse(code)
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            return supported_keys(code, symbol), f'{origin}:{symbol}'
        if isinstance(node, ast.ImportFrom):
            candidates = [alias for alias in node.names
                          if alias.name == '*' or (alias.asname or alias.name) == symbol]
            for alias in candidates:
                if node.level:
                    parent = module.split('.')[:-node.level]
                    target = '.'.join(parent + (node.module.split('.') if node.module else []))
                else:
                    target = node.module or ''
                imported = symbol if alias.name == '*' else alias.name
                try:
                    text, path = read_metadata_module(root, target, hashes)
                    return resolve_metadata(text, target, imported, root, hashes, path, seen)
                except MissingMetadata:
                    if alias.name != '*':
                        raise
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            if isinstance(node.value, ast.Name):
                return resolve_metadata(code, module, node.value.id, root, hashes, origin, seen)
            if isinstance(node.value, ast.Lambda):
                fn = ast.FunctionDef(name=symbol, args=node.value.args,
                                     body=[ast.Return(value=node.value.body)], decorator_list=[])
                ast.copy_location(fn, node)
                rewritten = ast.Module(body=[n for n in tree.body if n is not node] + [fn], type_ignores=[])
                return supported_keys(ast.unparse(ast.fix_missing_locations(rewritten)), symbol), f'{origin}:{symbol}'
            raise RuntimeError(f'Unsupported metadata assignment: {origin}:{symbol}')
    raise MissingMetadata(f'No supported metadata declaration: {origin}:{symbol}')


def model_keys(raw, root, hashes):
    prm = json.loads(raw['prm']) if isinstance(raw['prm'], str) else raw['prm']
    model = prm.get('model') or raw['nn']
    module = 'ab.nn.nn.' + str(model)
    try:
        return resolve_metadata(raw['nn_code'], module, 'supported_hyperparameters', root, hashes,
                                'frozen nn_code', set())
    except MissingMetadata:
        # The original loader resolves the model module, including imported
        # metadata. Read that same declaration without importing model code.
        failures = []
        for name in dict.fromkeys((model, raw['nn'])):
            module = 'ab.nn.nn.' + str(name)
            try:
                text, path = read_metadata_module(root, module, hashes)
                return resolve_metadata(text, module, 'supported_hyperparameters', root, hashes, path, set())
            except MissingMetadata as exc:
                failures.append(str(exc))
        raise MissingMetadata('; '.join(failures))


def has_layer_data(record):
    payload = record.get('layer_stat')
    return (isinstance(payload, dict) and not any(str(k).isdigit() for k in payload)
            and isinstance(payload.get('layers'), list) and bool(payload['layers'])
            and all(isinstance(layer, dict) for layer in payload['layers']))


def extract_layers(raw, root, h, hashes, key_cache):
    prm = json.loads(raw['prm']) if isinstance(raw['prm'], str) else raw['prm']
    require(isinstance(prm, dict), 'Invalid frozen prm')
    code_hash = hashlib.sha256(raw['nn_code'].encode('utf-8')).hexdigest()
    metadata_key = (code_hash, prm.get('model') or raw['nn'])
    if metadata_key not in key_cache:
        key_cache[metadata_key] = model_keys(raw, root, hashes)
    keys, keys_source = key_cache[metadata_key]
    missing = sorted(k for k in keys if k not in prm or prm[k] in (None, ''))
    require(not missing, f'Missing declared hyperparameter values: {missing}')
    parameters = {k: copy.deepcopy(prm[k]) for k in sorted(keys)}
    model = prm.get('model') or raw['nn']
    components = [str(raw[k]) for k in ('task', 'dataset', 'metric')] + [str(model)]
    require(all(s and '/' not in s and '\\' not in s and s not in ('.', '..') for s in components),
            'Unsafe canonical configuration component')
    directory = root / 'nn-dataset/ab/nn/stat/train' / '_'.join(components)
    means = h.mean_table(raw['layer_stats'])
    by_epoch, order, selected = {}, [], []
    for epoch in (1, 2, 3):
        path = directory / f'{epoch}.json'
        data = h.load(path, hashes)
        records = data if isinstance(data, list) else [data]
        parameter_matches = [(i, record) for i, record in enumerate(records)
                             if isinstance(record, dict)
                             and all(record.get(k) == value for k, value in parameters.items())]
        matches = [(i, record) for i, record in parameter_matches if has_layer_data(record)]
        require(matches, f'{path}: no parameter-matching record with direct layer data; parameter indices={[i for i, _ in parameter_matches]}')
        index, record = matches[0]  # First eligible record in existing file order.
        payload = record.get('layer_stat')
        require(isinstance(payload, dict) and not any(str(k).isdigit() for k in payload),
                f'{path}: FIRST match {index} has no direct layer_stat')
        layers = payload.get('layers')
        require(isinstance(layers, list) and layers, f'{path}: FIRST match {index} has no layer list')
        values = {}
        for layer in layers:
            require(isinstance(layer, dict), f'{path}: invalid layer entry')
            name = layer.get('name', layer.get('layer_name'))
            if 'name' in layer and 'layer_name' in layer:
                require(layer['name'] == layer['layer_name'], 'Contradictory layer names')
            require(isinstance(name, str) and not any(c in name for c in '\t\r\n') and name not in values,
                    f'{path}: missing/duplicate/invalid layer name; fields={list(layer)}')
            value = layer.get('taylor_imp')
            require(value is None or (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value)), f'{path}: invalid Taylor value for {name}')
            values[name] = value
            if name not in order:
                order.append(name)
        # Verify the frozen mean AFTER selection. Do not rank candidates by
        # accuracy/timing/UID or skip a mean mismatch in favor of another record.
        finite = [v for v in values.values() if v is not None]
        mean = float(format(math.fsum(finite) / len(finite), '.6g')) if finite else None
        require(mean == means[epoch], f'{path}: FIRST layer-bearing match {index} Taylor mean {mean} differs from frozen {means[epoch]}; not trying another record')
        by_epoch[epoch] = values
        selected.append({'epoch': epoch, 'file': str(path), 'sha256': hashes[str(path)],
                         'record_index': index, 'matching_record_count': len(parameter_matches),
                         'eligible_layer_record_count': len(matches),
                         'uid_metadata_only': record.get('uid'), 'layer_count': len(values),
                         'finite_taylor_count': len(finite), 'verified_mean': mean})
    lines = ['LAYER_TAYLOR_BY_EPOCH', 'layer\tepoch_1\tepoch_2\tepoch_3']
    missing_cells = 0
    for name in order:
        cells = []
        for epoch in (1, 2, 3):
            value = by_epoch[epoch].get(name)
            missing_cells += value is None
            cells.append('NA' if value is None else format(value, '.6g'))
        lines.append(name + '\t' + '\t'.join(cells))
    return '\n'.join(lines), {
        'raw_id': raw['id'], 'snapshot_architecture_id': raw['nn'], 'canonical_model': model,
        'keys_source': keys_source,
        'nn_code_sha256': code_hash, 'matched_parameters': parameters,
        'selection_rule': 'first exact hyperparameter match WITH direct layer data in existing JSON order; not a uniqueness claim',
        'epochs': selected, 'layer_order': order, 'missing_cells': missing_cells,
    }


def build(root=HOST):
    h = helper(root)
    hashes = {}
    raw = h.jsonl(root / h.RAW, hashes)
    require(hashes[str(root / h.RAW)] == h.RAW_SHA, 'Frozen raw snapshot changed')
    require(len(raw) == len({r['id'] for r in raw}) == 1151, 'Unexpected raw IDs/count')
    manifest = h.load(root / h.MANIFEST, hashes)
    require(manifest['source_sha256'] == h.RAW_SHA, 'Manifest/snapshot mismatch')
    assignment = manifest['record_to_split']
    require(set(assignment) == {r['id'] for r in raw} and dict(Counter(assignment.values())) == h.COUNTS,
            'Frozen split membership/count changed')
    raw_by_id = {r['id']: r for r in raw}
    require(set(EXCLUDED_TRAIN_RECORDS) <= set(raw_by_id), 'Authorized exclusion missing from frozen snapshot')
    for rid in EXCLUDED_TRAIN_RECORDS:
        r = raw_by_id[rid]
        require(assignment[rid] == 'train' and r['nn'] == 'MaxVit'
                and r['dataset'] == 'cifar-10' and r['task'] == 'img-classification',
                'Exclusion identity/split differs; refusing broader exclusion')
    derived_counts = dict(h.COUNTS)
    derived_counts['train'] -= len(EXCLUDED_TRAIN_RECORDS)
    groups = {s: {r['nn'] for r in raw if assignment[r['id']] == s} for s in h.COUNTS}
    require(not (groups['train'] & groups['validation'] or groups['train'] & groups['test']
                 or groups['validation'] & groups['test']), 'Architecture split overlap')
    contract = h.load(root / h.CONTRACT, hashes)
    run = contract['runs']['3acc-taylor-s42']
    directory = from_pod(root, run['data'])
    files = {}
    for name, digest in run['input_sha256'].items():
        if not name.endswith('.jsonl'):
            continue
        require(Path(name).name == name, 'Unexpected source filename')
        prompts = h.jsonl(directory / name, hashes)
        require(hashes[str(directory / name)] == digest, f'Original prompts changed: {name}')
        splits = [s for s, n in h.COUNTS.items() if n == len(prompts)]
        require(len(splits) == 1 and splits[0] not in files, 'Ambiguous source split')
        files[splits[0]] = (name, prompts)
    require(set(files) == set(h.COUNTS), 'Missing original split file')
    pools = defaultdict(list)
    for record in raw:
        pools[(assignment[record['id']], record['nn'])].append(record)
    outputs, provenance, issues, seen, key_cache = {}, [], [], set(), {}
    excluded = []
    for split, (filename, prompts) in files.items():
        updated = []
        for position, prompt in enumerate(prompts):
            record = None
            try:
                parts = h.prompt_parts(prompt)
                record = h.matching_raw(prompt, pools[(split, prompt['architecture_id'])], parts)
                require(record['id'] not in seen, 'Repeated raw-record mapping')
                seen.add(record['id'])
                if record['id'] in EXCLUDED_TRAIN_RECORDS:
                    entry = {'raw_id': record['id'], 'architecture_id': record['nn'],
                             'dataset': record['dataset'], 'split': split, 'position': position,
                             'filename': filename, 'reason': EXCLUDED_TRAIN_RECORDS[record['id']]}
                    excluded.append(entry)
                    print('AUTHORIZED_EXCLUSION', json.dumps(entry), flush=True)
                    continue
                table, source = extract_layers(record, root, h, hashes, key_cache)
                updated.append(h.replace_prompt(prompt, parts, table))
                provenance.append({**source, 'split': split, 'position': position,
                                   'output_position': len(updated) - 1, 'filename': filename})
            except Exception as exc:
                issue = {'split': split, 'position': position, 'raw_id': record['id'] if record else None,
                         'architecture_id': prompt.get('architecture_id'), 'error': f'{type(exc).__name__}: {exc}'}
                issues.append(issue)
                print('ISSUE', json.dumps(issue), flush=True)
        outputs[filename] = updated
        print('JSON_AUDIT', split, len(updated), '/', derived_counts[split],
              'source_records=', len(prompts),
              'authorized_exclusions=', sum(e['split'] == split for e in excluded), flush=True)
    report = {'experiment': NAME, 'expected': derived_counts, 'source_split_counts': h.COUNTS,
        'excluded_records': excluded, 'expanded': len(provenance),
        'issues': len(issues), 'max_layers': max((len(p['layer_order']) for p in provenance), default=0),
        'epochs_with_multiple_parameter_matches': sum(e['matching_record_count'] > 1 for p in provenance for e in p['epochs']),
        'epochs_with_multiple_layer_matches': sum(e['eligible_layer_record_count'] > 1 for p in provenance for e in p['epochs']),
        'token_lengths_checked': False, 'training_launched': False}
    print('AUDIT_SUMMARY', json.dumps(report), flush=True)
    require(not issues, 'JSON audit blocked; no outputs written; no exclusions beyond the explicit authorized record')
    require(seen == {r['id'] for r in raw}, 'Incomplete source raw-record coverage')
    require({e['raw_id'] for e in excluded} == set(EXCLUDED_TRAIN_RECORDS)
            and len(excluded) == len(EXCLUDED_TRAIN_RECORDS), 'Exclusion coverage differs')
    require({p['raw_id'] for p in provenance} == set(raw_by_id) - set(EXCLUDED_TRAIN_RECORDS),
            'Derived dataset contains missing, extra or excluded records')
    require(dict(Counter(p['split'] for p in provenance)) == derived_counts, 'Derived split counts differ')
    for path, digest in hashes.items():
        require(sha(path) == digest, f'Source changed during audit: {path}')
    return outputs, provenance, hashes, report, {s: f[0] for s, f in files.items()}, run


def kube(*args):
    return subprocess.run(['kubectl', '-n', 'b-t-bose', *args], text=True, capture_output=True, check=True).stdout


def make_job(template, root):
    require(template['metadata']['name'] == 'init-taylor-pilot-v2', 'Unexpected saved Job template')
    job = copy.deepcopy(template)
    job['metadata'] = {'name': NAME, 'namespace': 'b-t-bose', 'labels': {'experiment-batch': NAME}}
    job['spec'].update(backoffLimit=0, activeDeadlineSeconds=7200)
    job['spec']['template']['metadata'] = {'labels': {'experiment-batch': NAME}}
    pod = job['spec']['template']['spec']
    require(len(pod['containers']) == 1 and not pod.get('initContainers'), 'Unexpected extra containers')
    container = pod['containers'][0]
    for limits in container['resources'].values():
        limits.pop('nvidia.com/gpu', None)
    volumes = {v['name']: v for v in pod['volumes']}
    require(volumes['source']['hostPath']['path'] == str(root), 'Wrong source mount')
    mounts = {m['name']: m for m in container['volumeMounts']}
    require(mounts['source']['mountPath'] == '/a/mm' and mounts['source'].get('readOnly'), 'Source mount must be read-only /a/mm')
    require(mounts['results']['mountPath'] == '/a/zcp-results', 'Unexpected result mount')
    volumes['results']['hostPath']['path'] = str(root / NAME / 'results')
    container['workingDir'] = '/tmp'
    container['args'] = [
        'set -euo pipefail\n'
        'export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false\n'
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/per-layer-hf\n'
        'export PYTHONPATH=/tmp/per-layer-tokenizer\n'
        'python -c "from transformers import AutoTokenizer" || '
        'python -m pip install --disable-pip-version-check --no-cache-dir '
        '--target /tmp/per-layer-tokenizer transformers==5.2.0\n'
        f'python -u /a/mm/{NAME}/runner.py tokens\n']
    return job


def prepare(root=HOST):
    bundle = root / NAME
    require(not bundle.exists(), f'Existing bundle preserved: {bundle}')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Job already exists')
    outputs, provenance, hashes, report, files, run = build(root)
    reference = from_pod(root, run['output'])
    tokenizer = reference / 'tuned_model'
    require(all((tokenizer / n).is_file() for n in ('tokenizer.json', 'tokenizer_config.json')), 'Saved tokenizer missing')
    effective_path = reference / 'effective_trainer_args.json'
    effective = load(effective_path)
    nominal = effective.get('max_seq_len')
    require(nominal is None or (type(nominal) is int and nominal > 0), 'Invalid recorded nominal sequence limit')
    args_path = tokenizer / 'training_args.bin'
    source_files = [root / HELPER, effective_path, root / 'init-taylor-pilot-v2/job.json']
    source_files.extend(root / PKG / name for name in ('train.py', 'common.py', 'eval.py'))
    source_files.extend(p for p in tokenizer.iterdir() if p.is_file() and
                        (p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'chat_template'))
                         or p.name in ('vocab.json', 'merges.txt', 'config.json', 'adapter_config.json')))
    if args_path.is_file():
        source_files.append(args_path)
    for path in source_files:
        hashes[str(path)] = sha(path)
    job = make_job(load(root / 'init-taylor-pilot-v2/job.json'), root)
    subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server', '-f', '-'],
                   input=json.dumps(job), text=True, check=True)
    for path, digest in hashes.items():
        require(sha(path) == digest, f'Source changed before writing: {path}')
    bundle.mkdir()
    (bundle / 'prompts').mkdir()
    (bundle / 'results').mkdir()
    for filename, prompts in outputs.items():
        write_jsonl(bundle / 'prompts' / filename, prompts)
    save(bundle / 'source_map.json', provenance)
    save(bundle / 'audit.json', report)
    save(bundle / 'excluded_records.json', report['excluded_records'])
    save(bundle / 'job.json', job)
    with (bundle / 'runner.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    prepared = [bundle / name for name in ('runner.py', 'job.json', 'source_map.json', 'audit.json', 'excluded_records.json')]
    prepared.extend(bundle / 'prompts' / name for name in outputs)
    contract = {'experiment': NAME, 'files': files, 'nominal_sequence_limit': nominal,
        'source_split_counts': report['source_split_counts'], 'split_counts': report['expected'],
        'excluded_raw_ids': list(EXCLUDED_TRAIN_RECORDS),
        'tokenizer': str(tokenizer.relative_to(root)),
        'saved_args': str(args_path.relative_to(root)) if args_path.is_file() else None,
        'source_hashes': {str(Path(p).relative_to(root)): digest for p, digest in hashes.items()},
        'prepared_hashes': {str(p.relative_to(bundle)): sha(p) for p in prepared},
        'selection': 'Frozen parameter values and split with one explicitly authorized training-record exclusion; model-specific keys plus batch/transform; first matching JSON record WITH layer data; frozen-mean post-selection validation'}
    save(bundle / 'contract.json', contract)
    check(root)
    print('PREPARED:', bundle)
    print('CPU token audit not submitted; no model training or database access performed.')


def check(root):
    bundle = root / NAME
    contract = load(bundle / 'contract.json')
    require(contract['experiment'] == NAME, 'Wrong experiment')
    for relative, digest in contract['source_hashes'].items():
        require(sha(root / relative) == digest, f'Source drift: {relative}')
    for relative, digest in contract['prepared_hashes'].items():
        require(sha(bundle / relative) == digest, f'Prepared artifact drift: {relative}')
    return contract


def submit(root=HOST):
    check(root)
    bundle = root / NAME
    require(not any((bundle / 'results').iterdir()), 'Outputs exist; do not rerun')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Job already exists')
    path = str(bundle / 'job.json')
    print(kube('create', '--dry-run=server', '-f', path).strip())
    print(kube('create', '-f', path).strip())


def load_function(path, name):
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1 and not nodes[0].decorator_list, f'Unexpected formatter: {name}')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def tokens(root=POD, output=Path('/a/zcp-results')):
    import transformers
    from transformers import AutoTokenizer
    contract = check(root)
    require(not any(output.iterdir()), 'Token-audit output directory not empty')
    package = root / PKG
    format_training = load_function(package / 'train.py', '_format_dataset')
    inference_messages = load_function(package / 'eval.py', '_test_build_prompt_messages')
    format_inference = load_function(package / 'common.py', '_apply_chat_template_for_inference')
    tokenizer = AutoTokenizer.from_pretrained(str(root / contract['tokenizer']), local_files_only=True)
    require(tokenizer.eos_token is not None, 'Saved tokenizer has no EOS token')
    actual, metadata_error = None, None
    if contract['saved_args']:
        try:
            import torch
            args = torch.load(root / contract['saved_args'], map_location='cpu', weights_only=False)
            lengths = [vars(args).get(k) for k in ('max_length', 'max_seq_length')]
            lengths = [v for v in lengths if type(v) is int and v > 0]
            actual = min(lengths) if lengths else None
        except Exception as exc:
            metadata_error = f'{type(exc).__name__}: {exc}'
    nominal = contract['nominal_sequence_limit']
    caps = sorted({1024} | ({nominal} if nominal else set()) | ({actual} if actual else set()))
    details, summary = {}, {}
    for split, filename in contract['files'].items():
        prompts = jsonl(root / NAME / 'prompts' / filename)
        lengths = []
        for position, prompt in enumerate(prompts):
            text = format_training({'messages': [prompt['messages']]}, tokenizer)['text'][0]
            inference = format_inference(tokenizer, inference_messages(prompt['messages']))
            with_specials = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            without_specials = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
            lengths.append({'position': position, 'architecture_id': prompt['architecture_id'],
                'training_with_specials': with_specials, 'training_without_specials': without_specials,
                'training_conservative': max(with_specials, without_specials),
                'inference_prompt': len(tokenizer.encode(inference, add_special_tokens=False, truncation=False))})
        details[split] = lengths
        summary[split] = {'records': len(lengths),
            'max_training_tokens': max(r['training_conservative'] for r in lengths),
            'max_inference_prompt_tokens': max(r['inference_prompt'] for r in lengths),
            'training_over_limit': {str(cap): sum(r['training_conservative'] > cap for r in lengths) for cap in caps},
            'inference_plus_64_over_nominal': None if nominal is None else sum(r['inference_prompt'] + 64 > nominal for r in lengths)}
        print('TOKEN_COUNTS', split, json.dumps(summary[split]), flush=True)
    check(root)
    save(output / 'token_lengths.json', details)
    report = {'experiment': NAME, 'transformers': transformers.__version__,
        'source_split_counts': contract['source_split_counts'], 'split_counts': contract['split_counts'],
        'excluded_raw_ids': contract['excluded_raw_ids'],
        'nominal_sequence_limit': nominal, 'saved_training_limit': actual,
        'training_metadata_error': metadata_error, 'historical_comparison_limit': 1024,
        'lengths': summary, 'training_launched': False, 'training_ready': False,
        'measurement': 'Existing training formatter including EOS; both added-special-token conventions; existing inference formatter; no truncation.',
        'next_gate': 'Review actual token lengths and agree training-length policy before training.',
        'token_lengths_sha256': sha(output / 'token_lengths.json'),
        'contract_sha256': sha(root / NAME / 'contract.json')}
    save(output / 'TOKEN_AUDIT.json', report)
    print('FROZEN_V6_TOKEN_AUDIT_COMPLETE', json.dumps(report), flush=True)


def show(root=HOST):
    check(root)
    bundle = root / NAME
    report = load(bundle / 'results/TOKEN_AUDIT.json')
    require(sha(bundle / 'contract.json') == report['contract_sha256'], 'Contract hash mismatch')
    require(sha(bundle / 'results/token_lengths.json') == report['token_lengths_sha256'], 'Token report hash mismatch')
    print(json.dumps(report, indent=2))
    print('PREPARED PROMPTS:', bundle / 'prompts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'prepare', 'submit', 'tokens', 'show'))
    args = parser.parse_args()
    if args.action == 'audit':
        build()
        print('JSON AUDIT PASSED; no files or Jobs changed.')
    else:
        {'prepare': prepare, 'submit': submit, 'tokens': tokens, 'show': show}[args.action]()
