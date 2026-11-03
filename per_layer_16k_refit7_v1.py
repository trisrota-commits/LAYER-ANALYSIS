#!/usr/bin/env python3
"""Six independent 16k per-layer Taylor refits, seeds 42-47, then test evaluation.

Seven epochs on 1,035 retained train+validation records; microbatch 1,
accumulation 32; no validation selection. Original data/checkpoints are read-only.
Failures remain per-seed; no automatic retries, cleanup or partial-seed averages.
"""
import argparse
from collections import Counter
import copy
import csv
from dataclasses import fields
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
import traceback

HOST = Path('/shared/ssd/home/b-t-bose/lmur')
POD = Path('/a/mm')
OUT = Path('/a/refit-output')
NAME = 'per-layer-16k-refit7-v1'
PREFLIGHT = 'per-layer-16k-preflight-v1'
DATA = 'per-layer-taylor-frozen-v6'
SEEDS = (42, 43, 44, 45, 46, 47)
LIMIT = 16384
EPOCHS = 7
METRICS = ('RMSE', 'MAE', 'Pearson', 'Spearman', 'R2')
T95 = 2.570581835636315


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False, default=str)
        f.write('\n')


def write_rows(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    require(rows(path) == data, 'Prepared dataset readback differs')


def host_path(root, pod_path):
    return root / str(PurePosixPath(pod_path).relative_to('/a/mm'))


def kube(*args):
    result = subprocess.run(['kubectl', '-n', 'b-t-bose', *args], capture_output=True, text=True)
    require(result.returncode == 0, result.stderr.strip() or result.stdout.strip())
    return result.stdout


def preflight(root):
    folder = root / PREFLIGHT
    c = load(folder / 'contract.json')
    require(sha(folder / 'runner.py') == c['prepared_hashes']['runner.py'], 'Preflight runner hash differs')
    spec = importlib.util.spec_from_file_location('verified_long_context_preflight', folder / 'runner.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    p.check(root)
    r = load(folder / 'results/report.json')
    require(r.get('status') == 'pass' and r.get('optimizer_steps') == 1, 'Successful preflight missing')
    require(r['contract_sha256'] == sha(folder / 'contract.json'), 'Preflight contract mismatch')
    require(r['token_check']['target_fully_supervised'] and r['token_check']['input_tokens'] <= LIMIT,
            'Preflight token/target checks missing')
    require(r['preflight_overrides']['max_length'] == LIMIT and r['node'], 'Preflight length/node missing')
    return p, c, r


def adapter_signature(config):
    require(not config.get('use_dora') and not config.get('modules_to_save')
            and not config.get('rank_pattern') and not config.get('alpha_pattern'), 'Unsupported adapter variant')
    return {'r': int(config['r']), 'lora_alpha': int(config['lora_alpha']),
            'lora_dropout': float(config['lora_dropout']), 'target_modules': sorted(config['target_modules']),
            'bias': config.get('bias', 'none'), 'use_rslora': bool(config.get('use_rslora', False))}


def make_job(template, root, seed, node, image, packages):
    job = copy.deepcopy(template)
    job_name = f'{NAME}-s{seed}'
    labels = {'experiment-batch': NAME, 'seed': str(seed)}
    job['metadata'] = {'name': job_name, 'namespace': 'b-t-bose', 'labels': labels}
    spec = job['spec']
    for key in ('selector', 'manualSelector', 'ttlSecondsAfterFinished', 'suspend'):
        spec.pop(key, None)
    spec.update(completions=1, parallelism=1, backoffLimit=0, activeDeadlineSeconds=172800)
    spec['template']['metadata'] = {'labels': labels.copy()}
    pod = spec['template']['spec']
    require(len(pod['containers']) == 1 and not pod.get('initContainers'), 'Unexpected containers')
    pod['nodeSelector'] = {'kubernetes.io/hostname': node}
    pod['restartPolicy'] = 'Never'
    pod['automountServiceAccountToken'] = False
    container = pod['containers'][0]
    require(container['name'] == 'trainer' and str(container['resources']['limits'].get('nvidia.com/gpu')) == '1', 'Unexpected preflight GPU template')
    container['image'] = image
    volumes = {v['name']: v for v in pod['volumes']}
    require(volumes['lmur']['hostPath']['path'] == str(root), 'Wrong source root')
    output_volume = volumes.pop('preflight-output')
    output_volume['name'] = 'refit-output'
    output_volume['hostPath']['path'] = str(root / NAME / 'results')
    for mount in container['volumeMounts']:
        if mount['name'] == 'lmur':
            require(mount['mountPath'] == '/a/mm' and mount.get('readOnly'), 'Source must stay read-only')
        elif mount['name'] == 'preflight-output':
            mount.update(name='refit-output', mountPath='/a/refit-output')
    container['workingDir'] = f'/a/refit-output/s{seed}/work'
    container['env'] = [e for e in container.get('env', []) if e['name'] in ('USER', 'LOGNAME', 'HOME')]
    env = {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': '/tmp/refit-hf',
        'TRANSFORMERS_CACHE': '/tmp/refit-hf/hub', 'XDG_CACHE_HOME': '/tmp/refit-cache',
        'TORCHINDUCTOR_CACHE_DIR': '/tmp/refit-inductor', 'RUN_SEED': str(seed)}
    container['env'].extend({'name': k, 'value': v} for k, v in env.items())
    container['env'].append({'name': 'NODE_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}})
    pins = ' '.join(f'{name}=={version}' for name, version in packages.items())
    require(all(re.fullmatch(r'[A-Za-z0-9_.+-]+', value) for value in list(packages) + list(packages.values())), 'Unsafe package specification')
    container['command'] = ['/bin/bash', '-lc']
    container['args'] = [
        'set -euo pipefail\n'
        'trap \'printf "FAILED at shell line %s\\n" "$LINENO" >&2\' ERR\n'
        'unset PYTHONPATH\n'
        'mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"\n'
        f'python -m pip install --disable-pip-version-check --no-cache-dir --user {pins}\n'
        'export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\n'
        f'python -u /a/mm/{NAME}/runner.py train --seed {seed}\n'
        f'python -u /a/mm/{NAME}/runner.py evaluate --seed {seed}\n']
    return job


def prepare(root=HOST):
    bundle = root / NAME
    require(not bundle.exists(), f'Existing bundle preserved: {bundle}')
    p, pc, pr = preflight(root)
    data = root / DATA
    dc = load(data / 'contract.json')
    require(dc['split_counts'] == {'train': 977, 'validation': 58, 'test': 115}, 'Unexpected retained split')
    input_rows, sources = {}, set()
    for split, filename in dc['files'].items():
        path = data / 'prompts' / filename
        require(sha(path) == dc['prepared_hashes']['prompts/' + filename], f'Prompt drift: {split}')
        input_rows[split] = rows(path)
        require(len(input_rows[split]) == dc['split_counts'][split], f'Split count differs: {split}')
        sources.add(path)
    training = input_rows['train'] + input_rows['validation']
    test = input_rows['test']
    require(len(training) == 1035 and len(test) == 115, 'Refit/test population differs')
    require(not ({r['architecture_id'] for r in training} & {r['architecture_id'] for r in test}), 'Test architecture overlap')
    sources.update(root / PREFLIGHT / n for n in ('contract.json', 'runner.py', 'job.json', 'results/report.json'))
    sources.add(data / 'contract.json')
    sources.update((root / rel) for rel in pc['source_hashes'])
    package = root / 'nn-gpt/ab/gpt/act/acc_predictor'
    sources.update(package.glob('*.py'))
    template = load(root / PREFLIGHT / 'job.json')
    image = template['spec']['template']['spec']['containers'][0]['image']
    # Prefer the actual successful preflight image digest when the retained pod exposes it.
    pod_reply = json.loads(kube('get', 'pods', '-l', f'experiment-batch={PREFLIGHT}', '-o', 'json'))
    digests = set()
    for pod in pod_reply.get('items', []):
        if pod.get('status', {}).get('phase') != 'Succeeded':
            continue
        for status in pod['status'].get('containerStatuses', []):
            value = status.get('imageID', '').split('://')[-1]
            if status['name'] == 'trainer' and '@sha256:' in value:
                digests.add(value)
    require(len(digests) <= 1, 'Ambiguous successful preflight image digests')
    if digests:
        image = next(iter(digests))
    packages = dict(pr['versions'])
    # Additional versions observed in the successful preflight installation log.
    packages.update({'peft': '0.21.2', 'accelerate': '1.7.0', 'datasets': '3.6.0',
                     'huggingface_hub': '1.33.0', 'tokenizers': '0.22.2',
                     'numpy': '2.2.5', 'pandas': '2.2.3', 'scipy': '1.15.3', 'tqdm': '4.67.1'})
    runs, jobs = {}, {}
    for seed in SEEDS:
        owner = 'zcp-off-toggle-refit7-v1' if seed <= 44 else 'taylor-3acc-zcp-off-seeds45-47-v1'
        contract_path = root / owner / 'contract.json'
        ref = load(contract_path)['runs'][f'3acc-taylor-s{seed}']
        folder = host_path(root, ref['output']) / 'tuned_model'
        args_path, adapter_path = folder / 'training_args.bin', folder / 'adapter_config.json'
        require(args_path.is_file() and adapter_path.is_file(), f'Missing reference artifacts: seed {seed}')
        require(adapter_signature(load(adapter_path)) == adapter_signature(load(root / pc['reference_adapter'])),
                f'Seed {seed} adapter differs from memory-preflight configuration')
        sources.update((contract_path, args_path, adapter_path))
        runs[str(seed)] = {'seed': seed, 'reference_args': args_path.relative_to(root).as_posix(),
                          'reference_adapter': adapter_path.relative_to(root).as_posix(), 'job': f'{NAME}-s{seed}'}
        jobs[seed] = make_job(template, root, seed, pr['node'], image, packages)
        subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server', '-f', '-'],
                       input=json.dumps(jobs[seed]), text=True, check=True)
    pinned = {path.relative_to(root).as_posix(): sha(path) for path in sources}
    bundle.mkdir()
    (bundle / 'data').mkdir()
    (bundle / 'jobs').mkdir()
    for seed in SEEDS:
        (bundle / 'results' / f's{seed}' / 'work').mkdir(parents=True)
        save(bundle / 'jobs' / f's{seed}.json', jobs[seed])
    write_rows(bundle / 'data/refit.jsonl', training)
    write_rows(bundle / 'data/test.jsonl', test)
    with (bundle / 'runner.py').open('xb') as f:
        f.write(Path(__file__).read_bytes())
    files = [bundle / 'runner.py', bundle / 'data/refit.jsonl', bundle / 'data/test.jsonl']
    files.extend(bundle / 'jobs' / f's{seed}.json' for seed in SEEDS)
    c = {'experiment': NAME, 'seeds': list(SEEDS), 'epochs': EPOCHS, 'max_length': LIMIT,
         'microbatch': 1, 'gradient_accumulation_steps': 32, 'training_records': 1035, 'test_records': 115,
         'node': pr['node'], 'image': image, 'packages': packages, 'preflight_versions': pr['versions'],
         'minimum_gpu_bytes': int(pr['gpu_total_gib'] * 1024**3) - 256 * 1024**2,
         'base_model': pc['base_model'], 'tokenizer': pc['tokenizer'], 'runs': runs,
         'excluded_raw_ids': dc['excluded_raw_ids'],
         'comparison_note': 'One authorized training record excluded; microbatch and context differ from the historical averaged-Taylor reference.',
         'source_hashes': pinned, 'prepared_hashes': {p.relative_to(bundle).as_posix(): sha(p) for p in files}}
    save(bundle / 'contract.json', c)
    save(bundle / 'run_registry.json', {'runs': list(runs.values())})
    check(root)
    print('PREPARED ALL SIX SEEDS; NOT SUBMITTED:', bundle)
    print('7 epochs; 1035 refit records; max_length 16384; batch 1 x accumulation 32; node', pr['node'])


def check(root):
    bundle = root / NAME
    c = load(bundle / 'contract.json')
    require(c['experiment'] == NAME and tuple(c['seeds']) == SEEDS, 'Wrong batch identity')
    for rel, digest in c['source_hashes'].items():
        require(sha(root / rel) == digest, f'Source drift: {rel}')
    for rel, digest in c['prepared_hashes'].items():
        require(sha(bundle / rel) == digest, f'Prepared artifact drift: {rel}')
    return c


def subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(subset(a, b) for a, b in zip(expected, actual))
    return expected == actual


def submit(root=HOST):
    check(root)
    failures = []
    for seed in SEEDS:
        try:
            path = root / NAME / 'jobs' / f's{seed}.json'
            desired = load(path)
            name = desired['metadata']['name']
            existing = kube('get', 'job', name, '--ignore-not-found', '-o', 'json').strip()
            if existing:
                actual = json.loads(existing)
                require(subset(desired['spec']['template'], actual['spec']['template']), f'Existing Job differs: {name}')
                print('EXISTS; NOT RESUBMITTED:', name)
                continue
            out = root / NAME / 'results' / f's{seed}'
            require({p.name for p in out.iterdir()} == {'work'} and not any((out / 'work').iterdir()),
                    f'Existing outputs without Job preserved: s{seed}')
            print(kube('create', '--dry-run=server', '-f', str(path)).strip())
            print(kube('create', '-f', str(path)).strip())
        except Exception as exc:
            failures.append({'seed': seed, 'error': str(exc)})
            print('SUBMISSION_FAILED', seed, str(exc), flush=True)
    require(not failures, f'Submission attempted for all seeds; unresolved: {failures}')


def configure(old, cls, seed, output):
    allowed = {f.name for f in fields(cls) if f.init}
    require('max_length' in allowed, 'SFTConfig.max_length missing')
    require(old.seed == seed, f'Reference training seed mismatch: expected {seed}')
    require(int(old.per_device_train_batch_size) * int(old.gradient_accumulation_steps) == 32, 'Reference effective batch differs')
    require(not getattr(old, 'assistant_only_loss', False), 'Reference loss objective differs')
    values = {k: getattr(old, k) for k in allowed - {'hub_token', 'push_to_hub_token'} if hasattr(old, k)}
    overrides = {'output_dir': str(output), 'logging_dir': str(output / 'logs'),
        'run_name': f'{NAME}-s{seed}', 'resume_from_checkpoint': None,
        'max_length': LIMIT, 'max_seq_length': LIMIT,
        'num_train_epochs': EPOCHS, 'max_steps': -1, 'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 32, 'seed': seed, 'eval_strategy': 'no', 'evaluation_strategy': 'no',
        'do_train': True, 'do_eval': False, 'eval_on_start': False, 'load_best_model_at_end': False,
        'save_strategy': 'epoch', 'save_total_limit': 2, 'save_safetensors': True,
        'report_to': [], 'push_to_hub': False, 'dataset_text_field': 'text', 'packing': False,
        'padding_free': False, 'assistant_only_loss': False, 'completion_only_loss': False,
        'dataloader_num_workers': 0, 'dataset_num_proc': 1, 'dataset_kwargs': None, 'model_init_kwargs': None,
        'logging_steps': 1, 'logging_first_step': True}
    values.update({k: v for k, v in overrides.items() if k in allowed})
    args = cls(**values)
    require(args.max_length == LIMIT and args.num_train_epochs == EPOCHS and args.max_steps == -1
            and args.seed == seed and args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 32,
            'Effective refit arguments differ')
    # Unlike the one-step preflight, these full-run optimization settings are preserved.
    for key in ('learning_rate', 'warmup_steps', 'warmup_ratio', 'weight_decay', 'max_grad_norm', 'optim', 'lr_scheduler_type'):
        if hasattr(old, key):
            require(getattr(args, key) == getattr(old, key), f'Reference setting changed: {key}')
    return args


def runtime(c):
    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    require(torch.cuda.is_available(), 'CUDA unavailable')
    versions = {name: importlib.metadata.version(name) for name in c['packages']}
    for name, expected in c['packages'].items():
        require(versions[name] == expected, f'Runtime version mismatch: {name}={versions[name]}, expected {expected}')
    require(torch.cuda.get_device_properties(0).total_memory >= c['minimum_gpu_bytes'], 'GPU smaller than preflight-tested capacity')
    return FastLanguageModel, torch, AutoTokenizer, set_seed, SFTConfig, SFTTrainer, Dataset, versions


def failure(output, seed, stage, exc):
    path = output / f'{stage}_failure.json'
    if not path.exists():
        save(path, {'seed': seed, 'stage': stage, 'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(), 'checkpoints_preserved': True})
    print('RUN_FAILED', seed, stage, str(exc), flush=True)


def train(seed, root=POD, output_root=OUT):
    require(seed in SEEDS, 'Unexpected seed')
    c = check(root)
    out = output_root / f's{seed}'
    require({p.name for p in out.iterdir()} == {'work'} and not any((out / 'work').iterdir()), 'Existing run output preserved; no automatic retraining')
    save(out / 'started.json', {'seed': seed, 'contract_sha256': sha(root / NAME / 'contract.json')})
    stage = 'initialization'
    try:
        p, _, _ = preflight(root)
        Fast, torch, AutoTokenizer, set_seed, SFTConfig, SFTTrainer, Dataset, versions = runtime(c)
        ref = c['runs'][str(seed)]
        old = torch.load(root / ref['reference_args'], map_location='cpu', weights_only=False)
        require(not getattr(old, 'bf16', False) or torch.cuda.is_bf16_supported(), 'GPU lacks reference BF16 support')
        adapter = load(root / ref['reference_adapter'])
        set_seed(seed)
        model, base_tokenizer = Fast.from_pretrained(model_name=str(root / c['base_model']),
            max_seq_length=LIMIT, dtype=None, load_in_4bit=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(str(root / c['tokenizer']), local_files_only=True, use_fast=True)
        require(tokenizer.is_fast and tokenizer.eos_token is not None and tokenizer.get_vocab() == base_tokenizer.get_vocab(), 'Tokenizer differs')
        model = Fast.get_peft_model(model, r=int(adapter['r']), lora_alpha=int(adapter['lora_alpha']),
            lora_dropout=float(adapter['lora_dropout']), target_modules=adapter['target_modules'],
            bias=adapter.get('bias', 'none'), use_gradient_checkpointing='unsloth', random_state=seed,
            use_rslora=bool(adapter.get('use_rslora', False)), loftq_config=adapter.get('loftq_config') or None)
        Fast.for_training(model)
        model.config.use_cache = False
        # Import the existing formatter, not the CLI and not data preprocessing.
        from ab.gpt.act.acc_predictor.train import _format_dataset
        examples = rows(root / NAME / 'data/refit.jsonl')
        require(len(examples) == c['training_records'], 'Refit count differs')
        texts = _format_dataset({'messages': [r['messages'] for r in examples]}, tokenizer)['text']
        expected = [tokenizer.encode(t, add_special_tokens=False, truncation=False) for t in texts]
        require(all(len(t) <= LIMIT for t in expected), 'An example exceeds 16k in the actual training runtime')
        args = configure(old, SFTConfig, seed, out / 'checkpoints')
        settings = args.to_dict()
        for key in ('hub_token', 'push_to_hub_token'):
            settings.pop(key, None)
        save(out / 'effective_trainer_args.json', {'seed': seed, 'adapter_seed': seed, 'training_args': settings,
            'training_records': len(examples), 'versions': versions, 'gpu': torch.cuda.get_device_name(0),
            'node': os.environ.get('NODE_NAME'), 'reference_args': ref['reference_args']})
        stage = 'token_verification'
        trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_dict({'text': texts}), processing_class=tokenizer)
        require(trainer.args.max_length == LIMIT and not trainer.args.packing and len(trainer.train_dataset) == len(examples), 'Trainer changed length/packing/coverage')
        for index, (example, text) in enumerate(zip(examples, texts)):
            item = trainer.train_dataset[index]
            actual = [int(v) for v in item['input_ids']]
            encodings = [tokenizer(text, add_special_tokens=flag, truncation=False, return_offsets_mapping=True) for flag in (False, True)]
            matched = [e for e in encodings if e['input_ids'] == actual]
            require(matched, f'Tokenization/truncation mismatch at training row {index}')
            answers = [m['content'] for m in example['messages'] if m['role'] == 'assistant']
            require(len(answers) == 1 and text.count(answers[0]) == 1, 'Assistant target not uniquely located')
            start = text.index(answers[0])
            batch = trainer.data_collator([item])
            require(batch['input_ids'][0].tolist() == actual, 'Collator changed tokens')
            p.verify_tokens(actual, [e['input_ids'] for e in encodings], matched[0]['offset_mapping'],
                            start, start + len(answers[0]), batch['labels'][0].tolist())
        del batch, expected
        save(out / 'token_verification.json', {'records': len(examples), 'max_length': LIMIT,
                                              'all_targets_supervised': True, 'truncated_records': 0})
        print('ALL_TRAINING_INPUTS_VERIFIED', seed, len(examples), flush=True)
        stage = 'training'
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        stage = 'checkpoint_save'
        checkpoint = out / 'tuned_model'
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(str(checkpoint))
        trainer.state.save_to_json(str(out / 'trainer_state.json'))
        require(float(trainer.state.epoch) >= EPOCHS - 1e-6, 'Refit did not reach seven epochs')
        require(math.isfinite(float(result.training_loss)), 'Nonfinite training loss')
        require(all(bool(torch.isfinite(param).all()) for param in model.parameters() if param.requires_grad), 'Nonfinite adapter weights')
        saved = torch.load(checkpoint / 'training_args.bin', map_location='cpu', weights_only=False)
        require(saved.seed == seed and saved.max_length == LIMIT and saved.per_device_train_batch_size == 1
                and saved.gradient_accumulation_steps == 32, 'Saved trainer arguments differ')
        weights = [checkpoint / n for n in ('adapter_model.safetensors', 'adapter_model.bin') if (checkpoint / n).is_file()]
        require(len(weights) == 1, 'Final adapter weights missing/ambiguous')
        check(root)
        save(out / 'TRAIN_COMPLETE.json', {'seed': seed, 'epochs': float(trainer.state.epoch),
            'optimizer_steps': trainer.state.global_step, 'training_loss': float(result.training_loss),
            'training_records': len(examples), 'max_length': LIMIT, 'microbatch': 1, 'accumulation': 32,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'adapter_file': weights[0].name, 'adapter_sha256': sha(weights[0]),
            'contract_sha256': sha(root / NAME / 'contract.json')})
        print('REFIT_COMPLETE', seed, flush=True)
    except Exception as exc:
        failure(out, seed, stage, exc)
        raise


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for index in order[i:j]:
            result[index] = (i + 1 + j) / 2
        i = j
    return result


def correlation(a, b):
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    ca = [v - mean_a for v in a]
    cb = [v - mean_b for v in b]
    va, vb = math.fsum(v*v for v in ca), math.fsum(v*v for v in cb)
    if math.sqrt(va / len(a)) <= 1e-12 or math.sqrt(vb / len(b)) <= 1e-12:
        return None
    den = math.sqrt(va * vb)
    return math.fsum(x*y for x, y in zip(ca, cb)) / den if den else None


def metric_values(y, pred):
    require(len(y) == len(pred) and len(y) >= 2 and all(math.isfinite(v) for v in y + pred), 'Invalid metric inputs')
    squared = math.fsum((p-t)**2 for p, t in zip(pred, y))
    mean_y = statistics.fmean(y)
    variance = math.fsum((v-mean_y)**2 for v in y)
    pearson = correlation(y, pred)
    return {'RMSE': math.sqrt(squared/len(y)), 'MAE': statistics.fmean(abs(p-t) for p, t in zip(pred, y)),
            'Pearson': pearson, 'Spearman': correlation(ranks(y), ranks(pred)) if pearson is not None else None,
            'R2': 1-squared/variance if variance > 1e-12 else None}


def verify_predictions(path, examples):
    with Path(path).open(newline='', encoding='utf-8') as f:
        predictions = list(csv.DictReader(f))
    require(len(predictions) == len(examples) == 115 and [int(r['idx']) for r in predictions] == list(range(115)), 'Test CSV coverage/order differs')
    targets, predicted = [], []
    for row, example in zip(predictions, examples):
        answer = json.loads(next(m['content'] for m in example['messages'] if m['role'] == 'assistant'))
        text = next(m['content'] for m in example['messages'] if m['role'] == 'user')
        datasets = re.findall(r'^dataset:\s*([^\r\n]+)', text, flags=re.M)
        require(len(datasets) == 1 and row['dataset'] == datasets[0].strip(), 'CSV dataset differs')
        require(float(row['gt_best_accuracy']) == float(answer['best_accuracy'])
                and float(row['gt_best_epoch']) == float(answer['best_epoch']), 'CSV targets differ')
        require(math.isfinite(float(row['pred_best_epoch'])), 'Invalid predicted epoch')
        targets.append(float(row['gt_best_accuracy']))
        predicted.append(float(row['pred_best_accuracy']))
    return metric_values(targets, predicted)


def evaluate(seed, root=POD, output_root=OUT):
    require(seed in SEEDS, 'Unexpected seed')
    c = check(root)
    out = output_root / f's{seed}'
    done = load(out / 'TRAIN_COMPLETE.json')
    require(done['seed'] == seed and done['contract_sha256'] == sha(root / NAME / 'contract.json'), 'Training completion identity differs')
    checkpoint = out / 'tuned_model'
    require(sha(checkpoint / done['adapter_file']) == done['adapter_sha256'], 'Saved adapter changed')
    csv_path = out / 'test_predictions.csv'
    require(not any((out / n).exists() for n in ('test_predictions.csv', 'test_metrics.json', 'COMPLETE.json')),
            'Existing evaluation artifacts preserved; no automatic overwrite')
    try:
        _, _, AutoTokenizer, set_seed, _, _, _, _ = runtime(c)
        set_seed(seed)
        evaluation = importlib.import_module('ab.gpt.act.acc_predictor.eval')
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        examples = rows(root / NAME / 'data/test.jsonl')
        from ab.gpt.act.acc_predictor.common import _apply_chat_template_for_inference
        for example in examples:
            messages = evaluation._test_build_prompt_messages(example['messages'])
            require(all(m['role'] != 'assistant' for m in messages), 'Target leaked into inference')
            text = _apply_chat_template_for_inference(tokenizer, messages)
            require(len(tokenizer.encode(text, add_special_tokens=False)) + 64 <= LIMIT, 'Test input/output budget exceeds 16k')
        measured = evaluation.evaluate_checkpoint(checkpoint, root / NAME / 'data/test.jsonl',
            max_seq_len=LIMIT, max_new_tokens=64, limit=None, show_progress=True, temperature=0.0,
            predictions_csv=csv_path, show_raw=False)
        evaluation._test_write_metrics_log(measured, csv_path, out / 'test_metrics.log')
        save(out / 'test_metrics.json', measured)
        require(measured['n_samples'] == measured['n_valid'] == 115 and measured['parse_failures'] == 0,
                'Test coverage/parse failures: outputs preserved, run not marked complete')
        verified = verify_predictions(csv_path, examples)
        mapping = {'RMSE': 'rmse', 'MAE': 'mae', 'Pearson': 'pearson_r', 'Spearman': 'spearman_r', 'R2': 'r2'}
        for name, suffix in mapping.items():
            a, b = verified[name], measured['best_accuracy_' + suffix]
            require(a is None and b is None or a is not None and b is not None and math.isclose(a, float(b), rel_tol=1e-8, abs_tol=1e-8),
                    f'CSV metric mismatch: {name}')
        check(root)
        save(out / 'COMPLETE.json', {'seed': seed, 'epochs': EPOCHS, 'training_records': 1035,
            'test_records': 115, 'metrics': verified, 'predictions_sha256': sha(csv_path),
            'contract_sha256': sha(root / NAME / 'contract.json')})
        print('SEED_COMPLETE', seed, json.dumps(verified), flush=True)
    except Exception as exc:
        failure(out, seed, 'evaluation', exc)
        raise


def show(root=HOST):
    check(root)
    examples = rows(root / NAME / 'data/test.jsonl')
    complete = []
    for seed in SEEDS:
        out = root / NAME / 'results' / f's{seed}'
        marker = out / 'COMPLETE.json'
        if not marker.is_file():
            failures = list(out.glob('*_failure.json'))
            print('FAILED' if failures else 'NOT COMPLETE', seed,
                  [load(p).get('error') for p in failures] if failures else '')
            continue
        record = load(marker)
        require(record['seed'] == seed and record['contract_sha256'] == sha(root / NAME / 'contract.json'), 'Completion identity differs')
        require(sha(out / 'test_predictions.csv') == record['predictions_sha256'], 'Prediction file changed')
        metrics = verify_predictions(out / 'test_predictions.csv', examples)
        for name in METRICS:
            a, b = metrics[name], record['metrics'][name]
            require(a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), 'Stored metrics differ')
        complete.append(metrics)
        print('VERIFIED', seed, json.dumps(metrics))
    if len(complete) != len(SEEDS):
        print('Batch incomplete: no partial-seed average presented.')
        return
    print('Mean [95% seed-level t CI], seeds 42-47; not paired-bootstrap inference:')
    for name in METRICS:
        values = [r[name] for r in complete]
        if any(v is None for v in values):
            print(name, 'undefined in at least one seed')
            continue
        mean = statistics.fmean(values)
        margin = T95 * statistics.stdev(values) / math.sqrt(len(values))
        print(f'{name}: {mean:.6f} [{mean-margin:.6f}, {mean+margin:.6f}]')
    print('One MaxVit training record excluded; 1035 refit / 115 test. All original inputs/results preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'train', 'evaluate', 'show'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action in ('train', 'evaluate'):
        require(args.seed is not None, '--seed is required')
        {'train': train, 'evaluate': evaluate}[args.action](args.seed)
    else:
        {'prepare': prepare, 'submit': submit, 'show': show}[args.action]()
