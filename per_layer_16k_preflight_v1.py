#!/usr/bin/env python3
"""One-step GPU memory preflight for the longest retained per-layer example.

Fresh Qwen3 base + fresh LoRA; max_length=16384; microbatch=1; accumulation=1;
one optimizer step. This is NOT a full refit or a predictive-performance run.
Old checkpoints, prompts and shared source are never written.
"""
import argparse
import ast
import copy
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback

HOST = Path('/shared/ssd/home/b-t-bose/lmur')
POD = Path('/a/mm')
NAME = 'per-layer-16k-preflight-v1'
DATA = 'per-layer-taylor-frozen-v6'
REFERENCE = 'zcp-off-toggle-refit7-v1'
RUN = '3acc-taylor-s42'
LIMIT = 16384
SEED = 42
OUTPUT = Path('/a/preflight-output')


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
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write('\n')


def host_path(root, path):
    return root / str(PurePosixPath(path).relative_to('/a/mm'))


def kube(*args):
    return subprocess.run(['kubectl', '-n', 'b-t-bose', *args], text=True,
                          capture_output=True, check=True).stdout


def find_base(root, reference, adapter, env):
    defaults = reference.get('defaults', {})
    candidates = [adapter.get('base_model_name_or_path'), env.get('BASE_MODEL_PATH'),
                  defaults.get('model_name'), defaults.get('base_model_path')]
    local = []
    for value in candidates:
        if not isinstance(value, str):
            continue
        if value.startswith('/a/mm/'):
            path = host_path(root, value)
        elif value.startswith(str(root) + '/'):
            path = Path(value)
        else:
            continue
        if path.is_dir() and (path / 'config.json').is_file():
            local.append(path)
    require(local, 'No verified local base-model path in saved adapter/Job/defaults; no remote-model fallback allowed')
    require(len({str(p.resolve()) for p in local}) == 1, f'Conflicting local base-model paths: {local}')
    path = local[0]
    require(path.resolve().is_relative_to(root.resolve()), 'Base model resolves outside the declared mounted root')
    config = load(path / 'config.json')
    require(config.get('model_type') == 'qwen3', 'Expected the reference Qwen3 model')
    native = config.get('max_position_embeddings')
    require(type(native) is int and native >= LIMIT,
            f'Base model native context does not establish 16k support: {native}; no implicit RoPE change')
    require(any(path.glob('*.safetensors')) or (path / 'model.safetensors.index.json').is_file(),
            'Base checkpoint weights/index missing')
    return path, native


def make_job(template, root, base):
    job = copy.deepcopy(template)
    require(job.get('kind') == 'Job' and job['metadata']['name'] == f'{REFERENCE}-{RUN}', 'Wrong reference GPU Job')
    job['metadata'] = {'name': NAME, 'namespace': 'b-t-bose', 'labels': {'experiment-batch': NAME}}
    spec = job['spec']
    for key in ('selector', 'manualSelector', 'ttlSecondsAfterFinished', 'suspend'):
        spec.pop(key, None)
    spec.update(backoffLimit=0, completions=1, parallelism=1, activeDeadlineSeconds=3600)
    spec['template']['metadata'] = {'labels': {'experiment-batch': NAME}}
    pod = spec['template']['spec']
    require(len(pod['containers']) == 1 and not pod.get('initContainers'), 'Unexpected containers')
    c = pod['containers'][0]
    require(c['name'] == 'trainer' and c['image'] == 'abrainone/ai-linux:cu12.6.3-latest', 'GPU runtime template changed')
    require(str(c['resources']['limits'].get('nvidia.com/gpu')) == '1', 'Expected one GPU')
    mounts = {m['name']: m for m in c['volumeMounts']}
    require(set(mounts) == {'lmur', 'dshm'} and mounts['lmur']['mountPath'] == '/a/mm', 'Unexpected reference mounts')
    volumes = {v['name']: v for v in pod['volumes']}
    require(volumes['lmur']['hostPath']['path'] == str(root), 'Wrong shared root')
    mounts['lmur']['readOnly'] = True
    c['volumeMounts'].append({'name': 'preflight-output', 'mountPath': OUTPUT.as_posix()})
    pod['volumes'].append({'name': 'preflight-output', 'hostPath': {
        'path': str(root / NAME / 'results'), 'type': 'Directory'}})
    c['workingDir'] = (OUTPUT / 'work').as_posix()
    c['env'] = [e for e in c.get('env', []) if e['name'] in ('USER', 'LOGNAME', 'HOME')]
    c['env'].extend({'name': k, 'value': v} for k, v in {
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HOME': '/tmp/longctx-hf', 'TRANSFORMERS_CACHE': '/tmp/longctx-hf/hub',
        'XDG_CACHE_HOME': '/tmp/longctx-cache', 'TORCHINDUCTOR_CACHE_DIR': '/tmp/longctx-inductor',
        'BASE_MODEL_PATH': '/a/mm/' + base.relative_to(root).as_posix(),
        'OUT_DIR': (OUTPUT / 'work').as_posix(), 'RUN_SEED': str(SEED),
    }.items())
    c['env'].append({'name': 'NODE_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}})
    c['command'] = ['/bin/bash', '-lc']
    c['args'] = [
        'set -euo pipefail\n'
        'trap \'printf "FAILED at shell line %s\\n" "$LINENO" >&2\' ERR\n'
        'unset PYTHONPATH\n'
        'mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"\n'
        'python -m pip install --disable-pip-version-check --no-cache-dir --user '
        'unsloth==2026.3.4 unsloth-zoo==2026.3.2 transformers==5.2.0 trl==0.24.0 torchao==0.16.0 '
        'datasets peft accelerate bitsandbytes numpy pandas scipy tqdm\n'
        f'python -u /a/mm/{NAME}/runner.py run\n']
    pod['restartPolicy'] = 'Never'
    pod['automountServiceAccountToken'] = False
    return job


def prepare(root=HOST):
    bundle = root / NAME
    require(not bundle.exists(), f'Existing preflight bundle preserved: {bundle}')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Preflight Job already exists')
    data_root = root / DATA
    data_contract = load(data_root / 'contract.json')
    audit = load(data_root / 'results/TOKEN_AUDIT.json')
    require(audit['contract_sha256'] == sha(data_root / 'contract.json'), 'Data contract changed since token audit')
    lengths_path = data_root / 'results/token_lengths.json'
    require(audit['token_lengths_sha256'] == sha(lengths_path), 'Token audit data changed')
    require(data_contract['split_counts'] == {'train': 977, 'validation': 58, 'test': 115}, 'Unexpected retained dataset')
    lengths = load(lengths_path)
    inputs = {}
    sources = [data_root / 'contract.json', data_root / 'results/TOKEN_AUDIT.json', lengths_path]
    for split, filename in data_contract['files'].items():
        path = data_root / 'prompts' / filename
        require(sha(path) == data_contract['prepared_hashes']['prompts/' + filename], 'Prepared prompt hash changed')
        inputs[split] = jsonl(path)
        require(len(inputs[split]) == data_contract['split_counts'][split] == len(lengths[split]), 'Input/token coverage mismatch')
        sources.append(path)
    longest = max((dict(row, split=split) for split in ('train', 'validation') for row in lengths[split]),
                  key=lambda r: r['training_conservative'])
    example = inputs[longest['split']][longest['position']]
    require(example['architecture_id'] == longest['architecture_id'], 'Longest-record identity mismatch')
    require(longest['training_conservative'] <= LIMIT, 'Measured longest example exceeds 16k')
    require(any('LAYER_TAYLOR_BY_EPOCH' in m['content'] for m in example['messages'] if m['role'] == 'user'),
            'Wrong input representation')
    reference_path = root / REFERENCE / 'contract.json'
    reference = load(reference_path)
    run = reference['runs'][RUN]
    reference_output = host_path(root, run['output'])
    adapter_path = reference_output / 'tuned_model/adapter_config.json'
    args_path = reference_output / 'tuned_model/training_args.bin'
    require(args_path.is_file(), 'Saved reference training arguments missing')
    adapter = load(adapter_path)
    require(not adapter.get('use_dora') and not adapter.get('modules_to_save')
            and not adapter.get('rank_pattern') and not adapter.get('alpha_pattern'), 'Unsupported nonuniform adapter settings')
    job_path = root / REFERENCE / 'jobs' / f'{RUN}.json'
    template = load(job_path)
    env = {e['name']: e['value'] for e in template['spec']['template']['spec']['containers'][0].get('env', []) if 'value' in e}
    base, native_context = find_base(root, reference, adapter, env)
    tokenizer = reference_output / 'tuned_model'
    require((tokenizer / 'tokenizer.json').is_file() and (tokenizer / 'tokenizer_config.json').is_file(), 'Saved reference tokenizer missing')
    sources.extend([reference_path, adapter_path, args_path, job_path, base / 'config.json',
                    root / 'nn-gpt/ab/gpt/act/acc_predictor/train.py'])
    sources.extend(p for p in tokenizer.iterdir() if p.is_file() and
                   (p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'chat_template'))
                    or p.name in ('vocab.json', 'merges.txt', 'config.json')))
    if (base / 'model.safetensors.index.json').is_file():
        sources.append(base / 'model.safetensors.index.json')
    pinned = {p.relative_to(root).as_posix(): sha(p) for p in sources}
    job = make_job(template, root, base)
    subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server', '-f', '-'],
                   input=json.dumps(job), text=True, check=True)
    for rel, digest in pinned.items():
        require(sha(root / rel) == digest, f'Source changed during preparation: {rel}')
    bundle.mkdir()
    (bundle / 'results/work').mkdir(parents=True)
    save(bundle / 'example.json', example)
    save(bundle / 'job.json', job)
    with (bundle / 'runner.py').open('xb') as f:
        f.write(Path(__file__).read_bytes())
    contract = {'experiment': NAME, 'sequence_length': LIMIT, 'seed': SEED,
        'microbatch': 1, 'gradient_accumulation_steps': 1, 'optimizer_steps': 1,
        'purpose': 'longest-example memory preflight only; not full training',
        'longest_example': longest, 'retained_split_counts': data_contract['split_counts'],
        'base_model': base.relative_to(root).as_posix(), 'native_model_context': native_context,
        'reference_adapter': adapter_path.relative_to(root).as_posix(),
        'reference_args': args_path.relative_to(root).as_posix(),
        'tokenizer': tokenizer.relative_to(root).as_posix(),
        'source_hashes': pinned,
        'prepared_hashes': {n: sha(bundle / n) for n in ('runner.py', 'example.json', 'job.json')}}
    save(bundle / 'contract.json', contract)
    check(root)
    print('PREFLIGHT PREPARED; NOT SUBMITTED:', bundle)
    print('Longest example:', json.dumps(longest))
    print('Native context:', native_context, '; requested trainer/model length:', LIMIT)
    print('One GPU, microbatch 1, accumulation 1, one optimizer step; no checkpoint saving.')


def check(root):
    bundle = root / NAME
    c = load(bundle / 'contract.json')
    require(c['experiment'] == NAME and c['sequence_length'] == LIMIT, 'Unexpected preflight contract')
    for rel, digest in c['source_hashes'].items():
        require(sha(root / rel) == digest, f'Source drift: {rel}')
    for name, digest in c['prepared_hashes'].items():
        require(sha(bundle / name) == digest, f'Prepared preflight changed: {name}')
    return c


def submit(root=HOST):
    check(root)
    out = root / NAME / 'results'
    require({p.name for p in out.iterdir()} == {'work'} and not any((out / 'work').iterdir()), 'Existing preflight output preserved')
    require(not kube('get', 'job', NAME, '--ignore-not-found', '-o', 'name').strip(), 'Job exists; inspect rather than resubmit')
    manifest = str(root / NAME / 'job.json')
    print(kube('create', '--dry-run=server', '-f', manifest).strip())
    print(kube('create', '-f', manifest).strip())


def training_args(old, config_class, output):
    allowed = {f.name for f in fields(config_class) if f.init}
    require('max_length' in allowed, 'Installed SFTConfig lacks max_length; stop rather than rely on a default')
    require(getattr(old, 'seed', None) == SEED, 'Reference seed differs from seed 42')
    require(not getattr(old, 'assistant_only_loss', False), 'Reference objective differs from full-conversation text SFT')
    excluded = {'hub_token', 'push_to_hub_token'}
    values = {k: getattr(old, k) for k in allowed - excluded if hasattr(old, k)}
    overrides = {'output_dir': str(output), 'max_length': LIMIT, 'max_seq_length': LIMIT,
        'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 1,
        'max_steps': 1, 'num_train_epochs': 1, 'warmup_steps': 0, 'warmup_ratio': 0.0,
        'save_strategy': 'no', 'eval_strategy': 'no', 'evaluation_strategy': 'no',
        'do_train': True, 'do_eval': False, 'eval_on_start': False,
        'load_best_model_at_end': False, 'report_to': [], 'push_to_hub': False,
        'logging_strategy': 'steps', 'logging_steps': 1, 'logging_first_step': True,
        'dataloader_num_workers': 0, 'dataset_num_proc': 1, 'dataset_text_field': 'text',
        'dataset_kwargs': None, 'model_init_kwargs': None, 'packing': False, 'padding_free': False,
        'assistant_only_loss': False, 'completion_only_loss': False, 'seed': SEED}
    values.update({k: v for k, v in overrides.items() if k in allowed})
    args = config_class(**values)
    require(args.max_length == LIMIT and args.max_steps == 1
            and args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 1,
            'Effective preflight settings differ')
    return args, {k: v for k, v in overrides.items() if k in allowed}


def verify_tokens(actual, candidates, offsets, answer_start, answer_end, labels):
    require(any(actual == candidate for candidate in candidates), 'Trainer changed/truncated the serialized example')
    require(len(actual) <= LIMIT and len(labels) == len(actual) == len(offsets), 'Batch lengths differ or exceed 16k')
    answer_positions = [i for i, (start, end) in enumerate(offsets) if end > answer_start and start < answer_end]
    require(answer_positions, 'Assistant target token span not found')
    require(all(labels[i] == actual[i] for i in answer_positions), 'Assistant target was truncated or masked')
    require(all(label == -100 or label == token for label, token in zip(labels, actual)), 'Unexpected training labels')
    return {'input_tokens': len(actual), 'supervised_tokens': sum(v != -100 for v in labels),
            'assistant_target_tokens': len(answer_positions), 'target_fully_supervised': True}


def run(root=POD, output=OUTPUT):
    report = {'experiment': NAME, 'status': 'failed', 'stage': 'preflight', 'full_training_launched': False}
    torch_module = None
    try:
        c = check(root)
        require({p.name for p in output.iterdir()} == {'work'} and not any((output / 'work').iterdir()), 'Preflight output not fresh')
        save(output / 'started.json', {'contract_sha256': sha(root / NAME / 'contract.json')})
        report['contract_sha256'] = sha(root / NAME / 'contract.json')
        report['stage'] = 'imports'
        from unsloth import FastLanguageModel  # Must precede transformers/trl imports.
        import torch
        torch_module = torch
        from transformers import AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset
        require(torch.cuda.is_available(), 'Allocated CUDA GPU unavailable')
        report['versions'] = {p: importlib.metadata.version(p) for p in
                              ('unsloth', 'unsloth-zoo', 'transformers', 'trl', 'torch', 'torchao', 'bitsandbytes')}
        require(report['versions']['transformers'] == '5.2.0' and report['versions']['trl'] == '0.24.0', 'Pinned trainer versions differ')
        import os
        report['node'] = os.environ.get('NODE_NAME')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['gpu_total_gib'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        report['stage'] = 'reference_settings'
        old = torch.load(root / c['reference_args'], map_location='cpu', weights_only=False)
        require(not getattr(old, 'bf16', False) or torch.cuda.is_bf16_supported(), 'Selected GPU lacks reference BF16 support')
        old_batch = int(old.per_device_train_batch_size)
        old_accum = int(old.gradient_accumulation_steps)
        report['reference_batch'] = {'microbatch': old_batch, 'accumulation': old_accum,
                                    'effective_batch_one_gpu': old_batch * old_accum}
        report['possible_full_run_batch'] = {'microbatch': 1, 'accumulation': old_batch * old_accum,
                                             'note': 'proposal only; no full run launched'}
        adapter = load(root / c['reference_adapter'])
        set_seed(SEED)
        report['stage'] = 'load_fresh_base'
        model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(root / c['base_model']), max_seq_length=LIMIT,
            dtype=None, load_in_4bit=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(str(root / c['tokenizer']), local_files_only=True, use_fast=True)
        require(tokenizer.get_vocab() == base_tokenizer.get_vocab(), 'Saved/base tokenizer vocabularies differ')
        require(tokenizer.is_fast and tokenizer.eos_token is not None, 'Fast reference tokenizer with EOS required')
        report['stage'] = 'fresh_lora'
        model = FastLanguageModel.get_peft_model(model, r=int(adapter['r']),
            lora_alpha=int(adapter['lora_alpha']), lora_dropout=float(adapter['lora_dropout']),
            target_modules=adapter['target_modules'], bias=adapter.get('bias', 'none'),
            use_gradient_checkpointing='unsloth', random_state=SEED,
            use_rslora=bool(adapter.get('use_rslora', False)), loftq_config=adapter.get('loftq_config') or None)
        FastLanguageModel.for_training(model)
        model.config.use_cache = False
        report['adapter'] = {k: adapter.get(k) for k in ('r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias', 'use_rslora')}
        example = load(root / NAME / 'example.json')
        source = (root / 'nn-gpt/ab/gpt/act/acc_predictor/train.py').read_text()
        funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_format_dataset']
        require(len(funcs) == 1 and not funcs[0].decorator_list, 'Unexpected training formatter')
        ns = {}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), 'reference_training_formatter', 'exec'), ns)
        text = ns['_format_dataset']({'messages': [example['messages']]}, tokenizer)['text'][0]
        encoded = [tokenizer(text, add_special_tokens=flag, truncation=False, return_offsets_mapping=True)
                   for flag in (False, True)]
        require(max(len(e['input_ids']) for e in encoded) <= LIMIT, 'Actual training-runtime tokenization exceeds 16k')
        answers = [m['content'] for m in example['messages'] if m['role'] == 'assistant']
        require(len(answers) == 1 and text.count(answers[0]) == 1, 'Assistant target not uniquely located in serialization')
        answer_start = text.index(answers[0]); answer_end = answer_start + len(answers[0])
        args, overrides = training_args(old, SFTConfig, output / 'work/trainer')
        report['preflight_overrides'] = overrides
        report['reference_cpu_audit'] = c['longest_example']
        report['stage'] = 'construct_trainer'
        trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_dict({'text': [text]}),
                             processing_class=tokenizer)
        require(trainer.args.max_length == LIMIT and not trainer.args.packing, 'Trainer effective limit/packing differs')
        require(len(trainer.train_dataset) == 1, 'Trainer split/dropped the longest example')
        actual = [int(value) for value in trainer.train_dataset[0]['input_ids']]
        matching = [e for e in encoded if e['input_ids'] == actual]
        require(matching, 'Trainer input differs from full untruncated serialization')
        batch = next(iter(trainer.get_train_dataloader()))
        require(batch['input_ids'].shape[0] == 1 and batch['input_ids'][0].tolist() == actual, 'Collator changed input tokens')
        report['token_check'] = verify_tokens(actual, [e['input_ids'] for e in encoded],
            matching[0]['offset_mapping'], answer_start, answer_end, batch['labels'][0].tolist())
        del batch
        torch.cuda.synchronize()
        report['load_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
        report['stage'] = 'one_optimizer_step'
        outcome = trainer.train()
        torch.cuda.synchronize()
        require(trainer.state.global_step == 1, 'Expected exactly one optimizer step')
        loss = float(outcome.training_loss)
        require(math.isfinite(loss), 'Nonfinite preflight loss')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad), 'Nonfinite trainable parameters')
        for entry in trainer.state.log_history:
            if 'grad_norm' in entry:
                require(math.isfinite(float(entry['grad_norm'])), 'Nonfinite gradient norm')
        report['optimizer_steps'] = trainer.state.global_step
        report['training_loss'] = loss
        report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report['train_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        report['train_peak_reserved_gib'] = torch.cuda.max_memory_reserved() / (1024 ** 3)
        report['device_free_gib_after_step'] = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        check(root)
        report.update(status='pass', stage='complete', checkpoint_saved=False,
                      interpretation='One longest-example optimizer step at batch 1 fits on this GPU. Not a full-run guarantee or accuracy result.')
        save(output / 'report.json', report)
        require(load(output / 'report.json')['status'] == 'pass', 'Report readback failed')
        print('LONG_CONTEXT_PREFLIGHT_PASS', json.dumps(report, default=str), flush=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        if torch_module is not None and torch_module.cuda.is_initialized():
            try:
                report['peak_allocated_gib_at_failure'] = torch_module.cuda.max_memory_allocated() / (1024 ** 3)
                report['peak_reserved_gib_at_failure'] = torch_module.cuda.max_memory_reserved() / (1024 ** 3)
            except Exception:
                pass
        if output.is_dir() and not (output / 'report.json').exists():
            save(output / 'report.json', report)
        print('LONG_CONTEXT_PREFLIGHT_FAILED', json.dumps(report, default=str), flush=True)
        raise


def show(root=HOST):
    check(root)
    report = load(root / NAME / 'results/report.json')
    require(report.get('contract_sha256') == sha(root / NAME / 'contract.json'), 'Report contract mismatch')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'submit', 'run', 'show'))
    a = p.parse_args()
    {'prepare': prepare, 'submit': submit, 'run': run, 'show': show}[a.action]()
