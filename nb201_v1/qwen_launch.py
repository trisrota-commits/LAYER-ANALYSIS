"""Build an isolated NB201 Qwen deployment; never edit repository sources."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

NAME = 'nb201-v1-qwen-matched-v1'
ROOT = Path(__file__).resolve().parent.parent
POD = '/a/mm/' + NAME
HOST = '/shared/ssd/home/b-t-bose/lmur'
SEEDS = tuple(range(42, 48))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))


def original_prompts():
    spec = importlib.util.spec_from_file_location('nb201_original_prompts', ROOT / 'nb201_v1/original/prompts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example(record, dataset, taylor):
    p = original_prompts()
    if dataset == 'cifar10':
        return p.example(record, ('taylor_imp',) if taylor else ())
    assert dataset == 'cifar100'
    assert record['target_kind'] == 'final_validation_accuracy' and record['target_epoch'] == 200
    system = ('You are a strict JSON generator.\nYou must output exactly ONE JSON object and nothing else.\n\n'
              'The JSON must contain exactly the keys:\nfinal_accuracy\ntarget_epoch\n\n'
              'Rules:\nfinal_accuracy must be a float in [0,100] rounded to 2 decimals.\n'
              'It represents validation accuracy at the fixed target epoch 200.\n'
              'target_epoch must be the integer 200.\n\nDo not explain.\nDo not add text.\n'
              'Stop immediately after the closing brace }.\n')
    parts = ['INPUT\ntask: img-classification\ndataset: cifar100\nmetric: acc',
             'TRAINING_BUDGET\nmax_epochs: 200\ntarget_epoch: 200',
             'EARLY_TRAINING_SIGNAL\n' + '\n'.join(f'epoch_{e}_accuracy: {round(record[f"accuracy_epoch_{e}"], 6)}' for e in (1,2,3))]
    if taylor:
        parts.append(p.layer_block({e: {'taylor_imp': record[f'taylor_epoch_{e}']} for e in (1,2,3)}, ('taylor_imp',)))
    parts.append('Using the early accuracy observations and per-epoch layer-statistics averages, if present, '
                 'estimate validation accuracy at epoch 200. The target is mean epoch-200 validation accuracy '
                 'over benchmark seeds 777 and 888.\n\nOUTPUT (JSON ONLY):')
    return dict(messages=[dict(role='system', content=system), dict(role='user', content='\n\n'.join(parts)),
                          dict(role='assistant', content=json.dumps(dict(final_accuracy=round(record['target_final_validation_accuracy'], 2), target_epoch=200), separators=(',', ':')))],
                architecture_id=record['arch_family'], record_id=record['record_id'], arch_index=record['architecture_id'])


WORKER_HELPERS = '''
def configure_evaluator(evaluator, run):
    # Only this process's parser is adapted; repository files stay unchanged.
    if run['dataset'] != 'cifar100':
        return
    original = evaluator._test_parse_assistant_json
    def parse_final(text):
        parsed = original(text)
        if not isinstance(parsed, dict):
            return None
        return dict(best_accuracy=parsed.get('final_accuracy'), best_epoch=parsed.get('target_epoch'))
    evaluator._test_parse_assistant_json = parse_final

def export_predictions(out, root, key, run):
    import csv
    registry = json.loads((root / 'test_registry.json').read_text())
    with (out / 'test_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(registry) == 275
    exported = []
    for i, (row, item) in enumerate(zip(rows, registry)):
        assert int(row['idx']) == i and row['dataset'] == item['dataset']
        assert float(row['gt_best_accuracy']) == item['target']
        value = float(row['pred_best_accuracy'])
        exported.append(dict(record_id=item['record_id'], prediction=value, target=item['target'], seed=run['seed']))
    path = out / 'per_record_predictions.csv'
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['record_id', 'prediction', 'target', 'seed'])
        writer.writeheader()
        writer.writerows(exported)
    with path.open(newline='') as f:
        saved = list(csv.DictReader(f))
    assert [r['record_id'] for r in saved] == [r['record_id'] for r in registry]
    for a, b in zip(exported, saved):
        assert a['target'] == float(b['target']) and a['seed'] == int(b['seed'])
        assert a['prediction'] == float(b['prediction']) or (math.isnan(a['prediction']) and math.isnan(float(b['prediction'])))
    (out / 'prediction_contract.json').write_text(json.dumps(dict(
        run=key, target_kind=run['target_kind'], legacy_columns='best_accuracy/best_epoch in raw evaluator CSV are internal compatibility names; use target_kind',
        prediction_sha256=sha(path), n=275,
        finite_accuracy_predictions=sum(math.isfinite(r['prediction']) for r in exported)), indent=2))
'''

RUNTIME_AUDIT = '''    import importlib.metadata
    from ab.gpt.act.acc_predictor.common import _apply_chat_template_for_inference
    cap = min(int(v) for v in saved_lengths.values())
    lengths = [len(tokenizer(row['text'], truncation=False, add_special_tokens=True)['input_ids']) for row in data]
    text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    test_rows = [json.loads(line) for line in (root / 'test_llm_dataset.jsonl').read_text().splitlines()]
    inference_lengths = [len(text_tokenizer.encode(_apply_chat_template_for_inference(text_tokenizer,
        [m for m in row['messages'] if m['role'] in ('system', 'user')]), add_special_tokens=False)) for row in test_rows]
    audit = dict(run=key, training_records=len(lengths), effective_cap=cap,
        max_training_tokens=max(lengths), over_cap_records=sum(n > cap for n in lengths),
        max_inference_tokens=max(inference_lengths), inference_context=seq,
        effective_seed=trainer.args.seed, adapter_seed=run['seed'],
        gpu=t.torch.cuda.get_device_name(0),
        packages={p: importlib.metadata.version(p) for p in ('torch','transformers','trl','unsloth','unsloth-zoo','datasets','peft')})
    (out / 'runtime_preflight.json').write_text(json.dumps(audit, indent=2))
    assert audit['training_records'] == 825 and audit['over_cap_records'] == 0
    assert max(inference_lengths) + config.TEST_MAX_NEW_TOKENS <= seq
    print('NB201_RUNTIME_PREFLIGHT_PASS', json.dumps(audit), flush=True)
'''


def worker_source():
    path = ROOT / 'evidence/epoch1-taylor-reference/extended/runner.py'
    assert sha(path) == 'ad4c81abc45e413e1b87d12a806f77004654444262020f432e6655962f4479d6'
    text = path.read_text()
    # Keep the 978-record assertion for the historical selection settings.
    # Recompute the NEW refit's warmup for 825 records.
    text = text.replace('1036', '825')
    replacements = [
        ("('train', 978), ('val', 58), ('test', 115)", "('train', 772), ('val', 53), ('test', 275)"),
        ("metrics['n_samples'] == 115", "metrics['n_samples'] == 275"),
        ('run_config.USE_LAYER_STATS = True', "run_config.USE_LAYER_STATS = run['condition'] == 'taylor'"),
        ("run_config.ACTIVE_LAYER_FEATURES = 'taylor_imp'", "run_config.ACTIVE_LAYER_FEATURES = 'taylor_imp' if run['condition'] == 'taylor' else None"),
        ('Frozen original 978-record training normalization', 'Frozen grouped NB201 split; no numerical feature normalization'),
        ('        metrics = evaluator.test_model(', '        configure_evaluator(evaluator, run)\n        metrics = evaluator.test_model('),
        ('        # Coverage is reported honestly, not used to retune or rerun a model.',
         '        export_predictions(out, root, key, run)\n        # Coverage is reported honestly, not used to retune or rerun a model.'),
        ('    trainer.train()  # Fresh run: no resume checkpoint or old optimizer/scheduler.',
         RUNTIME_AUDIT + '    trainer.train()  # Fresh run: no resume checkpoint or old optimizer/scheduler.'),
        ("if __name__ == '__main__':", WORKER_HELPERS + "\nif __name__ == '__main__':"),
    ]
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new, 1)
    compile(text, 'nb201_worker.py', 'exec')
    return text


CONTROL = '''import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = 'cifar10-taylor-s42'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def host(p):
    return ROOT / Path(p).relative_to('/a/mm')

def verify():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(HERE / name) == digest, ('Bundle drift', name)
    contract = json.loads((HERE / 'contract.json').read_text())
    for name, digest in contract['source_sha256'].items():
        assert sha(host(name)) == digest, ('Source/input drift', name)
    for run in contract['runs'].values():
        for name, digest in run['selection_sha256'].items():
            assert sha(host(run['selected']) / name) == digest, ('Saved settings drift', name)
    base = host(contract['base'])
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    for name in ('config.json', 'tokenizer.json', *set(index['weight_map'].values())):
        assert (base / name).is_file() and (base / name).stat().st_size > 0, name
    print('PREFLIGHT_PASS; free GiB:', round(shutil.disk_usage(HERE).free / 2**30, 2), flush=True)
    return contract

def completed(key):
    out = HERE / 'results' / key
    done = json.loads((out / 'test_complete.json').read_text())
    prediction = json.loads((out / 'prediction_contract.json').read_text())
    runtime = json.loads((out / 'runtime_preflight.json').read_text())
    assert done['run'] == prediction['run'] == runtime['run'] == key
    assert done['n_samples'] == done['n_valid'] == prediction['n'] == prediction['finite_accuracy_predictions'] == 275
    assert done['parse_failures'] == runtime['over_cap_records'] == 0
    assert sha(out / 'per_record_predictions.csv') == prediction['prediction_sha256']
    return done

def main():
    action = sys.argv[1]
    assert action in ('check', 'pilot', 'remaining', 'show')
    contract = verify()
    if action == 'check':
        return
    if action == 'show':
        for key in contract['runs']:
            out = HERE / 'results' / key
            if (out / 'test_complete.json').exists():
                print(key, json.loads((out / 'test_complete.json').read_text()))
            else:
                print(key, 'artifacts present' if out.exists() else 'not started')
        return
    if action == 'remaining':
        completed(PILOT)
        job = json.loads(subprocess.check_output(['kubectl', '-n', 'b-t-bose', 'get', 'job', HERE.name + '-' + PILOT, '-o', 'json'], text=True))
        assert job.get('status', {}).get('succeeded') == 1, 'Pilot Job not completed'
    keys = [PILOT] if action == 'pilot' else [k for k in contract['runs'] if k != PILOT]
    for key in keys:
        assert not (HERE / 'results' / key).exists(), ('Existing results preserved', key)
        existing = subprocess.check_output(['kubectl', '-n', 'b-t-bose', 'get', 'job', HERE.name + '-' + key, '--ignore-not-found', '-o', 'name'], text=True)
        assert not existing.strip(), ('Job already exists; do not resubmit', key)
        subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server', '-f', str(HERE / 'jobs' / (key + '.json'))], check=True)
    for key in keys:
        subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '-f', str(HERE / 'jobs' / (key + '.json'))], check=True)

if __name__ == '__main__':
    main()
'''


def build(dest):
    assert not dest.exists(), 'Existing bundle preserved'
    reference = json.loads((ROOT / 'evidence/epoch1-taylor-reference/extended/contract.json').read_text())
    template = json.loads((ROOT / 'evidence/epoch1-taylor-reference/original/3acc-taylor-s42.json').read_text())
    selected = reference['runs']['3acc-taylor-s45']
    contract = dict(epochs=7, base=reference['base'], source_sha256={}, runs={},
        scope='Matched NB201 Qwen: fixed external seven-epoch refit; no NB201 test tuning; same seed42-selected hyperparameters for every arm/seed',
        target_precision='Assistant targets rounded to 2 decimals as in original prompts; raw benchmark targets retained in registries',
        counts=dict(train=772, val=53, test=275, refit=825))
    for p in (ROOT / 'evidence/nb201-qwen-source-v1').glob('*.py'):
        key = '/a/mm/nn-gpt/ab/gpt/act/acc_predictor/' + p.name
        assert sha(p) == reference['source_sha256'][key], ('Current source drift', p.name)
        contract['source_sha256'][key] = sha(p)
    assert len(contract['source_sha256']) == 11
    frozen = {}
    for ds in ('cifar10', 'cifar100'):
        folder = ROOT / 'evidence/nb201-collected-v1' / ds
        snapshot = folder / 'snapshot.jsonl'
        assert sha(snapshot) == json.loads((folder / 'collection_complete.json').read_text())['snapshot_sha256']
        contract['source_sha256'][f'/a/mm/nb201-v1-{ds}-full/snapshot.jsonl'] = sha(snapshot)
        records = [json.loads(l) for l in snapshot.read_text().splitlines()]
        cohort = json.loads((ROOT / 'nb201_v1' / ('cohort.json' if ds == 'cifar10' else 'cifar100/cohort.json')).read_text())
        key = 'arch_index' if ds == 'cifar10' else 'architecture_id'
        byid = {r[key]: r for r in records}
        assert len(byid) == len(records) == 1100
        assert set(byid) == {i for ids in cohort['splits'].values() for i in ids}
        frozen[ds] = (byid, cohort)
    worker = worker_source()
    compile(CONTROL, 'control.py', 'exec')
    dest.mkdir(parents=True)
    put(dest / 'worker.py', worker)
    put(dest / 'control.py', CONTROL)
    contract['source_sha256'][POD + '/worker.py'] = sha(dest / 'worker.py')
    for ds, (byid, cohort) in frozen.items():
        for cond in ('no-layer', 'taylor'):
            local = dest / 'prompts' / ds / cond
            for split, count in [('train', 772), ('val', 53), ('test', 275)]:
                rows = [example(byid[i], ds, cond == 'taylor') for i in cohort['splits'][split]]
                assert len(rows) == len({r['record_id'] for r in rows}) == count
                put(local / f'{split}_llm_dataset.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
                if split == 'test':
                    registry = []
                    for i, row in zip(cohort['splits'][split], rows):
                        answer = json.loads(row['messages'][2]['content'])
                        raw = byid[i]
                        registry.append(dict(record_id=row['record_id'], architecture_id=i,
                            family=row['architecture_id'], dataset=raw['dataset'],
                            target=answer['best_accuracy' if ds == 'cifar10' else 'final_accuracy'],
                            raw_target=raw['best_accuracy' if ds == 'cifar10' else 'target_final_validation_accuracy']))
                    put(local / 'test_registry.json', registry)
            digests = {p.name: sha(p) for p in local.iterdir()}
            for seed in SEEDS:
                key = f'{ds}-{cond}-s{seed}'
                name = NAME + '-' + key
                run = dict(seed=seed, settings_seed=42, selected=selected['selected'],
                    selection_sha256=selected['selection_sha256'], data=f'{POD}/prompts/{ds}/{cond}',
                    output=f'{POD}/results/{key}', input_sha256=digests, dataset=ds, condition=cond,
                    zcp_enabled=False, target_kind='peak_mean_validation_curve' if ds == 'cifar10' else 'final_validation_accuracy')
                contract['runs'][key] = run
                job = copy.deepcopy(template)
                labels = {'experiment-batch': NAME, 'dataset': ds, 'condition': cond, 'run-seed': str(seed)}
                job['metadata'] = dict(name=name, namespace='b-t-bose', labels=labels)
                job['spec']['template']['metadata'] = dict(labels=labels)
                container = job['spec']['template']['spec']['containers'][0]
                env = {r['name']: r['value'] for r in container['env']}
                env.update(DATA_ROOT=run['data'], OUT_DIR=run['output'], RUN_SEED=str(seed),
                    LAYER_ON=str(cond == 'taylor').lower(), THIRD_ON='true',
                    XDG_CACHE_HOME='/tmp/cache-' + name, TORCHINDUCTOR_CACHE_DIR='/tmp/torchinductor-' + name)
                container['env'] = [dict(name=k, value=v) for k, v in env.items()]
                install = container['args'][0].split('export PYTHONPATH=')[0]
                container['args'] = [install + f'''export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset
export OUT_DIR={run['output']}
python -u {POD}/worker.py train {POD}/contract.json {key}
python -u {POD}/worker.py test {POD}/contract.json {key}
''']
                put(dest / 'jobs' / (key + '.json'), job)
    put(dest / 'contract.json', contract)
    put(dest / 'manifest.json', {p.relative_to(dest).as_posix(): sha(p) for p in dest.rglob('*') if p.is_file()})
    print('PREPARED', dest, '24 Jobs; NONE submitted', flush=True)


if __name__ == '__main__':
    dest = ROOT / 'evidence' / NAME
    archive = dest.with_suffix('.tar.gz')
    assert not archive.exists(), 'Existing archive preserved'
    build(dest)
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(dest, arcname=NAME)
    print('ARCHIVE', archive, 'SHA256', sha(archive), flush=True)
