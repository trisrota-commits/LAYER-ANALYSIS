"""Recover only seed-44 unfinished inner fits. Never delete old outputs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

NAME = 'lemur-nested-qwen-s44-inner-recovery-v1'
OLD = 'lemur-nested-qwen-s44-inner-p12-v1'
BASE = 'lemur-nested-qwen-full-v1'
EVAL = ('inner-f1-c06', 'inner-f1-c07')
TRAIN = ('inner-f1-c05', 'inner-f2-c00', 'inner-f2-c03', 'inner-f2-c04',
         'inner-f2-c05', 'inner-f2-c06', 'inner-f2-c07')
EXPECTED = {
    'control.py': 'c6d9733240e8bd4930abf03a25476102ee085c26ff3d434a0aba9d3bc03908dc',
    'worker.py': 'fe6309053b6b88f46f744515c8238031705751159a093659da789fd6bc745dad',
    'storage-safe-continuation-v1/worker_no_intermediate.py': 'cea1d1a216ef90840bbaeee484a6253aebc93387f7513511c2d64493aa00980a',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def js(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def controller(root):
    base = root / BASE
    for name, digest in EXPECTED.items():
        require(sha(base / name) == digest, 'Original source changed: ' + name)
    source = (base / 'control.py').read_text()
    anchor = 'args["seed"] == 42'
    require(source.count(anchor) == 1, 'Unexpected verifier source')
    scope = {'__name__': 'recovery_verifier', '__file__': str(base / 'control.py')}
    exec(compile(source.replace(anchor, 'args["seed"] == run["seed"]'), str(base / 'control.py'), 'exec'), scope)
    return scope


def plan(original):
    require(len(original['runs']) == 24 and original['epochs'] == 7, 'Unexpected original contract')
    result = copy.deepcopy(original)
    result['runs'] = {}
    for key in EVAL + TRAIN:
        run = copy.deepcopy(original['runs'][key])
        require(run['seed'] == 44 and run['phase'] == 'inner', 'Wrong seed or phase')
        require(run['output'] == '/a/mm/' + OLD + '/results/' + key, 'Unexpected old output')
        run['output'] = '/a/mm/' + NAME + '/results/' + key
        result['runs'][key] = run
    return result


def check_fit(out, key, run):
    receipt = js(out / 'refit_complete.json')
    args = js(out / 'effective_trainer_args.json')
    require(receipt['run'] == key and receipt['epochs'] == 7 and receipt['training_records'] == run['fit_n'], 'Invalid fitting receipt: ' + key)
    require(args['seed'] == 44 and args['num_train_epochs'] == 7 and args['learning_rate'] == run['learning_rate'] and args['save_strategy'] == 'no', 'Fitting settings mismatch: ' + key)
    model = out / 'tuned_model'
    require(not list(model.glob('checkpoint-*')), 'Unexpected intermediate checkpoints')
    require({'adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'}.issubset(receipt['model_sha256']), 'Required model hashes missing')
    for name, digest in receipt['model_sha256'].items():
        path = (model / name).resolve()
        require(path.is_relative_to(model.resolve()), 'Unsafe model receipt path')
        require(sha(path) == digest, 'Model checksum mismatch: ' + key + '/' + name)
    return receipt


def make_job(c, key, run, bootstrap):
    job = c['job_object'](key, run, 'inner_contract.json')
    jobname = NAME + '-' + key
    labels = {'experiment-batch': NAME, 'phase': 'inner', 'recovery-mode': 'eval' if key in EVAL else 'train'}
    job['metadata'] = {'name': jobname, 'namespace': 'b-t-bose', 'labels': labels}
    job['spec']['template']['metadata'] = {'labels': labels}
    job['spec']['backoffLimit'] = 0
    job['spec']['parallelism'] = 1
    job['spec']['completions'] = 1
    pod = job['spec']['template']['spec']
    require(len(pod['containers']) == 1, 'Expected one trainer')
    trainer = pod['containers'][0]
    overrides = {'RUN_SEED': '44', 'DATA_ROOT': run['data'], 'OUT_DIR': run['output'],
                 'XDG_CACHE_HOME': '/tmp/cache-' + jobname,
                 'TORCHINDUCTOR_CACHE_DIR': '/tmp/torchinductor-' + jobname}
    trainer['env'] = [e for e in trainer.get('env', []) if e['name'] not in overrides]
    trainer['env'] += [{'name': k, 'value': v} for k, v in overrides.items()]
    trainer['args'] = [bootstrap + '\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\npython -B -u /a/mm/' + NAME + '/recover.py run ' + key + '\n']
    return job


def run(key):
    here = Path(__file__).resolve().parent
    root = here.parent
    require(here.name == NAME, 'Wrong recovery directory')
    for name, digest in js(here / 'pins.json').items():
        require(sha(here / name) == digest, 'Recovery input drift: ' + name)
    c = controller(root)
    c['verify']()
    contract = js(here / 'inner_contract.json')
    run_spec = contract['runs'][key]
    output = Path(run_spec['output'])
    require(output == here / 'results' / key, 'Output scope mismatch')
    require(run_spec['seed'] == 44 and run_spec['phase'] == 'inner', 'Seed/phase mismatch')
    require(shutil.disk_usage(here).free >= 8 * 1024**3, 'Less than 8 GiB free; recovery stopped without deletion')
    with (here / (key + '.claim')).open('x') as f:
        f.write('eval-only' if key in EVAL else 'fresh-fit')
    if key in EVAL:
        check_fit(output, key, run_spec)
        modes = ('test',)
    else:
        require(key in TRAIN and not output.exists(), 'Existing output preserved')
        modes = ('train', 'test')
    wrapper = root / BASE / 'storage-safe-continuation-v1/worker_no_intermediate.py'
    for mode in modes:
        subprocess.run([sys.executable, '-B', '-u', str(wrapper), mode,
                        str(here / 'inner_contract.json'), key], check=True)
    check_fit(output, key, run_spec)
    rows = c['read_predictions'](key, run_spec)
    metrics = c['score'](rows)
    require(js(output / 'nested_runtime.json')['packages'] == js(root / BASE / 'environment_lock.json'), 'Environment mismatch')
    put(here / (key + '-verified.json'), {
        'seed': 44, 'run': key, 'mode': 'evaluation-only' if key in EVAL else 'fresh-training',
        'metrics': metrics, 'prediction_sha256': sha(output / 'test_predictions.csv'),
        'contract_sha256': sha(here / 'inner_contract.json'),
    })
    print('RECOVERY_INNER_VERIFIED', key, json.dumps(metrics), flush=True)


def launch():
    root = Path('/shared/ssd/home/b-t-bose/lmur')
    old = root / OLD
    dest = root / NAME
    require(not dest.exists(), 'Recovery directory exists; preserve it and inspect before retrying')
    require(shutil.disk_usage(root).free >= 16 * 1024**3, 'Need 16 GiB free to prepare recovery')
    status = subprocess.run(['kubectl','-n','b-t-bose','get','job',OLD,'-o','json'], check=True, capture_output=True, text=True)
    original_job = json.loads(status.stdout)
    require(any(x['type'] == 'Failed' and x['status'] == 'True' for x in original_job['status'].get('conditions', [])), 'Original Job is not confirmed failed')
    pods = subprocess.run(['kubectl','-n','b-t-bose','get','pods','-l','job-name='+OLD,'-o','json'], check=True, capture_output=True, text=True)
    require(not any(p['status'].get('phase') in ('Pending','Running','Unknown') for p in json.loads(pods.stdout)['items']), 'Original pods still active; no recovery submitted')
    c = controller(root); c['verify']()
    original = js(old / 'inner_contract.json')
    contract = plan(original)
    for key, run_spec in original['runs'].items():
        out = old / 'results' / key
        if key in contract['runs']:
            require(not (old / (key+'-verified.json')).exists() and not (out / 'test_complete.json').exists(), 'Recovery would duplicate a completed run: '+key)
            if key in EVAL:
                check_fit(out, key, run_spec)
            else:
                require(not (out / 'refit_complete.json').exists(), 'New fitting receipt found; inspect before retraining: '+key)
        else:
            receipt = js(old / (key+'-verified.json'))
            require(receipt['run'] == key and receipt['seed'] == 44, 'Completed-run identity mismatch')
            require(receipt['prediction_sha256'] == sha(out/'test_predictions.csv'), 'Completed predictions changed: '+key)
            require((out/'refit_complete.json').is_file() and (out/'test_complete.json').is_file(), 'Completed-run receipt missing')
    settings = js(root / BASE / 'settings.json')
    jobs = [make_job(c, key, run_spec, settings['bootstrap']) for key, run_spec in contract['runs'].items()]
    payload = {'apiVersion': 'v1', 'kind': 'List', 'items': jobs}
    subprocess.run(['kubectl','-n','b-t-bose','create','--dry-run=server','-f','-'], input=json.dumps(payload), text=True, check=True)
    dest.mkdir(); (dest/'results').mkdir()
    put(dest/'inner_contract.json', contract)
    shutil.copyfile(Path(__file__).resolve(), dest/'recover.py')
    for key in EVAL:
        src = old/'results'/key; dst = dest/'results'/key; dst.mkdir()
        shutil.copytree(src/'tuned_model', dst/'tuned_model', symlinks=False)
        for file in (src/'tuned_model').rglob('*'):
            if file.is_file():
                require(sha(file) == sha(dst/'tuned_model'/file.relative_to(src/'tuned_model')), 'Copied adapter changed')
        for name in ('refit_complete.json','effective_trainer_args.json','nested_runtime.json','run_metadata.json'):
            shutil.copyfile(src/name, dst/name)
        check_fit(dst, key, contract['runs'][key])
        put(dst/'recovery_provenance.json', {'mode':'evaluation-only','original_output':str(src),'original_fit_receipt_sha256':sha(src/'refit_complete.json'),'old_test_started_preserved':True})
    put(dest/'bindings.json', {key:{'original_output':original['runs'][key]['output'],'recovery_output':run_spec['output'],'mode':'evaluation-only' if key in EVAL else 'fresh-training'} for key,run_spec in contract['runs'].items()})
    put(dest/'jobs.json', payload)
    put(dest/'pins.json', {name:sha(dest/name) for name in ('recover.py','inner_contract.json','bindings.json')})
    subprocess.run(['kubectl','-n','b-t-bose','create','-f',str(dest/'jobs.json')], check=True)
    print('SUBMITTED: two evaluation-only Jobs and seven fresh-training Jobs. Originals preserved; no outer stage or deletion.')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        launch()
    else:
        require(len(sys.argv) == 3 and sys.argv[1] == 'run', 'Unexpected arguments')
        run(sys.argv[2])
