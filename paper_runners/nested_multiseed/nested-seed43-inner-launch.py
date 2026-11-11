import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

NAME = 'lemur-nested-qwen-s43-inner-p12-v1'
RUNTIME = r'''import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

here = Path(__file__).resolve().parent
base = here.parent / 'lemur-nested-qwen-full-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = json.loads((here / 'pins.json').read_text())
for name, digest in pins.items():
    if sha(here / name) != digest:
        raise RuntimeError('Launch input changed: ' + name)
source = (base / 'control.py').read_text()
anchor = 'args["seed"] == 42'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected verifier source')
ns = {'__name__': 'seed43_verifier', '__file__': str(base / 'control.py')}
exec(compile(source.replace(anchor, 'args["seed"] == run["seed"]'), str(base / 'control.py'), 'exec'), ns)
ns['verify']()
contract = json.loads((here / 'inner_contract.json').read_text())
keys = sorted(contract['runs'])
if len(keys) != 24 or contract['epochs'] != 7:
    raise RuntimeError('Unexpected fitting budget')
summary = {}
index = int(os.environ['JOB_COMPLETION_INDEX'])
if not 0 <= index < len(keys):
    raise RuntimeError('Invalid completion index')
for key in [keys[index]]:
    run = contract['runs'][key]
    expected_output = here / 'results' / key
    if run['phase'] != 'inner' or run['seed'] != 43 or Path(run['output']) != expected_output:
        raise RuntimeError('Seed/phase/output scope mismatch')
    if expected_output.exists():
        raise RuntimeError('Existing output preserved: ' + str(expected_output))
    completed = sum((here / (k + '-verified.json')).is_file() for k in keys)
    if shutil.disk_usage(here).free < (8 + 0.5 * (len(keys) - completed)) * 1024**3:
        raise RuntimeError('Storage guard: stopped without deleting results')
    # Exclusive claim prevents duplicate pods for one index from training twice.
    with (here / (key + '.claim')).open('x') as f:
        f.write(str(index))
    env = dict(os.environ, RUN_SEED='43', DATA_ROOT=run['data'], OUT_DIR=run['output'])
    wrapper = base / 'storage-safe-continuation-v1/worker_no_intermediate.py'
    print('START_INNER', key, 'seed=43', flush=True)
    for mode in ('train', 'test'):
        subprocess.run([sys.executable, '-B', '-u', str(wrapper), mode,
                        str(here / 'inner_contract.json'), key], env=env, check=True)
    rows = ns['read_predictions'](key, run)
    args = json.loads((expected_output / 'effective_trainer_args.json').read_text())
    if args['save_strategy'] != 'no':
        raise RuntimeError('Intermediate checkpoint policy mismatch')
    if list((expected_output / 'tuned_model').glob('checkpoint-*')):
        raise RuntimeError('Unexpected intermediate checkpoints')
    summary[key] = ns['score'](rows)
    with (here / (key + '-verified.json')).open('x') as f:
        json.dump({'seed': 43, 'run': key, 'metrics': summary[key],
                   'prediction_sha256': sha(expected_output / 'test_predictions.csv'),
                   'contract_sha256': sha(here / 'inner_contract.json')}, f, indent=2)
    print('INNER_VERIFIED', key, json.dumps(summary[key]), flush=True)
print('SEED43_INNER_INDEX_COMPLETE', index, key, 'no outer fitting or cleanup', flush=True)
'''

def configure(contract, job, bootstrap):
    c = copy.deepcopy(contract)
    if len(c['runs']) != 24 or c['epochs'] != 7:
        raise RuntimeError('Unexpected contract')
    for key, run in c['runs'].items():
        if run['phase'] != 'inner' or run['seed'] != 42 or run['settings_seed'] != 42:
            raise RuntimeError('Unexpected original seed/phase')
        run['seed'] = 43
        run['output'] = '/a/mm/' + NAME + '/results/' + key
    j = copy.deepcopy(job)
    j['metadata'] = {'name': NAME, 'namespace': 'b-t-bose', 'labels': {'experiment-batch': NAME, 'phase': 'inner'}}
    j['spec']['backoffLimit'] = 0
    j['spec']['activeDeadlineSeconds'] = 604800
    j['spec']['parallelism'] = 12
    j['spec']['completions'] = 24
    j['spec']['completionMode'] = 'Indexed'
    j['spec']['template']['metadata'] = {'labels': j['metadata']['labels']}
    pod = j['spec']['template']['spec']
    if len(pod['containers']) != 1:
        raise RuntimeError('Expected one trainer container')
    trainer = pod['containers'][0]
    values = {'RUN_SEED': '43', 'XDG_CACHE_HOME': '/tmp/cache-' + NAME,
              'TORCHINDUCTOR_CACHE_DIR': '/tmp/torchinductor-' + NAME}
    # Rebuild only these explicit environment entries, preserving the others.
    trainer['env'] = [e for e in job['spec']['template']['spec']['containers'][0].get('env', []) if e['name'] not in values]
    trainer['env'] += [{'name': k, 'value': v} for k,v in values.items()]
    trainer['args'] = [bootstrap + '\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\npython -B -u /a/mm/' + NAME + '/run_inner.py\n']
    return c, j


def main():
    root = Path('/shared/ssd/home/b-t-bose/lmur')
    base = root / 'lemur-nested-qwen-full-v1'
    dest = root / NAME
    if (root / 'lemur-nested-qwen-s43-inner-v1').exists():
        raise SystemExit('Previous sequential launch directory exists; inspect it before switching launchers')
    if dest.exists():
        raise SystemExit('Existing launch directory preserved: ' + str(dest))
    expected = {
        'control.py': 'c6d9733240e8bd4930abf03a25476102ee085c26ff3d434a0aba9d3bc03908dc',
        'worker.py': 'fe6309053b6b88f46f744515c8238031705751159a093659da789fd6bc745dad',
        'storage-safe-continuation-v1/worker_no_intermediate.py': 'cea1d1a216ef90840bbaeee484a6253aebc93387f7513511c2d64493aa00980a',
    }
    for name,digest in expected.items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Original source changed: '+name)
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise SystemExit('Need at least 20 GiB free for this seed; nothing created')
    spec = importlib.util.spec_from_file_location('original_controller', base/'control.py')
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    controller.verify()
    old = json.loads((base/'inner_contract.json').read_text())
    original_job = controller.job_object('inner-f0-c00', old['runs']['inner-f0-c00'], 'inner_contract.json')
    settings = json.loads((base/'settings.json').read_text())
    contract, job = configure(old, original_job, settings['bootstrap'])
    compile(RUNTIME, 'run_inner.py', 'exec')
    subprocess.run(['kubectl','-n','b-t-bose','create','--dry-run=server','-f','-'], input=json.dumps(job), text=True, check=True)
    dest.mkdir()
    (dest/'inner_contract.json').write_text(json.dumps(contract, indent=2))
    (dest/'run_inner.py').write_text(RUNTIME)
    (dest/'job.json').write_text(json.dumps(job, indent=2))
    pins = {name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ['inner_contract.json','run_inner.py']}
    pins.update({'../lemur-nested-qwen-full-v1/'+name:digest for name,digest in expected.items()})
    (dest/'pins.json').write_text(json.dumps(pins,indent=2))
    subprocess.run(['kubectl','-n','b-t-bose','create','-f',str(dest/'job.json')],check=True)
    print('SUBMITTED', NAME, ': 24 indexed inner fits, up to 12 concurrent; no cleanup; no outer jobs')

if __name__ == '__main__':
    main()
