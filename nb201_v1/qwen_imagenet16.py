"""Isolated ImageNet16-120 continuation of the frozen CIFAR100 Qwen protocol.
Build only; never submit. Run python -B nb201_v1/qwen_imagenet16.py.
"""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
NAME = 'nb201-v1-qwen-imagenet16-v1'
POD = '/a/mm/' + NAME
SNAPSHOT_SHA256 = '542712257513aeb51ed970e902b0669270d7293c1c6b391474968307ddbe1a79'
spec = importlib.util.spec_from_file_location('frozen_qwen_launch', ROOT / 'nb201_v1/qwen_launch.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sha, put = prior.sha, prior.put


def frozen_inputs():
    snapshot = ROOT / 'evidence/nb201-imagenet16-collected-v1/snapshot.jsonl'
    assert sha(snapshot) == SNAPSHOT_SHA256, 'Snapshot checksum mismatch'
    folder = ROOT / 'nb201_v1/imagenet16-v2'
    cohort = json.loads((folder / 'cohort.json').read_text())
    with (folder / 'targets.csv').open(newline='') as f:
        targets = list(csv.DictReader(f))
    records = [json.loads(line) for line in snapshot.read_text().splitlines()]
    byid = {r['architecture_id']: r for r in records}
    architectures = {r['arch_index']: r for r in cohort['architectures']}
    target_map = {int(r['architecture_id']): r for r in targets}
    splits = {i: split for split, ids in cohort['splits'].items() for i in ids}
    assert len(records) == len(byid) == len(architectures) == len(targets) == len(target_map) == len(splits) == 1100
    assert sum(map(len, cohort['splits'].values())) == 1100
    assert set(byid) == set(architectures) == set(target_map) == set(splits)
    assert cohort['dataset'] == 'ImageNet16-120'
    assert cohort['counts'] == dict(train=772, val=53, test=275)
    families = {}
    for i, r in byid.items():
        a, t = architectures[i], target_map[i]
        assert r['record_id'] == a['id'] == t['record_id']
        assert r['architecture_string'] == a['arch_str'] and r['arch_family'] == a['arch_family']
        assert r['dataset'] == t['dataset'] == 'ImageNet16-120'
        assert r['split'] == t['split'] == splits[i]
        assert r['target_kind'] == 'final_validation_accuracy'
        assert r['target_epoch'] == int(t['target_epoch']) == 200
        assert r['benchmark_target_seeds'] == [777, 888] and r['candidate_seed'] == 777
        assert r['target_final_validation_accuracy'] == float(t['target_final_validation_accuracy'])
        assert r['target_final_validation_accuracy'] == (float(t['benchmark_seed_777_accuracy']) + float(t['benchmark_seed_888_accuracy'])) / 2
        assert families.setdefault(r['arch_family'], r['split']) == r['split']
        for e in (1, 2, 3):
            assert math.isfinite(r[f'accuracy_epoch_{e}']) and 0 <= r[f'accuracy_epoch_{e}'] <= 100
            assert math.isfinite(r[f'taylor_epoch_{e}'])
    for split, count in cohort['counts'].items():
        assert len(cohort['splits'][split]) == count
    return byid, cohort


def example(record, taylor):
    result = prior.example(record, 'cifar100', taylor)
    result['messages'][1]['content'] = result['messages'][1]['content'].replace('dataset: cifar100', 'dataset: ImageNet16-120')
    return result


def worker_source():
    text = prior.worker_source()
    assert text.count("run['dataset'] != 'cifar100'") == 1
    return text.replace("run['dataset'] != 'cifar100'", "run['dataset'] != 'imagenet16'")


CONTROL = prior.CONTROL.replace("PILOT = 'cifar10-taylor-s42'", "PILOT = 'imagenet16-taylor-s42'")
CONTROL = CONTROL.replace('import csv\n', 'import csv\nimport os\n', 1)
CONTROL = CONTROL.replace("    base = host(contract['base'])", """    work = HERE / 'work'
    assert work.is_dir() and os.access(work, os.W_OK | os.X_OK), 'Dedicated work directory absent/not writable'
    probe = work / '.control-write-probe'
    with probe.open('x') as f:
        f.write('writable')
    probe.unlink()
    base = host(contract['base'])""")
# Reject all preexisting pending outputs/jobs before even starting dry runs.
CONTROL = CONTROL.replace("        subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server'", "    for key in keys:\n        subprocess.run(['kubectl', '-n', 'b-t-bose', 'create', '--dry-run=server'")


def build(dest):
    assert not dest.exists(), 'Existing bundle preserved'
    byid, cohort = frozen_inputs()  # all validation precedes output creation
    old = ROOT / 'evidence/nb201-v1-qwen-matched-v1'
    previous = json.loads((old / 'contract.json').read_text())
    selected = previous['runs']['cifar100-taylor-s42']
    template = json.loads((old / 'jobs/cifar100-taylor-s42.json').read_text())
    # The completed export is the authority for the frozen worker and settings.
    completed_archive = ROOT / 'evidence/nb201-qwen-predictions-v1.tar.gz'
    with tarfile.open(completed_archive) as tar:
        completed = json.load(tar.extractfile('contract.json'))
        assert completed == previous
        assert tar.extractfile('worker.py').read() == (old / 'worker.py').read_bytes()
        assert tar.extractfile('worker.py').read().decode() == prior.worker_source()
        done = json.load(tar.extractfile('results/cifar100-taylor-s42/test_complete.json'))
        assert done['n_samples'] == done['n_valid'] == 275 and done['parse_failures'] == 0
    contract = dict(epochs=7, base=previous['base'], counts=previous['counts'],
        target_precision=previous['target_precision'], runs={}, source_sha256={},
        scope='Fixed seven-epoch refit on train+val; no test selection; frozen CIFAR100 protocol',
        provenance_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / 'nb201_v1/qwen_launch.py', ROOT / 'nb201_v1/imagenet16-v2/cohort.json',
            ROOT / 'nb201_v1/imagenet16-v2/targets.csv', completed_archive,
            old / 'jobs/cifar100-taylor-s42.json')},
        image_caveat='Completed local CIFAR bundle specifies a mutable image tag, not an image digest; no recorded imageID available. Dependency install also retains original unpinned packages.')
    for p in (ROOT / 'evidence/nb201-qwen-source-v1').glob('*.py'):
        key = '/a/mm/nn-gpt/ab/gpt/act/acc_predictor/' + p.name
        assert sha(p) == previous['source_sha256'][key]
        contract['source_sha256'][key] = sha(p)
    assert len(contract['source_sha256']) == 11
    contract['source_sha256']['/a/mm/nb201-v1-imagenet16-full/snapshot.jsonl'] = SNAPSHOT_SHA256
    worker = worker_source()
    compile(worker, 'worker.py', 'exec')
    compile(CONTROL, 'control.py', 'exec')
    dest.mkdir(parents=True)
    put(dest / 'work/README.txt', 'Dedicated writable working directory; required before submission.\n')
    put(dest / 'worker.py', worker)
    put(dest / 'control.py', CONTROL)
    contract['source_sha256'][POD + '/worker.py'] = sha(dest / 'worker.py')
    for cond in ('no-layer', 'taylor'):
        local = dest / 'prompts/imagenet16' / cond
        for split in ('train', 'val', 'test'):
            rows = [example(byid[i], cond == 'taylor') for i in cohort['splits'][split]]
            put(local / f'{split}_llm_dataset.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
            if split == 'test':
                put(local / 'test_registry.json', [dict(record_id=byid[i]['record_id'], architecture_id=i,
                    family=byid[i]['arch_family'], dataset='ImageNet16-120',
                    target=round(byid[i]['target_final_validation_accuracy'], 2),
                    raw_target=byid[i]['target_final_validation_accuracy']) for i in cohort['splits'][split]])
        for seed in prior.SEEDS:
            key = f'imagenet16-{cond}-s{seed}'
            name = NAME + '-' + key
            run = copy.deepcopy(selected)
            run.update(seed=seed, dataset='imagenet16', condition=cond,
                data=f'{POD}/prompts/imagenet16/{cond}', output=f'{POD}/results/{key}',
                input_sha256={p.name: sha(p) for p in local.iterdir()})
            contract['runs'][key] = run
            job = copy.deepcopy(template)
            labels = {'experiment-batch': NAME, 'dataset': 'imagenet16', 'condition': cond, 'run-seed': str(seed)}
            job['metadata'] = dict(name=name, namespace='b-t-bose', labels=labels)
            job['spec']['template']['metadata'] = dict(labels=labels)
            c = job['spec']['template']['spec']['containers'][0]
            c['workingDir'] = POD + '/work'
            env = {r['name']: r['value'] for r in c['env']}
            env.update(DATA_ROOT=run['data'], OUT_DIR=run['output'], RUN_SEED=str(seed), RUN_EPOCHS='7',
                LAYER_ON=str(cond == 'taylor').lower(), THIRD_ON='true',
                XDG_CACHE_HOME='/tmp/cache-' + name, TORCHINDUCTOR_CACHE_DIR='/tmp/torchinductor-' + name)
            c['env'] = [dict(name=k, value=v) for k, v in env.items()]
            install = c['args'][0].split('export PYTHONPATH=')[0].replace('cd /a/mm\n', f'cd {POD}/work\n')
            c['args'] = [install + f'export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\nexport OUT_DIR={run["output"]}\npython -u {POD}/worker.py train {POD}/contract.json {key}\npython -u {POD}/worker.py test {POD}/contract.json {key}\n']
            assert 'nb201-v1-qwen-matched-v1' not in json.dumps(job)
            put(dest / 'jobs' / (key + '.json'), job)
    put(dest / 'contract.json', contract)
    put(dest / 'manifest.json', {p.relative_to(dest).as_posix(): sha(p) for p in dest.rglob('*') if p.is_file()})
    print('PREPARED', dest, '12 Jobs; NONE submitted')


if __name__ == '__main__':
    dest = ROOT / 'evidence' / NAME
    archive = dest.with_suffix('.tar.gz')
    assert not archive.exists(), 'Existing archive preserved'
    build(dest)
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(dest, arcname=NAME)
    print('ARCHIVE', archive, 'SHA256', sha(archive))
