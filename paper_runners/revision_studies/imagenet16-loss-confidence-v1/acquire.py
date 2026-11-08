from pathlib import Path
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
NAME = 'imagenet16-loss-confidence-v1'
PILOT = 'nb201-v1-imagenet16-full'
BASE = 'nb201-v1-imagenet16-full'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def js(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def put(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def load_module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def verify(root):
    c = js(root / NAME / 'contract.json')
    for (relative, expected) in c['files_sha256'].items():
        require(sha(root / NAME / relative) == expected, 'Experiment source changed: ' + relative)
    for (relative, expected) in c['sources_sha256'].items():
        require(sha(root / relative) == expected, 'Original source changed: ' + relative)
    return c

def inputs(root):
    cohort = js(root / BASE / 'cohort.json')
    members = {int(r['arch_index']): r for r in cohort['architectures']}
    with (root / BASE / 'targets.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    targets = {int(r['architecture_id']): r for r in rows}
    require(len(rows) == len(targets) == len(members) == 1100, 'Incomplete or duplicate frozen inputs')
    require(set(members) == set(targets), 'Frozen identity mismatch')
    return (cohort, members, targets)

def instrument(base):

    def measure(model, batch, analyzer_class, device):
        import torch
        seen = []

        class Observed(analyzer_class):

            def taylor_importance(self, batches, loss_fn, num_batches=1):

                def observed_loss(output, labels):
                    loss = loss_fn(output, labels)
                    with torch.no_grad():
                        logits = output[1].detach().double()
                        require(logits.ndim == 2 and logits.shape[1] == 120, 'Expected ImageNet16-120 logits')
                        require(bool(torch.isfinite(logits).all()), 'Nonfinite logits')
                        logp = logits.log_softmax(dim=1)
                        probs = logp.exp()
                        seen.append({'probe_loss': float(loss.detach()), 'true_confidence': float(probs.gather(1, labels[:, None]).mean()), 'max_confidence': float(probs.max(dim=1).values.mean()), 'entropy': float(-(probs * logp).sum(dim=1).mean()), 'probe_examples': int(labels.numel())})
                    return loss
                return super().taylor_importance(batches, observed_loss, num_batches=num_batches)
        result = base(model, batch, Observed, device)
        require(len(seen) == 1 and seen[0]['probe_examples'] == 256, 'Unexpected probe forwards/batch size')
        return dict(result, **seen[0])
    return measure

def validate(root, output, index):
    directory = output / 'candidates' / f'{index:05d}'
    receipt = js(directory / 'complete.json')
    require(receipt['status'] == 'complete' and receipt['architecture_id'] == index and (receipt['contract_sha256'] == sha(root / NAME / 'contract.json')), 'Candidate receipt mismatch')
    for (relative, expected) in receipt['files_sha256'].items():
        require(sha(directory / relative) == expected, 'Candidate output changed: ' + relative)
    (_, members, targets) = inputs(root)
    record = js(directory / 'record.json')
    check_record(record, index, members, targets)
    return record

def collect(root, output):
    c = verify(root)
    for (k, ids) in enumerate(c['shards']):
        report = js(output / 'shards' / f's{k}' / 'report.json')
        require(report['architecture_ids'] == ids and report['complete'] == len(ids) and (not report['failures']), 'Incomplete shard; no partial-cohort fitting')
    (cohort, members, targets) = inputs(root)
    records = [validate(root, output, i) for i in c['architectures']]
    require(len(records) == len({r['record_id'] for r in records}) == 1100, 'Incomplete/duplicate collection')
    counts = {split: sum((r['split'] == split for r in records)) for split in ('train', 'val', 'test')}
    require(counts == {'train': 772, 'val': 53, 'test': 275}, 'Frozen split changed')
    families = {split: {r['family'] for r in records if r['split'] == split} for split in counts}
    require(not families['train'] & families['val'] and (not families['train'] & families['test']) and (not families['val'] & families['test']), 'Family leakage')
    with (output / 'snapshot.jsonl').open('x') as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False) + '\n')
    require([json.loads(l) for l in (output / 'snapshot.jsonl').read_text().splitlines()] == records, 'Snapshot readback mismatch')
    provenance = {str(i): sha(output / 'candidates' / f'{i:05d}' / 'complete.json') for i in c['architectures']}
    put(output / 'provenance.json', provenance)
    put(output / 'collection_complete.json', {'status': 'complete', 'n': 1100, 'counts': counts, 'dataset': 'ImageNet16-120', 'snapshot_sha256': sha(output / 'snapshot.jsonl'), 'provenance_sha256': sha(output / 'provenance.json'), 'contract_sha256': sha(root / NAME / 'contract.json'), 'target': 'Mean epoch200 x-valid accuracy over fixed benchmark seeds777/888', 'input_policy': 'All early signals from the same fresh local candidate run'})
    sources = [output / 'snapshot.jsonl', output / 'provenance.json', output / 'collection_complete.json', root / NAME / 'contract.json', root / BASE / 'cohort.json', root / BASE / 'targets.csv', root / BASE / 'contract.json']
    put(output / 'histgb_contract.json', {'histgb_worker_sha256': sha(root / NAME / 'histgb_worker.py'), 'packages': c['packages'], 'sources_sha256': {p.relative_to(root).as_posix(): sha(p) for p in sources}})
    with (output / 'histgb_ready').open('x'):
        pass
    print('IMAGENET16_COLLECTION_COMPLETE', flush=True)

def shard(root, data, output, k):
    c = verify(root)
    ids = c['shards'][k]
    state = output / 'shards' / f's{k}'
    state.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for (position, index) in enumerate(ids, 1):
        space = os.statvfs(output)
        require(space.f_bavail * space.f_frsize >= 1024 ** 3 and space.f_favail >= 10000, 'STORAGE_GUARD_STOP: preserve completed outputs')
        try:
            with (state / f'arch-{index}.log').open('xb') as log:
                result = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'one', '--root', str(root), '--data', str(data), '--output', str(output), '--index', str(index)], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            require(result.returncode == 0, f'Candidate exited {result.returncode}')
            validate(root, output, index)
            row = {'architecture_id': index, 'status': 'complete'}
        except Exception as exc:
            row = {'architecture_id': index, 'status': 'failed', 'error': str(exc)}
        outcomes.append(row)
        with (state / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print('IMAGENET16_SHARD', k, f'{position}/{len(ids)}', index, row['status'], flush=True)
    failures = [r for r in outcomes if r['status'] != 'complete']
    put(state / 'report.json', {'shard': k, 'architecture_ids': ids, 'complete': len(ids) - len(failures), 'failures': failures})
    with (state / 'ready').open('x'):
        pass
    require(not failures, 'Candidate failures preserved; no silent exclusions')
    if all(((output / 'shards' / f's{j}' / 'ready').exists() for j in range(len(c['shards'])))):
        try:
            (output / 'collection-lock').mkdir()
        except FileExistsError:
            return
        try:
            collect(root, output)
        except Exception:
            put(output / 'collection_failed.json', {'traceback': traceback.format_exc()})
            raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['one', 'shard'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int)
    p.add_argument('--shard', type=int)
    a = p.parse_args()
    if a.action == 'one':
        one(a.root, a.data, a.output, a.index)
    else:
        shard(a.root, a.data, a.output, a.shard)

def check_record(r, index, members, targets):
    (member, target) = (members[index], targets[index])
    require(r['record_id'] == member['id'] == target['record_id'] and r['architecture_id'] == index and (r['architecture_string'] == member['arch_str']) and (r['family'] == member['arch_family']) and (r['dataset'] == 'ImageNet16-120') and (r['split'] == target['split']) and (r['candidate_seed'] == 777) and (r['target_kind'] == 'final_validation_accuracy') and (r['target_epoch'] == 200) and (r['benchmark_target_seeds'] == [777, 888]) and (r['target'] == float(target['target_final_validation_accuracy'])), 'Frozen record/target mismatch')
    require([e['epoch'] for e in r['epochs']] == [1, 2, 3], 'Incomplete early epochs')
    for e in r['epochs']:
        require(e['architecture_id'] == index and e['candidate_seed'] == 777, 'Epoch identity mismatch')
        require(all((math.isfinite(e[k]) for k in ('validation_accuracy', 'taylor', 'train_loss', 'probe_loss', 'true_confidence', 'max_confidence', 'entropy'))), 'Nonfinite signal')
        require(0 <= e['validation_accuracy'] <= 100, 'Invalid accuracy')
        require(e['taylor'] >= 0 and e['train_loss'] >= 0 and (e['probe_loss'] >= 0), 'Invalid Taylor/loss')
        require(0 <= e['true_confidence'] <= e['max_confidence'] <= 1, 'Invalid confidence')
        require(0 <= e['entropy'] <= math.log(120) + 1e-12, 'Invalid entropy')

def one(root, data, output, index):
    verify(root)
    (cohort, members, targets) = inputs(root)
    require(index in members, 'Architecture outside frozen cohort')
    directory = output / 'candidates' / f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'raw'
    raw.mkdir()
    engine = load_module(root / BASE / 'engine.py', 'original_imagenet16_engine')
    engine.NAME = BASE
    engine.ARCHS = {index: members[index]['arch_str']}
    engine.measure_taylor = instrument(engine.measure_taylor)
    start = time.monotonic()
    engine.worker(root, data, raw)
    done = js(raw / 'complete.json')
    require(done['status'] == 'complete' and done['architectures'] == [index] and (done['epochs'] == [1, 2, 3]), 'Engine did not complete')
    for (relative, expected) in done['files_sha256'].items():
        require(sha(raw / relative) == expected, 'Raw acquisition changed')
    acquired = js(raw / f'arch-{index}-complete.json')
    (member, target) = (members[index], targets[index])
    require(acquired['architecture_id'] == index and acquired['architecture_string'] == member['arch_str'] and (acquired['dataset'] == 'ImageNet16-120') and (acquired['candidate_seed'] == 777), 'Acquired identity mismatch')
    require([e['epoch'] for e in acquired['epochs']] == [1, 2, 3], 'Acquired epochs incomplete')
    epochs = []
    for (epoch, original_epoch) in zip((1, 2, 3), acquired['epochs']):
        row = js(raw / f'arch-{index}-epoch-{epoch}.json')
        require(all((row[key] == value for (key, value) in original_epoch.items())), 'Epoch file and acquisition record differ')
        epochs.append(row)
    record = {'record_id': member['id'], 'architecture_id': index, 'architecture_string': member['arch_str'], 'family': member['arch_family'], 'dataset': 'ImageNet16-120', 'split': target['split'], 'candidate_seed': 777, 'target_kind': 'final_validation_accuracy', 'target_epoch': 200, 'benchmark_target_seeds': [777, 888], 'target': float(target['target_final_validation_accuracy']), 'early_accuracy_policy': 'Local x-valid accuracy from the same fresh candidate run as loss/confidence and Taylor', 'epochs': epochs, 'elapsed_seconds': time.monotonic() - start}
    check_record(record, index, members, targets)
    put(directory / 'record.json', record)
    verify(root)
    put(directory / 'complete.json', {'status': 'complete', 'architecture_id': index, 'contract_sha256': sha(root / NAME / 'contract.json'), 'files_sha256': {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*') if p.is_file()}})
    validate(root, output, index)
    print('IMAGENET16_CANDIDATE_COMPLETE', index, flush=True)
if __name__ == '__main__':
    main()
