"""Fit no-Taylor classical baselines from a frozen export, reusing saved Qwen outputs.

The original fitting implementation is loaded from its checksum-pinned archive.
Only its feature list and input matrix are projected; grids and fitting logic
remain unchanged. This mode trains CPU regressors, never Qwen or candidate nets.
"""
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile

EXPECTED = 'e3629395b79631261ea11675a41c46d5ddc6a2b5b76c87015dab955d2249bd1e'
PUBLIC_ARCHIVE = '3a420140ff64a851bd64fe916c37651ff7016bc93b87fad41ce9309f05448ebd'
FEATURES = ['accuracy_1', 'accuracy_2', 'accuracy_3', 'log_params', 'depth']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    if str(value).strip().lower() in ('', 'none', 'nan', 'na', 'null'):
        return None
    value = float(value)
    require(math.isfinite(value), 'Nonfinite feature')
    return value


def check_versions(expected):
    for name, version in expected.items():
        require(importlib.metadata.version(name) == version,
                'Version mismatch: ' + name + '; install requirements-classical-no-taylor.txt in a separate environment')


def load(archive):
    archive_hash = sha(archive.read_bytes())
    require(archive_hash in {EXPECTED, PUBLIC_ARCHIVE}, 'Archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        prefix = 'files/classical-eight-features-v1/'
        complete = json.loads(z.read(prefix + 'complete.json'))
        require(complete['status'] == 'complete', 'Incomplete source experiment')

        def checked(name):
            content = z.read(prefix + name)
            require(sha(content) == complete['files_sha256'][name], 'Checksum mismatch: ' + name)
            return content

        old = json.loads(checked('contract.json'))
        records = checked('records.csv')
        rows = list(csv.DictReader(io.StringIO(records.decode())))
        source_bytes = z.read('files/run_classical_baselines_v1.py')
        require(sha(source_bytes) == old['script_sha256'], 'Original fitting script changed')
        require(len({r['record_id'] for r in rows}) == len(rows), 'Duplicate record IDs')
        data, projected, splits, architectures = {}, [], {}, {}
        for split, count in (('train', 978), ('val', 58), ('test', 115)):
            part = sorted((r for r in rows if r['split'] == split), key=lambda r: int(r['position']))
            require(len(part) == count, 'Wrong split count: ' + split)
            require([int(r['position']) for r in part] == list(range(count)), 'Wrong record order: ' + split)
            x = [[number(r[f]) for f in FEATURES] for r in part]
            y = [float(r['target']) for r in part]
            require(all(math.isfinite(v) and 0 <= v <= 100 for v in y), 'Invalid target')
            data[split] = dict(x=x, y=y, ids=[r['record_id'] for r in part], datasets=[r['dataset'] for r in part])
            architectures[split] = {r['architecture_id'] for r in part}
            splits[split] = dict(records=count, missing={f: sum(values[i] is None for values in x) for i, f in enumerate(FEATURES)})
            projected.extend({k: v for k, v in r.items() if k not in ('taylor_1', 'taylor_2', 'taylor_3')} for r in part)
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            require(not architectures[a] & architectures[b], f'Architecture overlap: {a}/{b}')
        reference, reference_hashes = {}, {}
        for seed in range(42, 48):
            name = f'qwen_reference-s{seed}-predictions.csv'
            content = checked(name)
            predictions = list(csv.DictReader(io.StringIO(content.decode())))
            require([r['record_id'] for r in predictions] == data['test']['ids'], f'Qwen ID/order mismatch: seed {seed}')
            require([float(r['target']) for r in predictions] == data['test']['y'], f'Qwen target mismatch: seed {seed}')
            require(all(int(r['seed']) == seed for r in predictions), 'Qwen seed mismatch')
            values = [float(r['prediction']) for r in predictions]
            require(all(math.isfinite(v) for v in values), 'Invalid prediction')
            reference[seed] = values
            reference_hashes[name] = sha(content)
    receipt = dict(
        features=FEATURES, seeds=list(range(42, 48)), splits=splits,
        target_policy=old['target_policy'], preprocessing=old['preprocessing'], selection=old['selection'],
        comparison_note='Saved Qwen+Taylor versus classical predictors without Taylor. Classical inputs retain three accuracies, log_params and depth. Qwen retains its original metadata and Taylor inputs. This compares complete approaches, not Taylor in isolation.',
        source_archive_sha256=archive_hash, source_records_sha256=sha(records),
        qwen_prediction_sha256=reference_hashes, gpu_required=False, qwen_retrained=False,
        adapter_sha256=sha(Path(__file__).read_bytes()),
    )
    return data, projected, receipt, reference, source_bytes, old['packages']


def run(archive, out):
    archive, out = Path(archive), Path(out)
    require(not out.exists(), 'Existing output preserved: ' + str(out))
    data, projected, receipt, reference, source_bytes, packages = load(archive)
    check_versions(packages)
    # A single CPU thread avoids unexpectedly oversubscribing workstation resources.
    from threadpoolctl import threadpool_limits
    with tempfile.TemporaryDirectory(prefix='classical-notaylor-source-') as temp:
        source = Path(temp) / 'run_classical_baselines_v1.py'
        source.write_bytes(source_bytes)
        spec = importlib.util.spec_from_file_location('archived_classical_fitter', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.FEATURES = FEATURES
        with threadpool_limits(limits=1):
            module.run_fits(data, projected, receipt, reference, out)
    require(sha(archive.read_bytes()) == receipt['source_archive_sha256'], 'Input archive changed during fitting')
