"""Reproduce the recovered initialization-proxy analysis from its frozen inputs.

The original Python body is preserved separately. This wrapper changes only
its ROOT assignment to an isolated extraction directory, verifies the recorded
results, and writes new outputs after those checks pass. No network training.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE_SHA = '750fb543431de939c527c5936ee04715ce3a45d14dcc347e3e5205966f195be2'
SOURCE_SHA = 'f1add7a6186c1575eb6653239c78b7ba47ed0e659d2da2ba0415d51373848f23'
ROOT_LINE = 'ROOT = Path("/shared/ssd/home/b-t-bose/lmur")'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_values(text, models, metrics):
    means, pairs = {}, {}
    contrast = None
    for line in text.splitlines():
        if line.startswith('Candidate: '):
            candidate, reference = line.removeprefix('Candidate: ').split('; reference: ')
            contrast = (candidate, reference)
        cells = [v.strip() for v in line.split('|')]
        if len(cells) == 6 and cells[0] in models:
            means[cells[0]] = [float(v) for v in cells[1:]]
        elif contrast and len(cells) == 4 and cells[0] in metrics:
            endpoints = [float(v.strip()) for v in cells[2].strip('[]').split(',')]
            pairs[contrast + (cells[0],)] = (float(cells[1]), *endpoints, int(cells[3]))
    require(len(means) == 5 and len(pairs) == 20, 'Incomplete historical reference report')
    return means, pairs


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=REPO / 'evidence/init-proxy-combination-inputs-v1.tar.gz')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing paths are refused')
    args = parser.parse_args()
    require(not args.out.exists(), 'Existing output preserved: ' + str(args.out))
    original = (HERE / 'original_analysis.py').read_bytes()
    require(hashlib.sha256(original).hexdigest() == SOURCE_SHA, 'Recovered source hash changed')
    archive = args.archive.read_bytes()
    require(hashlib.sha256(archive).hexdigest() == ARCHIVE_SHA, 'Original archive hash changed')
    reference_path = REPO / 'evidence/thesis-reports/initialization_proxy_combinations.log'
    reference = reference_path.read_text(encoding='utf-8')
    expected_hashes = dict(re.findall(r'(?m)^([^\n]+)\n([0-9a-f]{64})$', reference))
    require(len(expected_hashes) == 6, 'Expected six historical input hashes')
    code = original.decode('utf-8')
    require(code.count(ROOT_LINE) == 1, 'Expected one original ROOT assignment')
    capture = io.StringIO()
    with tempfile.TemporaryDirectory(prefix='hermes-verify-init-proxy-') as tmp:
        root = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as handle:
            members = handle.getmembers()
            require(len(members) == 6 and {m.name for m in members} == set(expected_hashes), 'Archive membership changed')
            for member in members:
                require(member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts,
                        'Unsafe archive member')
                data = handle.extractfile(member).read()
                require(hashlib.sha256(data).hexdigest() == expected_hashes[member.name], 'Input hash mismatch: ' + member.name)
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        namespace = {'__name__': '__main__'}
        relocated = code.replace(ROOT_LINE, 'ROOT = Path(' + repr(str(root)) + ')', 1)
        with contextlib.redirect_stdout(capture):
            exec(compile(relocated, str(HERE / 'original_analysis.py'), 'exec'), namespace)
        for name, expected in expected_hashes.items():
            require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'Input modified during execution')
    means, pairs = historical_values(reference, namespace['models'], namespace['metric_names'])
    for name, values in namespace['observed'].items():
        require(np.allclose(values, means[name], rtol=0, atol=0.500001e-6), 'Historical metric mismatch: ' + name)
    paired_rows = []
    for j, (candidate, baseline) in enumerate(namespace['contrasts']):
        delta = namespace['direction'] * (namespace['observed'][candidate] - namespace['observed'][baseline])
        for k, metric in enumerate(namespace['metric_names']):
            valid = namespace['draws'][:, j, k]
            valid = valid[np.isfinite(valid)]
            low, high = np.quantile(valid, [.025, .975])
            expected = pairs[(candidate, baseline, metric)]
            require(np.allclose([delta[k], low, high], expected[:3], rtol=0, atol=0.500001e-6)
                    and len(valid) == expected[3], 'Historical paired comparison mismatch: ' + str((candidate, baseline, metric)))
            paired_rows.append(dict(candidate=candidate, reference=baseline, metric=metric,
                                    improvement=float(delta[k]), low=float(low), high=float(high), valid_draws=len(valid)))
    result = {
        'status': 'verified_against_historical_report_at_six_decimal_places',
        'training_records': len(namespace['train_ids']), 'test_records': len(namespace['test_ids']),
        'bootstrap_draws': namespace['N_BOOT'], 'bootstrap_seed': 0,
        'original_source_sha256': SOURCE_SHA, 'archive_sha256': ARCHIVE_SHA,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'historical_report_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'input_sha256': namespace['hashes'],
        'source_recovery': {'session': '20261020_085613_2ea473', 'command_message': 45352, 'execution_output_message': 45353},
        'adaptation': 'Only ROOT is relocated in memory; original calculation code is unchanged.',
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy']},
        'training_ids': namespace['train_ids'], 'test_ids': namespace['test_ids'],
        'feature_order': ['Taylor', 'signed_log1p(SynFlow)', 'NASWOT'],
        'training_feature_mean': namespace['mean'].tolist(), 'training_feature_scale': namespace['scale'].tolist(),
        'calibrators': namespace['coefficients'],
        'metrics': {name: dict(zip(namespace['metric_names'], map(float, values))) for name, values in namespace['observed'].items()},
        'paired_comparisons': paired_rows,
        'scope': 'Five affine calibrators refitted from saved scores; no candidate/Qwen training or new feature extraction. Intervals unadjusted, conditional on fixed fitted calibrators and successful-extraction cohort.'
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (args.out / 'stdout.log').write_text(capture.getvalue(), encoding='utf-8')
    write_csv(args.out / 'paired_comparisons.csv', list(paired_rows[0]), paired_rows)
    rows = [dict(record_id=rid, target=float(namespace['ytest'][i]),
                 **{name: float(pred[i]) for name, pred in namespace['predictions'].items()})
            for i, rid in enumerate(namespace['test_ids'])]
    write_csv(args.out / 'predictions.csv', list(rows[0]), rows)
    np.savez_compressed(args.out / 'bootstrap_draws.npz', draws=namespace['draws'])
    receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('PASS: 852 calibration / 86 test records; all 25 metric values and all 20 paired comparisons match the historical report at six decimal places.')
    print('Results:', args.out.resolve())


if __name__ == '__main__':
    main()
