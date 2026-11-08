
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
contract = json.loads((HERE / 'resume_contract.json').read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

action = sys.argv[1]
if action == 'space':
    state = os.statvfs(HERE)
    required = int(sys.argv[2])
    assert state.f_bavail * state.f_frsize >= required, (
        'STORAGE_GUARD_STOP: preserve outputs; insufficient free space'
    )
    assert state.f_favail >= 10000, 'STORAGE_GUARD_STOP: insufficient inodes'
    print('STORAGE_GUARD_PASS', flush=True)

elif action == 'verify':
    key = sys.argv[2]
    run = contract['runs'][key]
    out = Path(run['output'])
    fit = json.loads((out / 'refit_complete.json').read_text())
    test = json.loads((out / 'test_complete.json').read_text())

    assert fit['run'] == test['run'] == key
    assert fit['epochs'] == test['epochs'] == 7
    assert fit['training_records'] == run['fit_n']
    assert test['n_samples'] == run['eval_n']

    for filename, expected in fit['model_sha256'].items():
        assert sha(out / 'tuned_model' / filename) == expected

    with (out / 'test_predictions.csv').open(newline='') as f:
        predictions = list(csv.DictReader(f))
    registry = json.loads(
        (Path(run['data']) / 'registry.json').read_text()
    )
    assert len(predictions) == len(registry) == run['eval_n']
    for i, (prediction, item) in enumerate(zip(predictions, registry)):
        assert int(prediction['idx']) == i
        assert prediction['dataset'] == item['dataset']
        assert float(prediction['gt_best_accuracy']) == item['target']

    with (out / 'storage_resume_complete.json').open('x') as f:
        json.dump({
            'status': 'complete',
            'run': key,
            'prediction_sha256': sha(out / 'test_predictions.csv'),
            'refit_receipt_sha256': sha(out / 'refit_complete.json'),
            'test_receipt_sha256': sha(out / 'test_complete.json'),
            'resume_provenance_sha256': sha(out / 'resume_provenance.json'),
        }, f, indent=2)
    print('STORAGE_RESUME_COMPLETE', key, flush=True)
