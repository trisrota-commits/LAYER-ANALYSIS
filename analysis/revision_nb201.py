"""Supporting post-hoc NB201 Qwen intervals on all test architectures.

CPU-only port of analyze_nb201_qwen_extra_metrics.py. No model refitting.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis import revision_common
from analysis.revision_common import METRICS, metrics

SPEC = {'CIFAR-10-valid': 'evidence/nb201-qwen-primary-v1/cifar10/inputs',
        'CIFAR-100': 'evidence/nb201-qwen-primary-v1/cifar100/inputs',
        'ImageNet16-120': 'evidence/nb201-imagenet16-primary-v1/inputs'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset(repo: Path, dataset: str, relative: str, frozen: dict):
    """Load exactly the paired registry order, checking immutable input hashes."""
    folder = repo / relative
    hashes = {}
    for name in ['panel.csv', 'registry.json']:
        p = folder / name
        digest = sha(p)
        require(digest == frozen[relative + '/' + name], 'Frozen input mismatch: ' + str(p))
        hashes[relative + '/' + name] = digest
    registry = json.loads((folder / 'registry.json').read_text())
    require(len(registry) == 275 and len({r['architecture_id'] for r in registry}) == 275,
            'Registry architecture population mismatch')
    require(len({r['record_id'] for r in registry}) == 275, 'Registry record identity mismatch')
    dataset_key = {'CIFAR-10-valid': 'cifar10-valid', 'CIFAR-100': 'cifar100',
                   'ImageNet16-120': 'ImageNet16-120'}[dataset]
    require(all(r['dataset'] == dataset_key for r in registry), 'Registry dataset mismatch')
    ids = [r['architecture_id'] for r in registry]
    y = np.array([r['target'] for r in registry])
    families = np.array([r['family'] for r in registry])
    with (folder / 'panel.csv').open(newline='') as f:
        panel = list(csv.DictReader(f))
    lookup = {(r['condition'], int(r['seed']), int(r['idx'])): r for r in panel}
    require(len(lookup) == len(panel) == 2 * 6 * 275, 'Duplicate or missing panel records')
    pred = []
    for condition in ['no-layer', 'taylor']:
        seeds = []
        for seed in range(42, 48):
            selected = [lookup[condition, seed, idx] for idx in ids]
            require(np.array_equal([float(r['gt_best_accuracy']) for r in selected], y),
                    'Panel targets do not exactly match registry')
            require(all(r['dataset'] == registry[i]['dataset'] for i, r in enumerate(selected)),
                    'Panel datasets do not match registry')
            seeds.append([float(r['pred_best_accuracy']) for r in selected])
        pred.append(seeds)
    pred = np.array(pred)
    require(np.isfinite(pred).all() and np.isfinite(y).all(), 'Nonfinite predictions or targets')
    return y, pred, families, hashes


def run(repo: Path, out: Path):
    """Write the original report schema into a new, never-overwritten directory."""
    repo, out = Path(repo), Path(out)
    if out.exists():
        raise RuntimeError('Existing output preserved: ' + str(out))
    frozen = json.loads((repo / 'INPUT_MANIFEST.json').read_text())['files']
    rows = []
    arrays = {}
    hashes = {}
    for dataset, relative in SPEC.items():
        y, pred, families, dataset_hashes = load_dataset(repo, dataset, relative, frozen)
        hashes.update(dataset_hashes)
        point = metrics(y, pred).mean(axis=1)
        sign = np.array([-1, -1, 1, 1, 1])
        gain = (point[1] - point[0]) * sign
        groups = np.unique(families)
        blocks = [np.flatnonzero(families == g) for g in groups]
        sampled = np.random.default_rng(0).integers(len(groups), size=(10000, len(groups)))
        draws = np.empty((10000, 5))
        for i, sample in enumerate(sampled):
            idx = np.concatenate([blocks[g] for g in sample])
            values = metrics(y[idx], pred[:, :, idx]).mean(axis=1)
            draws[i] = (values[1] - values[0]) * sign
        require(np.isfinite(draws).all(), 'Undefined draws; do not silently exclude')
        arrays[dataset] = draws
        for k, metric in enumerate(METRICS):
            if metric == 'MAE':
                continue
            low, high = np.quantile(draws[:, k], [.025, .975])
            adjlow, adjhigh = np.quantile(draws[:, k], [.025 / 12, 1 - .025 / 12])
            rows.append(dict(dataset=dataset, population='all', n=275, groups=len(groups), metric=metric,
                             no_layer=float(point[0, k]), taylor=float(point[1, k]), improvement=float(gain[k]),
                             ci95_low=float(low), ci95_high=float(high),
                             bonferroni12_low=float(adjlow), bonferroni12_high=float(adjhigh)))
        print('COMPUTED', dataset, '275 records', len(groups), 'groups; 10000 valid draws', flush=True)
    for name, digest in hashes.items():
        require(sha(repo / name) == digest, 'Input changed during analysis: ' + name)
    out.mkdir(parents=True)
    with (out / 'intervals.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / 'bootstrap_differences.npz', **arrays)
    report = dict(status='complete', results=rows, draws=10000, rng_seed=0, input_sha256=hashes,
                  metric_helper_sha256=sha(Path(revision_common.__file__)), analysis_script_sha256=sha(Path(__file__)),
                  method='Paired operation-family cluster bootstrap. Sample groups with replacement and retain all their records with multiplicity. Metrics per fitted predictor then mean over seeds42-47. No seed resampling or model refitting.',
                  multiplicity='Bonferroni12 across four non-MAE metrics x three datasets; primary MAE analysis unchanged.',
                  interpretation='Supporting post-hoc inference conditional on frozen split, acquisition and fitted predictors; not adjustment for all analyses in the manuscript.')
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', files_sha256={
        p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
    for r in rows:
        print(f"{r['dataset']} | {r['metric']} | {r['no_layer']:.4f} | {r['taylor']:.4f} | {r['improvement']:+.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | [{r['bonferroni12_low']:.4f}, {r['bonferroni12_high']:.4f}]")
    return report
