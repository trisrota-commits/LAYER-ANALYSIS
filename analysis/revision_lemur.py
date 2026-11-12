"""Rows10/14/17: matched saved predictions, common two-decimal targets, Bonferroni20.

CPU-only port of analyze_lemur_extra_paired.py. No model refitting.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from analysis import revision_common
from analysis import thesis_lemur as existing
from analysis.revision_common import METRICS, metrics

REFERENCES = ['All seven / ZCP-OFF', 'XGBoost / 3 accuracies',
              'Calibrated SynFlow', 'Calibrated NASWOT']
ARCHIVE_SHA256 = 'e3629395b79631261ea11675a41c46d5ddc6a2b5b76c87015dab955d2249bd1e'
PUBLIC_ARCHIVE_SHA256 = '3a420140ff64a851bd64fe916c37651ff7016bc93b87fad41ce9309f05448ebd'
PACKET = 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(repo: Path, archive: Path):
    """Check archive integrity and exact paired identities before computing metrics."""
    manifest = json.loads((repo / 'INPUT_MANIFEST.json').read_text())['files']
    require(digest((repo / PACKET).read_bytes()) == manifest[PACKET], 'Frozen packet SHA256 mismatch')
    require(digest(archive.read_bytes()) in {ARCHIVE_SHA256, PUBLIC_ARCHIVE_SHA256}, 'Archive SHA256 mismatch')
    # The existing pool is cached: clear it so a previous run against this path
    # cannot supply stale bytes after the manifest check above.
    existing.pools.cache_clear()
    pool = existing.pools(repo)
    data, y, sources = existing.load_main(repo)
    with zipfile.ZipFile(archive) as z:
        prefix = 'files/classical-eight-features-v1/'
        done = json.loads(z.read(prefix + 'complete.json'))
        records = z.read(prefix + 'records.csv')
        require(digest(records) == done['files_sha256']['records.csv'], 'Registry member SHA256 mismatch')
        registry = sorted([r for r in csv.DictReader(io.StringIO(records.decode())) if r['split'] == 'test'],
                          key=lambda r: int(r['position']))
        require(len(registry) == len({r['record_id'] for r in registry}) == 115, 'Registry record population mismatch')
        require(len({r['architecture_id'] for r in registry}) == 115, 'Registry architecture population mismatch')
        require(np.array_equal([float(r['target']) for r in registry], y), 'Registry target mismatch')
        qwen = data[3, 'off', 'taylor']
        for si, seed in enumerate(range(42, 48)):
            name = f'qwen_reference-s{seed}-predictions.csv'
            blob = z.read(prefix + name)
            require(digest(blob) == done['files_sha256'][name], 'Qwen member SHA256 mismatch: ' + name)
            rows = list(csv.DictReader(io.StringIO(blob.decode())))
            require([r['record_id'] for r in rows] == [r['record_id'] for r in registry], 'Qwen record identity mismatch')
            require(np.array_equal([float(r['target']) for r in rows], y), 'Qwen target mismatch')
            require(np.array_equal([float(r['prediction']) for r in rows], qwen[si]), 'Qwen prediction mismatch')
    hashes = {}
    baselines = []

    def baseline(suffix):
        name, blob = existing.get(pool['baseline'], suffix)
        rows = list(csv.DictReader(io.StringIO(blob.decode())))
        require(len(rows) == 115, 'Baseline population mismatch')
        require([r['run_id'] for r in rows] == [r['record_id'] for r in registry], 'Baseline record identity mismatch')
        require([r['architecture_id'] for r in rows] == [r['architecture_id'] for r in registry],
                'Baseline architecture mismatch')
        require([r['dataset'] for r in rows] == [r['dataset'] for r in registry], 'Baseline dataset mismatch')
        require(np.array_equal([float(r['gt_best_accuracy']) for r in rows],
                               [float(r['raw_target']) for r in registry]), 'Baseline raw target mismatch')
        require(np.array_equal([round(float(r['gt_best_accuracy']), 2) for r in rows], y),
                'Baseline rounded target mismatch')
        hashes[name] = digest(blob)
        return [float(r['pred_best_accuracy']) for r in rows]

    baselines.append(data[3, 'off', 'all-seven'])
    baselines.append(np.array([baseline(f'xgboost_3acc/predictions-s{s}.csv') for s in range(42, 48)]))
    for suffix in ['/synflow_predictions.csv', '/nwot_predictions.csv']:
        # Fixed calibrator: repeat for metric alignment, not independent seed replicates.
        baselines.append(np.repeat(np.array(baseline(suffix))[None, :], 6, axis=0))
    pred = np.array([qwen, *baselines])
    require(pred.shape == (5, 6, 115) and np.isfinite(pred).all() and np.isfinite(y).all(),
            'Invalid paired prediction shape or nonfinite inputs')
    return pred, y, registry, sources, hashes, manifest[PACKET]


def run(repo: Path, out: Path, archive: Path):
    """Write paired intervals into a new directory using an explicitly supplied archive."""
    repo, out, archive = Path(repo), Path(out), Path(archive)
    if out.exists():
        raise RuntimeError('Existing output preserved: ' + str(out))
    archive_hash = digest(archive.read_bytes())
    pred, y, registry, sources, hashes, packet_hash = load_inputs(repo, archive)
    names = METRICS
    labels = REFERENCES
    sign = np.array([-1, -1, 1, 1, 1])
    point = metrics(y, pred).mean(axis=1)
    gain = (point[0] - point[1:]) * sign
    rng = np.random.default_rng(0)
    draws = np.empty((4, 10000, 5))
    for start in range(0, 10000, 250):
        idx = rng.integers(len(y), size=(250, len(y)))
        value = metrics(y[idx], pred[:, :, idx]).mean(axis=1)
        draws[:, start:start + 250] = (value[0][None, :, :] - value[1:]) * sign
    require(np.isfinite(draws).all(), 'Undefined resample metric; no silent dropping')
    ci = np.quantile(draws, [.025, .975], axis=1)
    adjusted = np.quantile(draws, [.025 / 20, 1 - .025 / 20], axis=1)
    output = []
    for i, label in enumerate(labels):
        for k, name in enumerate(names):
            output.append(dict(reference=label, metric=name, reference_value=float(point[i + 1, k]),
                               qwen_taylor=float(point[0, k]), improvement=float(gain[i, k]),
                               ci95_low=float(ci[0, i, k]), ci95_high=float(ci[1, i, k]),
                               bonferroni20_low=float(adjusted[0, i, k]),
                               bonferroni20_high=float(adjusted[1, i, k]), draws=10000, n=115))
    require(digest(archive.read_bytes()) == archive_hash, 'Archive changed during analysis')
    require(digest((repo / PACKET).read_bytes()) == packet_hash, 'Packet changed during analysis')
    out.mkdir(parents=True)
    with (out / 'intervals.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    np.savez_compressed(out / 'bootstrap.npz', differences=draws)
    np.savez_compressed(out / 'paired_predictions.npz', predictions=pred, target=y)
    (out / 'test_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    report = dict(status='complete', results=output, methods=['Qwen Taylor / ZCP-OFF', *labels], metrics=names,
                  main_sources={k: sources[k] for k in ['3/off/taylor', '3/off/all-seven']}, baseline_sources_sha256=hashes,
                  source_archive_sha256=archive_hash, source_packet_sha256=packet_hash,
                  metric_helper_sha256=digest(Path(revision_common.__file__).read_bytes()),
                  script_sha256=digest(Path(__file__).read_bytes()),
                  method='10000 paired test-record bootstrap draws; numpy default_rng(0); fixed fitted models; metrics per seed then averaged; no seed resampling; Bonferroni20 across four contrasts x five metrics',
                  target_policy='Same two-decimal Qwen targets for every method. Baseline raw targets exactly match registry raw_target and round to these targets. Predictions unchanged.',
                  proxy_policy='One fixed calibrated prediction vector per proxy; repeating across seed axis is algebraic alignment only, not independent repetitions.',
                  limitations=['Post-hoc supporting inference; does not correct prior Taylor feature selection', 'Initialization proxies and early-training Qwen have unequal information budgets; not an isolated Taylor-effect comparison', 'Calibrator/predictor fitting uncertainty is not included'])
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', files_sha256={
        p.name: digest(p.read_bytes()) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
    for r in output:
        print(f"{r['reference']} | {r['metric']} | {r['reference_value']:.4f} | {r['qwen_taylor']:.4f} | {r['improvement']:+.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | [{r['bonferroni20_low']:.4f}, {r['bonferroni20_high']:.4f}]")
    return report
