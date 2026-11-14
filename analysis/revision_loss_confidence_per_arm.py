"""S48-style pointwise per-arm family-bootstrap intervals; fixed saved models only."""
import csv
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from analysis import revision_loss_confidence as source
from analysis.revision_loss_confidence import ARMS, METRICS, metrics, require, sha

DRAWS = 10000
PAPER_METRICS = ('MAE', 'RMSE', 'Spearman')


def render_markdown(rows, selected_metrics=METRICS):
    """Render full metrics or the paper's MAE/RMSE/Spearman column subset."""
    require(bool(selected_metrics) and set(selected_metrics) <= set(METRICS),
            'Unknown or empty metric selection')
    lines = ['# Per-arm fixed-model performance', '',
             'All records; pointwise unadjusted 95% operation-family percentile intervals.',
             'No Bonferroni correction; fixed fitted models, not training/seed uncertainty.', '',
             '| Dataset | Arm | ' + ' | '.join(selected_metrics) + ' |',
             '| --- | --- | ' + ' | '.join(['---'] * len(selected_metrics)) + ' |']
    lookup = {(r['dataset'], r['arm'], r['metric']): r for r in rows}
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        for arm in ARMS:
            cells = []
            for metric in selected_metrics:
                r = lookup[dataset, arm, metric]
                cells.append(f"{r['estimate']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]")
            lines.append('| ' + ' | '.join([dataset, arm] + cells) + ' |')
    return '\n'.join(lines) + '\n'


def run(archive, out):
    """Verify frozen evidence and export 60 unadjusted per-arm metric intervals."""
    archive, out = Path(archive), Path(out)
    require(not out.exists(), 'Existing output preserved: ' + str(out))
    loaded = source.load(archive)
    require(set(loaded) == set(source.BUNDLES), 'Unexpected or missing datasets')
    rows, arrays, checks = [], {}, []
    for dataset, (y, p, families, saved, _) in loaded.items():
        values = metrics(y, p)
        require(np.isfinite(values).all(), 'Undefined full-sample metric')
        for ai, arm in enumerate(ARMS):
            expected = [saved['all'][arm]['mean_metrics'][m] for m in METRICS]
            np.testing.assert_allclose(values[ai], expected, rtol=1e-10, atol=1e-10)
        groups = np.unique(families)
        blocks = [np.flatnonzero(families == group) for group in groups]
        sampled = np.random.default_rng(0).integers(len(groups), size=(DRAWS, len(groups)))
        boot = np.empty((DRAWS, len(ARMS), len(METRICS)))
        for i, draw in enumerate(sampled):
            idx = np.concatenate([blocks[g] for g in draw])
            boot[i] = metrics(y[idx], p[:, idx])
        require(np.isfinite(boot).all(), 'Undefined bootstrap metric; no silent draw removal')
        intervals = np.quantile(boot, [.025, .975], axis=0)
        for ai, arm in enumerate(ARMS):
            for mi, metric in enumerate(METRICS):
                rows.append(dict(dataset=dataset, population='all', arm=arm, metric=metric,
                                 n=len(y), groups=len(groups), estimate=float(values[ai, mi]),
                                 ci95_low=float(intervals[0, ai, mi]),
                                 ci95_high=float(intervals[1, ai, mi]), draws=DRAWS))
        arrays[dataset] = boot
        checks.append(dict(dataset=dataset, population='all',
                           all_descriptive_metrics_match=True, valid_draws=DRAWS))
        print('VERIFIED', dataset, 'all records; per-arm five-metric bootstrap complete', flush=True)
    require(sha(archive.read_bytes()) == source.EXPECTED, 'Input archive changed')
    source_files = {f'analysis/{p.name}': sha(p.read_bytes())
                    for p in [Path(__file__), Path(source.__file__)]}
    versions = dict(python=platform.python_version(),
                    **{n: importlib.metadata.version(n) for n in ['numpy', 'scipy']})
    provenance = dict(source_archive_sha256=source.EXPECTED,
                      source_files_sha256=source_files, versions=versions)
    result = dict(analysis='S48-style per-arm pointwise 95% confidence intervals',
                  **provenance, checks=checks, results=rows,
                  bootstrap=dict(draws=DRAWS, rng='numpy default_rng(0) reset per dataset',
                                 unit='operation-multiset family', replacement=True,
                                 shared_across='all four arms and all five metrics',
                                 records='all records in sampled families, with multiplicity',
                                 array_axes=['draw', 'arm', 'metric'], arms=ARMS, metrics=METRICS,
                                 quantiles=[.025, .975], quantile_method='linear'),
                  correction='None: pointwise unadjusted 95%; not simultaneous intervals',
                  identical_seed_predictions='Exact equality across seeds42-47 verified by shared loader; one fixed vector per arm',
                  limitations=['Fixed fitted predictors and acquisitions; no training or seed uncertainty',
                               'ImageNet16-120 is an ordered-truncation partial cohort'])
    out.mkdir(parents=True)
    with (out / 'per_arm_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / 'bootstrap_per_arm.npz', **arrays)
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'per_arm_metrics.md').write_text(render_markdown(rows), encoding='utf-8')
    (out / 'paper_three_metrics.md').write_text(render_markdown(rows, PAPER_METRICS), encoding='utf-8')
    files = {p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', **provenance,
                                                     files_sha256=files), indent=2) + '\n', encoding='utf-8')
    print('OUTPUT', out, flush=True)
