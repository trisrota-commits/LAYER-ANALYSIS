"""All six HistGB arm pairs, five metrics, all records; no fitting."""
import csv
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from analysis import revision_loss_confidence as source

ARMS = source.ARMS
METRICS = source.METRICS
PAIRS = list(combinations(range(len(ARMS)), 2))
DRAWS = 10000


def run(archive, out):
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise FileExistsError('Existing output preserved: ' + str(out))
    loaded = source.load(archive)
    family_size = len(PAIRS) * len(METRICS)
    sign = np.array([-1. if m in ('MAE', 'RMSE') else 1. for m in METRICS])
    rows, descriptive, arrays = [], [], {}
    report = ['# Loss/confidence: all-pairs confidence intervals', '',
              'All test records; A = early accuracy, LC = loss/confidence, T = Taylor.',
              'Positive improvement favors the candidate. Bonferroni-30 is applied separately per dataset.', '']
    for dataset, (y, pred, families, saved, _) in loaded.items():
        values = source.metrics(y, pred)
        source.require(np.isfinite(values).all(), 'Undefined observed metric')
        for ai, arm in enumerate(ARMS):
            np.testing.assert_allclose(values[ai], [saved['all'][arm]['mean_metrics'][m] for m in METRICS],
                                       rtol=1e-10, atol=1e-10)
            descriptive.append(dict(dataset=dataset, arm=arm, n=len(y),
                                    **{m: float(values[ai, mi]) for mi, m in enumerate(METRICS)}))
        point = np.stack([(values[b] - values[a]) * sign for a, b in PAIRS])
        groups = np.unique(families)
        blocks = [np.flatnonzero(families == g) for g in groups]
        sampled = np.random.default_rng(0).integers(len(groups), size=(DRAWS, len(groups)))
        boot = np.empty((DRAWS, len(PAIRS), len(METRICS)))
        for i, draw in enumerate(sampled):
            idx = np.concatenate([blocks[g] for g in draw])
            v = source.metrics(y[idx], pred[:, idx])
            boot[i] = np.stack([(v[b] - v[a]) * sign for a, b in PAIRS])
        source.require(np.isfinite(boot).all(), 'Undefined bootstrap metric; no draws discarded')
        ci = np.quantile(boot, [.025, .975], axis=0)
        adjusted = np.quantile(boot, [.025 / family_size, 1 - .025 / family_size], axis=0)
        arrays[dataset] = boot
        report += ['## ' + dataset, '', '| Arm | ' + ' | '.join(METRICS) + ' |',
                   '|---|' + '---:|' * len(METRICS)]
        for arm, v in zip(ARMS, values):
            report.append('| ' + arm + ' | ' + ' | '.join(f'{x:.4f}' for x in v) + ' |')
        report += ['', '| Reference -> Candidate | Metric | Improvement | 95% CI | Bonferroni-30 CI | Conclusion |',
                   '|---|---|---:|---|---|---|']
        for pi, (a, b) in enumerate(PAIRS):
            for mi, metric in enumerate(METRICS):
                lo, hi = ci[:, pi, mi]
                alo, ahi = adjusted[:, pi, mi]
                conclusion = 'Candidate better' if alo > 0 else 'Candidate worse' if ahi < 0 else 'Inconclusive'
                contrast = ARMS[a] + ' -> ' + ARMS[b]
                rows.append(dict(dataset=dataset, population='all', contrast=contrast, metric=metric,
                                 n=len(y), groups=len(groups), reference=float(values[a, mi]),
                                 candidate=float(values[b, mi]), improvement=float(point[pi, mi]),
                                 ci95_low=float(lo), ci95_high=float(hi), bonferroni30_low=float(alo),
                                 bonferroni30_high=float(ahi), draws=DRAWS, conclusion=conclusion))
                report.append(f'| {contrast} | {metric} | {point[pi, mi]:+.4f} | [{lo:.4f}, {hi:.4f}] '
                              f'| [{alo:.4f}, {ahi:.4f}] | {conclusion} |')
        report.append('')
        print('VERIFIED', dataset, len(y), 'records;', DRAWS, 'paired operation-family draws', flush=True)
    source.require(len(rows) == len(loaded) * family_size, 'Incomplete comparison coverage')
    source.require(source.sha(archive.read_bytes()) == source.EXPECTED, 'Input archive changed')
    limitations = ['Post-hoc, fixed fitted predictors/acquisitions; no seed resampling or training uncertainty.',
                   'Six seed-labeled predictions are identical within each arm, verified by the loader.',
                   'ImageNet16-120 is an ordered-truncation partial cohort.',
                   'Adjusted percentile tails have limited Monte Carlo precision; near-zero endpoints need caution.',
                   'Intervals including zero are inconclusive, not evidence of equivalence.',
                   'This all-record Bonferroni-30 family does not replace the existing two-population Bonferroni-40 analysis.']
    report += ['## Scope', ''] + ['- ' + note for note in limitations]
    result = dict(analysis='Post-hoc all-pairs five-metric operation-family bootstrap',
                  source_archive_sha256=source.EXPECTED, metrics=METRICS,
                  pairs=[[ARMS[a], ARMS[b]] for a, b in PAIRS], population='all', draws=DRAWS,
                  bootstrap='Paired whole operation-multiset families with replacement; default_rng(0) reset per dataset; same resamples for all arms.',
                  correction='Bonferroni30 per dataset: six pairs x five metrics; not joint across datasets.',
                  positive_improvement='Reference minus candidate for MAE/RMSE; candidate minus reference for Pearson/Spearman/R2.',
                  limitations=limitations, descriptive=descriptive, results=rows,
                  source_sha256={p.name: source.sha(p.read_bytes()) for p in (Path(__file__), Path(source.__file__))},
                  versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy')})
    out.mkdir(parents=True, exist_ok=False)
    for name, records in [('all_comparisons.csv', rows), ('descriptive.csv', descriptive)]:
        with (out / name).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    with (out / 'bootstrap_differences.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    for name, text in [('results.json', json.dumps(result, indent=2, allow_nan=False) + '\n'),
                       ('results.md', '\n'.join(report) + '\n')]:
        with (out / name).open('x', encoding='utf-8') as f:
            f.write(text)
    receipt = dict(status='complete', files_sha256={p.name: source.sha(p.read_bytes()) for p in out.iterdir() if p.is_file()})
    with (out / 'complete.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    print('COMPLETE:', len(rows), 'metric comparisons. Output:', out, flush=True)
