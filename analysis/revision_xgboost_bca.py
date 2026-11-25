"""Final-paper XGBoost sensitivity: 100,000 paired-record BCa draws, Bonferroni5.

Preserves the operator-executed inline analysis: default_rng(0), batches of 250,
metrics per fitted seed then averaged, paired record jackknife, common targets.
"""
import csv
import json
from pathlib import Path
import numpy as np
from analysis.revision_bca import interval
from analysis.revision_common import METRICS, metrics
from analysis.revision_lemur import load_inputs, PACKET
from analysis.thesis_common import require, sha


def calculate(predictions, target, n_resamples=100000):
    p, y = np.asarray(predictions), np.asarray(target)
    require(p.shape == (2, 6, len(y)) and len(y) > 3, 'Expected two methods and six seeds')
    require(np.isfinite(p).all() and np.isfinite(y).all(), 'Nonfinite predictions/targets')
    require(n_resamples >= 1000, 'Too few BCa resamples')
    sign = np.array([-1, -1, 1, 1, 1])
    means = metrics(y, p).mean(axis=1)
    point = (means[0]-means[1])*sign
    jack = np.empty((len(y), 5))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        value = metrics(y[keep], p[:, :, keep]).mean(axis=1)
        jack[i] = (value[0]-value[1])*sign
    rng = np.random.default_rng(0)
    draws = np.empty((n_resamples, 5))
    for start in range(0, n_resamples, 250):
        end = min(start+250, n_resamples)
        idx = rng.integers(len(y), size=(end-start, len(y)))
        value = metrics(y[idx], p[:, :, idx]).mean(axis=1)
        draws[start:end] = (value[0]-value[1])*sign
    require(np.isfinite(draws).all() and np.isfinite(jack).all(), 'Undefined metric; no silent dropping')
    rows = []
    for k, metric in enumerate(METRICS):
        ci = interval(draws[:, k], point[k], jack[:, k], .95)
        adjusted = interval(draws[:, k], point[k], jack[:, k], .99)
        previous = interval(draws[:n_resamples//2, k], point[k], jack[:, k], .99)
        rows.append(dict(metric=metric, xgboost=float(means[1, k]), qwen_taylor=float(means[0, k]),
                         improvement=float(point[k]), ci95_low=ci['low'], ci95_high=ci['high'],
                         bonferroni5_low=adjusted['low'], bonferroni5_high=adjusted['high'],
                         bias_z0=ci['bias_z0'], acceleration=ci['acceleration'],
                         adjusted_effective_tail_draws=adjusted['effective_tail_draws'],
                         sparse_adjusted_tail=adjusted['sparse_tail'],
                         half_draw_adjusted_endpoint_change=max(abs(adjusted[t]-previous[t]) for t in ('low','high'))))
    return rows, draws, jack


def run(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Existing output preserved: '+str(out))
    require(not any(out.is_relative_to(repo/p) for p in ('analysis','evidence','tests','paper_runners')),
            'Output overlaps source/evidence')
    archive = repo/'evidence/two-experiments-cpu-inputs-v1.zip'
    before = {str(p.relative_to(repo)): sha(p) for p in (archive, repo/PACKET)}
    pred, y, registry, sources, hashes, packet_hash = load_inputs(repo, archive)
    rows, draws, jack = calculate(pred[[0, 2]], y)
    require(all(sha(repo/n) == h for n,h in before.items()), 'Input changed during calculation')
    out.mkdir(parents=True, exist_ok=False)
    with (out/'intervals.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'bootstrap.npz', differences=draws, jackknife=jack)
    (out/'test_registry.json').write_text(json.dumps(registry, indent=2)+'\n', encoding='utf-8')
    report = dict(status='complete', results=rows, n=115, seeds=list(range(42,48)),
                  draws=100000, rng_seed=0, batch=250, method='paired-record BCa',
                  correction_family=5, seed_aggregation='mean of seed-level metrics; no ensemble',
                  target_policy='Common two-decimal Qwen targets; raw baseline targets verified',
                  input_sha256=before, baseline_sources_sha256=hashes,
                  qwen_source=sources['3/off/taylor'],
                  source_sha256={n:sha(repo/n) for n in ['analysis/revision_xgboost_bca.py',
                      'analysis/revision_bca.py','analysis/revision_common.py','analysis/revision_lemur.py']},
                  limitations=['Post-hoc sensitivity, separate from percentile Bonferroni20',
                               'Fixed fitted predictors and test set; no correction for feature selection',
                               'Half-draw endpoint comparison is a stability diagnostic, not proof of convergence'])
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status='complete', files_sha256={
        p.name:sha(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n', encoding='utf-8')
    for row in rows:
        print(row, flush=True)
    return report
