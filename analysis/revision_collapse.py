"""S25 from frozen matched scores; original class-stratified draw order."""
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import rankdata
from analysis.thesis_common import require, sha

ARCHIVE_SHA = '2b233d441517d5fca8c4d6d564abce824706c8d2828aae0dbf48d867c895e249'
SOURCE_SHA = 'e7b27941c9b3a3050c8d6f60807c5e2109e76dfff0224beefde7c69ffc2720c9'


def run(repo, out):
    repo, out = Path(repo), Path(out)
    if out.exists():
        raise FileExistsError('Existing output preserved: ' + str(out))
    archive = repo / 'evidence/thesis-new-logs-20261112-084434.zip'
    require(sha(archive) == ARCHIVE_SHA, 'Frozen evidence changed')
    source = repo / 'paper_runners/collapse_baseline/analyze.py'
    require(sha(source) == SOURCE_SHA, 'Recovered source changed')
    prefix = 'workspace_evidence/lemur-collapse-baseline-v1/'
    with zipfile.ZipFile(archive) as z:
        payload = z.read(prefix + 'matched_scores.csv')
        original = json.loads(z.read(prefix + 'results.json'))
        saved_arrays = z.read(prefix + 'bootstrap.npz')
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    ids = [r['record_id'] for r in rows]
    target = np.array([float(r['target']) for r in rows])
    labels = target < 20
    require(len(ids) == len(set(ids)) == 115 and labels.sum() == 9, 'Collapse cohort mismatch')
    require(np.array_equal(labels, [int(r['collapsed']) for r in rows]), 'Collapse labels differ')
    scores = {k: np.array([float(r[k]) for r in rows]) for k in ('epoch3_accuracy', 'no-layer', 'taylor')}
    require(all(np.isfinite(s).all() for s in scores.values()), 'Nonfinite scores')
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    matrices = {k: (s[pos, None] > s[neg][None, :]).astype(float) + .5 * (s[pos, None] == s[neg][None, :]) for k, s in scores.items()}
    rng = np.random.default_rng(1)
    draws = {k: np.empty(10000) for k in scores}
    # Preserve the original 128-draw batching: it controls RNG consumption order.
    for start in range(0, 10000, 128):
        n = min(128, 10000 - start)
        pi = rng.integers(len(pos), size=(n, len(pos)))
        ni = rng.integers(len(neg), size=(n, len(neg)))
        for k, m in matrices.items():
            draws[k][start:start+n] = m[pi[:, :, None], ni[:, None, :]].mean(axis=(1, 2))
    result = dict(n=115, collapsed=9, threshold=20, draws=10000, seed=1,
                  method=original['method'], scores={}, paired={},
                  source_sha256=SOURCE_SHA, input_archive_sha256=ARCHIVE_SHA,
                  source_recovery='paper_runners/collapse_baseline/analyze.py; same draw algorithm, frozen matched-score input replaces original source joins')
    table = []
    with np.load(io.BytesIO(saved_arrays), allow_pickle=False) as old:
        for k, s in scores.items():
            auc = float((rankdata(s)[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
            np.testing.assert_allclose(auc, matrices[k].mean(), atol=1e-14)
            np.testing.assert_array_equal(draws[k], old[k])
            ci = np.quantile(draws[k], [.025, .975]).tolist()
            result['scores'][k] = dict(auc=auc, ci95=ci)
            np.testing.assert_allclose([auc, *ci], [original['scores'][k]['auc'], *original['scores'][k]['ci95']], atol=1e-14)
            table.append(dict(kind='score', condition=k, estimate=auc, low=ci[0], high=ci[1]))
            if k != 'epoch3_accuracy':
                delta = auc - result['scores']['epoch3_accuracy']['auc']
                lo, hi = np.quantile(draws[k] - draws['epoch3_accuracy'], [.025, .975])
                result['paired'][k] = dict(auc_difference=delta, ci95=[float(lo), float(hi)])
                np.testing.assert_allclose([delta, lo, hi], [original['paired'][k]['auc_difference'], *original['paired'][k]['ci95']], atol=1e-14)
                table.append(dict(kind='paired_vs_epoch3', condition=k, estimate=delta, low=float(lo), high=float(hi)))
    result['verification'] = 'all original bootstrap arrays and intervals reproduced'
    require(sha(archive) == ARCHIVE_SHA, 'Input changed during calculation')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'S25.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with (out / 'bootstrap.npz').open('xb') as f:
        np.savez_compressed(f, **draws)
    with (out / 'results.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (out / 'results.md').open('x', encoding='utf-8') as f:
        f.write('# S25: collapse detection\n\nPointwise unadjusted 95% class-stratified record-bootstrap intervals; fixed scores.\n\n| Kind | Condition | Estimate | 95% CI |\n|---|---|---:|---|\n')
        for r in table:
            f.write(f"| {r['kind']} | {r['condition']} | {r['estimate']:.6f} | [{r['low']:.6f}, {r['high']:.6f}] |\n")
    hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    with (out / 'complete.json').open('x', encoding='utf-8') as f:
        json.dump(dict(status='complete', files_sha256=hashes), f, indent=2)
    print('COMPLETE: S25 exact score, interval and bootstrap reproduction;', out, flush=True)
    return result
