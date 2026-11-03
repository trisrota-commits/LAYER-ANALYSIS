"""Fresh, fixed-model paired-record bootstrap evidence; never overwrites output.
Run --inspect to list relevant archive members without creating the report.
Requires Python 3.11+, NumPy and SciPy. No extraction, fitting, or training.
"""
from pathlib import Path
import csv
import hashlib
import io
import platform
import random
import sys
import tarfile
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.stats import rankdata, t

import argparse
_parser = argparse.ArgumentParser(description='Reproduce archived main-study paired tables; never overwrite reports.')
_parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent / 'article-tables-only-20261014-111938.tar.gz')
_parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'paired_main_tables_recomputed.log')
_parser.add_argument('--inspect', action='store_true')
_cli = _parser.parse_args()
ROOT = Path(__file__).resolve().parent
ARCHIVE = _cli.archive
OUTPUT = _cli.output
SEEDS = list(range(42, 48))
METRICS = ['RMSE', 'MAE', 'Pearson', 'Spearman', 'R2']
B = 10000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def correlation(a, b):
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum(ac * bc, axis=-1) / np.sqrt(np.sum(ac * ac, axis=-1) * np.sum(bc * bc, axis=-1))


def metrics(pred, target):
    """Last axis is records, preceding axes can be seeds and bootstrap draws."""
    err = pred - target
    mse = np.mean(err * err, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - np.sum(err * err, axis=-1) / np.sum((target - target.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    return np.stack([np.sqrt(mse), np.mean(np.abs(err), axis=-1), correlation(pred, target), correlation(rankdata(pred, axis=-1), rankdata(target, axis=-1)), r2], axis=-1)


def main():
    if '--inspect' not in sys.argv and OUTPUT.exists():
        raise FileExistsError(f'Refusing overwrite: {OUTPUT}')
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('/test_predictions.csv')]
        if '--inspect' in sys.argv:
            for m in members:
                if any(x in m.name for x in ['round2', 'round3', 'readmission', 'taylor-sensitivity-zcp', 'per-layer-16k', 'zcp-off-toggle', 'taylor-3acc', 'layer-ablation', 'tables9-12']):
                    print(m.name)
            print('VERSIONS', sys.version, np.__version__, scipy.__version__)
            return

        def resolve(batch, run):
            suffix = f'{batch}/results/{run}/test_predictions.csv'
            found = [m for m in members if m.name.endswith(suffix)]
            if len(found) != 1:
                raise ValueError(f'Expected one member for {suffix}; got {[m.name for m in found]}')
            return found[0]

        conditions = {}
        for label, feature in [('T+S', 'taylor-sensitivity'), ('T+A', 'taylor-activation'), ('T+C', 'taylor-cka')]:
            conditions[label] = [resolve('round2-taylor-pairs-3acc-off-v1', f'{feature}-s{s}') for s in SEEDS]
        for label, feature in [('T+S+A', 'taylor-sensitivity-activation'), ('T+S+C', 'taylor-sensitivity-cka')]:
            conditions[label] = [resolve('round3-triples-3acc-off-v1', f'{feature}-s{s}') for s in SEEDS]
        for label, feature in [('T+G', 'taylor-gradient-flow'), ('T+E', 'taylor-effective-rank'), ('T+S+G', 'taylor-sensitivity-gradient-flow'), ('T+S+E', 'taylor-sensitivity-effective-rank')]:
            conditions[label] = [resolve('readmission-3acc-off-v1', f'{feature}-s{s}') for s in SEEDS]
        conditions['T OFF'] = [resolve('zcp-off-toggle-refit7-v1', f'3acc-taylor-s{s}') if s < 45 else resolve('taylor-3acc-zcp-off-seeds45-47-v1', f'3acc-taylor-s{s}') for s in SEEDS]
        conditions['T ON'] = [resolve('layer-ablation-refit7-v1', f'taylor-s{s}') if s < 45 else resolve('tables9-12-seeds45-47-v1', f'3acc-zcp-on-taylor-s{s}') for s in SEEDS]
        conditions['T+S ON'] = [resolve('taylor-sensitivity-zcp-on-v1', f'taylor-sensitivity-s{s}') for s in SEEDS]
        conditions['individual T OFF'] = [resolve('per-layer-16k-refit7-v1', f's{s}') for s in SEEDS]
        data, provenance = {}, []
        canonical = None
        for label, group in conditions.items():
            predictions = []
            for seed, member in zip(SEEDS, group):
                raw = archive.extractfile(member).read()
                rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
                assert len(rows) == 115, (member.name, len(rows))
                indices = [int(r['idx']) for r in rows]
                assert len(set(indices)) == 115, member.name
                assert all(r['run_id'] == '' for r in rows), member.name
                keys = [(int(r['idx']), r['dataset'], float(r['gt_best_accuracy'])) for r in rows]
                if canonical is None:
                    canonical = keys
                assert keys == canonical, f'Ordered idx/dataset/target mismatch: {member.name}'
                pred = np.array([float(r['pred_best_accuracy']) for r in rows])
                assert np.isfinite(pred).all() and np.isfinite([k[2] for k in keys]).all()
                predictions.append(pred)
                provenance.append(f'{label}\tseed={seed}\tsha256={sha(raw)}\tbytes={len(raw)}\t{member.name}')
            data[label] = np.array(predictions)
    target = np.array([k[2] for k in canonical])
    lines = ['FRESH RECOMPUTATION FROM ARCHIVED CSVs — NOT ORIGINAL EXPERIMENT STDOUT',
             f'UTC={datetime.now(timezone.utc).isoformat()}', f'archive={ARCHIVE}',
             f'archive_sha256={hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest()}',
             f'script={Path(__file__).resolve()}', f'script_sha256={sha(Path(__file__).read_bytes())}',
             f'Python={sys.version}', f'platform={platform.platform()}', f'numpy={np.__version__}; scipy={scipy.__version__}',
             'Validation: all members n=115, unique idx, empty run_id, finite predictions/targets; identical ordered idx/dataset/float targets across every seed and arm.',
             'Conditions mapped by requested archived run paths; prediction checks do not independently verify training flags.',
             'Seeds=42,43,44,45,46,47; metrics computed per fitted seed then averaged, never prediction ensembling.',
             'RMSE=sqrt(mean squared error); MAE=mean absolute error; Pearson=centered correlation; Spearman=Pearson of average-tie ranks recomputed per resample; R2=1-SSE/SST, undefined constant targets -> NaN.',
             'Descriptive 95% CI = seed mean +/- t.ppf(.975,5)*sample_SD/sqrt(6), no clipping.',
             'Bootstrap: 10000 paired record resamples of size 115 with replacement; same indices for both arms and all six fixed fitted seeds; no retraining or seed resampling.',
             'Positive favors candidate: reference-candidate for RMSE/MAE; candidate-reference for Pearson/Spearman/R2.',
             'Finite-all-six policy per metric: discard a draw if either arm is nonfinite in any of six seeds; otherwise average six paired differences. Report retained count separately for every metric.',
             '95% percentile intervals use numpy.quantile([.025,.975], method="linear"); unadjusted, conditional on fitted models.',
             'Main pairs/readmission/ZCP RNG: new random.Random(0) EACH contrast, 115 successive randrange(115) calls per draw, draw-major order.',
             'Per-layer RNG: new numpy.random.default_rng(0), integers(0,115,size=(10000,115)), draw-major order.',
             'Abbreviations: T=Taylor, S=sensitivity, A=activation, C=CKA, G=gradient-flow, E=effective-rank. Unmarked arms are ZCP OFF; T+S is the OFF reference.',
             '\nSOURCE MEMBERS (raw CSV SHA256)', *provenance, '\nDESCRIPTIVE SIX-SEED RESULTS (mean [95% t CI], SD; full seed metric rows)']
    originals = {}
    for label, pred in data.items():
        values = metrics(pred, target)
        assert np.isfinite(values).all()
        originals[label] = values
        lines.append(label)
        for seed, row in zip(SEEDS, values):
            lines.append(f'  seed={seed} ' + ' '.join(f'{m}={v:.12f}' for m, v in zip(METRICS, row)))
        for j, metric in enumerate(METRICS):
            avg, sd = values[:, j].mean(), values[:, j].std(ddof=1)
            half = t.ppf(.975, 5) * sd / np.sqrt(6)
            lines.append(f'  {metric}: {avg:.12f} [{avg-half:.12f}, {avg+half:.12f}] SD={sd:.12f}')
    tables = [
        ('tab:taylor-paired-bootstrap-rmse', [('T+S', 'T OFF'), ('T+A', 'T OFF'), ('T+C', 'T OFF'), ('T+S+A', 'T+S'), ('T+S+C', 'T+S'), ('T+S+A', 'T OFF'), ('T+S+C', 'T OFF')]),
        ('tab:readmission-paired-five-metrics', [('T+G', 'T OFF'), ('T+E', 'T OFF'), ('T+S+G', 'T+S'), ('T+S+E', 'T+S')]),
        ('tab:zcp-paired-five-metrics', [('T ON', 'T OFF'), ('T+S ON', 'T+S')]),
        ('tab:per-layer-versus-averaged-taylor', [('individual T OFF', 'T OFF')])]
    results = {}
    sign = np.array([-1, -1, 1, 1, 1])
    for table, contrasts in tables:
        lines.append('\n' + table + ' (ALL FIVE METRICS)')
        for candidate, reference in contrasts:
            if table == 'tab:per-layer-versus-averaged-taylor':
                indices = np.random.default_rng(0).integers(0, 115, size=(B, 115))
            else:
                rng = random.Random(0)
                indices = np.array([[rng.randrange(115) for _ in range(115)] for _ in range(B)])
            draws = np.full((B, 5), np.nan)
            for start in range(0, B, 200):
                ix = indices[start:start+200]
                cm = metrics(data[candidate][:, ix], target[ix])
                rm = metrics(data[reference][:, ix], target[ix])
                valid = np.isfinite(cm).all(axis=0) & np.isfinite(rm).all(axis=0)
                delta = ((cm-rm)*sign).mean(axis=0)
                draws[start:start+len(ix)] = np.where(valid, delta, np.nan)
            point = ((originals[candidate]-originals[reference])*sign).mean(axis=0)
            lines.append(f'candidate={candidate}; reference={reference}')
            for j, metric in enumerate(METRICS):
                finite = draws[np.isfinite(draws[:, j]), j]
                assert len(finite) > 0
                lo, hi = np.quantile(finite, [.025, .975], method='linear')
                results[(candidate, reference, metric)] = (point[j], lo, hi)
                lines.append(f'  {metric}: delta={point[j]:.12f} CI95=[{lo:.12f}, {hi:.12f}] finite_draws={len(finite)}/{B}')
            print(f'Completed {candidate} / {reference}', flush=True)
    lines.append('\nSPOTCHECKS (compare rounded historical values; mismatches reported, never forced)')
    checks = [('T+S', 'T OFF', 'RMSE', (-.1436, -.5132, .2619), 4), ('T+A', 'T OFF', 'RMSE', (-.4808, -.8596, -.0899), 4), ('individual T OFF', 'T OFF', 'RMSE', (-.5717, -1.2748, .1059), 4)]
    for c, r, m, expected, digits in checks:
        actual = results[(c, r, m)]
        status = 'MATCH' if tuple(round(float(v), digits) for v in actual) == expected else 'MISMATCH'
        lines.append(f'{status}: {c}/{r} {m} actual={actual}; expected rounded={expected}')
    upper = results[('T+S+G', 'T+S', 'Spearman')][2]
    lines.append(f'{"MATCH" if round(float(upper), 6) == .000002 else "MISMATCH"}: T+S+G/T+S Spearman upper={upper:.12f}; expected rounded=0.000002; includes zero={results[("T+S+G", "T+S", "Spearman")][1] <= 0 <= upper}')
    text = '\n'.join(lines) + '\n'
    # Exclusive creation also prevents races; no other outputs are written.
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        out.write(text)
    assert OUTPUT.read_text(encoding='utf-8') == text
    print('\n'.join(lines[-5:]))
    print(f'VERIFIED report={OUTPUT}; bytes={OUTPUT.stat().st_size}; sha256={sha(OUTPUT.read_bytes())}')


if __name__ == '__main__':
    main()
