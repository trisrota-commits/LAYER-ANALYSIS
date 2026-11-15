"""Render retained dependence/Pareto figures (archival S3--S6), without fitting.

Pilot runtimes are archived Trainer.train_runtime (training, including in-training
validation, excluding held-out generation), NOT original workstation wall times.
The replay contracts explicitly warn that the cluster runtime differs. Figures
are reconstructions, not digitizations of the older PNGs embedded in the paper.
"""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from analysis.thesis_measurements import build_tables
from analysis.thesis_figures import FEATURE_ORDER, DATASET_ORDER, LABELS

PACKET = 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip'
NESTED = 'overleaf-experiment-logs-20261015-100424.zip'
COST = {'no-layer': None, 'weightwatcher': 'weightwatcher', 'cka': 'cka_similarity',
        'effective-rank': 'effective_rank', 'taylor': 'taylor_importance',
        'gradient-flow': 'gradient_flow', 'sensitivity': 'sensitivity',
        'activation': 'activation_statistics', 'all-seven': 'all_seven'}
DISPLAY = {'activation': 'Dead-activation fraction', 'cka': 'CKA redundancy',
           'effective-rank': 'Effective rank ratio', 'no-layer': 'No layer',
           'taylor': 'Taylor', 'gradient-flow': 'Gradient flow',
           'weightwatcher': 'WeightWatcher', 'all-seven': 'All seven', 'sensitivity': 'Sensitivity'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(root, name):
    root = Path(root)
    path = root / 'tables' / (name + '.json')
    if not path.exists():
        path = root / (name + '.json')
    rows = json.loads(path.read_text(encoding='utf-8'))
    return rows, path


def validate_metrics(rows, seeds, n_test):
    for r in rows:
        require(r['seeds'] == seeds and r['n_test'] == n_test,
                'Unexpected seed/test cohort: ' + r['group'])
        values = np.asarray(r['seed_values'], float)
        require(len(values) == len(seeds) and np.isfinite(values).all(), 'Invalid repetition values')
        mean = float(values.mean())
        half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        require(np.allclose([r['mean'], r['low'], r['high']], [mean, mean-half, mean+half], atol=1e-10, rtol=0),
                'Metric mean/Student t CI disagrees: ' + r['group'])


def pilot_data(repo, root, observations):
    rows, path = table(root, 'S2')
    rows = [r for r in rows if r['observations'] == observations and r['zcp'] == 'on']
    validate_metrics(rows, [42, 42, 42], 22)
    require(len(rows) == 10 and {r['feature'] for r in rows} == {'no-layer', 'all-seven'}, 'Pilot requires both arms and all five metrics')
    runtimes = []
    with zipfile.ZipFile(repo / PACKET) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(NESTED))) as archive:
            for feature in ['no-layer', 'all-seven']:
                row = next(r for r in rows if r['feature'] == feature and r['metric'] == 'RMSE')
                logs, values, digests = [], [], []
                for source in row['sources']:
                    name = source.removeprefix('extra:').rsplit('/', 1)[0] + '/full.log'
                    raw = archive.read(name)
                    matches = re.findall(r"['\"]train_runtime['\"]\s*:\s*['\"]?([0-9.eE+\-]+)", raw.decode())
                    require(len(matches) == 1, f'Expected one Trainer train_runtime in {name}; found {len(matches)}')
                    values.append(float(matches[0])); logs.append(name)
                    digests.append(hashlib.sha256(raw).hexdigest())
                mean = float(np.mean(values)); half = float(t.ppf(.975, 2)*np.std(values, ddof=1)/np.sqrt(3))
                runtimes.append(dict(feature=feature, mean=mean, low=mean-half, high=mean+half,
                                     values=values, sources=logs, source_sha256=digests))
    return dict(metrics=rows, runtimes=runtimes, observations=observations,
                table=str(path), runtime_boundary='Trainer.train_runtime seconds; includes training validation; excludes held-out generation',
                cohort='22 test records; three nominal seed-42 cluster replay repetitions; not original workstation timings',
                interval='95% Student t over repetitions (df=2); not independent-seed uncertainty')


def pilot_figure(data):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.7), layout='constrained')
    arms = ['no-layer', 'all-seven']
    colors = ['#2878A5', '#D17732']
    groups = [('Runtime', ['runtime']), ('Prediction error', ['RMSE', 'MAE']),
              ('Correlation', ['Pearson', 'Spearman']), ('Predictive fit', ['R2'])]
    for ax, (title, metrics) in zip(axes, groups):
        for i, arm in enumerate(arms):
            rr = []
            for metric in metrics:
                rr.append(next(r for r in (data['runtimes'] if metric == 'runtime' else data['metrics'])
                               if r['feature'] == arm and (metric == 'runtime' or r['metric'] == metric)))
            means = np.array([r['mean'] for r in rr])
            ax.bar(np.arange(len(metrics)) + (i-.5)*.34, means, width=.34, color=colors[i],
                   yerr=[means-[r['low'] for r in rr], [r['high'] for r in rr]-means], capsize=3,
                   label=DISPLAY[arm])
        ax.set_xticks(range(len(metrics)), ['Training' if m == 'runtime' else m for m in metrics])
        ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.set_ylabel('Seconds' if title == 'Runtime' else 'Percentage points' if title == 'Prediction error' else 'Dimensionless')
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Pilot reconstruction: {data['observations']} accuracy observations, ZCP on\n"
                 '22 test records; three same-seed replay repetitions; bars: means and 95% t intervals\n'
                 'Runtime: archived Trainer training seconds, not candidate acquisition or held-out generation', fontsize=11)
    return fig


def checked_cells(cells):
    require(len(cells) == 196, 'Expected 196 same-epoch dCor cells')
    expected = {(d, e, f) for d in DATASET_ORDER for e in (1, 2, 3, 50) for f in FEATURE_ORDER}
    require({(r['dataset'], int(r['source_epoch']), r['feature']) for r in cells} == expected, 'dCor grid incomplete/duplicated')
    p = np.array([float(r['analytical_p_value']) for r in cells]); order = np.argsort(p)
    q = np.empty(len(p)); q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1])
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid analytical p values')
    require(np.allclose(q, [float(r['global_bh_fdr_q_value']) for r in cells], rtol=0, atol=1e-12), 'Global 196-test BH values disagree')
    require(all(0 <= float(r['dcor']) <= 1 and r['status'] == 'ok' for r in cells), 'Invalid dCor')
    return cells


def dependence_figure(cells, epochs=None):
    checked_cells(cells)
    if epochs is None:
        points = sorted([(f, float(np.mean([float(r['dcor']) for r in cells if r['feature'] == f]))) for f in FEATURE_ORDER], key=lambda x:x[1])
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        ax.barh([p[0] for p in points], [p[1] for p in points], color='#2878A5')
        ax.set_xlabel('Mean same-epoch distance correlation (dimensionless)')
        ax.set_title('28 dataset–checkpoint estimates per feature\nSeven datasets × epochs 1, 2, 3, 50', fontsize=11)
        return fig, dict(points=[dict(feature=f, mean=v, cells=28) for f,v in points], cells=cells)
    fig, axes = plt.subplots(1, len(epochs), figsize=(6*len(epochs)+1, 5.3), squeeze=False, layout='constrained')
    grids = []
    multi=len(epochs)>1
    tick_size=13 if multi else 9
    cell_size=13 if multi else 9
    for panel,(ax, epoch) in enumerate(zip(axes[0], epochs)):
        selected = [next(r for r in cells if r['dataset'] == d and r['feature'] == f and int(r['source_epoch']) == epoch) for f in FEATURE_ORDER for d in DATASET_ORDER]
        values = np.array([float(r['dcor']) for r in selected]).reshape(7,7)
        q = np.array([float(r['global_bh_fdr_q_value']) for r in selected]).reshape(7,7)
        image = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(7), LABELS, rotation=45, ha='right', fontsize=tick_size)
        ax.set_yticks(range(7), FEATURE_ORDER if panel==0 else ['']*7, fontsize=tick_size)
        ax.set_title(f'Epoch {epoch}',fontsize=17 if multi else 12)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{values[i,j]:.3f}' + ('*' if q[i,j] <= .05 else ''), ha='center', va='center', fontsize=cell_size, color='white' if values[i,j] < .55 else 'black')
        grids.append(dict(epoch=epoch, values=values.tolist(), q=q.tolist(), cells=selected))
    bar=fig.colorbar(image, ax=list(axes[0]), shrink=.8)
    bar.set_label('Same-epoch dCor',fontsize=tick_size);bar.ax.tick_params(labelsize=tick_size)
    fig.suptitle('* Global BH q ≤ 0.05 across 196 analytical tests; complete-case architecture records, not images', fontsize=14 if multi else 10)
    return fig, dict(grids=grids, datasets=DATASET_ORDER, features=FEATURE_ORDER, bh_family=196)


def nondominated(points):
    return sorted([p for p in points if not any(q['seconds'] <= p['seconds'] and q['rmse'] <= p['rmse'] and
                  (q['seconds'] < p['seconds'] or q['rmse'] < p['rmse']) for q in points)], key=lambda p:p['seconds'])


def pareto_data(root, timing):
    rows, path = table(root, 'S12')
    rows = [r for r in rows if r['observations'] == 3 and r['zcp'] == 'on' and r['metric'] == 'RMSE']
    validate_metrics(rows, [42,43,44,45,46,47], 115)
    require(len(rows) == 9 and {r['feature'] for r in rows} == set(COST), 'Expected nine ZCP-on arms')
    costs = {r['operation']:r for r in timing}
    points = []
    for r in rows:
        operation = COST[r['feature']]
        if operation is not None:
            require(costs[operation]['n'] == 96 and costs[operation]['boundary'] == 'one checkpoint', 'Incorrect timing cohort/boundary')
        points.append(dict(feature=r['feature'], seconds=0. if operation is None else costs[operation]['seconds'], rmse=r['mean'], metric_source=r))
    return dict(points=points, frontier=[r['feature'] for r in nondominated(points)], timing=timing, table=str(path),
                interpretation='Means only, not statistical superiority; 115 prediction architectures, six seeds; separate 96-candidate matched layer timing cohort; zero means no additional layer extraction; excludes training and ZCP costs')


def pareto_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={'width_ratios':[1.6,1]}, layout='constrained')
    points = data['points']; frontier = nondominated(points)
    for ax in axes:
        ax.plot([p['seconds'] for p in frontier], [p['rmse'] for p in frontier], '--', color='#2878A5')
        for p in points:
            front = p in frontier
            ax.scatter(p['seconds'], p['rmse'], marker='D' if front else 'o', color='#2878A5' if front else '#888888')
            if ax is axes[0] or front:
                # Deliberately separate the tightly packed low-cost labels.
                offsets = {'activation': (5,14), 'taylor': (5,0), 'gradient-flow': (5,-13), 'no-layer': (6,7)}
                ax.annotate(DISPLAY[p['feature']], (p['seconds'], p['rmse']), xytext=offsets.get(p['feature'], (4,7)), textcoords='offset points', fontsize=8)
        ax.set_xlabel('Additional layer extraction per checkpoint (s)'); ax.set_ylabel('Mean RMSE (percentage points)'); ax.grid(alpha=.2)
    axes[0].set_title('Full range: ◆ non-dominated means; ● dominated means', fontsize=10)
    axes[0].set_xlim(-.6, max(p['seconds'] for p in points)*1.18)
    axes[0].set_ylim(min(p['rmse'] for p in points)-.07, max(p['rmse'] for p in points)+.05)
    axes[1].set_title('Low-cost frontier (zoom)', fontsize=10)
    axes[1].set_xlim(-.03, max(p['seconds'] for p in frontier)*1.5)
    ys = [p['rmse'] for p in frontier]; axes[1].set_ylim(min(ys)-.05,max(ys)+.05)
    fig.suptitle('Three accuracy observations, ZCP on: six-seed prediction means (115 architectures)\n'
                 'Extraction: 96 matched benchmark candidates; zero is no additional layer cost, not free prediction', fontsize=11)
    return fig


def render(repo, tables_root, out):
    """Create a NEW directory; return archival S3..S6 with exact blockers.

    Every successful figure has PDF/PNG, full plotted data/provenance JSON, and
    SHA256 receipts. Source bytes are checked again after rendering. Missing or
    inconsistent evidence blocks only dependent figures; never fabricated data.
    """
    repo, tables_root, out = Path(repo), Path(tables_root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [repo / PACKET]
    for name in ('S12',):
        try:
            _, path = table(tables_root, name); sources.append(path)
        except (OSError, ValueError):
            pass
    before = {str(p):sha(p) for p in sources if p.exists()}
    inventory = {}
    measurements = None
    def measurement():
        nonlocal measurements
        if measurements is None:
            measurements = build_tables(repo)
        return measurements
    for number in range(3,7):
        key = f'S{number}'; fig = None
        try:
            if number <= 5:
                _, cells = measurement()
                fig, data = dependence_figure(cells, None if number == 3 else [1,2,3] if number == 4 else [50])
            else:
                tables, _ = measurement()
                data = pareto_data(tables_root, tables['6.13']); fig = pareto_figure(data)
            data['input_sha256'] = before
            files = {}
            for ext in ('pdf', 'png'):
                path = out / f'{key}.{ext}'
                fig.savefig(path, dpi=200, bbox_inches='tight')
                files[ext] = dict(path=path.name, sha256=sha(path))
            path = out / f'{key}.json'
            with path.open('x', encoding='utf-8') as f:
                json.dump(data, f, indent=2, allow_nan=False)
            files['data'] = dict(path=path.name, sha256=sha(path))
            inventory[key] = dict(status='rendered', files=files, outputpaths=[files[e]['path'] for e in ('pdf','png')], datafiles=[path.name])
        except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
            inventory[key] = dict(status='blocked', blocker=f'{type(exc).__name__}: {exc}', outputpaths=[], datafiles=[])
        finally:
            if fig is not None:
                plt.close(fig)
    require(all(sha(p) == h for p,h in before.items()), 'INPUT INTEGRITY FAILURE: source changed during rendering')
    with (out / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump(dict(figures=inventory, input_sha256=before, input_integrity='unchanged'), f, indent=2)
    return inventory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path); parser.add_argument('tables_root', type=Path); parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.repo, args.tables_root, args.out), indent=2))
