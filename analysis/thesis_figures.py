"""Render thesis figures from existing schematic source and regenerated table values."""
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

from analysis.thesis_common import require, run_script, sha

FEATURE_ORDER = ['Sensitivity', 'CKA similarity', 'Effective rank', 'Taylor importance', 'Gradient flow', 'WeightWatcher', 'Activation statistics']
DATASET_ORDER = ['celeba-gender', 'cifar-10', 'cifar-100', 'imagenette', 'mnist', 'places365', 'svhn']
LABELS = ['CelebA-Gender', 'CIFAR-10', 'CIFAR-100', 'ImageNette', 'MNIST', 'Places365', 'SVHN']


def save(fig, directory, number):
    stem = 'figure_' + number.replace('.', '_')
    for ext in ['pdf', 'png']:
        fig.savefig(directory / (stem + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_figures(repo, out, work, tables, cells):
    out.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    # The three methodology figures are existing source, executed in an isolated copy.
    diagrams = work / 'methodology'
    diagrams.mkdir()
    script = diagrams / 'make_diagrams.py'
    shutil.copyfile(repo / 'figures/methodology_v4/make_diagrams.py', script)
    run_script(script, [], work / 'methodology.log')
    for number, name in [('4.1', 'early_epoch_timeline'), ('5.2', 'taylor_calculation'), ('5.3', 'checkpoint_storage')]:
        for ext in ['pdf', 'png']:
            shutil.copyfile(diagrams / f'{name}.{ext}', out / ('figure_' + number.replace('.', '_') + '.' + ext))
    workflow = work / 'workflow'
    workflow.mkdir()
    source = repo / 'figures/workflow/implementation_workflow.tex'
    shutil.copyfile(source, workflow / source.name)
    with (work / 'workflow.log').open('x', encoding='utf-8') as log:
        command = [shutil.which('pdflatex'), '-interaction=nonstopmode', '-halt-on-error', source.name]
        result = subprocess.run(command, cwd=workflow, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Workflow LaTeX failed; inspect ' + str(work / 'workflow.log'))
    pdf = workflow / 'implementation_workflow.pdf'
    with pymupdf.open(pdf) as doc:
        require(len(doc) == 1, 'Workflow page count changed')
        doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(out / 'figure_5_1.png')
    shutil.copyfile(pdf, out / 'figure_5_1.pdf')
    # Summary uses the freshly recomputed direct comparison and primary analyses.
    summary = [next(row for row in tables['6.5'] if row['metric'] == 'MAE')]
    names = ['LEMUR']
    for row in tables['6.15']:
        # Adapter's flat rows are normalized by the caller to this view.
        names.append(row['dataset'])
        summary.append(dict(improvement=row['improvement'], low=row['record_ci_low'], high=row['record_ci_high']))
    require(len(summary) == 4, 'Summary requires exactly four comparisons')
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    for i, row in enumerate(summary):
        ax.errorbar(row['improvement'], i, xerr=[[row['improvement']-row['low']], [row['high']-row['improvement']]],
                    fmt='o', capsize=4, color='#356B9A')
    ax.set_yticks(range(4), names); ax.invert_yaxis(); ax.set_xlim(0, 1.6)
    ax.set_xlabel('MAE improvement with Taylor (percentage points)'); ax.grid(axis='x', alpha=.2)
    save(fig, out, '6.1')
    # Re-aggregate exported same-epoch dCor cells; do not claim new estimator runs.
    features = sorted({row['feature'] for row in cells}, key=lambda f: -np.mean([float(r['dcor']) for r in cells if r['feature'] == f]))
    means = [np.mean([float(r['dcor']) for r in cells if r['feature'] == f]) for f in features]
    fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    ax.barh(features[::-1], means[::-1], color='#356B9A'); ax.set_xlabel('Mean same-epoch distance correlation')
    save(fig, out, '6.2')
    for number, epochs in [('6.3', [1, 2, 3]), ('6.4', [50])]:
        fig, axes = plt.subplots(1, len(epochs), figsize=(5.2*len(epochs), 5.0), squeeze=False, constrained_layout=True)
        for ax, epoch in zip(axes[0], epochs):
            a = np.empty((7, 7)); q = np.empty((7, 7))
            for i, feature in enumerate(FEATURE_ORDER):
                for j, dataset in enumerate(DATASET_ORDER):
                    match = [r for r in cells if r['feature'] == feature and r['dataset'] == dataset and int(r['source_epoch']) == epoch]
                    require(len(match) == 1, 'Missing/duplicate heatmap cell')
                    a[i, j] = float(match[0]['dcor']); q[i, j] = float(match[0]['global_bh_fdr_q_value'])
            image = ax.imshow(a, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            ax.set_xticks(range(7), LABELS, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(7), FEATURE_ORDER, fontsize=8); ax.set_title(f'Epoch {epoch}')
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, f'{a[i,j]:.3f}' + ('*' if q[i,j] <= .05 else ''), ha='center', va='center',
                            fontsize=7, color='white' if a[i,j] < .55 else 'black')
        fig.colorbar(image, ax=list(axes[0]), label='Distance correlation', shrink=.8)
        save(fig, out, number)
    # Six-seed ZCP-ON cost plot from freshly recomputed means, not the older 3-seed CSV.
    timing = {row['operation']: row['seconds'] for row in tables['6.13']}
    mapping = {'no-layer': None, 'weightwatcher': 'weightwatcher', 'cka': 'cka_similarity',
               'effective-rank': 'effective_rank', 'taylor': 'taylor_importance', 'gradient-flow': 'gradient_flow',
               'sensitivity': 'sensitivity', 'activation': 'activation_statistics', 'all-seven': 'all_seven'}
    points = [(row['feature'], 0. if mapping[row['feature']] is None else timing[mapping[row['feature']]], row['RMSE'])
              for row in tables['6.4'] if row['observations'] == 3 and row['zcp'] == 'on']
    require(len(points) == 9, 'Cost plot condition count changed')
    frontier = sorted([p for p in points if not any(q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]) for q in points)], key=lambda p: p[1])
    require({p[0] for p in frontier} == {'no-layer', 'activation', 'taylor', 'gradient-flow'}, 'Pareto frontier differs from thesis')
    fig, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    ax.plot([p[1] for p in frontier], [p[2] for p in frontier], '--D', color='#356B9A', label='Non-dominated means')
    for name, cost, error in points:
        ax.scatter(cost, error, color='#356B9A' if any(p[0] == name for p in frontier) else '0.5')
        ax.annotate(name, (cost, error), xytext=(4, 6), textcoords='offset points', fontsize=8)
    ax.set_xlim(-.6, 28); ax.set_ylim(8.72, 9.5); ax.grid(alpha=.2)
    ax.set_xlabel('Mean extraction time per checkpoint (s)'); ax.set_ylabel('Mean prediction RMSE'); ax.legend()
    save(fig, out, '6.5')
    numbers = ['4.1', '5.1', '5.2', '5.3', '6.1', '6.2', '6.3', '6.4', '6.5']
    return {n: {ext: dict(path='figure_' + n.replace('.', '_') + '.' + ext,
                          sha256=sha(out / ('figure_' + n.replace('.', '_') + '.' + ext))) for ext in ['pdf', 'png']} for n in numbers}
