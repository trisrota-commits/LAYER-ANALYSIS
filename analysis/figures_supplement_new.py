"""New data-grounded S10--S12 renderers; not recovered historical generators.

No fitting or acquisition. S48/S50 are verified-table inputs, not independently
recomputed predictions here. S54/S56 are cross-checked against hash-pinned raw
acquisition evidence. Output creation is exclusive. Matplotlib imports are lazy.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

DATASETS = ('CIFAR-10-valid', 'CIFAR-100', 'ImageNet16-120 (partial)')
LABELS = ('CIFAR-10', 'CIFAR-100', 'ImageNet16-120*')
ARMS = ('A', 'A_LC', 'A_T', 'A_LC_T')
ARM_LABELS = ('A', 'A+LC', 'A+T', 'A+LC+T')
METRICS = ('MAE', 'RMSE', 'Pearson', 'Spearman', 'R2')
ROBUST_DATASETS = ('cifar10', 'cifar100', 'imagenette')
ROBUST_LABELS = ('CIFAR-10', 'CIFAR-100', 'ImageNette')
ALTERNATIVES = ((32, 'eval', 'on'), (32, 'train', 'off'),
                (128, 'eval', 'off'), (128, 'train', 'off'))
ALT_LABELS = ('Augmentation on', 'Train, batch 32', 'Eval, batch 128', 'Train, batch 128')
COLORS = ('#8aa3b8', '#9fc5ee', '#2879d7', '#194c8c')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique(rows, **keys):
    found = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one row for {keys}; got {len(found)}')
    return found[0]


def _excludes(low, high):
    return low > 0 or high < 0


def _interval(row, prefix):
    low, high = row[prefix + '_low'], row[prefix + '_high']
    if not all(math.isfinite(v) for v in (low, high)) or low > high:
        raise ValueError('Invalid interval')
    return low, high


def prepare(repo, tables_root):
    """Return full-precision coordinates and provenance before any output writes."""
    from analysis.revision_point4_tables import _verified, _numeric_tables, ROOT, ARCHIVE_SHA256
    repo, root = Path(repo), Path(tables_root)
    if (root / 'tables').is_dir():
        root = root / 'tables'
    tables, sources = {}, {}
    for name in ('S48', 'S50', 'S54', 'S56'):
        path = root / (name + '.json')
        data = path.read_bytes()
        tables[name] = json.loads(data)
        sources[name] = {'path': str(path.resolve()), 'sha256': _sha(data)}
    archive = repo / 'evidence/paper-missing-evidence-v1.tar.gz'
    files, receipts = _verified(archive)
    for dataset in ROBUST_DATASETS:
        base = f'{ROOT}/lemur-robustness-all3-v1-{dataset}/results/analysis/'
        raw = list(csv.DictReader(io.StringIO(files[base + 'measurement_summary.csv'].decode())))
        _, summaries, _ = _numeric_tables(raw, dataset)
        for r in summaries:
            if r != _unique(tables['S56'], dataset=dataset, epoch=r['epoch']):
                raise ValueError('S56 differs from pinned raw measurements')
        archived_ranks = json.loads(files[base + 'rank_agreement.json'])
        expected = [dict(dataset=dataset, **r) for r in archived_ranks if r['loss_reduction'] == 'mean']
        if expected != [r for r in tables['S54'] if r['dataset'] == dataset]:
            raise ValueError('S54 differs from pinned rank evidence')
    coords = {'S10': [], 'S11': [], 'S12': []}
    for metric in ('MAE', 'RMSE', 'Spearman'):
        for d, dataset in enumerate(DATASETS):
            rows = [_unique(tables['S48'], dataset=dataset, population='all', arm=a, metric=metric) for a in ARMS]
            best = (max if metric == 'Spearman' else min)(r['estimate'] for r in rows)
            for a, r in enumerate(rows):
                _interval(r, 'ci95')
                if r['n'] != (275, 275, 198)[d] or not math.isfinite(r['estimate']):
                    raise ValueError('S10 cohort/value mismatch')
                coords['S10'].append(dict(r, panel=metric, x=d + (a - 1.5) * .19,
                                         y=r['estimate'], best=r['estimate'] == best))
    for m, metric in enumerate(METRICS):
        for d, dataset in enumerate(DATASETS):
            r = _unique(tables['S50'], dataset=dataset, population='target_ge20',
                        contrast='A_LC -> A_LC_T', metric=metric)
            lo, hi = _interval(r, 'ci95')
            blo, bhi = _interval(r, 'bonferroni40')
            if r['n'] != (273, 272, 187)[d] or not blo <= lo <= hi <= bhi:
                raise ValueError('S11 cohort/interval mismatch')
            expected = r['reference'] - r['candidate'] if metric in ('MAE', 'RMSE') else r['candidate'] - r['reference']
            if not math.isclose(expected, r['improvement'], abs_tol=1e-12):
                raise ValueError('S11 improvement orientation mismatch')
            coords['S11'].append(dict(r, panel=m, x=r['improvement'], y=2-d,
                filled=_excludes(lo, hi), corrected_excludes_zero=_excludes(blo, bhi)))
    for d, dataset in enumerate(ROBUST_DATASETS):
        for epoch in (1, 2, 3):
            r = _unique(tables['S56'], dataset=dataset, epoch=epoch)
            for kind in ('raw', 'rescaled'):
                coords['S12'].append(dict(r, panel='magnitude', kind=kind,
                    x=r[kind + '_median'], y=8-(d*3+epoch-1)))
            for a, (batch, mode, aug) in enumerate(ALTERNATIVES):
                r = _unique(tables['S54'], dataset=dataset, epoch=epoch,
                    batch_size=batch, mode=mode, augmentation=aug, loss_reduction='mean')
                if len(r['candidate_indices']) != r['n_matched_architectures'] or not -1 <= r['spearman'] <= 1:
                    raise ValueError('S12 rank coverage/value mismatch')
                coords['S12'].append(dict(r, panel='rank', x=epoch-1, y=a, alternative=ALT_LABELS[a]))
    provenance = dict(renderer='new data-grounded renderer, not original historical source',
        reference='project_final_4_-21.pdf, pages 40 and 44; visually inspected',
        sources=sources, acquisition_archive={'path': str(archive.resolve()), 'sha256': ARCHIVE_SHA256,
                                             'verified_receipts': receipts},
        verification='S48/S50 table-based; S56 raw recomputation; S54 pinned archived rank agreement',
        notes=['S10: all retained test records; CIFAR test n=275 each; partial ImageNet16-120 test n=198 from cohort 799/1100.',
               'S11: target >=20%; operation-group paired bootstrap; positive favours Taylor; 40-interval Bonferroni family per dataset.',
               'S12: mean loss; magnitude eval/augmentation-off; matched eligible completed candidates, six successful repeats and stable coverage.',
               'S12 rank comparisons have distinct matched subsets; train batch 128 changes both mode and batch size.',
               'Other S12 printed-label differences retained from verified S54: CIFAR-100 augmentation-on epoch 3 is 0.999 (printed 1.000); ImageNette train batch32 epoch 1 is 0.983 (printed 0.984); ImageNette train batch128 epoch 3 is 0.956 (printed 0.957).',
               'Full precision retained: CIFAR-100 epoch-2 raw discrepancy rounds to 1.978, not printed 1.979.',
               'Full precision retained: CIFAR-10 train batch32 epoch-3 rho rounds to 0.938, not printed 0.939.',
               'No original generator found by bounded repository filename-reference search.'])
    return coords, provenance


def _clean(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=3)


def build_figures(coords):
    """Build editable Matplotlib figures; return figure dictionary for testing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figures = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, metric in zip(axes, ('MAE', 'RMSE', 'Spearman')):
            rows = [r for r in coords['S10'] if r['metric'] == metric]
            ymax = max(r['y'] for r in rows)
            for a, arm in enumerate(ARMS):
                chosen = [r for r in rows if r['arm'] == arm]
                bars = ax.bar([r['x'] for r in chosen], [r['y'] for r in chosen], width=.18,
                              color=COLORS[a], label=ARM_LABELS[a])
                for bar, r in zip(bars, chosen):
                    ax.text(bar.get_x()+bar.get_width()/2, r['y']+.025*ymax, f"{r['y']:.2f}",
                            rotation=90, ha='center', va='bottom', fontsize=7,
                            weight='bold' if r['best'] else 'normal')
            ax.set_xticks(range(3), LABELS, fontsize=8)
            ax.set_ylim(0, ymax*1.25)
            ax.set_title(metric + (' ↑' if metric == 'Spearman' else ' ↓'))
            _clean(ax)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
        fig.subplots_adjust(top=.82, bottom=.19, wspace=.24)
        fig.text(.5, .02, 'All retained test records: n=275, 275, 198. *Partial ImageNet16-120 cohort (799/1100).', ha='center', fontsize=8)
        figures['S10'] = fig
        fig, axes = plt.subplots(1, 5, figsize=(12, 3.5))
        for m, (ax, metric) in enumerate(zip(axes, METRICS)):
            for r in [r for r in coords['S11'] if r['panel'] == m]:
                ax.hlines(r['y'], r['bonferroni40_low'], r['bonferroni40_high'], color='.60', lw=1.2)
                ax.hlines(r['y'], r['ci95_low'], r['ci95_high'], color=COLORS[2], lw=4)
                ax.plot(r['x'], r['y'], 'o', color='black', mfc='black' if r['filled'] else 'white', ms=5)
            ax.axvline(0, color='.45', ls='--', lw=.8)
            ax.set_yticks([2, 1, 0], LABELS if m == 0 else ['', '', ''])
            ax.set_ylim(-.6, 2.6)
            ax.set_title('R²' if metric == 'R2' else metric)
            if metric in ('MAE', 'RMSE'):
                ax.set_xlabel('points')
            _clean(ax)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
        handles = [Line2D([], [], color=COLORS[2], lw=4, label='95% paired interval'),
            Line2D([], [], color='.60', lw=1.2, label='Bonferroni-adjusted (40 per dataset)'),
            Line2D([], [], marker='o', color='black', ls='', label='unadjusted excludes zero'),
            Line2D([], [], marker='o', color='black', mfc='white', ls='', label='includes zero')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53,.09), ncol=4, frameon=False, fontsize=8)
        fig.subplots_adjust(left=.15, right=.99, bottom=.34, top=.88, wspace=.35)
        fig.text(.5, .025, 'Targets ≥20% (n=273, 272, 187); positive favours Taylor. *Partial cohort.', ha='center', fontsize=8)
        figures['S11'] = fig
        fig = plt.figure(figsize=(12.8, 4.3))
        gs = fig.add_gridspec(1, 5, width_ratios=[2.6, .9, 1, 1, 1], wspace=.18)
        ax = fig.add_subplot(gs[0])
        rows = [r for r in coords['S12'] if r['panel'] == 'magnitude']
        for r in rows:
            if r['kind'] == 'raw':
                ax.hlines(r['y'], r['rescaled_median'], r['raw_median'], color='#d7e6f7', lw=3)
        for kind, color in (('raw', COLORS[0]), ('rescaled', COLORS[2])):
            chosen = [r for r in rows if r['kind'] == kind]
            ax.scatter([r['x'] for r in chosen], [r['y'] for r in chosen], color=color, s=24,
                       label='raw' if kind == 'raw' else 'rescaled by batch size', zorder=3)
        ax.set_yticks(range(8,-1,-1), [f'{d}, ep {e}' if e == 1 else f'ep {e}' for d in ROBUST_LABELS for e in (1,2,3)])
        ax.set_xlabel('median |log₂(T₁₂₈/T₃₂)|')
        ax.axvline(2, color='.5', ls='--', lw=.8)
        ax.set_xlim(left=0)
        ax.set_ylim(-.6,8.7)
        ax.legend(loc='lower center', bbox_to_anchor=(.5,1.01), ncol=2, frameon=False, fontsize=8)
        _clean(ax)
        for d, dataset in enumerate(ROBUST_DATASETS):
            ax = fig.add_subplot(gs[d+2])
            rows = [r for r in coords['S12'] if r['panel'] == 'rank' and r['dataset'] == dataset]
            matrix = [[next(r['spearman'] for r in rows if r['x'] == x and r['y'] == y) for x in range(3)] for y in range(4)]
            ax.imshow(matrix, cmap='Blues', vmin=.8, vmax=1, aspect='auto')
            for r in rows:
                ax.text(r['x'], r['y'], f"{r['spearman']:.3f}", ha='center', va='center',
                        fontsize=8, color='white' if r['spearman'] >= .94 else 'black')
            ax.set_xticks(range(3), ['ep 1','ep 2','ep 3'])
            ax.set_yticks(range(4), ALT_LABELS if d == 0 else ['', '', '', ''])
            ax.set_title(ROBUST_LABELS[d], fontsize=10)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=.12, right=.99, top=.77, bottom=.20)
        fig.text(.12,.94,'(a) Cross-batch magnitude, batch 32 vs 128', fontsize=10)
        fig.text(.50,.94,'(b) Rank agreement with reference (Spearman ρ)', fontsize=10)
        fig.text(.5,.035,'Mean loss; reference: batch 32, eval, augmentation off. Matched subsets differ by comparison.', ha='center', fontsize=8)
        figures['S12'] = fig
    return figures


def render(repo, tables_root, out):
    """Export three one-page PDFs, PNGs, coordinate JSON/CSV and provenance."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Existing output preserved: {out}')
    coords, provenance = prepare(repo, tables_root)
    figures = build_figures(coords)
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    inventory = {'figures': {}, 'provenance': 'provenance.json'}
    try:
        for name, stem in (('S10','fig_lc_arms'), ('S11','fig_lc_increment'), ('S12','fig_robust')):
            fig = figures[name]
            for ext in ('pdf', 'png'):
                fig.savefig(out / f'{stem}.{ext}', dpi=220, facecolor='white')
            (out / f'{stem}.plot-data.json').write_text(json.dumps(coords[name], indent=2, allow_nan=False), encoding='utf-8')
            keys = list(dict.fromkeys(k for row in coords[name] for k in row))
            with (out / f'{stem}.plot-data.csv').open('x', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in coords[name])
            inventory['figures'][name] = {'pdf': f'{stem}.pdf', 'png': f'{stem}.png',
                'json': f'{stem}.plot-data.json', 'csv': f'{stem}.plot-data.csv', 'coordinates': len(coords[name])}
        provenance['renderer_sha256'] = _sha(Path(__file__).read_bytes())
        provenance['output_sha256'] = {p.name: _sha(p.read_bytes()) for p in sorted(out.iterdir())}
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        (out / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    finally:
        for fig in figures.values():
            plt.close(fig)
    return inventory
