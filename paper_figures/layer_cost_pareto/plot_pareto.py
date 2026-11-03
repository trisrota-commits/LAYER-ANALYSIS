"""Plot measured checkpoint extraction cost versus updated primary-study RMSE.

Input: user-reported 96-candidate matched timing means and recorded 115-test
prediction means (seeds 42/43/44, seven-epoch final refit). No-layer cost is
zero additional extraction by definition. These are different cohorts.
Untimed pairs and the separate two-versus-three-accuracy comparison are excluded.
This is a frontier of mean estimates, not a test of statistical superiority.
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
BLUE = '#0072B2'
GRAY = '#777777'


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['arm'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty, uniquely named arms')
    for row in rows:
        for key in ('extraction_seconds', 'rmse_mean'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError((row['arm'], key, row[key]))
        for key, expected in (('timing_candidates', 96), ('test_records', 115), ('prediction_seeds', 3)):
            if int(row[key]) != expected:
                raise ValueError(('Unexpected reporting cohort', row['arm'], key))
    return rows


def pareto_frontier(rows):
    """Minimize both cost and RMSE; ties do not dominate one another."""
    def dominates(a, b):
        ax, ay = a['extraction_seconds'], a['rmse_mean']
        bx, by = b['extraction_seconds'], b['rmse_mean']
        return ax <= bx and ay <= by and (ax < bx or ay < by)
    return sorted(
        [r for r in rows if not any(dominates(other, r) for other in rows)],
        key=lambda r: (r['extraction_seconds'], r['rmse_mean']),
    )


def plot(rows):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    frontier = pareto_frontier(rows)
    efficient = {r['arm'] for r in frontier}
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.87)
    ax.set_axisbelow(True)
    ax.grid(color='#E4E4E4', linewidth=0.6)
    ax.set_xlim(-0.65, 26.0)
    ax.set_ylim(8.81, 9.49)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([8.9, 9.0, 9.1, 9.2, 9.3, 9.4])
    ax.set_xlabel('Mean extraction time per checkpoint (s; lower is better)', labelpad=9)
    ax.set_ylabel('Mean prediction RMSE (lower is better)', labelpad=9)

    def points(axes, size):
        for row in rows:
            is_frontier = row['arm'] in efficient
            axes.scatter(row['extraction_seconds'], row['rmse_mean'],
                         s=size, marker='D' if is_frontier else 'o',
                         color=BLUE if is_frontier else GRAY,
                         edgecolors='white', linewidths=0.7, zorder=4)
        axes.plot([r['extraction_seconds'] for r in frontier],
                  [r['rmse_mean'] for r in frontier],
                  color=BLUE, linestyle='--', linewidth=1.2, zorder=3)

    points(ax, 48)
    positions = {
        'No layer statistics': ((0.80, 9.047), 'No layer statistics', 'left'),
        'WeightWatcher': ((3.9, 8.918), 'WeightWatcher', 'left'),
        'CKA similarity': ((14.0, 9.365), 'CKA similarity', 'left'),
        'Effective rank': ((1.3, 9.346), 'Effective rank', 'left'),
        'Taylor importance': ((4.1, 9.005), 'Taylor importance', 'left'),
        'Gradient flow': ((1.55, 8.830), 'Gradient flow', 'left'),
        'Sensitivity': ((6.25, 9.239), 'Sensitivity', 'left'),
        'Activation statistics': ((1.3, 9.181), 'Activation statistics', 'left'),
        'All seven': ((23.8, 9.455), 'All seven', 'right'),
    }
    for row in rows:
        xytext, label, ha = positions[row['arm']]
        ax.annotate(label, (row['extraction_seconds'], row['rmse_mean']),
                    xytext=xytext, textcoords='data', ha=ha, va='center',
                    fontsize=10, color=BLUE if row['arm'] in efficient else '#333333',
                    arrowprops={'arrowstyle': '-', 'color': '#8A8A8A', 'lw': 0.65},
                    zorder=5)

    legend = [
        Line2D([0], [0], color=BLUE, linestyle='--', marker='D', markersize=6,
               linewidth=1.2, label='Pareto frontier (mean estimates)'),
        Line2D([0], [0], color=GRAY, linestyle='None', marker='o', markersize=6,
               label='Dominated mean estimates'),
    ]
    ax.legend(handles=legend, loc='lower left', bbox_to_anchor=(0, 1.015),
              frameon=False, ncol=2, fontsize=10, borderaxespad=0)

    # Show the small cost/error differences without distorting the main linear axes.
    zoom = ax.inset_axes([0.56, 0.14, 0.41, 0.45])
    zoom.set_facecolor('white')
    zoom.set_axisbelow(True)
    zoom.grid(color='#E8E8E8', linewidth=0.5)
    zoom.set_xlim(-0.035, 0.63)
    zoom.set_ylim(8.830, 9.020)
    zoom.set_xticks([0, 0.25, 0.5])
    zoom.set_yticks([8.85, 8.90, 8.95, 9.00])
    zoom.tick_params(labelsize=8)
    zoom.set_xlabel('Extraction (s)', fontsize=8.5, labelpad=3)
    zoom.set_ylabel('RMSE', fontsize=8.5, labelpad=3)
    zoom.set_title('Low-cost frontier (zoom)', fontsize=10, pad=6)
    points(zoom, 31)
    offsets = {
        'No layer statistics': ((7, 6), 'No layer', 'left'),
        'Taylor importance': ((-8, -15), 'Taylor importance', 'right'),
        'Gradient flow': ((-9, -1), 'Gradient flow', 'right'),
    }
    for row in frontier:
        offset, text, ha = offsets[row['arm']]
        zoom.annotate(text, (row['extraction_seconds'], row['rmse_mean']),
                      xytext=offset, textcoords='offset points', ha=ha,
                      va='center', color=BLUE, fontsize=8.5,
                      arrowprops={'arrowstyle': '-', 'color': '#888888', 'lw': 0.5})
    ax.add_patch(Rectangle((0, 8.830), 0.63, 0.190,
                           facecolor=BLUE, edgecolor=BLUE, alpha=0.07,
                           linewidth=0.8, zorder=1))
    fig.text(0.11, 0.04,
             'Timing: 96 matched candidates. Prediction: 115 test records, mean over three seeds.',
             fontsize=9, color='#444444')
    return fig, frontier


def main():
    rows = load_rows(ROOT / 'pareto_data.csv')
    figure, frontier = plot(rows)
    for suffix in ('png', 'pdf'):
        path = ROOT / ('layer_cost_pareto.' + suffix)
        figure.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(figure)
    print('Pareto frontier:', ', '.join(r['arm'] for r in frontier))
    print('Scope: measured per-checkpoint extraction costs; no extrapolation, no pair timings, no significance claim.')


if __name__ == '__main__':
    main()
