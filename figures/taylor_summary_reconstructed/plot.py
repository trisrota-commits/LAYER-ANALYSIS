"""Reconstruct the thesis Taylor MAE summary from published numerical evidence.

This is new plotting code, not the recovered historical generator. LEMUR's
interval uses the four-decimal values printed in thesis Table 6.5; NB201 values
are read from the committed primary-analysis results and rounded to four decimals.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def plot_rows():
    rows = [dict(dataset='LEMUR', improvement=0.4026, low=0.0507, high=0.7563,
                 n=115, interval='Unadjusted paired test-record bootstrap',
                 source='thesis_full_project-4.pdf, Table 6.5; displayed four-decimal values')]
    hashes = {}
    for label, relative in [
        ('CIFAR-10-valid', 'evidence/nb201-qwen-primary-v1/cifar10/analysis/results.json'),
        ('CIFAR-100', 'evidence/nb201-qwen-primary-v1/cifar100/analysis/results.json'),
        ('ImageNet16-120', 'evidence/nb201-imagenet16-primary-v1/analysis/results.json'),
    ]:
        path = REPO / relative
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        record = json.loads(data)['primary']['record']
        rows.append(dict(dataset=label, improvement=round(record['improvement'], 4),
                         low=round(record['ci95'][0], 4), high=round(record['ci95'][1], 4),
                         n=record['n'], interval='Unadjusted record-paired t interval; target >=20%', source=relative))
    # Independently check the LEMUR plotted point against saved prediction metrics.
    source = REPO / 'evidence/two-experiments-final-v1/descriptive.csv'
    with source.open(newline='', encoding='utf-8') as handle:
        means = {row['model']: float(row['mean']) for row in csv.DictReader(handle) if row['metric'] == 'MAE'}
    if round(means['no_layer'] - means['qwen_reference'], 4) != rows[0]['improvement']:
        raise ValueError('LEMUR point no longer matches saved descriptive metrics')
    hashes[source.relative_to(REPO).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    if any(not row['low'] <= row['improvement'] <= row['high'] for row in rows):
        raise ValueError('Invalid interval ordering')
    return rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwrite existing artwork')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, hashes = plot_rows()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    for y, row in enumerate(rows):
        ax.errorbar(row['improvement'], y,
                    xerr=[[row['improvement'] - row['low']], [row['high'] - row['improvement']]],
                    fmt='o', color='#356B9A', capsize=4, markersize=6, linewidth=1.4)
    ax.set_yticks(range(len(rows)), [row['dataset'] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.6)
    ax.set_xlabel('MAE improvement with Taylor (percentage points)')
    ax.axvline(0, color='0.4', linewidth=0.8)
    ax.grid(axis='x', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    args.out.mkdir(parents=True, exist_ok=False)
    for ext in ['pdf', 'png']:
        fig.savefig(args.out / ('taylor_summary.' + ext), dpi=200)
    plt.close(fig)
    with (args.out / 'plot_data.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(status='reconstructed plotting source; not historical code',
                      source_sha256=hashes, plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      precision='four decimal places, matching thesis table display',
                      caveat='LEMUR is exploratory and uses all 115 test records; NB201 uses target>=20 primary populations. Intervals differ in construction and are unadjusted.',
                      outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()})
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Created reconstructed PDF/PNG, plotted values and provenance:', args.out.resolve())


if __name__ == '__main__':
    main()
