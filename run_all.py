"""Single entry point for the exported-data thesis reproduction workflow."""
import argparse
import json
from pathlib import Path
import importlib.util
import importlib.metadata
import shutil
import sys
import traceback

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
TABLES = [f'4.{i}' for i in range(1, 5)] + [f'6.{i}' for i in range(1, 17)] + [f'A.{i}' for i in range(1, 6)]
FIGURES = ['4.1', '5.1', '5.2', '5.3'] + [f'6.{i}' for i in range(1, 6)]


def preflight():
    from analysis.thesis_common import require, sha, safe_name
    for name in ['numpy', 'scipy', 'matplotlib', 'pandas', 'statsmodels', 'sklearn', 'pymupdf']:
        require(importlib.util.find_spec(name) is not None, 'Missing dependency: ' + name + '; install requirements-reproduce.txt')
    require(shutil.which('pdflatex') is not None, 'pdflatex is required for the recovered workflow diagram')
    manifest = json.loads((ROOT / 'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    require(manifest.get('schema') == 1 and bool(manifest.get('files')), 'Invalid input manifest')
    expected = manifest['files']
    for name, digest in expected.items():
        safe_name(name)
        require((ROOT / name).is_file() and sha(ROOT / name) == digest, 'Missing/changed frozen input: ' + name)
    return dict(status='ready', database_required=False, model_training=False, input_sha256=expected)


def reproduce(out, checked):
    from analysis.thesis_common import require, unpack_zip, write_table, sha
    from analysis import thesis_metadata, thesis_lemur, thesis_measurements, thesis_paired, thesis_nb201, thesis_figures
    out.mkdir(parents=True, exist_ok=False)
    tables_dir = out / 'tables'; tables_dir.mkdir()
    work = out / 'work'; work.mkdir()
    notes = ['Dependence summaries and heatmaps reaggregate exported per-cell estimates; raw dCor/MIC estimators and permutations are not rerun.',
             'Design tables 4.2-4.4 render documented protocol definitions; they are not new empirical measurements.',
             'Figures are regenerated with documented source/style differences, not a claim of pixel-identical thesis layout.',
             'Existing statistical analysis algorithms are reused, including distinct Python/NumPy RNG conventions.',
             'No candidate training, Qwen fitting, database operation, feature extraction, or initialization-calibrator refit is performed.']
    print('[1/7] Verify and unpack the numerical evidence packet', flush=True)
    packet = unpack_zip(ROOT / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip', work / 'packet')
    packet_manifest = json.loads((packet / 'reproduction_packet_manifest.json').read_text())['files']
    for name, meta in packet_manifest.items():
        require(sha(packet / name) == meta['sha256'], 'Packet payload hash mismatch: ' + name)
    print('[2/7] Corpus, LEMUR descriptive results and direct paired comparison', flush=True)
    tables = thesis_metadata.build_tables(ROOT)
    main_tables, sources = thesis_lemur.build_tables(ROOT)
    tables.update(main_tables)
    (out / 'main_sources.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    initial = next(r for r in tables['6.2'] if r['split'] == 'initial' and r['observations'] == 2 and r['layer'] == 'no-layer')
    if f"{initial['RMSE']:.4f}" != '11.4716':
        notes.append(f"Known manuscript last-digit discrepancy: initial 2-accuracy/no-layer RMSE recomputes to {initial['RMSE']:.12f}; thesis prints 11.4716. Recomputed value is retained.")
    print('[3/7] Dependence aggregation and measured timing cohorts', flush=True)
    measurements, cells = thesis_measurements.build_tables(ROOT)
    tables.update(measurements)
    print('[4/7] Fresh 10,000-draw combination, per-layer, LODO and LOFO analyses', flush=True)
    tables.update(thesis_paired.build_tables(ROOT, packet, work / 'paired'))
    blobs = [thesis_lemur.get(thesis_lemur.pools(ROOT)['main'], f'per-layer-16k-refit7-v1/results/s{s}/test_predictions.csv')[1] for s in range(42, 48)]
    pred, target, _ = thesis_lemur.matrix(blobs, 115)
    individual = thesis_lemur.metric_function(ROOT)(pred, target).mean(axis=0)
    averaged = next(r for r in tables['6.4'] if (r['observations'], r['zcp'], r['feature']) == (3, 'off', 'taylor'))
    for i, row in enumerate(tables['6.8']):
        row.update(individual=float(individual[i]), averaged=averaged[row['metric']])
    print('[5/7] NB201 primary, supporting mixed models, and classical paired analyses', flush=True)
    nb = thesis_nb201.build_tables(ROOT, work / 'nb201')
    classical_metrics = [r for r in nb['6.14'] if r['model'] != 'qwen']
    nb['6.14'] = [r for r in nb['6.14'] if r['model'] == 'qwen']
    for row in nb['6.16']:
        for condition in ['no-layer', 'taylor']:
            match = next(m for m in classical_metrics if m['dataset'] == row['dataset'] and m['model'] == row['model'] and m['condition'] == condition)
            row[condition + '_MAE'] = match['MAE']
    for row in nb['6.15']:
        report = json.loads(Path(row['report_path']).read_text())
        row.update(improvement=row['record_improvement'], record_ci_low=row['record_ci95_low'], record_ci_high=row['record_ci95_high'],
                   relative_reduction_percent=100 * row['record_improvement'] / row['no-layer_MAE'])
        for name in ['record_bootstrap', 'two_way_bootstrap', 'family_bootstrap']:
            low, high = report['robustness'][name]['ci95']
            row[name + '_low'], row[name + '_high'] = low, high
    failures = []
    for number, rows in nb.items():
        for row in rows:
            if row.get('verification_status') != 'verified':
                failures.append(f"{number}/{row.get('dataset')}/{row.get('model')}: numerical mismatch")
            if row.get('mixed_verification_status', 'verified') != 'verified':
                failures.append(f"{number}/{row.get('dataset')}: mixed-model mismatch/unverified fit")
            if 'report_path' in row:
                row['report_path'] = Path(row['report_path']).relative_to(out).as_posix()
    tables.update(nb)
    require(not failures, '; '.join(failures))
    require(set(tables) == set(TABLES), 'Table coverage mismatch: ' + str(set(TABLES) ^ set(tables)))
    print('[6/7] Write all 25 tables as CSV, Markdown and LaTeX', flush=True)
    table_manifest = {}
    for number in TABLES:
        provenance = ('Frozen exported snapshot count' if number == '4.1' else 'Documented design metadata' if number.startswith('4.') else
                      'Aggregation of exported per-cell dependence estimates' if number == '6.1' else
                      'Mean of actual saved timing measurements' if number == '6.13' else
                      'Recomputed from saved predictions; see source manifests and fresh analysis reports')
        table_manifest[number] = write_table(tables_dir, number, tables[number], provenance)
    (out / 'tables.json').write_text(json.dumps(tables, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('[7/7] Regenerate all nine figures as PDF and PNG', flush=True)
    figures = thesis_figures.build_figures(ROOT, out / 'figures', work / 'figure_build', tables, cells)
    require(set(figures) == set(FIGURES), 'Figure coverage mismatch')
    for name, digest in checked['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Frozen input changed during execution: ' + name)
    source_files = [ROOT / 'run_all.py', *sorted((ROOT / 'analysis').glob('thesis_*.py'))]
    result = dict(status='complete', tables=table_manifest, figures=figures, notes=notes,
                  input_sha256=checked['input_sha256'], source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
                  versions={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'statsmodels', 'matplotlib', 'scikit-learn', 'pymupdf']},
                  database_required=False, model_training=False)
    (out / 'complete.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (out / 'README.md').write_text('# Reproduced thesis artifacts\n\n25 tables (CSV/Markdown/LaTeX), 9 figures (PDF/PNG).\n\n' +
                                  '\n'.join('- ' + note for note in notes) + '\n\nSee complete.json for source hashes, coverage and environment.\n', encoding='utf-8')
    print('COMPLETE: 25 tables and 9 figures; original inputs unchanged. Output: ' + str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Print the thesis table/figure inventory')
    parser.add_argument('--check', action='store_true', help='Check frozen inputs and dependencies without producing results')
    parser.add_argument('--out', type=Path, help='New output directory, never an existing path')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(dict(tables=TABLES, figures=FIGURES)))
    elif args.check:
        print(json.dumps(preflight(), indent=2))
    else:
        if args.out is None:
            parser.error('Select --list or provide --out')
        if args.out.exists():
            parser.error('Existing output preserved: ' + str(args.out))
        checked = preflight()
        out = args.out.resolve()
        try:
            reproduce(out, checked)
        except Exception:
            if out.exists():
                (out / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise


if __name__ == '__main__':
    main()
