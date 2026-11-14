"""Portable, standard-library-only reporting of pinned recovered evidence.

No extraction, candidate imports, pickle loading, training or predictor fitting.
Receipts attest export integrity, not independent replication of the experiment.
"""
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import median

ARCHIVE_SHA256 = '1b354d5d45e65e287052f14e141169e1f23e11427d0be0f128b028344a105884'
DATASETS = ('cifar10', 'cifar100', 'imagenette')
ROOT = 'point4-all3-no-checkpoints-v1'
REQUIRED = {'study_report.json', 'measurement_summary.csv', 'candidate_outcomes.json',
            'rank_agreement.json', 'measurement_failures.json', 'scaling_checks.json'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe(name):
    if (not name or '\\' in name or ':' in name or name.startswith('/')
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Unsafe evidence path: ' + name)
    return name


def _verified(archive):
    # Hash and consume the same bytes (avoid a file replacement race).
    data = Path(archive).read_bytes()
    if _sha(data) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 differs from pinned recovered evidence')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for member in tar:
            name = _safe(member.name)
            if not member.isfile() or name in files:
                raise ValueError('Non-regular or duplicate archive member: ' + name)
            files[name] = tar.extractfile(member).read()
    manifest = json.loads(files['EXPORT_MANIFEST.json'])['files_sha256']
    if set(manifest) != set(files) - {'EXPORT_MANIFEST.json'}:
        raise ValueError('EXPORT_MANIFEST member inventory mismatch')
    for name, digest in manifest.items():
        _safe(name)
        if _sha(files[name]) != digest:
            raise ValueError('EXPORT_MANIFEST hash mismatch: ' + name)
    receipts = {}
    for dataset in DATASETS:
        base = f'{ROOT}/lemur-robustness-all3-v1-{dataset}/results/analysis/'
        receipt = json.loads(files[base + 'complete.json'])
        hashes = receipt.get('files_sha256', {})
        if receipt.get('status') != 'complete' or not REQUIRED.issubset(hashes):
            raise ValueError('Incomplete analysis receipt: ' + dataset)
        for name, digest in hashes.items():
            _safe(name)
            if '/' in name or _sha(files.get(base + name, b'')) != digest:
                raise ValueError('Analysis receipt hash/path mismatch: ' + name)
        receipts[dataset] = _sha(files[base + 'complete.json'])
    return files, receipts


def _stats(values):
    values = list(values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite statistic')
    return dict(n=len(values), median=median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def _eligible(row):
    return (row['candidate_status'] == 'complete'
            and int(row['n_successful_measurements']) == 6
            and int(row['coverage_variants_at_checkpoint']) == 1)


def _reference(row):
    return (row['mode'], row['augmentation'], row['loss_reduction']) == ('eval', 'off', 'mean')


def _numeric_tables(rows, dataset):
    eligible = [r for r in rows if _eligible(r) and _reference(r)]
    indexed = {}
    for r in eligible:
        key = (r['candidate'], int(r['epoch']), int(r['batch_size']))
        if key in indexed:
            raise ValueError('Ambiguous eligible coverage: ' + str(key))
        indexed[key] = r
    cvs, discrepancies, details = [], [], []
    for epoch in (1, 2, 3):
        refs = [r for r in eligible if int(r['epoch']) == epoch and int(r['batch_size']) == 32]
        values = [float(r['raw_cv']) for r in refs if r['raw_cv'] and math.isfinite(float(r['raw_cv']))]
        cvs.append(dict(dataset=dataset, epoch=epoch, unit='ratio', **_stats(values)))
        pairs = []
        for r in refs:
            other = indexed.get((r['candidate'], epoch, 128))
            if other is None or r['coverage_id'] != other['coverage_id']:
                continue
            numbers = [float(x[field]) for field in ('raw_mean', 'rescaled_mean') for x in (r, other)]
            if not all(math.isfinite(v) and v > 0 for v in numbers):
                continue
            raw = abs(math.log2(numbers[1] / numbers[0]))
            scaled = abs(math.log2(numbers[3] / numbers[2]))
            pairs.append((raw, scaled))
            details.append(dict(dataset=dataset, epoch=epoch, candidate=int(r['candidate']),
                                coverage_id=r['coverage_id'], raw=raw, rescaled=scaled))
        stats = dict(dataset=dataset, epoch=epoch, n=len(pairs), closer=sum(scaled < raw for raw, scaled in pairs))
        for i, label in enumerate(('raw', 'rescaled')):
            stats.update({label + '_' + k: v for k, v in _stats(p[i] for p in pairs).items() if k != 'n'})
        discrepancies.append(stats)
    return cvs, discrepancies, details


def run(archive, out):
    """Verify all evidence before creating out; return the full JSON report."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('Existing report preserved: ' + str(out))
    files, receipts = _verified(archive)
    tables = {name: [] for name in ('S47', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58')}
    sources, pair_details, timing_details, flags = {}, [], [], []
    for dataset in DATASETS:
        root = f'{ROOT}/lemur-robustness-all3-v1-{dataset}/results/'
        base = root + 'analysis/'
        source = {name[:-5]: json.loads(files[base + name]) for name in REQUIRED if name.endswith('.json')}
        sources[dataset] = source
        rows = list(csv.DictReader(io.StringIO(files[base + 'measurement_summary.csv'].decode())))
        outcomes = source['candidate_outcomes']
        completed = {r['index'] for r in outcomes if r['status'] == 'complete'}
        study = source['study_report']
        coverage = dict(dataset=dataset, selected_architectures=len(outcomes), completed_architectures=len(completed),
                        failed_architectures=sum(r['status'] == 'failed' for r in outcomes),
                        measurement_failures=len(source['measurement_failures']), summary_rows=len(rows),
                        eligible_summary_rows=sum(_eligible(r) for r in rows),
                        scaling_checks=len(source['scaling_checks']),
                        scaling_disagreements=sum(not r['approximately_equal'] for r in source['scaling_checks']))
        failures = source['measurement_failures']
        coverage.update(oom_batch32=sum(r['batch_size'] == 32 and 'out of memory' in r['error'].lower() for r in failures),
                        oom_batch128=sum(r['batch_size'] == 128 and 'out of memory' in r['error'].lower() for r in failures),
                        output_contract_failures=sum('Candidate must return a [batch,' in r['error'] for r in failures),
                        completed_missing_modules=len({r['candidate'] for r in rows if r['candidate_status'] == 'complete' and r['missing_modules']}))
        for key in ('selected_architectures', 'completed_architectures', 'measurement_failures', 'summary_rows', 'scaling_checks', 'scaling_disagreements'):
            if coverage[key] != study[key]:
                flags.append(dict(table='S53', dataset=dataset, field=key, computed=coverage[key], archived=study[key]))
        tables['S53'].append(coverage)
        tables['S54'].extend(dict(dataset=dataset, **r) for r in source['rank_agreement'] if r['loss_reduction'] == 'mean')
        cv, discrepancy, details = _numeric_tables(rows, dataset)
        tables['S55'].extend(cv)
        tables['S56'].extend(discrepancy)
        pair_details.extend(details)
        times = defaultdict(list)
        for name, data in files.items():
            if not name.startswith(root):
                continue
            match = re.fullmatch(r'candidate-(\d+)/measurements-e\d+\.jsonl', name[len(root):])
            if not match or int(match[1]) not in completed:
                continue
            for line in data.splitlines():
                r = json.loads(line)
                if r.get('status') == 'ok' and _reference(r) and r['batch_size'] in (32, 128):
                    value = float(r['seconds'])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('Invalid measurement time')
                    times[int(match[1]), r['batch_size']].append(value)
        for batch in (32, 128):
            medians = []
            for (candidate, b), values in sorted(times.items()):
                if b == batch:
                    medians.append(median(values))
                    timing_details.append(dict(dataset=dataset, batch_size=batch, candidate=candidate,
                                               successful_measurements=len(values), median_seconds=median(values)))
            tables['S57'].append(dict(dataset=dataset, batch_size=batch, unit='seconds', **_stats(medians)))
    # Explicitly retain the known paper discrepancy, never overwrite evidence.
    row = next(r for r in tables['S56'] if r['dataset'] == 'cifar100' and r['epoch'] == 2)
    if round(row['raw_median'], 3) != 1.979:
        flags.append(dict(table='S56', dataset='cifar100', epoch=2, field='raw_median',
                          computed=row['raw_median'], paper_display=1.979,
                          note='Recovered value rounds to 1.978, not the paper 1.979.'))
    for name in ('cifar10-loss-confidence-v1', 'nb201-loss-confidence-resume-v1', 'imagenet16-loss-confidence-partial-histgb-v1'):
        counts = Counter((r['dataset'], r['split']) for r in map(json.loads, files[name + '/results/snapshot.jsonl'].splitlines()))
        tables['S47'].extend(dict(source=name, dataset=d, split=s, n=n, metadata_only=True) for (d, s), n in sorted(counts.items()))
    for fold in json.loads(files['lemur-nested-qwen-v1/folds.json']):
        tables['S58'].append(dict(fold=fold['fold'], metadata_only=True,
                                 **{k.removesuffix('_ids') + '_n': len(v) for k, v in fold.items() if k.endswith('_ids')}))
    notes = [
        'Recovered-evidence reporting only; no new training, candidate execution, pickle loading or predictor fits.',
        'Archive pin, exact EXPORT_MANIFEST inventory/member hashes and all three analysis receipts verified before output.',
        'S53 recomputes archived coverage/failure/scaling counts; full source reports retain limitations and failures.',
        'S54 exports authenticated archived mean-loss rank agreements, including matched candidate IDs; not newly fitted ranks.',
        'S55: batch32/eval/off/mean, complete candidate, six successful repeats, one checkpoint coverage; within-candidate raw CV ratio, not percent.',
        'S56: same eligibility at both batch sizes, matched candidate/epoch/coverage, positive means; abs(log2(mean128/mean32)), separately rescaled.',
        'S57: median across candidates of each candidate median successful eval/off/mean measurement seconds, pooled over epochs; completed candidates only. Incomplete repeat coverage is allowed.',
        'S47/S58 are snapshot/fold metadata counts, not new training or validation of nested outer model results.',
        'Full precision is retained. Flags identify known paper/source mismatches; no exhaustive PDF comparison is claimed.',
    ]
    report = dict(archive_sha256=ARCHIVE_SHA256, payload_members=len(files)-1,
                  source_receipt_sha256=receipts, script_sha256=_sha(Path(__file__).read_bytes()),
                  notes=notes, flags=flags, tables=tables, source_reports=sources,
                  S56_pairs=pair_details, S57_candidate_medians=timing_details)
    # Serialize before writing, including finite-number validation.
    full_json = json.dumps(report, indent=2, allow_nan=False) + '\n'
    out.mkdir(parents=True)
    lines = ['# Recovered Point 4 and cohort/fold tables', ''] + ['- ' + n for n in notes]
    lines += ['', '## Mismatch flags', json.dumps(flags, indent=2), '']
    for table, rows in tables.items():
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / (table + '.csv')).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
        lines.extend(['## ' + table, '', '| ' + ' | '.join(fields) + ' |', '|' + '|'.join('---' for _ in fields) + '|'])
        for r in rows:
            lines.append('| ' + ' | '.join(str(r.get(k, '')).replace('|', '\\|') for k in fields) + ' |')
        lines.append('')
    (out / 'full_results.json').write_text(full_json, encoding='utf-8')
    (out / 'full_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    hashes = {p.name: _sha(p.read_bytes()) for p in sorted(out.iterdir())}
    receipt = dict(status='complete', archive_sha256=ARCHIVE_SHA256, payload_members=len(files)-1,
                   files_sha256=hashes, flags=len(flags), table_rows={k: len(v) for k, v in tables.items()})
    (out / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = run(args.archive, args.out)
    print(json.dumps({'output': str(args.out), 'flags': result['flags']}, indent=2))
