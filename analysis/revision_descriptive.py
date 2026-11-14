"""Prediction-only descriptive paper tables; no fitting or legacy entrypoints."""
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from analysis import thesis_lemur as lemur
from analysis.thesis_common import require, sha

ARCHIVE = 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip'
SEEDS = list(range(42, 48))


def seed_rows(repo, pred, target, seeds, group, sources, **labels):
    require(len(pred) == len(seeds) == len(sources), 'Seed/source count mismatch')
    values = lemur.metric_function(repo)(pred, target)
    require(np.isfinite(values).all(), 'Nonfinite seed metric')
    rows = lemur.seed_intervals(repo, pred, target, group=group, **labels)
    for i, row in enumerate(rows):
        row.update(seeds=seeds, repetitions=list(range(1, len(seeds)+1)),
                   seed_values=values[:, i].tolist(), sources=sources, n_test=len(target),
                   df=len(seeds)-1, interval_kind='student_t95_repetitions' if len(set(seeds)) == 1 else 'student_t95_seeds')
    return rows


def baseline_functions(repo):
    """Compile only three audited arithmetic functions, never training/top-level code."""
    with zipfile.ZipFile(repo / ARCHIVE) as packet:
        with tarfile.open(fileobj=io.BytesIO(packet.read('article-timing-baselines-20261014-113919.tar.gz'))) as archive:
            members = [m for m in archive if m.name.endswith('/baseline_ci_85510_v1.py')]
            require(len(members) == 1, 'Baseline source missing/ambiguous')
            source = archive.extractfile(members[0]).read()
    tree = ast.parse(source)
    names = {'metric_values', 'seed_summary', 'bootstrap_metrics'}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in tree.body} == names, 'Baseline arithmetic functions changed')
    namespace = dict(require=require, math=math, METRICS=lemur.METRICS, SEEDS=SEEDS)
    exec(compile(tree, members[0].name, 'exec'), namespace)
    return namespace, dict(member=members[0].name, sha256=hashlib.sha256(source).hexdigest())


def run(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output already exists; refusing overwrite: '+str(out))
    require(not out.is_relative_to(repo / 'evidence') and not out.is_relative_to(repo / 'analysis')
            and not out.is_relative_to(repo / 'tests'), 'Output overlaps source inputs')
    frozen = json.loads((repo / 'INPUT_MANIFEST.json').read_text())['files']
    names = [ARCHIVE, 'evidence/reproduce_main_paired_tables.py']
    names += [f'evidence/two-experiments-final-v1/{model}-s{s}-predictions.csv'
              for model in ['epoch1', 'ridge', 'random_forest', 'hist_gradient_boosting'] for s in SEEDS]
    before = {name: sha(repo / name) for name in names}
    require(all(before[n] == frozen[n] for n in names), 'Frozen descriptive input changed')
    lemur.pools.cache_clear()
    lemur.metric_function.cache_clear()
    data, y, paths = lemur.load_main(repo)
    pools = lemur.pools(repo)
    tables, source_hashes = {}, {}

    def load(pool, suffixes, n=115):
        found = [lemur.get(pools[pool], suffix) for suffix in suffixes]
        sources = [pool+':'+name for name, _ in found]
        for key, (_, blob) in zip(sources, found):
            source_hashes[key] = hashlib.sha256(blob).hexdigest()
        p, target, keys = lemur.matrix([blob for _, blob in found], n)
        return p, target, sources

    def add(table, p, target, group, sources, seeds=SEEDS, **labels):
        tables.setdefault(table, []).extend(seed_rows(repo, p, target, seeds, group, sources, **labels))

    def main(table, obs, zcp, feature):
        sources = paths[f'{obs}/{zcp}/{feature}']
        for key in sources:
            pool, name = key.split(':', 1)
            source_hashes[key] = hashlib.sha256(pools[pool][name]).hexdigest()
        add(table, data[obs, zcp, feature], y, f'{obs}/{zcp}/{feature}', sources,
            observations=obs, zcp=zcp, feature=feature)

    for obs in [2, 3]:
        for arm, feature in [('no-layer', 'no-layer'), ('layer', 'all-seven')]:
            suffixes = [f'historical-22test-additions-v1/results/{obs}acc-{arm}-r{r}/test_predictions.csv'
                        if obs == 3 or (arm == 'no-layer' and r == 3)
                        else f'historical-2acc-reproduction-v2/results/{arm}-r{r}/test_predictions.csv' for r in [1, 2, 3]]
            p, target, sources = load('extra', suffixes, 22)
            add('S2', p, target, f'{obs}/on/{feature}', sources, [42]*3, observations=obs, zcp='on', feature=feature)
            main('S3', obs, 'on', feature)
    for table, obs, zcp in [('S10', 2, 'on'), ('S11', 2, 'off'), ('S12', 3, 'on'), ('S13', 3, 'off'), ('MainIV', 3, 'off')]:
        for feature in lemur.FEATURES:
            main(table, obs, zcp, feature)
    for zcp in ['off', 'on']:
        main('S14', 3, zcp, 'taylor')
    for feature in ['taylor', 'no-layer']:
        main('S15', 3, 'off', feature)
    for table, model in [('S15', 'epoch1'), ('MainVII', 'ridge'), ('MainVII', 'random_forest'), ('MainVII', 'hist_gradient_boosting')]:
        sources = [f'evidence/two-experiments-final-v1/{model}-s{s}-predictions.csv' for s in SEEDS]
        p, target, _ = lemur.matrix([(repo / n).read_bytes() for n in sources], 115)
        add(table, p, target, model, sources)
    main('MainVII', 3, 'off', 'taylor')
    combinations = [
        ('taylor-sensitivity', 'round2-taylor-pairs-3acc-off-v1', ['S17', 'S18', 'S20', 'S22']),
        ('taylor-activation', 'round2-taylor-pairs-3acc-off-v1', ['S17']),
        ('taylor-cka', 'round2-taylor-pairs-3acc-off-v1', ['S17']),
        ('taylor-sensitivity-activation', 'round3-triples-3acc-off-v1', ['S20']),
        ('taylor-sensitivity-cka', 'round3-triples-3acc-off-v1', ['S20']),
        ('taylor-gradient-flow', 'readmission-3acc-off-v1', ['S22']),
        ('taylor-effective-rank', 'readmission-3acc-off-v1', ['S22']),
        ('taylor-sensitivity-gradient-flow', 'readmission-3acc-off-v1', ['S22']),
        ('taylor-sensitivity-effective-rank', 'readmission-3acc-off-v1', ['S22']),
        ('taylor-sensitivity', 'taylor-sensitivity-zcp-on-v1', ['S18'])]
    for feature, batch, targets in combinations:
        p, target, sources = load('main', [f'{batch}/results/{feature}-s{s}/test_predictions.csv' for s in SEEDS])
        require(np.array_equal(target, y), 'Combination target mismatch')
        zcp = 'on' if batch == 'taylor-sensitivity-zcp-on-v1' else 'off'
        for table in targets:
            add(table, p, target, f'3/{zcp}/{feature}', sources, observations=3, zcp=zcp, feature=feature)
    for table in ['S17', 'S20', 'S22']:
        main(table, 3, 'off', 'taylor')
    main('S17', 3, 'off', 'no-layer')
    original, baseline_source = baseline_functions(repo)
    for obs in [2, 3]:
        p, target, sources = load('baseline', [f'xgboost_{obs}acc/predictions-s{s}.csv' for s in SEEDS])
        add('S4', p, target, f'XGBoost [{obs}]', sources)
        reference = original['seed_summary']([original['metric_values'](target, pred) for pred in p])
        for row in tables['S4'][-5:]:
            stats = reference[row['metric']]
            row.update(mean=stats['estimate'], low=stats['lower'], high=stats['upper'])
    for proxy in ['synflow', 'nwot']:
        p, target, sources = load('baseline', [f'/{proxy}_predictions.csv'])
        stats = original['bootstrap_metrics'](target, p[0], 10000, 20261007)
        for metric, stat in stats.items():
            tables['S4'].append(dict(group=proxy, metric=metric, mean=stat['estimate'], low=stat['lower'], high=stat['upper'],
                interval_kind='pointwise_record_percentile95', n_test=115, seeds=[0], seed_values=[stat['estimate']], sources=sources,
                bootstrap_seed=20261007, bootstrap_replicates=10000, valid_resamples=stat['valid_resamples'], invalid_resamples=stat['invalid_resamples']))
    # Timing is a measured cost table, not a prediction-metric or seed-CI table.
    from analysis.thesis_measurements import build_tables
    measured, _ = build_tables(repo)
    tables['S16'] = [dict(r, group=r['operation'], interval_kind='measured_mean_no_interval',
                         paper_location='caption' if r['operation'] in ['training', 'evaluation'] else 'body')
                     for r in measured['6.13']]
    tables['S16'].append(dict(group='no-layer', operation='no-layer', seconds=0.,
        boundary='zero additional layer extraction per checkpoint', paper_location='body',
        interval_kind='defined_zero_additional_cost'))
    require(before == {name: sha(repo / name) for name in names}, 'Inputs changed during calculation')
    out.mkdir(parents=True, exist_ok=False)
    caveat = ('Full precision of saved predictions/targets; no refitting, ensembling, target rerounding or clipping. '
              'Qwen targets retain source two-decimal precision, baselines retain raw target precision. '
              'Seed t intervals (df=5) describe fixed-test training variation; pilot nominal seed42 repetitions use df=2. '
              'Proxy intervals are pointwise paired-record percentiles, 10000 draws, default_rng(20261007), reset per proxy, '
              'batches of 500, finite draws per metric, original archived arithmetic. S15/S18/S20/MainIV/MainVII include '
              'additional descriptive seed intervals even where the paper prints means only. S16 is measured timing, not five prediction metrics.')
    outputs = {}
    for table, rows in tables.items():
        stem = out / ('table_'+table)
        stem.with_suffix('.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with stem.with_suffix('.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
        md = f'# {table}\n\n{caveat}\n\n| Group | Metric | Mean | 95% interval |\n|---|---|---:|---|\n'
        for row in rows:
            interval = f"[{row['low']:.10g}, {row['high']:.10g}]" if 'low' in row else 'not applicable'
            md += f"| {row['group']} | {row.get('metric', 'seconds')} | {row.get('mean', row.get('seconds')):.10g} | {interval} |\n"
        stem.with_suffix('.md').write_text(md, encoding='utf-8')
        outputs[table] = dict(rows=len(rows), groups=len({r['group'] for r in rows}), files=[stem.with_suffix(ext).name for ext in ['.csv', '.json', '.md']])
    require(before == {name: sha(repo / name) for name in names}, 'Inputs changed during export')
    receipt = dict(tables=outputs, frozen_inputs=before, input_preservation_verified=True,
                   member_sha256=source_hashes, baseline_source=baseline_source, caveats=caveat)
    receipt['output_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir())}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    (out / 'README.md').write_text('# Descriptive paper tables\n\n'+caveat+'\n\n'+ '\n'.join(f"- {t}: {v['groups']} groups, {v['rows']} long rows" for t, v in outputs.items())+'\n', encoding='utf-8')
    return receipt
