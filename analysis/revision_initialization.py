"""S41--S45 from frozen predictions; does not refit initialization calibrators."""
import csv
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from analysis import thesis_lemur as source
from analysis.thesis_common import csv_rows, require, sha

METRICS = source.METRICS
DRAWS = 10000
INPUTS = ['evidence/overleaf-table-evidence-reproducible-20261019-v2.zip',
          'evidence/initialization-proxy-combinations-reproduced-v1/predictions.csv',
          'evidence/initialization-proxy-combinations-reproduced-v1/results.json',
          'evidence/reproduce_main_paired_tables.py']


def run(repo, out):
    repo, out = Path(repo), Path(out)
    if out.exists():
        raise FileExistsError('Existing output preserved: ' + str(out))
    frozen = json.loads((repo / 'INPUT_MANIFEST.json').read_text())['files']
    hashes = {name: sha(repo / name) for name in INPUTS}
    require(all(digest == frozen[name] for name, digest in hashes.items()), 'Frozen initialization inputs changed')
    pool = source.pools(repo)
    rows = csv_rows(source.get(pool['main'], 'init-taylor-comparison-v1/results-successful-only-v1/test_predictions.csv')[1])
    ids = [r['run_id'] for r in rows]
    require(len(ids) == len(set(ids)) == 86, 'Initialization cohort mismatch')
    rounded = np.array([float(r['gt_best_accuracy']) for r in rows])
    init = np.array([[float(r['pred_best_accuracy']) for r in rows]])
    qwen = np.array([[float(r[f'taylor_off_s{s}']) for r in rows] for s in range(42, 48)])
    proxies = {name: {r['run_id']: r for r in csv_rows(source.get(pool['baseline'], f'/{name}_predictions.csv')[1])}
               for name in ('synflow', 'nwot')}
    target = np.array([float(proxies['synflow'][rid]['gt_best_accuracy']) for rid in ids])
    require(np.array_equal(np.round(target, 2), rounded), 'Rounded/unrounded target mismatch')
    predictions = {'Taylor': init}
    for name, label in [('synflow', 'SynFlow'), ('nwot', 'NASWOT')]:
        for i, rid in enumerate(ids):
            r = proxies[name][rid]
            require(float(r['gt_best_accuracy']) == target[i] and r['architecture_id'] == rows[i]['architecture_id']
                    and r['dataset'] == rows[i]['dataset'], 'Proxy identity/target mismatch')
        predictions[label] = np.array([[float(proxies[name][rid]['pred_best_accuracy']) for rid in ids]])
    common_rows = csv_rows(repo / INPUTS[1])
    require([r['record_id'] for r in common_rows] == ids, 'Common calibration identity mismatch')
    require(np.array_equal([float(r['target']) for r in common_rows], target), 'Common calibration target mismatch')
    common = {name: np.array([[float(r[name]) for r in common_rows]])
              for name in ('Taylor', 'SynFlow', 'NASWOT', 'Taylor+SynFlow', 'Taylor+NASWOT')}
    contexts = [dict(table='S41', target=rounded, predictions={'Early-training Taylor/Qwen': qwen, 'Initialization Taylor': init},
                     pairs=[('Early-training Taylor/Qwen', 'Initialization Taylor')], precision='two-decimal target'),
                dict(table='S42', paired_table='S43', target=target, predictions=predictions,
                     pairs=[('SynFlow', 'Taylor'), ('NASWOT', 'Taylor')], precision='unrounded target'),
                dict(table='S44', paired_table='S45', target=target, predictions=common,
                     pairs=[('SynFlow', 'Taylor+SynFlow'), ('Taylor', 'Taylor+SynFlow'),
                            ('NASWOT', 'Taylor+NASWOT'), ('Taylor', 'Taylor+NASWOT')], precision='unrounded target')]
    metric = source.metric_function(repo)
    indices = np.random.default_rng(0).integers(0, len(ids), size=(DRAWS, len(ids)))
    sign = np.array([-1., -1., 1., 1., 1.])
    descriptive, paired, arrays = [], [], {}
    for context in contexts:
        y = context['target']; values = {}; bootstrap = {}
        for name, pred in context['predictions'].items():
            require(np.isfinite(pred).all() and np.isfinite(y).all(), 'Nonfinite initialization input')
            values[name] = metric(pred, y).mean(axis=0)
            chunks = []
            for start in range(0, DRAWS, 200):
                ix = indices[start:start + 200]
                chunks.append(metric(pred[:, ix], y[ix]).mean(axis=0))
            bootstrap[name] = np.concatenate(chunks)
            for mi, m in enumerate(METRICS):
                descriptive.append(dict(table=context['table'], model=name, metric=m, value=float(values[name][mi]),
                                        n=len(ids), target_precision=context['precision']))
        for reference, candidate in context['pairs']:
            table = context.get('paired_table', context['table'])
            point = (values[candidate] - values[reference]) * sign
            draws = (bootstrap[candidate] - bootstrap[reference]) * sign
            arrays[f'{table}__{reference}__{candidate}'] = draws
            for mi, m in enumerate(METRICS):
                finite = draws[:, mi][np.isfinite(draws[:, mi])]
                require(len(finite) > 0, 'No finite bootstrap differences')
                low, high = np.quantile(finite, [.025, .975])
                paired.append(dict(table=table, reference=reference, candidate=candidate, metric=m,
                                   improvement=float(point[mi]), low=float(low), high=float(high), n=len(ids),
                                   draws=DRAWS, valid_draws=len(finite), target_precision=context['precision']))
        print('COMPUTED', context['table'], context.get('paired_table', ''), flush=True)
    require(len(descriptive) == 50 and len(paired) == 35, 'Incomplete initialization coverage')
    require(all(sha(repo / name) == digest for name, digest in hashes.items()), 'Input changed during analysis')
    notes = ['Predictions and calibrations are fixed; this mode does not refit the 852-record calibrators.',
             'S41 uses the historical two-decimal targets; S42--S45 use common unrounded targets.',
             'S41 reports means over the six fitted Qwen seeds, not an ensemble prediction.',
             'All intervals are unadjusted paired test-record percentile intervals; no seed resampling.',
             'Positive differences favor the candidate; valid-draw counts are metric-specific.']
    result = dict(descriptive=descriptive, paired=paired, record_ids=ids, input_sha256=hashes,
                  source_sha256={p.name: sha(p) for p in [Path(__file__), Path(source.__file__)]},
                  method=f'default_rng(0), {DRAWS} paired record draws; metrics per fitted predictor then mean.',
                  notes=notes, versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy')})
    out.mkdir(parents=True, exist_ok=False)
    for name, data in [('descriptive.csv', descriptive), ('paired_intervals.csv', paired)]:
        with (out / name).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    with (out / 'bootstrap_differences.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    text = ['# Initialization comparisons', ''] + ['- ' + note for note in notes]
    text += ['', '| Table | Model | Metric | Value |', '|---|---|---|---:|']
    text += [f"| {r['table']} | {r['model']} | {r['metric']} | {r['value']:.6f} |" for r in descriptive]
    text += ['', '| Table | Reference -> candidate | Metric | Improvement | 95% CI | Valid draws |', '|---|---|---|---:|---|---:|']
    text += [f"| {r['table']} | {r['reference']} -> {r['candidate']} | {r['metric']} | {r['improvement']:+.6f} | [{r['low']:.6f}, {r['high']:.6f}] | {r['valid_draws']} |" for r in paired]
    for name, content in [('results.json', json.dumps(result, indent=2, allow_nan=False) + '\n'), ('results.md', '\n'.join(text) + '\n')]:
        with (out / name).open('x', encoding='utf-8') as f:
            f.write(content)
    with (out / 'complete.json').open('x', encoding='utf-8') as f:
        json.dump(dict(status='complete', files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'complete.json'}), f, indent=2)
    print('COMPLETE: S41--S45, 50 descriptive and 35 paired metric rows;', out, flush=True)
