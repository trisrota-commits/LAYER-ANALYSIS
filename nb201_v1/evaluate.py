"""NB201 classical fast tier using the unchanged supervisor panel implementation.

The internal legacy best_accuracy key holds the explicitly labelled final target
for CIFAR100. No prompt generation uses this adaptation. Originals are read-only.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'original'))
import panel


def paired_mae(y, control, taylor, families, draws=10000):
    """Positive = Taylor improvement; average seed-wise absolute errors.

    IID-record and operation-multiset cluster percentile bootstraps, seed 0.
    Eight endpoints: two datasets x two classical models x two populations.
    No seed resampling, prediction ensembling, or claim of primary-script parity.
    """
    assert control.shape == taylor.shape == (6, len(y))
    delta = (np.abs(control - y) - np.abs(taylor - y)).mean(axis=0)
    assert np.isfinite(delta).all() and len(delta) > 1
    rng = np.random.default_rng(0)
    iid = delta[rng.integers(len(y), size=(draws, len(y)))].mean(axis=1)
    groups = np.unique(families)
    assert len(groups) > 1
    families = np.asarray(families)
    sums = np.array([delta[families == g].sum() for g in groups])
    sizes = np.array([(families == g).sum() for g in groups])
    indices = rng.integers(len(groups), size=(draws, len(groups)))
    clustered = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    result = dict(n=len(y), families=len(groups), improvement=float(delta.mean()))
    for name, values in [('record', iid), ('family', clustered)]:
        result[name + '_ci95'] = np.quantile(values, [.025, .975]).tolist()
        result[name + '_bonferroni8'] = np.quantile(values, [.025 / 8, 1 - .025 / 8]).tolist()
    return result, np.stack([iid, clustered])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_panel(rows, test, draws=10000):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    lookup = {(r['condition'], int(r['seed']), int(r['idx'])): r for r in rows}
    assert len(lookup) == len(rows) == 12 * len(test)
    expected = {(c, s, r['arch_index']) for c in panel.CONDITIONS for s in panel.SEEDS for r in test}
    assert set(lookup) == expected
    y = np.array([r['best_accuracy'] for r in test])
    families = np.array([r['arch_family'] for r in test])
    predictions = {}
    for cond in panel.CONDITIONS:
        predictions[cond] = np.array([[lookup[cond, s, r['arch_index']]['pred_best_accuracy'] for r in test] for s in panel.SEEDS])
        assert np.isfinite(predictions[cond]).all()
        for s in panel.SEEDS:
            assert all(lookup[cond, s, r['arch_index']]['gt_best_accuracy'] == r['best_accuracy'] for r in test)
    metrics, results, arrays = [], {}, {}
    for population, mask in [('all', np.ones(len(y), dtype=bool)), ('target_ge20', y >= 20.)]:
        target = y[mask]
        for cond, pred in predictions.items():
            for seed, prediction in zip(panel.SEEDS, pred[:, mask]):
                metrics.append(dict(population=population, condition=cond, seed=seed, n=len(target),
                    MAE=float(mean_absolute_error(target, prediction)),
                    RMSE=float(np.sqrt(mean_squared_error(target, prediction))),
                    Pearson=float(pearsonr(target, prediction).statistic),
                    Spearman=float(spearmanr(target, prediction).statistic),
                    R2=float(r2_score(target, prediction))))
        result, array = paired_mae(target, predictions['no-layer'][:, mask], predictions['taylor'][:, mask], families[mask], draws)
        result['mean_metrics'] = {c: {metric: float(np.mean([r[metric] for r in metrics if r['population'] == population and r['condition'] == c])) for metric in ['MAE', 'RMSE', 'Pearson', 'Spearman', 'R2']} for c in panel.CONDITIONS}
        result['unique_prediction_vectors_by_condition'] = {c: len(np.unique(p[:, mask], axis=0)) for c, p in predictions.items()}
        results[population], arrays[population] = result, array
    return metrics, results, arrays


def load_records(root, dataset):
    folder = root / 'evidence/nb201-collected-v1' / dataset
    cp = root / 'nb201_v1' / ('cohort.json' if dataset == 'cifar10' else 'cifar100/cohort.json')
    tp = root / 'nb201_v1' / ('benchmark_targets.csv' if dataset == 'cifar10' else 'cifar100/targets.csv')
    sp = folder / 'snapshot.jsonl'
    receipt = json.loads((folder / 'collection_complete.json').read_text())
    assert sha(sp) == receipt['snapshot_sha256']
    cohort = json.loads(cp.read_text())
    with tp.open() as f:
        targets = {int(r['architecture_id']): r for r in csv.DictReader(f)}
    members = {r['arch_index']: r for r in cohort['architectures']}
    raw = [json.loads(line) for line in sp.read_text().splitlines()]
    records = []
    for row in raw:
        r = dict(row)
        if dataset == 'cifar100':
            assert r['target_kind'] == 'final_validation_accuracy' and r['target_epoch'] == 200
            assert r['benchmark_target_seeds'] == [777, 888]
            r.update(id=r['record_id'], arch_index=r['architecture_id'], arch_str=r['architecture_string'],
                     best_accuracy=r['target_final_validation_accuracy'],
                     layer_stats='epoch\ttaylor_imp\n' + '\n'.join(f'{e}\t{r[f"taylor_epoch_{e}"]!r}' for e in (1, 2, 3)))
        i = r['arch_index']
        t, m = targets[i], members[i]
        assert r['id'] == t['record_id'] == m['id']
        assert r['arch_str'] == m['arch_str'] and r['arch_family'] == m['arch_family']
        assert r['best_accuracy'] == float(t['best_accuracy' if dataset == 'cifar10' else 'target_final_validation_accuracy'])
        if dataset == 'cifar10':
            for e in (1, 2, 3):
                assert r[f'accuracy_epoch_{e}'] == float(t[f'accuracy_epoch_{e}'])
        assert r['candidate_seed'] == 777
        records.append(r)
    assert len(records) == len({r['id'] for r in records}) == len(members) == 1100
    assert {r['arch_index'] for r in records} == set(members) == set(targets)
    tr, va, te = panel.split_by_family(records, 2026, frac=panel.fracs(.25))
    for name, rows in [('train', tr), ('val', va), ('test', te)]:
        assert {r['arch_index'] for r in rows} == set(cohort['splits'][name])
        assert all(targets[r['arch_index']]['split'] == name for r in rows)
    x, _ = panel.featurize(records, ('taylor_imp',))
    assert np.isfinite(x).all() and (x[:, 5:] >= 0).all()
    return records, cohort, {str(p.relative_to(root)): sha(p) for p in [sp, cp, tp, folder / 'collection_complete.json']}


def main():
    import platform
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits
    root = Path(__file__).resolve().parent.parent
    out = root / 'evidence/nb201-classical-matched-v1'
    assert not out.exists(), 'Existing results preserved; do not overwrite or blindly rerun'
    loaded = {ds: load_records(root, ds) for ds in ['cifar10', 'cifar100']}
    out.mkdir()
    contract = dict(tier='Classical directional comparison, NOT Qwen replication or supervisor primary_endpoint.py',
        seeds=list(panel.SEEDS), train=772, validation=53, test=275, fit_records=825,
        model_policy='Unmodified original/panel.py classical(): RidgeCV 13 logspace alphas; HistGradientBoosting defaults. Both fit train+val. No test selection.',
        inputs='acc1, acc2, acc3, acc3-acc2, acc2-acc1; Taylor arm additionally taylor epochs1-3; no static features or ZCP',
        target_policy={'cifar10': 'peak of available-seed mean benchmark x-valid curve', 'cifar100': 'mean epoch200 x-valid over seeds777/888'},
        legacy_schema='panel.csv gt_best_accuracy is a legacy column name only: CIFAR100 is final accuracy, never peak accuracy',
        uncertainty='10000 percentile bootstrap draws, seed0; paired seed-averaged absolute-error differences; no seed resampling; IID records and family clusters; positive favors Taylor',
        multiplicity='Bonferroni8 percentile intervals over 2 datasets x 2 models x 2 populations, separately for each resampling scheme; not supervisor Holm tests',
        population='All records plus target >=20 percent per attached proposal; fixed threshold is not a universal collapse definition',
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        sources={str(p.relative_to(root)): sha(p) for p in [Path(__file__), Path(__file__).with_name('test_evaluate.py'), Path(panel.__file__), Path(panel.__file__).with_name('prompts.py')]})
    for _, _, sources in loaded.values():
        contract['sources'].update(sources)
    (out / 'contract.json').write_text(json.dumps(contract, indent=2))
    all_results = {}
    for ds, (records, cohort, sources) in loaded.items():
        for kind in ['ridge', 'gbm']:
            dest = out / ds / kind
            dest.mkdir(parents=True)
            print('FITTING', ds, kind, flush=True)
            with threadpool_limits(limits=1):
                rows, test = panel.classical(records, 2026, kind, .25)
            metrics, results, arrays = analyze_panel(rows, test)
            for filename, data in [('panel.csv', rows), ('metrics_by_seed.csv', metrics)]:
                with (dest / filename).open('x', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
            meta = {r['arch_index']: r for r in test}
            for cond in panel.CONDITIONS:
                for seed in panel.SEEDS:
                    with (dest / f'{cond}-s{seed}-predictions.csv').open('x', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['record_id', 'prediction', 'target', 'seed'])
                        writer.writeheader()
                        writer.writerows(dict(record_id=meta[r['idx']]['id'], prediction=r['pred_best_accuracy'], target=r['gt_best_accuracy'], seed=seed) for r in rows if r['condition'] == cond and r['seed'] == seed)
            (dest / 'test_registry.json').write_text(json.dumps([dict(record_id=r['id'], architecture_id=r['arch_index'], family=r['arch_family'], target=r['best_accuracy']) for r in test], indent=2))
            (dest / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
            np.savez_compressed(dest / 'bootstrap_draws.npz', **arrays)
            all_results[ds + '/' + kind] = results
            print(json.dumps({ds + '/' + kind: results}), flush=True)
    assert all(sha(root / path) == digest for path, digest in contract['sources'].items()), 'Input/source changed during evaluation'
    (out / 'results.json').write_text(json.dumps(all_results, indent=2, allow_nan=False))
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', sha256=files), indent=2))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
