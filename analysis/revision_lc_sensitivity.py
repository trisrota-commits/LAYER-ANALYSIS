"""Post-hoc LC/Taylor cluster BCa and unclustered Zou sensitivity analyses."""
import numpy as np
from scipy.stats import bootstrap, norm


def require(ok, message):
    if not ok:
        raise ValueError(message)


DIRECTION = np.array([-1., -1., 1., 1., 1.])
DRAWS = 100000
ADJUSTED_CONFIDENCE = 1 - .05/15


def cluster_intervals(y, predictions, families, metric_function, n_resamples=DRAWS):
    """Record-weighted paired improvements, resampling whole families only."""
    import warnings
    groups = np.unique(families)
    require(len(groups) > 2, 'Too few operation families')
    blocks = [np.flatnonzero(families == g) for g in groups]
    full = metric_function(y, predictions)
    require(np.isfinite(full).all(), 'Undefined complete-sample metric')
    observed = (full[1]-full[0])*DIRECTION

    def statistic(sampled_groups):
        indices = np.concatenate([blocks[int(g)] for g in sampled_groups])
        values = metric_function(y[indices], predictions[:, indices])
        require(np.isfinite(values).all(), 'Undefined resampled metric; refusing silent draw removal')
        return (values[1]-values[0])*DIRECTION

    indices = np.arange(len(groups))
    np.testing.assert_allclose(statistic(indices), observed, atol=1e-10, rtol=1e-10)
    options = dict(data=(indices,), statistic=statistic, vectorized=False, paired=False, batch=100, method='BCa')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        marginal = bootstrap(**options, confidence_level=.95, n_resamples=n_resamples, rng=np.random.default_rng(0))
        adjusted = bootstrap(**options, confidence_level=ADJUSTED_CONFIDENCE, n_resamples=0, bootstrap_result=marginal)
    draws = np.asarray(marginal.bootstrap_distribution)
    require(draws.shape == (5, n_resamples) and np.isfinite(draws).all(), 'Invalid bootstrap draws')
    require(np.array_equal(draws, adjusted.bootstrap_distribution), 'Draw reuse failed')
    jackknife = np.array([statistic(np.delete(indices, j)) for j in indices])
    influence = jackknife.mean(axis=0)-jackknife
    denominator = 6*np.sum(influence**2, axis=0)**1.5
    require(np.all(denominator > 0), 'Undefined BCa acceleration')
    acceleration = np.sum(influence**3, axis=0)/denominator
    bias_probability = ((draws < observed[:, None]).sum(axis=1)+(draws <= observed[:, None]).sum(axis=1))/(2*n_resamples)
    require(np.all((bias_probability > 0) & (bias_probability < 1)), 'Infinite BCa bias correction')
    z0 = norm.ppf(bias_probability)
    tail = .05/(2*15)
    z = norm.ppf([tail, 1-tail])[:, None]
    denominator = 1-acceleration*(z0+z)
    require(np.all(denominator > 0), 'Unstable BCa transformation')
    probabilities = norm.cdf(z0+(z0+z)/denominator)
    bca_adjusted = np.array(adjusted.confidence_interval)
    reconstructed = np.array([np.quantile(draws[j], probabilities[:, j]) for j in range(5)]).T
    np.testing.assert_allclose(reconstructed, bca_adjusted, rtol=1e-9, atol=1e-10)
    result = dict(full=full, observed=observed, draws=draws, jackknife=jackknife,
                  BCa95=np.array(marginal.confidence_interval), BCa_Bonf15=bca_adjusted,
                  percentile95=np.quantile(draws, [.025, .975], axis=1),
                  percentile_Bonf15=np.quantile(draws, [tail, 1-tail], axis=1),
                  acceleration=acceleration, bias_probability=bias_probability, z0=z0,
                  adjusted_probabilities=probabilities,
                  lower_tail_draws=n_resamples*probabilities[0], upper_tail_draws=n_resamples*(1-probabilities[1]),
                  warnings=[w.category.__name__+': '+str(w.message) for w in caught])
    result['sparse_tail'] = np.minimum(result['lower_tail_draws'], result['upper_tail_draws']) < 100
    for name in ['BCa95', 'BCa_Bonf15', 'percentile95', 'percentile_Bonf15']:
        require(np.isfinite(result[name]).all() and np.all(result[name][0] <= result[name][1]), 'Invalid interval: '+name)
    return result


def zou_overlapping(r_taylor, r_baseline, r_predictions, n):
    """Zou (2007) overlapping Pearson difference; unadjusted, NOT clustered."""
    require(n > 3, 'Too few observations')
    r1, r2, r12 = r_taylor, r_baseline, r_predictions
    require(np.isfinite([r1, r2, r12]).all(), 'Nonfinite correlations')
    require(abs(r1) < 1 and abs(r2) < 1, 'Perfect target correlation')
    require(abs(r12) <= 1, 'Invalid prediction correlation')
    matrix = np.array([[1., r1, r2], [r1, 1., r12], [r2, r12, 1.]])
    require(np.linalg.eigvalsh(matrix).min() >= -1e-10, 'Invalid correlation matrix')
    h = norm.ppf(.975) / np.sqrt(n - 3)
    l1, u1 = np.tanh(np.arctanh(r1) + np.array([-h, h]))
    l2, u2 = np.tanh(np.arctanh(r2) + np.array([-h, h]))
    dependence = ((r12 - .5*r1*r2)*(1-r1**2-r2**2-r12**2) + r12**3) / ((1-r1**2)*(1-r2**2))
    require(abs(dependence) <= 1+1e-10, 'Invalid estimated correlation dependence')
    lv = (r1-l1)**2 + (u2-r2)**2 - 2*dependence*(r1-l1)*(u2-r2)
    uv = (u1-r1)**2 + (r2-l2)**2 - 2*dependence*(u1-r1)*(r2-l2)
    require(min(lv, uv) >= -1e-12, 'Negative interval radicand beyond rounding tolerance')
    difference = r1-r2
    lower, upper = difference-np.sqrt(max(0., lv)), difference+np.sqrt(max(0., uv))
    require(np.isfinite([difference, lower, upper]).all() and lower <= difference <= upper, 'Invalid interval')
    return difference, lower, upper


LOADER_SHA = 'b4b78239913d1768c0880f6af403b128cdbe60d08406c8d0fc004cdb851221e6'
ARCHIVE_SHA = '6f852836ce55e57d4eea49d54b484b33d78a9d0f26b7b12d24f4435487007121'


def run(repo, out):
    """Verify pinned local inputs, compute sensitivities, write a new output only."""
    import csv
    import hashlib
    import json
    import platform
    from pathlib import Path
    import scipy

    repo, out = Path(repo), Path(out)
    require(not out.exists(), 'Existing output preserved')
    loader_path = repo/'analysis'/'revision_loss_confidence.py'
    archive = repo/'evidence'/'loss-confidence-all-metrics-inputs-v1.tar.gz'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = loader_path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == LOADER_SHA, 'Loader checksum mismatch')
    require(sha(archive) == ARCHIVE_SHA, 'Archive checksum mismatch')
    module = {'__name__': 'verified_loss_confidence_loader'}
    exec(compile(source, 'analysis/revision_loss_confidence.py', 'exec'), module)
    data = module['load'](archive)
    metric_names = ['MAE', 'RMSE', 'Pearson', 'Spearman', 'R2']
    require(module['METRICS'] == metric_names and module['ARMS'][1] == 'A_LC' and module['ARMS'][3] == 'A_LC_T', 'Unexpected metric/arm ordering')
    require(len(data)*len(metric_names) == 15, 'Unexpected comparison count')
    rows, zou_rows, arrays, checks = [], [], {}, []
    for di, (dataset, (y, all_predictions, families, saved, _)) in enumerate(data.items()):
        predictions = all_predictions[[1, 3]]
        print('Running', dataset, len(y), 'records;', len(np.unique(families)), 'families', flush=True)
        result = cluster_intervals(y, predictions, families, module['metrics'])
        for ai, arm in enumerate(['A_LC', 'A_LC_T']):
            np.testing.assert_allclose(result['full'][ai], [saved['all'][arm]['mean_metrics'][m] for m in metric_names], rtol=1e-10, atol=1e-10)
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                arrays[f'dataset_{di}__{key}'] = value
        checks.append(dict(dataset=dataset, array_prefix=f'dataset_{di}', warnings=result['warnings'], valid_draws=result['draws'].shape[1], identical_draw_reuse=True, archived_metrics_match=True, bca_endpoint_reconstruction=True))
        for j, metric in enumerate(metric_names):
            row = dict(dataset=dataset, records=len(y), families=len(np.unique(families)), metric=metric,
                       A_LC=float(result['full'][0,j]), A_LC_T=float(result['full'][1,j]), improvement=float(result['observed'][j]))
            for name in ['percentile95', 'BCa95', 'percentile_Bonf15', 'BCa_Bonf15']:
                low, high = result[name][:, j]
                row[name+'_low'], row[name+'_high'] = float(low), float(high)
                row[name+'_direction'] = 'positive_interval' if low > 0 else ('negative_interval' if high < 0 else 'includes_zero')
            for key in ['acceleration', 'bias_probability', 'z0', 'lower_tail_draws', 'upper_tail_draws']:
                row[key] = float(result[key][j])
            row['sparse_tail'] = bool(result['sparse_tail'][j])
            rows.append(row)
            if row['sparse_tail']:
                print('TAIL WARNING', dataset, metric, 'fewer than 100 expected draws in an adjusted BCa endpoint tail', flush=True)
            print(f"  {metric}: improvement={row['improvement']:+.8f}; BCa95=[{row['BCa95_low']:+.8f}, {row['BCa95_high']:+.8f}]; BCa Bonf15=[{row['BCa_Bonf15_low']:+.8f}, {row['BCa_Bonf15_high']:+.8f}]", flush=True)
        require(np.std(y) > 0 and np.all(np.std(predictions, axis=1) > 0), 'Constant Pearson input')
        matrix = np.corrcoef(np.stack([y, predictions[1], predictions[0]]))
        r1, r2, r12 = matrix[0,1], matrix[0,2], matrix[1,2]
        np.testing.assert_allclose([r2,r1], result['full'][:,2], atol=1e-10, rtol=1e-10)
        delta, lower, upper = zou_overlapping(r1, r2, r12, len(y))
        np.testing.assert_allclose(zou_overlapping(r2, r1, r12, len(y)), [-delta, -upper, -lower], atol=1e-12)
        zou_rows.append(dict(dataset=dataset, records=len(y), operation_families=len(np.unique(families)), Pearson_A_LC=float(r2), Pearson_A_LC_T=float(r1), correlation_between_predictions=float(r12), Pearson_improvement=float(delta), Zou_ci95_low=float(lower), Zou_ci95_high=float(upper)))
        print(f'{dataset}: delta Pearson={delta:+.8f}, Zou 95% CI=[{lower:+.8f}, {upper:+.8f}]', flush=True)
    require(sha(loader_path) == LOADER_SHA and sha(archive) == ARCHIVE_SHA, 'Input changed during run')
    out.mkdir(parents=True, exist_ok=False)
    for name, values in [('cluster_intervals.csv', rows), ('zou_intervals.csv', zou_rows)]:
        with (out/name).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    np.savez_compressed(out/'bootstrap_arrays.npz', **arrays)
    report = dict(analysis='Post-hoc A_LC -> A_LC_T sensitivity; all records; positive favours Taylor',
                  source_commit='311a75aca52b12dd992de23ca5782b590d777ba8',
                  source_loader_sha256=LOADER_SHA, source_archive_sha256=ARCHIVE_SHA,
                  source_script_sha256=sha(Path(__file__)), inputs_unchanged=True,
                  bootstrap=dict(draws_per_dataset=[c['valid_draws'] for c in checks], rng_seed=0, rng_reset_per_dataset=True, batch=100, vectorized=False, paired=False, pairing='Inside statistic; whole families with multiplicity; record weighted', jackknife='Delete one whole family', adjusted_reuse_n_resamples=0),
                  correction='Bonferroni15: five metrics x three datasets; 99.66666666666667% per interval; approximate 95% simultaneous coverage only for these 15 post-hoc comparisons, not exploration history',
                  seed_handling='Exact equality across seeds42-47 verified by pinned loader; collapsed to one vector',
                  zou_method='Zou 2007 overlapping Pearson; analytic unadjusted 95%; independent observational rows assumed; NOT operation-family adjusted; no Spearman inference',
                  limitations=['Fixed fitted predictors and acquisitions; no candidate-seed uncertainty', 'ImageNet16-120 partial ordered-truncation cohort', 'Sparse adjusted BCa tails (<100 expected draws) have limited endpoint precision', 'Cluster-bootstrap validity requires appropriate family sampling assumptions'],
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__), checks=checks, results=rows, zou=zou_rows)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    require(sha(loader_path) == LOADER_SHA and sha(archive) == ARCHIVE_SHA, 'Input changed before completion')
    receipt = dict(status='complete', inputs_unchanged=True, files_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'complete.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.repo, args.out)
