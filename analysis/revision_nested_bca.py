"""Post-hoc nested BCa: original draw-major/fold-major RNG and stratified jackknife.

Portable implementation of the executed sensitivity procedure, not a new search.
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
from scipy.stats import bootstrap
from analysis.revision_common import METRICS, metrics
from analysis.revision_nested import load_analyzer
from analysis.thesis_common import require, sha


def calculate(y, p, blocks, n_resamples=10000):
    require(len(blocks)==3 and p.shape==(2,6,len(y)), 'Expected three folds and two six-seed arms')
    require(np.isfinite(y).all() and np.isfinite(p).all(), 'Nonfinite input')
    ids=tuple(np.arange(len(b)) for b in blocks)
    require(all(len(i)>2 for i in ids), 'Insufficient fold groups')
    require(sorted(np.concatenate([g for b in blocks for g in b]).tolist())==list(range(len(y))),
            'Groups must partition the records exactly once')
    sign=np.array([-1,-1,1,1,1])
    means=metrics(y,p).mean(axis=1)
    observed=(means[1]-means[0])*sign
    def statistic(*sample):
        idx=np.concatenate([b[int(g)] for b,s in zip(blocks,sample) for g in s])
        value=metrics(y[idx],p[:,:,idx]).mean(axis=1)
        require(np.isfinite(value).all(), 'Undefined metric; no dropped samples')
        return (value[1]-value[0])*sign
    np.testing.assert_allclose(statistic(*ids),observed,rtol=1e-10,atol=1e-10)
    rng=np.random.default_rng(0)
    draws=np.empty((n_resamples,5))
    for i in range(n_resamples):
        draws[i]=statistic(*(rng.integers(len(b),size=len(b)) for b in blocks))
    cache=SimpleNamespace(bootstrap_distribution=draws.T)
    intervals=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for confidence in (.95,.99):
            result=bootstrap(ids,statistic,vectorized=False,paired=False,n_resamples=0,
                             batch=50,method='BCa',confidence_level=confidence,bootstrap_result=cache)
            require(np.array_equal(result.bootstrap_distribution,draws.T), 'BCa changed cached draws')
            ci=np.array(result.confidence_interval)
            require(np.isfinite(ci).all() and (ci[0]<=ci[1]).all(), 'Invalid BCa intervals')
            intervals.append(ci)
    ci,adjusted=intervals
    percentile=np.quantile(draws,[.025,.975,.005,.995],axis=0)
    rows=[dict(metric=name,no_layer=float(means[0,k]),taylor=float(means[1,k]),
               improvement=float(observed[k]),ci95_low=float(ci[0,k]),ci95_high=float(ci[1,k]),
               bonferroni5_low=float(adjusted[0,k]),bonferroni5_high=float(adjusted[1,k]),
               percentile95_low=float(percentile[0,k]),percentile95_high=float(percentile[1,k]),
               percentile_bonferroni5_low=float(percentile[2,k]),percentile_bonferroni5_high=float(percentile[3,k]),
               warnings=[str(w.message) for w in caught]) for k,name in enumerate(METRICS)]
    return rows,draws


def run(repo,out):
    repo,out=Path(repo).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    require(not any(out.is_relative_to(repo/p) for p in ('analysis','evidence','tests','paper_runners')),
            'Output overlaps source/evidence')
    archive=repo/'evidence/nested-outer-six-seeds-v1.tar.gz'
    analyzer=load_analyzer(repo,archive,out)
    require(sha(archive)==analyzer.EXPECTED,'Nested archive checksum mismatch')
    y,p,selected,folds,groups,registry,selections,fold_metrics=analyzer.load()
    require(analyzer.METRICS==METRICS and analyzer.SEEDS==list(range(42,48))
            and analyzer.ARMS==['no-layer','taylor'] and p.shape==(2,6,1139),'Unexpected nested contract')
    blocks=analyzer.group_blocks(folds,groups)
    require([len(b) for b in blocks]==[212,215,212],'Unexpected fold group counts')
    rows,draws=calculate(y,p,blocks)
    # Independent historical helper checks the same first draw and point estimates.
    idx=analyzer.sample_indices(blocks,np.random.default_rng(0))
    value=analyzer.metrics(y[idx],p[:,:,idx]).mean(axis=1)
    np.testing.assert_allclose(draws[0],(value[1]-value[0])*[-1,-1,1,1,1],atol=1e-12)
    require(sha(archive)==analyzer.EXPECTED,'Input changed during calculation')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'intervals.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'bootstrap.npz',differences=draws)
    report=dict(status='complete',results=rows,n=1139,seeds=list(range(42,48)),draws=10000,rng_seed=0,
                groups_per_fold=[len(b) for b in blocks],method='stratified multi-sample BCa; SciPy paired=False',
                rng_order='draw-major, then fold-major; original sample_indices schedule',
                correction_family=5,input_archive_sha256=analyzer.EXPECTED,
                source_sha256={n:sha(repo/n) for n in ['analysis/revision_nested_bca.py',
                    'analysis/revision_common.py','paper_runners/nested_multiseed/analyze_nested_outer_six_seeds.py']},
                limitations=['Fixed folds, models and inner selections; no repeated search or seed resampling',
                             'BCa jackknife treats folds as three samples, not one pooled group list',
                             'Adjusted tails have limited Monte Carlo resolution with 10000 draws'])
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256={
        p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
    print('NESTED BCA COMPLETE',rows,flush=True)
    return report
