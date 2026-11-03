"""Independent implementation of project_final_4_.pdf III-D / V-A.

Not the unavailable supervisor script. Positive differences favor Taylor.
No training, model selection, or automatic removal of invalid predictions.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import io
from pathlib import Path
import warnings

import numpy as np
from scipy import stats

SEEDS = tuple(range(42, 48))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_panel(panel_path, registry_path):
    registry = json.loads(Path(registry_path).read_text())
    require(isinstance(registry, list) and len(registry) > 1, 'Expected a nonempty test registry list')
    ids = [str(r.get('idx', r.get('architecture_id', r['record_id']))) for r in registry]
    require(len(ids) == len(set(ids)), 'Duplicate registry IDs')
    lookup = {key: i for i, key in enumerate(ids)}
    y = np.array([r['target'] for r in registry], dtype=float)
    require(np.isfinite(y).all(), 'Non-finite registry targets')
    predictions = np.full((2,len(ids),6), np.nan)
    seen, datasets = set(), {}
    with Path(panel_path).open(newline='') as f:
        for row in csv.DictReader(f):
            require(row['condition'] in ('no-layer','taylor'), 'Unexpected condition')
            c = ('no-layer','taylor').index(row['condition'])
            s = int(row['seed'])
            require(s in SEEDS and row['idx'] in lookup, 'Unexpected seed or test ID')
            i = lookup[row['idx']]
            key = (c,i,s)
            require(key not in seen, 'Duplicate prediction')
            seen.add(key)
            require(float(row['gt_best_accuracy']) == y[i], 'Target differs from frozen registry')
            value = float(row['pred_best_accuracy'])
            require(np.isfinite(value), 'Non-finite prediction; no silent complete-case filtering')
            dataset = row['dataset']
            require(datasets.setdefault(i,dataset) == dataset, 'Dataset identity mismatch')
            require(registry[i].get('dataset',dataset) == dataset, 'Dataset differs from registry')
            predictions[c,i,SEEDS.index(s)] = value
    require(len(seen) == 12*len(ids) and np.isfinite(predictions).all(), 'Incomplete condition/seed/test-ID matrix')
    return y, predictions, registry


def paired_t(values):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 1 and np.isfinite(values).all()
    result = dict(n=len(values), improvement=float(values.mean()))
    if np.ptp(values) < 1e-12:
        return dict(result, t=None, p=None, ci95=None, status='undefined: zero variance in paired differences')
    test = stats.ttest_1samp(values, 0., alternative='two-sided')
    interval = stats.t.interval(.95, df=len(values)-1, loc=values.mean(), scale=stats.sem(values))
    return dict(result, t=float(test.statistic), p=float(test.pvalue), ci95=list(map(float, interval)), status='ok')


def holm(pvalues):
    """Keep missing tests in the declared family conservatively as p=1."""
    p = np.array([1. if v is None else v for v in pvalues], dtype=float)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = adjusted
    return [None if original is None else float(value) for original, value in zip(pvalues, out)]


def bootstrap_summary(values):
    return dict(ci95=np.quantile(values,[.025,.975]).tolist(),
                p=float(min(1.,max(1/len(values),2*min(np.mean(values<=0),np.mean(values>=0))))))


def crossed_mixed(delta):
    import pandas as pd
    import statsmodels.formula.api as smf
    n,s=delta.shape
    frame=pd.DataFrame(dict(delta=delta.ravel(),record=np.repeat(np.arange(n),s),seed=np.tile(np.arange(s),n),group=1))
    attempts=[]
    for optimizer in ('lbfgs','powell'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model=smf.mixedlm('delta ~ 1',frame,groups='group',re_formula='0',
                                 vc_formula={'record':'0+C(record)','seed':'0+C(seed)'},use_sparse=True)
                fit=model.fit(reml=True,method=optimizer,maxiter=300,disp=False)
                messages=sorted(set(str(w.message) for w in caught))
                invalid=any(any(term in text.lower() for term in ('not positive definite','singular','failed to converge')) for text in messages)
                valid=bool(fit.converged and not invalid and np.isfinite(fit.pvalues['Intercept']) and np.isfinite(fit.bse['Intercept']) and fit.bse['Intercept']>0)
                result=dict(status='ok' if valid else 'unverified fit',converged=bool(fit.converged),optimizer=optimizer,
                            improvement=float(fit.params['Intercept']),p=float(fit.pvalues['Intercept']) if valid else None,
                            ci95=fit.conf_int().loc['Intercept'].tolist() if valid else None,warnings=messages)
            except Exception as exc:
                result=dict(status='failed',optimizer=optimizer,p=None,error=repr(exc),warnings=sorted(set(str(w.message) for w in caught)))
        attempts.append(dict(result))
        if result['status']=='ok':
            break
    result['attempts']=attempts
    result['method']='REML intercept-only paired-error-difference model with crossed record/seed random intercepts; Wald normal p. L-BFGS first, Powell only after invalid convergence/Hessian; never chosen by p-value.'
    return result


def analyze(y,predictions,registry,draws=10000,seed=0,threshold=20.,mixed=True):
    require(predictions.shape==(2,len(y),6) and np.isfinite(predictions).all(), 'Expected two complete arms x records x six seeds')
    require(draws>=100, 'At least 100 draws required')
    keep=y>=threshold
    require(keep.sum()>1, 'Too few non-collapsed records')
    errors=np.abs(predictions-y[None,:,None])
    delta=(errors[0]-errors[1])[keep]
    record=paired_t(delta.mean(axis=1))
    deterministic=all(np.allclose(p,p[:,:1],atol=1e-12,rtol=0) for p in predictions)
    seedtest=paired_t(delta.mean(axis=0))
    if deterministic:
        seedtest.update(p=None,t=None,ci95=None,status='not applicable: identical predictions across seeds; not independent training evidence')
    for test,adjusted in zip((record,seedtest),holm([record['p'],seedtest['p']])):
        test['holm_p']=adjusted
    result=dict(primary=dict(record=record,seed=seedtest),
                counts=dict(all=len(y),noncollapsed=int(keep.sum()),collapsed=int((~keep).sum())),
                deterministic_seed_predictions=deterministic,
                seed_improvements=delta.mean(axis=0).tolist(),
                robust_location={c:dict(median_seed_mean_absolute_error=float(np.median(errors[i,keep].mean(axis=1))),
                    trimmed20_seed_mean_absolute_error=float(stats.trim_mean(errors[i,keep].mean(axis=1),.2))) for i,c in enumerate(('no-layer','taylor'))})
    rng=np.random.default_rng(seed)
    samples={k:np.empty(draws) for k in ('record_bootstrap','two_way_bootstrap','cell_signflip')}
    n,s=delta.shape
    for start in range(0,draws,128):
        count=min(128,draws-start); stop=start+count
        ix=rng.integers(n,size=(count,n)); sx=rng.integers(s,size=(count,s))
        samples['record_bootstrap'][start:stop]=delta.mean(axis=1)[ix].mean(axis=1)
        samples['two_way_bootstrap'][start:stop]=delta[ix[:,:,None],sx[:,None,:]].mean(axis=(1,2))
        signs=2*rng.integers(2,size=(count,n,s))-1
        samples['cell_signflip'][start:stop]=(delta*signs).mean(axis=(1,2))
    robustness={k:bootstrap_summary(samples[k]) for k in ('record_bootstrap','two_way_bootstrap')}
    robustness['cell_signflip']=dict(p=float((1+np.sum(np.abs(samples['cell_signflip'])>=abs(delta.mean())))/(draws+1)),
        warning='Independent cell flips assume exchangeability of record-seed cells despite shared records/seeds; supporting sensitivity check only',
        method='two-sided Monte Carlo absolute statistic with plus-one correction')
    if deterministic:
        robustness['cell_signflip'].update(p=None,status='not applicable: repeated identical seed predictions would create pseudoreplication')
    d=delta.mean(axis=1)
    robustness['wilcoxon']=dict(p=1. if np.all(d==0) else float(stats.wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='auto').pvalue))
    robustness['crossed_mixed_model']=(dict(status='not applicable: deterministic seed predictions',p=None) if deterministic else
        crossed_mixed(delta) if mixed else dict(status='not run (explicit skip)',p=None))
    groups=[r.get('family') for r in np.asarray(registry,dtype=object)[keep]]
    if all(g is not None for g in groups) and len(set(groups))>1:
        names=sorted(set(groups)); sums=np.array([d[np.array(groups)==g].sum() for g in names]);sizes=np.array([groups.count(g) for g in names])
        ix=rng.integers(len(names),size=(draws,len(names)))
        samples['family_bootstrap']=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
        robustness['family_bootstrap']=dict(bootstrap_summary(samples['family_bootstrap']),groups=len(names),method='Additional NB201 operation-multiset cluster check; record-weighted mean')
    result['robustness']=robustness
    result['collapse_auc']={}
    positive=np.where(~keep)[0];negative=np.where(keep)[0]
    for i,condition in enumerate(('no-layer','taylor')):
        if not len(positive) or not len(negative):
            result['collapse_auc'][condition]=dict(auc=None,status='undefined: only one class')
            continue
        score=-predictions[i].mean(axis=1)
        comparisons=(score[positive,None]>score[negative][None,:]).astype(float)+.5*(score[positive,None]==score[negative][None,:])
        auc_samples=np.empty(draws);auc_rng=np.random.default_rng(seed+1)
        for start in range(0,draws,128):
            count=min(128,draws-start)
            pos=auc_rng.integers(len(positive),size=(count,len(positive)))
            neg=auc_rng.integers(len(negative),size=(count,len(negative)))
            auc_samples[start:start+count]=comparisons[pos[:,:,None],neg[:,None,:]].mean(axis=(1,2))
        result['collapse_auc'][condition]=dict(auc=float(comparisons.mean()),ci95=np.quantile(auc_samples,[.025,.975]).tolist(),
            collapsed=len(positive),noncollapsed=len(negative),method='Negative six-seed mean prediction; class-stratified record bootstrap, RNG seed+1',
            warning='Very few collapsed records; interval does not establish reliable stopping decisions' if len(positive)<10 else None)
        samples['auc_'+condition]=auc_samples
    result['descriptive']={}
    for population,mask in [('all',np.ones(len(y),dtype=bool)),('noncollapsed',keep),('collapsed',~keep)]:
        if not mask.any():
            continue
        result['descriptive'][population]={}
        target=y[mask]
        for i,condition in enumerate(('no-layer','taylor')):
            rows=[]
            for j,s in enumerate(SEEDS):
                p=predictions[i,mask,j];error=p-target;ss=float(np.sum((target-target.mean())**2))
                corr=len(target)>1 and np.ptp(target)>0 and np.ptp(p)>0
                rows.append(dict(seed=s,MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt((error**2).mean())),
                    Pearson=float(stats.pearsonr(target,p).statistic) if corr else None,
                    Spearman=float(stats.spearmanr(target,p).statistic) if corr else None,
                    R2=float(1-np.sum(error**2)/ss) if ss>0 else None))
            result['descriptive'][population][condition]=dict(n=len(target),per_seed=rows,
                mean={k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in ('MAE','RMSE','Pearson','Spearman','R2')})
    result['protocol']=dict(provenance='Independent implementation from project_final_4_.pdf pages3-4,7; not the missing original script',
        draws=draws,seed=seed,collapse_threshold=threshold,primary_family='Two two-sided paired t-tests; by seed-averaged record errors and by seed MAE; Holm within this panel',
        bootstrap_p='2 min(sign probabilities), clipped to [1/B,1]; percentile CI',
        unspecified_choices='Mixed model fit on differences with REML, lbfgs then Powell only on numerical failure; independent record/seed resampling; plus-one cell-flip p; class-stratified AUC bootstrap. Declared choices, not claimed identical to unavailable source.',
        inference_limit='Record tests condition on trained models; seed tests condition on held-out records. Not independent replications of the same sampling uncertainty. Cell-flip assumptions are stronger. Use family-cluster sensitivity for NB201.',
        multiplicity='Robustness p-values are supporting, unadjusted; per-panel Holm2 does not correct selection across datasets/models.')
    return result,samples


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_panel(panel,registry,out,draws=10000,seed=0,threshold=20.,mixed=True):
    out=Path(out)
    require(not out.exists(),'Existing report preserved: '+str(out))
    y,pred,records=load_panel(panel,registry)
    sources={str(Path(p).resolve()):sha(p) for p in (panel,registry,__file__)}
    result,samples=analyze(y,pred,records,draws,seed,threshold,mixed)
    result['input_sha256']=sources
    result['versions']={p:importlib.metadata.version(p) for p in ('numpy','scipy')}
    if mixed and not result['deterministic_seed_predictions']:
        result['versions']['statsmodels']=importlib.metadata.version('statsmodels')
    text=json.dumps(result,indent=2,allow_nan=False)
    require(all(sha(p)==h for p,h in sources.items()),'Input changed during analysis')
    out.mkdir(parents=True)
    (out/'results.json').write_text(text,encoding='utf-8')
    np.savez_compressed(out/'resamples.npz',**samples)
    with (out/'paired_record_errors.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['record_id','target','noncollapsed','no_layer_mae','taylor_mae','improvement'])
        writer.writeheader()
        errors=np.abs(pred-y[None,:,None]).mean(axis=2)
        for i,r in enumerate(records):
            writer.writerow(dict(record_id=r['record_id'],target=y[i],noncollapsed=bool(y[i]>=threshold),
                no_layer_mae=errors[0,i],taylor_mae=errors[1,i],improvement=errors[0,i]-errors[1,i]))
    saved=json.loads((out/'results.json').read_text())
    require(saved==result,'Report readback failed')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(dict(output=str(out),primary=result['primary'],robustness=result['robustness']),indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel',type=Path,nargs='?')
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--qwen-root',type=Path)
    parser.add_argument('--dataset',choices=['cifar10','cifar100'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--threshold',type=float,default=20.)
    parser.add_argument('--skip-mixed',action='store_true',help='Explicitly mark mixed model unrun, never pretend it passed')
    a=parser.parse_args()
    if a.qwen_root:
        require(a.dataset and not a.panel and not a.registry,'Use --qwen-root with --dataset, without panel/registry')
        require(not a.out.exists(),'Existing output preserved')
        text,registry,sources=qwen_inputs(a.qwen_root,a.dataset)
        inputs=a.out/'inputs';inputs.mkdir(parents=True)
        (inputs/'panel.csv').write_text(text,encoding='utf-8')
        (inputs/'registry.json').write_text(json.dumps(registry,indent=2))
        (inputs/'source_hashes.json').write_text(json.dumps(sources,indent=2))
        a.panel,a.registry=inputs/'panel.csv',inputs/'registry.json'
        a.out=a.out/'analysis'
    require(a.panel and a.registry,'Supply panel and --registry, or --qwen-root and --dataset')
    run_panel(a.panel,a.registry,a.out,a.draws,a.seed,a.threshold,not a.skip_mixed)


def qwen_inputs(root,dataset):
    root=Path(root)
    needed=[root/'contract.json',root/'manifest.json']
    for cond in ('no-layer','taylor'):
        needed.append(root/'prompts'/dataset/cond/'test_registry.json')
        for seed in SEEDS:
            needed.extend(root/'results'/f'{dataset}-{cond}-s{seed}'/name for name in
                ('per_record_predictions.csv','prediction_contract.json','test_complete.json','runtime_preflight.json'))
    missing=[str(p.relative_to(root)) for p in needed if not p.is_file()]
    require(not missing,'Missing Qwen artifacts (no analysis or retraining performed):\n'+'\n'.join(missing))
    contract=json.loads((root/'contract.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    require(sha(root/'contract.json')==manifest['contract.json'],'Contract hash mismatch')
    expected_n=contract['counts']['test']
    sources={str(p.resolve()):sha(p) for p in needed}
    output=[];registry=None
    for cond in ('no-layer','taylor'):
        path=root/'prompts'/dataset/cond/'test_registry.json'
        current=json.loads(path.read_text())
        if registry is None:
            registry=current
        require(current==registry and len(current)==expected_n,'Different arm registries')
        ids={r['record_id']:r for r in registry}
        require(len(ids)==expected_n,'Duplicate registry IDs')
        for seed in SEEDS:
            key=f'{dataset}-{cond}-s{seed}';run=contract['runs'][key];folder=root/'results'/key
            require(run['dataset']==dataset and run['condition']==cond and run['seed']==seed,'Run identity mismatch')
            require(sha(path)==run['input_sha256']['test_registry.json'],'Registry hash mismatch')
            done=json.loads((folder/'test_complete.json').read_text())
            pred=json.loads((folder/'prediction_contract.json').read_text())
            runtime=json.loads((folder/'runtime_preflight.json').read_text())
            require(done['run']==pred['run']==runtime['run']==key,'Receipt run mismatch')
            require(done['n_samples']==done['n_valid']==pred['n']==pred['finite_accuracy_predictions']==expected_n,'Incomplete prediction coverage')
            require(done['parse_failures']==runtime['over_cap_records']==0,'Parse failures or training truncation')
            require(runtime['effective_seed']==runtime['adapter_seed']==seed,'Effective seed mismatch')
            require(pred['target_kind']==run['target_kind'],'Target semantics mismatch')
            csvpath=folder/'per_record_predictions.csv'
            require(sha(csvpath)==pred['prediction_sha256'],'Prediction hash mismatch')
            with csvpath.open(newline='') as f:
                rows=list(csv.DictReader(f))
            require(len(rows)==expected_n and {r['record_id'] for r in rows}==set(ids),'Prediction IDs missing or duplicated')
            for r in rows:
                m=ids[r['record_id']]
                require(int(r['seed'])==seed and float(r['target'])==m['target'],'Seed/target mismatch')
                require(np.isfinite(float(r['prediction'])),'Non-finite prediction')
                output.append(dict(idx=m['architecture_id'],dataset=m['dataset'],gt_best_accuracy=m['target'],
                    pred_best_accuracy=float(r['prediction']),condition=cond,seed=seed))
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    require(all(sha(p)==h for p,h in sources.items()),'Qwen input changed while reading')
    return text.getvalue(),registry,sources


if __name__=='__main__':
    main()
