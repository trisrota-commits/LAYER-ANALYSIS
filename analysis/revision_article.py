"""Import-safe, prediction-only adapter for the recovered article analyses.

The original registry is deliberately never executed. Preserved scripts run with
canonical condition arrays and a confined artifact path; no prediction training.
"""
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from scipy import stats
from analysis import thesis_lemur as lemur


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_inputs(repo):
    frozen=json.loads((Path(repo)/'INPUT_MANIFEST.json').read_text())['files']
    names=['evidence/overleaf-table-evidence-reproducible-20261019-v2.zip',
           'evidence/reproduce_main_paired_tables.py']
    expected={name:frozen[name] for name in names}
    expected['evidence/thesis-new-logs-20261112-084434.zip']='2b233d441517d5fca8c4d6d564abce824706c8d2828aae0dbf48d867c895e249'
    for name,digest in expected.items():
        require(sha((Path(repo)/name).read_bytes())==digest,'Frozen article input changed: '+name)
    return expected


def source_dir(repo):
    return Path(repo) / 'paper_runners/article_analysis'


def verify_sources(repo):
    base=source_dir(repo)
    manifest=json.loads((base/'sources.json').read_text())
    for name, record in manifest['files'].items():
        require(sha((base/'original'/name).read_bytes())==record['sha256'], 'Original source checksum mismatch: '+name)
    import zipfile
    audit_member='workspace_evidence/registry-audit-corrected-v1/verified/audit_results.json'
    with zipfile.ZipFile(Path(repo)/'evidence/thesis-new-logs-20261112-084434.zip') as packet:
        audit_bytes=packet.read(audit_member)
    published=json.loads(audit_bytes)['source_script_sha256']
    require({r['member']:r['sha256'] for r in manifest['files'].values()}==published,
            'Recovered source manifest differs from published source audit')
    manifest['published_audit']={'archive':'evidence/thesis-new-logs-20261112-084434.zip',
                                 'member':audit_member,'sha256':sha(audit_bytes)}
    return manifest


def functions(repo, name, namespace):
    """Extract function definitions only: never execute top-level legacy I/O."""
    path=source_dir(repo)/'original'/name
    tree=ast.parse(path.read_bytes(),filename=str(path))
    tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))]
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def load_conditions(repo):
    repo=Path(repo)
    # Invalidate inherited archive caches: provenance must refer to current bytes.
    lemur.pools.cache_clear()
    lemur.metric_function.cache_clear()
    data,y,paths=lemur.load_main(repo)
    pools=lemur.pools(repo)
    sources=[]
    def blob(path):
        pool,member=path.split(':',1)
        payload=pools[pool][member]
        sources.append(dict(pool=pool,member=member,sha256=sha(payload)))
        return payload
    first=paths['3/off/taylor']
    _,_,canonical=lemur.matrix([blob(p) for p in first],115)
    keys=[(int(k[0]),k[1]) for k in canonical]
    for members in paths.values():
        _,target,identity=lemur.matrix([blob(p) for p in members],115)
        require(identity==canonical and np.array_equal(target,y),'Main identity changed')
    combos=[('taylor-sensitivity','round2-taylor-pairs-3acc-off-v1'),
            ('taylor-activation','round2-taylor-pairs-3acc-off-v1'),
            ('taylor-cka','round2-taylor-pairs-3acc-off-v1'),
            ('taylor-gradient-flow','readmission-3acc-off-v1'),
            ('taylor-effective-rank','readmission-3acc-off-v1'),
            ('taylor-sensitivity-activation','round3-triples-3acc-off-v1'),
            ('taylor-sensitivity-cka','round3-triples-3acc-off-v1'),
            ('taylor-sensitivity-gradient-flow','readmission-3acc-off-v1'),
            ('taylor-sensitivity-effective-rank','readmission-3acc-off-v1'),
            ('taylor-sensitivity','taylor-sensitivity-zcp-on-v1')]
    for feature,batch in combos:
        blobs=[]
        for s in range(42,48):
            member,_=lemur.get(pools['main'],f'{batch}/results/{feature}-s{s}/test_predictions.csv')
            blobs.append(blob('main:'+member))
        p,target,identity=lemur.matrix(blobs,115)
        require(identity==canonical and np.array_equal(y,target),'Combo identity mismatch')
        data[3,'on' if batch=='taylor-sensitivity-zcp-on-v1' else 'off',feature]=p
    cond={k:{s:(keys,p[i],y) for i,s in enumerate(range(42,48))} for k,p in data.items()}
    return cond,list({(r['pool'],r['member']):r for r in sources}.values()),keys


def execute_original(repo,name,cond,out):
    """Execute audited analytical statements, excluding legacy import plumbing.

    No sys.modules or cwd mutation. 'paths' is an explicit confined dependency.
    Only this fixed allowlist of non-training source programs is executable.
    """
    require(name in {'paired.py','selection.py','ensemble.py','diagnosis.py','within.py','signtest2.py','substitution_and_budget.py'},'Script not approved')
    def artifact(filename):
        require(Path(filename).name==filename,'Unsafe artifact name')
        return str(out/filename)
    ns={'__name__':'recovered_article','cond':cond,'paths':SimpleNamespace(artifact=artifact)}
    captures=[]
    ns['__capture_collapse']=lambda threshold,n,auc,lo,hi: captures.append(
        dict(threshold=float(threshold),n_positive=int(n),auroc=float(auc),ci=[float(lo),float(hi)],
             bootstrap='original diagnosis.py: stratified, 5000 draws, RNG 0 reset per threshold'))
    functions(repo,'metrics.py',ns)
    path=source_dir(repo)/'original'/name
    tree=ast.parse(path.read_bytes(),filename=str(path))
    kept=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='metrics': continue
        if isinstance(node,ast.Import):
            node.names=[a for a in node.names if a.name not in {'paths','sys'}]
            if not node.names: continue
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='sys.path.insert': continue
        if name=='diagnosis.py' and isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='thr':
            node.body.extend(ast.parse('__capture_collapse(thr,n1,auc,lo,hi)').body)
        kept.append(node)
    tree.body=kept
    ast.fix_missing_locations(tree)
    log=io.StringIO()
    with contextlib.ExitStack() as stack:
        def confined_open(filename,mode='r',*args,**kwargs):
            path=Path(filename).resolve()
            require(path.parent==out.resolve() and path.name in {'paired_results.json','ensemble_results.json'} and mode=='w',
                    'Unapproved legacy file access: '+str(path))
            # Exclusive creation and deterministic close without modifying source bytes.
            return stack.enter_context(path.open('x',*args,**kwargs))
        ns['open']=confined_open
        with contextlib.redirect_stdout(log): exec(compile(tree,str(path),'exec'),ns)
    ns['collapse_captures']=captures
    (out/(Path(name).stem+'.log')).write_text(log.getvalue(),encoding='utf-8')
    return ns


def fit_is_valid(converged, information_eigenvalues, messages, standard_errors):
    eig=np.asarray(information_eigenvalues); se=np.asarray(standard_errors)
    return bool(converged and eig.size and np.isfinite(eig).all() and (eig>0).all()
                and np.isfinite(se).all() and (se>0).all()
                and not any(any(x in m.lower() for x in ('hessian','singular','converg','boundary')) for m in messages))


def mixed_fit(sub):
    """Original statsmodels formula/default optimizer, with added validity gate."""
    result={'formula':'d_ae ~ 1','groups':'one','vc_formula':{'rec':'0+C(idx)','sd':'0+C(seed)'},
            'reml':True,'optimizer':'statsmodels default','paper_p':0.0107,'p':None,'status':'unavailable'}
    messages=[]
    try:
        import statsmodels.formula.api as smf
        import statsmodels
        result['statsmodels_version']=statsmodels.__version__
        s=sub.copy(); s['one']=1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r=smf.mixedlm('d_ae ~ 1',s,groups='one',vc_formula={'rec':'0+C(idx)','sd':'0+C(seed)'}).fit(reml=True)
            hessian,singular=r.model.hessian(r.params_object)
            eig=np.linalg.eigvalsh(-hessian)
            se=np.asarray(r.bse)
            messages=[str(w.message) for w in caught]
        valid=fit_is_valid(r.converged,eig,messages,se) and not singular
        result.update(status='valid' if valid else 'invalid_fit_inference_withheld',converged=bool(r.converged),
                      singular=bool(singular),information_eigenvalues=eig.tolist(),standard_errors=se.tolist(),
                      gain=float(r.params['Intercept']),variance_components=r.vcomp.tolist(),
                      p=float(r.pvalues['Intercept']) if valid else None,
                      reconciliation='reproduced' if valid and abs(float(r.pvalues['Intercept'])-.0107)<.00005 else 'gap: paper p=0.0107 not reproduced with valid inference')
    except Exception as exc:
        result['error']=str(exc)
        result['reconciliation']='gap: mixed-model inference unavailable; paper p is not substituted'
    result['warnings']=messages
    return result


def primary(repo,cond,out):
    import pandas as pd
    ns={'np':np,'pd':pd,'stats':stats,'SEED':20261022,'B_BOOT':20000,'COLLAPSE_THRESHOLD':20.0}
    functions(repo,'primary_endpoint.py',ns)
    rows=[]
    for feature,label in [('no-layer','no-layer'),('taylor','taylor')]:
        for seed,(keys,p,y) in cond[3,'off',feature].items():
            rows.extend(dict(idx=k[0],dataset=k[1],seed=seed,condition=label,gt_best_accuracy=float(g),pred_best_accuracy=float(v)) for k,g,v in zip(keys,y,p))
    pd.DataFrame(rows).to_csv(out/'panel.csv',index=False)
    df,w=ns['load'](out/'panel.csv'); nc=w[w.non_collapsed]; co=w[~w.non_collapsed]
    R={'n_records':int(w.idx.nunique()),'n_non_collapsed':int(nc.idx.nunique()),'n_collapsed':int(co.idx.nunique()),'n_seeds':6}
    R['primary_by_record']=ns['by_record'](nc)
    R['primary_by_seed']=ns['by_seed'](nc)
    R['primary_holm_adjusted']=dict(zip(['by_record','by_seed'],ns['holm']([R['primary_by_record']['p'],R['primary_by_seed']['p']]).tolist()))
    R['supporting']={name:ns[fn](sub,*args) for name,fn,sub,args in [
        ('mae_all_by_record','by_record',w,()),('mae_all_by_seed','by_seed',w,()),
        ('mae_collapsed_by_record','by_record',co,()),('rmse_non_collapsed_by_seed','by_seed',nc,('rmse',)),('rmse_all_by_seed','by_seed',w,('rmse',))]}
    # Only exclude the original unchecked mixed-model try block. All resampling,
    # order, RNG state consumption, pooled cells and location functions unchanged.
    path=source_dir(repo)/'original/primary_endpoint.py'
    tree=ast.parse(path.read_bytes()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='robustness')
    fn.body=[n for n in fn.body if not isinstance(n,ast.Try)]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    R['robustness_non_collapsed']=ns['robustness'](nc)
    R['robustness_non_collapsed']['crossed_mixed_model']=mixed_fit(nc)
    R['collapse_detection']=ns['collapse_auroc'](df)
    R['means']={}
    for label,sub in [('all',w),('non_collapsed',nc),('collapsed',co)]:
        R['means']['mae_'+label]={'no_layer':float(sub.ae_no.mean()),'taylor':float(sub.ae_ta.mean())}
        R['means']['rmse_'+label]={arm:float(sub.groupby('seed')[col].mean().pow(.5).mean()) for arm,col in [('no_layer','se_no'),('taylor','se_ta')]}
    dz=R['primary_by_record']['dz']; R['power']={'dz':dz,'n_for_80pct':ns['power_n'](dz,.8),'n_for_90pct':ns['power_n'](dz,.9),'n_available':R['n_non_collapsed']}
    R['contract']={'bootstrap_draws':20000,'seed':20261022,'robustness_seed':20261023,'pooled_cells':len(nc),'mixed_model':'original formula and fit; independent inference validity gate'}
    return R


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.integer,np.floating,np.bool_)): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def dump(path,value):
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')


def run(repo,out):
    """Recompute article results into a NEW directory. Returns the result dict."""
    repo,out=Path(repo).resolve(),Path(out).resolve()
    require(not out.exists(),'Existing output preserved: '+str(out))
    frozen_inputs=verify_inputs(repo)
    manifest=verify_sources(repo)
    archive=repo/'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip'
    before=sha(archive.read_bytes())
    cond,sources,keys=load_conditions(repo)
    import pandas as pd
    require(tuple(map(int,pd.__version__.split('.')[:2])) >= (2,2),'pandas >=2.2 required by original include_groups=False calls')
    out.mkdir(parents=True,exist_ok=False)
    results={'primary_endpoint':primary(repo,cond,out)}
    paired=execute_original(repo,'paired.py',cond,out)
    rows=[]
    # Preserve original bootstrap-mean deltas, but also expose the paper's actual
    # point estimand. Do not relabel the legacy bootstrap mean as observed delta.
    mapping={label:(ref,cand) for comps in paired['FAMILIES'].values() for ref,cand,label in comps}
    metric=lemur.metric_function(repo)
    for row,q in zip(paired['rowsall'],paired['b']):
        fam,label,m,d,lo,hi,p,h=row; ref,cand=mapping[label]; j=lemur.METRICS.index(m)
        a=np.array([cond[ref][s][1] for s in range(42,48)]); b=np.array([cond[cand][s][1] for s in range(42,48)])
        y=cond[ref][42][2]
        delta=(metric(b,y).mean(0)[j]-metric(a,y).mean(0)[j])*(-1 if j<2 else 1)
        rows.append(dict(family=fam,comparison=label,metric=m,delta=float(delta),bootstrap_mean_delta=d,low=lo,high=hi,p=p,p_holm=h,q_bh=float(q)))
    require(len(rows)==90,'Incomplete multiplicity family')
    write_csv(out/'multiplicity.csv',rows); results['multiplicity']=rows
    selection=execute_original(repo,'selection.py',cond,out)
    selrows=[]
    for zcp in ['off','on']:
        for feature in selection['order']:
            key=(3,zcp,feature)
            if key not in cond: continue
            aggregate,seeds=selection['agg'](key,selection['mask_c100'])
            for name,(mean,se) in aggregate.items():
                selrows.append(dict(observations=3,zcp=zcp,feature=feature,population='cifar-100',seeds=len(seeds),metric=name,mean=mean,se=se))
    results['selection']={'rows':selrows,'random_reference':selection['rand']}; write_csv(out/'selection.csv',selrows)
    ensemble=execute_original(repo,'ensemble.py',cond,out); results['ensemble']=ensemble['res']
    import itertools
    sweep=[]
    for feature in ['taylor','no-layer']:
        P=np.array([cond[3,'off',feature][s][1] for s in range(42,48)]); y=cond[3,'off',feature][42][2]
        for size in range(1,7):
            mse=np.array([np.mean((P[list(c)].mean(0)-y)**2) for c in itertools.combinations(range(6),size)])
            sweep.append(dict(feature=feature,size=size,subsets=len(mse),mean_rmse=float(np.sqrt(mse).mean()),sqrt_mean_mse=float(np.sqrt(mse.mean())),min_rmse=float(np.sqrt(mse).min()),max_rmse=float(np.sqrt(mse).max())))
    results['ensemble_size_sweep']=sweep; write_csv(out/'ensemble_size.csv',sweep)
    diagnosis=execute_original(repo,'diagnosis.py',cond,out)
    strata=[]
    for label,mask in [('all',diagnosis['allsel']),('collapsed',diagnosis['g']<20),('non_collapsed',diagnosis['g']>=20),('cifar-100',diagnosis['m100']),('cifar-100_non_collapsed',diagnosis['m100']&(diagnosis['g']>=20))]:
        for arm,p in [('taylor',diagnosis['mu']),('no-layer',diagnosis['mu0'])]:
            strata.append(dict(population=label,condition=arm,n=int(mask.sum()),**diagnosis['five'](p[mask],diagnosis['g'][mask])))
    results['stratification']=strata; write_csv(out/'stratification.csv',strata)
    results['collapse_detection_stratified_bootstrap']=diagnosis['collapse_captures']
    results['variance_decomposition']={key:float(diagnosis[key]/diagnosis['tot']) for key in ['bet','betc','betb']}
    within=execute_original(repo,'within.py',cond,out)
    withinrows=[]
    for zcp in ['off','on']:
        for feature in ['no-layer','taylor','sensitivity','gradient-flow','weightwatcher']:
            for population,mask in [('all',np.ones(len(keys),dtype=bool)),('cifar-100',within['m'])]:
                values=[within['five'](p[mask],g[mask]) for _,p,g in cond[3,zcp,feature].values()]
                withinrows.append(dict(zcp=zcp,feature=feature,population=population,n=int(mask.sum()),
                                       **{m:float(np.mean([v[m] for v in values])) for m in lemur.METRICS}))
    results['within_dataset']=withinrows; write_csv(out/'within_dataset.csv',withinrows)
    sign=execute_original(repo,'signtest2.py',cond,out)
    results['zcp_sign_test']={'n':sign['n'],'off_better':sign['k'],'p':float(stats.binomtest(sign['k'],sign['n'],.5).pvalue),'rows':sign['rows']}
    substitution=execute_original(repo,'substitution_and_budget.py',cond,out)
    results['substitution']={'features':substitution['feats'],'standalone_strength':substitution['x'],
                             'zcp_benefit':substitution['y'],'pearson':list(substitution['r']),
                             'spearman':list(substitution['rs'])}
    results['s24_reconciliation']={'mixed_model_paper_p':.0107,
        'mixed_model':results['primary_endpoint']['robustness_non_collapsed']['crossed_mixed_model']['reconciliation'],
        'trimmed_mean_paper_rounded':{'no_layer':5.1560,'taylor':4.7160},
        'trimmed_mean_computed':results['primary_endpoint']['robustness_non_collapsed']['location_trimmed_20pct'],
        'note':'Pooled-cell estimands retained; paper rounding discrepancy recorded, not forced.'}
    results['coverage']={'original_executed':['primary_endpoint (functions; guarded mixed fit)','paired','selection','ensemble','diagnosis','within','signtest2','substitution_and_budget'],
                         'independently_implemented':['canonical routing/identity verification','flat CSV projections','observed multiplicity point deltas','mixed-fit validity diagnostics','ensemble-size CSV (original expressions)'],
                         'source_available_not_executed':['build_panel (replaced canonical input routing)','registry (stale seed mapping; prohibited)','features','baselines (training prohibited)','figures','figures_panel'],
                         'limitations':['Not a claim to export every article/supplement table. Baseline training, figures, hardcoded LODO/cost data are not regenerated here.', 'Budget simulation detailed readouts remain in the original rounded console log; within-dataset and both collapse-bootstrap variants also have machine-readable full-precision outputs.', 'S24 paper mixed-model p=.0107 is not substituted for invalid/unavailable inference.']}
    dump(out/'results.json',results)
    require(sha(archive.read_bytes())==before,'Prediction archive changed during analysis')
    require(verify_inputs(repo)==frozen_inputs,'Frozen inputs changed during analysis')
    require(verify_sources(repo)==manifest,'Original sources changed during analysis')
    provenance={'original_sources':manifest,'input_sha256':frozen_inputs,'prediction_sources':sources,'prediction_archive':{'path':str(archive.relative_to(repo)),'sha256':before,'unchanged_after_run':True},'identities':keys,'dependencies':{}}
    import importlib.metadata
    for package in ['numpy','scipy','pandas','statsmodels']:
        try: provenance['dependencies'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: provenance['dependencies'][package]=None
    provenance['adapter_sha256']=sha(Path(__file__).read_bytes())
    dump(out/'provenance.json',provenance)
    mixed=results['primary_endpoint']['robustness_non_collapsed']['crossed_mixed_model']
    (out/'report.md').write_text('# Recovered article analyses\n\nCanonical 115 identities; six seeds (42–47); no network or predictor training.\n\n'
        'Original byte-preserved functions/programs executed with explicit canonical arrays. See results.json coverage and provenance.json.\n\n'
        f"## S24 reconciliation\n\n20,000 draws; original RNG streams and 636 pooled cells. Mixed-model status: **{mixed['status']}**. {mixed['reconciliation']}.\n\n"
        'Pooled location summaries retain full precision; printed paper rounding is not forced. Multiplicity CSV separates observed delta from original bootstrap-mean delta.\n\n'
        '## Uncaptured tables\n\nNo baseline refitting, figure rendering, or hardcoded LODO/cost-table regeneration. Within/substitution/budget and diagnostic detailed readouts are in named logs. This mode does not assert complete paper table coverage.\n',encoding='utf-8')
    receipt={'status':'complete','mixed_model_status':mixed['status'],'files_sha256':{p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
    dump(out/'receipt.json',receipt)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=run(args.repo,args.out)
    print(json.dumps(clean({'primary':result['primary_endpoint'],'multiplicity_tests':len(result['multiplicity'])}),indent=2))
