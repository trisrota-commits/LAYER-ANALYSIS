"""Recompute both experiments and four manuscript tables from verified archived CSVs."""
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parent/'two-experiments-inputs-v1.zip'
EXPECTED='e3629395b79631261ea11675a41c46d5ddc6a2b5b76c87015dab955d2249bd1e'
MODELS=['qwen_reference','ridge','random_forest','hist_gradient_boosting','epoch1','no_layer']
METRICS=['RMSE','MAE','Pearson','Spearman','R2']
B=20000
SEED=20261025

def correlation(y,p):
    yy=y-y.mean(axis=-1,keepdims=True);pp=p-p.mean(axis=-1,keepdims=True)
    return (yy*pp).sum(axis=-1)/np.sqrt((yy*yy).sum(axis=-1)*(pp*pp).sum(axis=-1))

def metrics(y,p):
    e=p-y;v=((y-y.mean(axis=-1,keepdims=True))**2).mean(axis=-1)
    return np.stack([np.sqrt((e*e).mean(axis=-1)),abs(e).mean(axis=-1),correlation(y,p),
                     correlation(stats.rankdata(y,axis=-1),stats.rankdata(p,axis=-1)),
                     1-(e*e).mean(axis=-1)/v],axis=-1)

def write_csv(name,rows):
    with (HERE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest=json.loads(z.read('manifest.json'))
        for name,item in manifest['files'].items():
            assert hashlib.sha256(z.read('files/'+name)).hexdigest()==item['sha256'],name
        def js(name):return json.loads(z.read('files/'+name))
        def cr(name):return list(csv.DictReader(io.StringIO(z.read('files/'+name).decode())))
        def jr(name):return [json.loads(line) for line in z.read('files/'+name).decode().splitlines() if line.strip()]
        reg=js('taylor-epoch1-3acc-off-v1/test_registry.json')
        ids=[r['record_id'] for r in reg];y=np.array([r['target'] for r in reg]);assert len(set(ids))==len(y)==115
        pred=np.zeros((6,6,115));seeds=list(range(42,48))
        for mi,model in enumerate(MODELS[:4]):
            for si,seed in enumerate(seeds):
                rr=cr(f'classical-eight-features-v1/{model}-s{seed}-predictions.csv')
                assert [r['record_id'] for r in rr]==ids
                assert all(int(r['seed'])==seed for r in rr)
                assert np.array_equal([float(r['target']) for r in rr],y)
                pred[mi,si]=[float(r['prediction']) for r in rr]
        contracts={n:js(n+'/contract.json') for n in ['taylor-epoch1-3acc-off-v1','zcp-off-toggle-refit7-v1','tables9-12-seeds45-47-v1']}
        # Verify all prompt records differ only by the layer table content (or its omission).
        for split in ['train','val','test']:
            full=jr(f'zcp-off-toggle-refit7-v1/prompts/3acc-taylor/{split}_llm_dataset.jsonl')
            one=jr(f'taylor-epoch1-3acc-off-v1/prompts/{split}_llm_dataset.jsonl')
            no=jr(f'zcp-off-toggle-refit7-v1/prompts/3acc-no-layer/{split}_llm_dataset.jsonl')
            assert len(full)==len(one)==len(no)
            def strip_layer(row):
                row=json.loads(json.dumps(row))
                for msg in row['messages']:
                    if msg['role']=='user':
                        msg['content']=re.sub(r'(?m)^LAYER_STATS_MEAN_BY_EPOCH[^\n]*\n(?:[^\n]+\n)+\n','',msg['content'])
                return row
            for a,b,c in zip(full,one,no):
                assert strip_layer(a)==strip_layer(b)==strip_layer(c),'Prompt mismatch beyond layer block'
        for si,run in enumerate(manifest['runs']):
            seed=run['seed'];assert seed==seeds[si]
            rr=cr(run['epoch1_output']+'/per_record_predictions.csv')
            assert [r['record_id'] for r in rr]==ids
            assert np.array_equal([float(r['target']) for r in rr],y)
            assert all(int(r['seed'])==seed for r in rr)
            pred[4,si]=[float(r['prediction']) for r in rr]
            rr=cr(run['no_layer_output']+'/test_predictions.csv')
            assert len(rr)==115
            for i,row in enumerate(rr):
                assert int(row['idx'])==i and row['dataset']==reg[i]['dataset'] and float(row['gt_best_accuracy'])==y[i]
            pred[5,si]=[float(r['pred_best_accuracy']) for r in rr]
            for out in [run['epoch1_output'],run['no_layer_output']]:
                args=js(out+'/effective_trainer_args.json');done=js(out+'/test_complete.json')
                assert args['seed']==seed and args['num_train_epochs']==7
                assert done['n_valid']==done['n_samples']==115 and done['parse_failures']==0
        assert np.isfinite(pred).all()
        original=cr('classical-eight-features-v1/metrics_by_seed.csv')
        source_metrics={(r['model'],int(r['seed'])):r for r in original}
        return pred,y,reg,source_metrics,manifest

def main():
    assert not (HERE/'results.json').exists(),'Refusing existing outputs'
    pred,y,reg,old,manifest=load()
    per_seed=metrics(y,pred)
    for mi,name in enumerate(MODELS[:4]):
        for si,seed in enumerate(range(42,48)):
            np.testing.assert_allclose(per_seed[mi,si],[float(old[name,seed][k]) for k in METRICS],atol=1e-12,rtol=1e-10)
    mean=per_seed.mean(axis=1)
    width=stats.t.ppf(.975,5)*per_seed.std(axis=1,ddof=1)/np.sqrt(6)
    # Candidate/reference pairs. Positive improvement always favors candidate.
    pairs=[(0,1),(0,2),(0,3),(4,0),(4,5)]
    sign=np.array([-1,-1,1,1,1])
    point=np.array([(mean[a]-mean[b])*sign for a,b in pairs])
    boot=np.empty((B,5,5));rng=np.random.default_rng(SEED)
    for start in range(0,B,250):
        end=min(start+250,B);idx=rng.integers(0,115,size=(end-start,115));yy=y[idx][:,None,:]
        scores=np.array([metrics(yy,pp[:,idx].transpose(1,0,2)).mean(axis=1) for pp in pred])
        boot[start:end]=np.stack([(scores[a]-scores[b])*sign for a,b in pairs],axis=1)
    assert np.isfinite(boot).all()
    ci=np.quantile(boot,[.025,.975],axis=0)
    descriptive=[];paired=[]
    for mi,name in enumerate(MODELS):
        for ki,k in enumerate(METRICS):
            descriptive.append(dict(model=name,metric=k,mean=float(mean[mi,ki]),low=float(mean[mi,ki]-width[mi,ki]),high=float(mean[mi,ki]+width[mi,ki])))
    for pi,(a,b) in enumerate(pairs):
        family=15 if pi<3 else 10
        adj=np.quantile(boot[:,pi,:],[.05/(2*family),1-.05/(2*family)],axis=0)
        for ki,k in enumerate(METRICS):
            paired.append(dict(candidate=MODELS[a],reference=MODELS[b],metric=k,improvement=float(point[pi,ki]),low=float(ci[0,pi,ki]),high=float(ci[1,pi,ki]),adjusted_low=float(adj[0,ki]),adjusted_high=float(adj[1,ki]),family_size=family))
    write_csv('descriptive.csv',descriptive);write_csv('paired_bootstrap.csv',paired)
    write_csv('metrics_by_seed.csv',[dict(model=MODELS[mi],seed=seed,**dict(zip(METRICS,per_seed[mi,si]))) for mi in range(6) for si,seed in enumerate(range(42,48))])
    # Standardized CSVs for all conditions, retaining raw target precision in registry.
    for mi,name in enumerate(MODELS):
        for si,seed in enumerate(range(42,48)):
            write_csv(f'{name}-s{seed}-predictions.csv',[dict(record_id=r['record_id'],prediction=float(p),target=float(t),seed=seed) for r,p,t in zip(reg,pred[mi,si],y)])
    with (HERE/'test_registry.json').open('x') as f:json.dump(reg,f,indent=2)
    np.savez_compressed(HERE/'bootstrap_differences.npz',improvement=boot,observed=point)
    report=dict(archive_sha256=EXPECTED,payloads_verified=len(manifest['files']),n_test=115,seeds=list(range(42,48)),bootstrap_resamples=B,rng_seed=SEED,
                sign='Reference minus candidate for errors; candidate minus reference for correlations/R2; positive favors candidate',
                intervals='Descriptive: 95% seed-level t, df5. Paired: 95% unadjusted record-bootstrap percentile, fitted seeds fixed. Bonferroni adjusted columns separately provided, families15/10.',
                descriptive=descriptive,paired=paired)
    with (HERE/'results.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('VERIFIED: archive payloads, ordered IDs, targets, effective seeds and full prompt parity after layer-block removal')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
