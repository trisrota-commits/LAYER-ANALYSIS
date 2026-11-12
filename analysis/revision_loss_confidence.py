"""Post-hoc five-metric operation-family bootstrap; saved predictions only."""
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np
from scipy.stats import rankdata

EXPECTED = '6f852836ce55e57d4eea49d54b484b33d78a9d0f26b7b12d24f4435487007121'
BUNDLES = {
    'CIFAR-10-valid': 'cifar10-loss-confidence-v1/histgb-recovery-v1/results',
    'CIFAR-100': 'nb201-loss-confidence-histgb-v1/results',
    'ImageNet16-120 (partial)': 'imagenet16-loss-confidence-partial-histgb-v1/histgb_results',
}
ARMS = ['A', 'A_LC', 'A_T', 'A_LC_T']
METRICS = ['MAE', 'RMSE', 'Pearson', 'Spearman', 'R2']
PAIRS = [(1,3),(0,1),(0,2),(2,3)]
SEEDS = list(range(42,48))
DRAWS = 10000


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def corr(y,p):
    a=y-y.mean(); b=p-p.mean(axis=-1,keepdims=True)
    return (b*a).sum(axis=-1)/np.sqrt((a*a).sum()*(b*b).sum(axis=-1))


def metrics(y,p):
    e=p-y
    return np.stack([abs(e).mean(axis=-1),np.sqrt((e*e).mean(axis=-1)),corr(y,p),
                     corr(rankdata(y),rankdata(p,axis=-1)),
                     1-(e*e).sum(axis=-1)/((y-y.mean())**2).sum()],axis=-1)


def improvements(values):
    sign=np.array([-1.,-1.,1.,1.,1.])
    return np.stack([(values[b]-values[a])*sign for a,b in PAIRS])


def load(archive):
    require(sha(archive.read_bytes())==EXPECTED,'Archive checksum mismatch')
    loaded={}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        require(len({m.name for m in members})==len(members),'Duplicate archive members')
        require(all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members),'Unsafe archive member')
        for dataset,folder in BUNDLES.items():
            receipt=json.load(tar.extractfile(folder+'/complete.json'))
            require(receipt['status']=='complete','Incomplete experiment')
            payload={}
            for name in ['predictions.csv','results.json','comparisons.json','fit_contract.json']:
                data=tar.extractfile(folder+'/'+name).read()
                require(sha(data)==receipt['files_sha256'][name],'Payload changed: '+folder+'/'+name)
                payload[name]=data
            rows=list(csv.DictReader(io.StringIO(payload['predictions.csv'].decode())))
            require(set(r['condition'] for r in rows)==set(ARMS),'Unexpected arms')
            require(set(int(r['seed']) for r in rows)==set(SEEDS),'Unexpected seeds')
            grouped={}
            for r in rows:
                key=(r['condition'],int(r['seed']))
                rid=r['record_id']; bucket=grouped.setdefault(key,{})
                require(rid not in bucket,'Duplicate prediction identity')
                bucket[rid]=r
            first=grouped[ARMS[0],SEEDS[0]]; ids=sorted(first)
            require(len({r['architecture_id'] for r in first.values()})==len(ids),'Duplicate architecture')
            y=np.array([float(first[r]['target']) for r in ids])
            families=np.array([first[r]['family'] for r in ids])
            pred=[]
            for arm in ARMS:
                vectors=[]
                for seed in SEEDS:
                    bucket=grouped[arm,seed]
                    require(set(bucket)==set(ids),'Unmatched record IDs')
                    for rid in ids:
                        require(bucket[rid]['architecture_id']==first[rid]['architecture_id'] and bucket[rid]['family']==first[rid]['family'] and float(bucket[rid]['target'])==float(first[rid]['target']),'Identity/target/family mismatch')
                    vectors.append([float(bucket[r]['prediction']) for r in ids])
                vectors=np.asarray(vectors)
                require(np.array_equal(vectors,np.repeat(vectors[:1],len(SEEDS),axis=0)),'Nonidentical seed predictions: do not collapse')
                pred.append(vectors[0])
            pred=np.asarray(pred)
            require(np.isfinite(y).all() and np.isfinite(pred).all(),'Nonfinite input')
            require(len(ids)==receipt['test_records'],'Test count mismatch')
            loaded[dataset]=(y,pred,families,json.loads(payload['results.json']),json.loads(payload['comparisons.json']))
    return loaded


def run(archive, out):
    ARCHIVE, OUTPUT = Path(archive), Path(out)
    require(not OUTPUT.exists(),'Existing output preserved: '+str(OUTPUT))
    loaded=load(ARCHIVE)
    rows=[]; arrays={}; checks=[]
    for dataset,(targets,predictions,family_labels,saved,saved_comparisons) in loaded.items():
        for population,mask in [('all',np.ones(len(targets),dtype=bool)),('target_ge20',targets>=20)]:
            y=targets[mask];p=predictions[:,mask];families=family_labels[mask]
            values=metrics(y,p);gain=improvements(values)
            require(np.isfinite(values).all(),'Undefined full-sample metric')
            for ai,arm in enumerate(ARMS):
                expected=[saved[population][arm]['mean_metrics'][name] for name in METRICS]
                np.testing.assert_allclose(values[ai],expected,rtol=1e-10,atol=1e-10)
            groups=np.unique(families)
            blocks=[np.flatnonzero(families==g) for g in groups]
            sampled=np.random.default_rng(0).integers(len(groups),size=(DRAWS,len(groups)))
            boot=np.empty((DRAWS,len(PAIRS),len(METRICS)))
            for i,draw in enumerate(sampled):
                idx=np.concatenate([blocks[g] for g in draw])
                boot[i]=improvements(metrics(y[idx],p[:,idx]))
            require(np.isfinite(boot).all(),'Undefined bootstrap metric; no silent draw removal')
            # Reproduce all historical MAE intervals before extending inference.
            for ci,(a,b) in enumerate(PAIRS):
                name=ARMS[a]+' -> '+ARMS[b]
                historical=saved_comparisons[population][name]
                np.testing.assert_allclose(gain[ci,0],historical['MAE_improvement'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(np.quantile(boot[:,ci,0],[.025,.975]),historical['family_ci95'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(np.quantile(boot[:,ci,0],[.025/8,1-.025/8]),historical['family_bonferroni8'],atol=1e-10,rtol=1e-10)
                for mi,metric in enumerate(METRICS):
                    low,high=np.quantile(boot[:,ci,mi],[.025,.975])
                    adjlow,adjhigh=np.quantile(boot[:,ci,mi],[.025/40,1-.025/40])
                    rows.append(dict(dataset=dataset,population=population,contrast=name,metric=metric,n=len(y),groups=len(groups),reference=float(values[a,mi]),candidate=float(values[b,mi]),improvement=float(gain[ci,mi]),ci95_low=float(low),ci95_high=float(high),bonferroni40_low=float(adjlow),bonferroni40_high=float(adjhigh),draws=DRAWS))
            arrays[dataset+'__'+population]=boot
            checks.append(dict(dataset=dataset,population=population,all_descriptive_metrics_match=True,all_historical_MAE_intervals_match=True,valid_draws=DRAWS))
            print('VERIFIED',dataset,population,'historical MAE reproduced; five-metric bootstrap complete',flush=True)
    require(sha(ARCHIVE.read_bytes())==EXPECTED,'Input archive changed')
    OUTPUT.mkdir(parents=True)
    for filename,subset in [('all_comparisons.csv',rows),('primary_all_metrics.csv',[r for r in rows if r['population']=='target_ge20' and r['contrast']=='A_LC -> A_LC_T'])]:
        with (OUTPUT/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(subset[0]));w.writeheader();w.writerows(subset)
    np.savez_compressed(OUTPUT/'bootstrap_differences.npz',**arrays)
    result=dict(analysis='Post-hoc expansion to five metrics; not a new preregistered primary test',source_archive_sha256=EXPECTED,
                source_script_sha256=sha(Path(__file__).read_bytes()),checks=checks,results=rows,
                bootstrap='10000 paired operation-multiset-family resamples with replacement; numpy default_rng(0) reset per dataset/population; all records of each sampled group included with multiplicity; no seed resampling',
                identical_seed_predictions='Verified exact equality within every arm across seeds42-47; computing one vector equals averaging seedwise metrics',
                positive_improvement='Reference minus candidate for MAE/RMSE; candidate minus reference for Pearson/Spearman/R2',
                correction='Bonferroni40 within each dataset: 4 contrasts x 2 populations x 5 metrics. Not simultaneous across three datasets. Original Bonferroni8 MAE results reproduced but not overwritten.',
                limitations=['Fixed fitted predictors and acquisitions; no candidate-seed uncertainty','ImageNet16-120 is an ordered-truncation partial cohort','Bootstrap percentile tails at Bonferroni40 are estimated from relatively few extreme draws; near-zero boundaries need cautious interpretation'],
                versions={n:importlib.metadata.version(n) for n in ['numpy','scipy']})
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    files={p.name:sha(p.read_bytes()) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256=files),indent=2)+'\n')
    print('OUTPUT',OUTPUT,flush=True)
    for r in rows:
        if r['population']=='target_ge20' and r['contrast']=='A_LC -> A_LC_T':
            print(r['dataset'],r['metric'],f"{r['reference']:.6f} -> {r['candidate']:.6f}; gain {r['improvement']:+.6f}; CI [{r['ci95_low']:.6f},{r['ci95_high']:.6f}]; Bonf40 [{r['bonferroni40_low']:.6f},{r['bonferroni40_high']:.6f}]")
