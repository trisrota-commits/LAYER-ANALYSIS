from pathlib import Path
import csv, json, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

B = Path(__file__).resolve().parents[1]
out = Path(__file__).resolve().parent
rp = B/'classical-eight-features-v1/records.csv'
pp = B/'primary-endpoint-independent-v1/lemur-inputs/panel.csv'
gp = B/'primary-endpoint-independent-v1/lemur-inputs/test_registry.json'
oldp = B/'primary-endpoint-independent-v1/lemur-reference-v2/results.json'
paths = [rp, pp, gp, oldp]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
registry = json.loads(gp.read_text())
records = [r for r in csv.DictReader(rp.open()) if r['split']=='test']
assert len(records)==len(registry)==115
lookup = {r['record_id']:r for r in records}
assert len(lookup)==115 and set(lookup)=={r['record_id'] for r in registry}
ids = [r['record_id'] for r in registry]
y = np.array([r['target'] for r in registry])
for r in registry:
    x = lookup[r['record_id']]
    assert float(x['target'])==r['target'] and x['dataset']==r['dataset']
assert len({lookup[i]['architecture_id'] for i in ids})==115
scores = {'epoch3_accuracy':-np.array([float(lookup[i]['accuracy_3']) for i in ids])}
rows = list(csv.DictReader(pp.open()))
assert len(rows)==1380
for c in ['no-layer','taylor']:
    predictions=[]
    for seed in range(42,48):
        subset=[r for r in rows if r['condition']==c and int(r['seed'])==seed]
        tab={r['idx']:r for r in subset}
        assert len(subset)==len(tab)==115 and set(tab)==set(ids)
        assert all(float(tab[i]['gt_best_accuracy'])==v for i,v in zip(ids,y))
        predictions.append([float(tab[i]['pred_best_accuracy']) for i in ids])
    scores[c]=-np.mean(predictions,axis=0)
assert all(np.isfinite(s).all() for s in scores.values())
labels=y<20
pos=np.flatnonzero(labels); neg=np.flatnonzero(~labels)
assert len(pos)==9 and len(neg)==106
rng=np.random.default_rng(1)
res={k:np.empty(10000) for k in scores}
matrices={k:(s[pos,None]>s[neg][None,:]).astype(float)+.5*(s[pos,None]==s[neg][None,:]) for k,s in scores.items()}
for start in range(0,10000,128):
    n=min(128,10000-start)
    pi=rng.integers(len(pos),size=(n,len(pos)))
    ni=rng.integers(len(neg),size=(n,len(neg)))
    for k,m in matrices.items():
        res[k][start:start+n]=m[pi[:,:,None],ni[:,None,:]].mean(axis=(1,2))
        if start==0:
            ix=np.concatenate([pos[pi[0]],neg[ni[0]]])
            np.testing.assert_allclose(res[k][0],roc_auc_score(labels[ix],scores[k][ix]))
result={'n':115,'collapsed':9,'threshold':20,'draws':10000,'seed':1,'method':'Paired class-stratified record bootstrap; fixed fitted six-seed prediction averages; positive differences favor learned predictor; exploratory unadjusted intervals','sources':hashes,'scores':{},'paired':{}}
old=json.loads(oldp.read_text())
for k in scores:
    auc=float(roc_auc_score(labels,scores[k])); np.testing.assert_allclose(auc,matrices[k].mean())
    ci=np.quantile(res[k],[.025,.975]).tolist()
    result['scores'][k]={'auc':auc,'ci95':ci}
    if k in ['no-layer','taylor']:
        np.testing.assert_allclose(auc,old['collapse_auc'][k]['auc'])
        np.testing.assert_allclose(ci,old['collapse_auc'][k]['ci95'])
        delta=res[k]-res['epoch3_accuracy']
        result['paired'][k]={'auc_difference':auc-roc_auc_score(labels,scores['epoch3_accuracy']),'ci95':np.quantile(delta,[.025,.975]).tolist()}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
with (out/'results.json').open('x') as f: json.dump(result,f,indent=2)
with (out/'bootstrap.npz').open('xb') as f: np.savez_compressed(f,**res)
with (out/'matched_scores.csv').open('x',newline='') as f:
    w=csv.writer(f); w.writerow(['record_id','target','collapsed',*scores])
    for j,i in enumerate(ids): w.writerow([i,y[j],int(labels[j]),*[scores[k][j] for k in scores]])
assert json.loads((out/'results.json').read_text())==result
print(json.dumps(result,indent=2))
print('VERIFIED: exact identity/target pairing, unique architecture IDs, original AUROCs and intervals reproduced, sklearn cross-check, source hashes unchanged, result readback.')
