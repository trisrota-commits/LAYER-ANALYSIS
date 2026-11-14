import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from features import load, featurize
from metrics import cond, five
import numpy as np, collections, json
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

recs=load()
key,_,gt=cond[(3,'off','taylor')][42]
gt_ep=None
import csv, os
# recover gt_best_epoch for tie-breaking
path=[r for r in json.load(open(paths.artifact("registry.json"))) if r['run']=='3acc-taylor-s42' and 'zcp-off-toggle' in r['exp']][0]['path']
rowsc=list(csv.DictReader(open(path)))
gt_ep=[float(r['gt_best_epoch']) for r in rowsc]

idx=collections.defaultdict(list)
for i,r in enumerate(recs): idx[(r['dataset'],round(r['y'],2))].append(i)
test_i=[]; amb=0
for (i_,ds),y,ep in zip(key,gt,gt_ep):
    c=idx[(ds,round(float(y),2))]
    if len(c)>1:
        amb+=1
        c2=[j for j in c if recs[j]['epochs_to_best']==int(ep)]
        c=c2 or c
    test_i.append(c[0])
test_i_unique=list(dict.fromkeys(test_i))
print(f"matched 115 test records ({amb} needed tie-breaking, {len(test_i_unique)} distinct snapshot rows)")
train_i=[i for i in range(len(recs)) if i not in set(test_i)]
print(f"train pool {len(train_i)}, test {len(test_i)}")

def build(stats):
    X,names=featurize(recs,stats=stats)
    return X,names
def run(stats,label):
    X,names=build(stats)
    Xtr,ytr=X[train_i],np.array([recs[i]['y'] for i in train_i])
    Xte,yte=X[test_i],np.array([float(v) for v in gt])
    out={}
    models={
     'Ridge':      lambda s: make_pipeline(SimpleImputer(),StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
     'kNN(k=10)':  lambda s: make_pipeline(SimpleImputer(),StandardScaler(),KNeighborsRegressor(n_neighbors=10)),
     'RandomForest':lambda s: make_pipeline(SimpleImputer(),RandomForestRegressor(n_estimators=400,random_state=s,n_jobs=-1)),
     'GradBoost':  lambda s: make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(random_state=s)),
    }
    for mname,mk in models.items():
        ms=[];preds=[]
        for s in range(42,48):
            m=mk(s); m.fit(Xtr,ytr); p=m.predict(Xte); preds.append(p); ms.append(five(p,yte))
        mean={k:float(np.mean([x[k] for x in ms])) for k in ms[0]}
        se={k:float(np.std([x[k] for x in ms],ddof=1)/np.sqrt(6)) for k in ms[0]}
        ens=five(np.mean(preds,0),yte)
        out[mname]=dict(mean=mean,se=se,ens=ens)
        print(f"  {label:22s} {mname:13s} RMSE={mean['RMSE']:7.4f}+-{se['RMSE']:.4f} MAE={mean['MAE']:6.4f} r={mean['Pearson']:.4f} rho={mean['Spearman']:.4f} R2={mean['R2']:.4f} | ens RMSE={ens['RMSE']:.4f}")
    return out

print("\n=== classical baselines on the same 115 test records ===")
res={}
res['no-layer']=run((),'3 acc + hyperparams')
res['taylor']=run(('taylor_imp',),'+ Taylor (3 epochs)')
res['taylor_e3']=None
# single-epoch Taylor (idea 8 / ablation) using classical model
X,names=featurize(recs,stats=('taylor_imp',),epochs=(3,))
def run_custom(X,label):
    Xtr,ytr=X[train_i],np.array([recs[i]['y'] for i in train_i]); Xte,yte=X[test_i],np.array([float(v) for v in gt])
    for mname,mk in [('Ridge',lambda s: make_pipeline(SimpleImputer(),StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13)))),
                     ('GradBoost',lambda s: make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(random_state=s)))]:
        ms=[]
        for s in range(42,48):
            m=mk(s); m.fit(Xtr,ytr); ms.append(five(m.predict(Xte),yte))
        mean={k:float(np.mean([x[k] for x in ms])) for k in ms[0]}
        print(f"  {label:22s} {mname:13s} RMSE={mean['RMSE']:7.4f} MAE={mean['MAE']:6.4f} r={mean['Pearson']:.4f} R2={mean['R2']:.4f}")
print()
run_custom(X,'+ Taylor (epoch 3 only)')
X7,_=featurize(recs,stats=('ww_alpha','grad_norm','dead_frac','taylor_imp','cka_redund','rank_ratio','sensitivity'))
run_custom(X7,'+ all seven stats')
json.dump(res,open(paths.artifact("baseline_results.json"),'w'),indent=1)
