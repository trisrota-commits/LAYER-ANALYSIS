import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from metrics import cond, five
import numpy as np, json
from scipy import stats
key,_,g=cond[(3,'off','taylor')][42]
g=np.array(g); DS=np.array([d for _,d in key]); m100=DS=='cifar-100'
B=10000; rng=np.random.default_rng(0); n=len(g); I=rng.integers(0,n,size=(B,n))
from scipy.stats import rankdata
def bm(p):
    P=p[I]; G=g[I]; E=P-G
    def pear(X,Y):
        Xc=X-X.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True)
        return (Xc*Yc).sum(1)/np.sqrt((Xc**2).sum(1)*(Yc**2).sum(1))
    return dict(RMSE=np.sqrt((E**2).mean(1)),MAE=np.abs(E).mean(1),Pearson=pear(P,G),
                Spearman=pear(rankdata(P,axis=1),rankdata(G,axis=1)),
                R2=1-(E**2).sum(1)/((G-G.mean(1,keepdims=True))**2).sum(1))
LOWER={'RMSE','MAE'}
print(f"{'condition':26s} {'single-seed mean':>17s} {'ensemble':>9s} {'gain':>7s} {'p':>7s}  {'ens r':>7s} {'ens rho':>8s}")
res={}
for cname in [(3,'off','taylor'),(3,'off','no-layer'),(3,'off','sensitivity'),(3,'on','taylor'),(3,'off','taylor-sensitivity'),(2,'off','taylor'),(2,'off','sensitivity')]:
    if cname not in cond: continue
    seeds=sorted(cond[cname]);
    if len(seeds)<6: continue
    P=np.array([cond[cname][s][1] for s in seeds])
    single=[five(P[i],g) for i in range(len(seeds))]
    sm={k:float(np.mean([x[k] for x in single])) for k in single[0]}
    mu=P.mean(0); ens=five(mu,g)
    # paired bootstrap: ensemble vs mean-over-seeds of single-seed RMSE
    bs=[bm(P[i]) for i in range(len(seeds))]
    be=bm(mu)
    d=np.mean([bs[i]['RMSE'] for i in range(len(seeds))],0)-be['RMSE']
    p=min(1.0,max(2*min((d<=0).mean(),(d>=0).mean()),1/B))
    lab=f"{cname[2]}/{cname[0]}acc/{cname[1]}"
    print(f"{lab:26s} {sm['RMSE']:17.4f} {ens['RMSE']:9.4f} {sm['RMSE']-ens['RMSE']:+7.4f} {p:7.4f}  {ens['Pearson']:7.4f} {ens['Spearman']:8.4f}")
    res[lab]=dict(single=sm,ens=ens,gain=sm['RMSE']-ens['RMSE'],p=float(p))
# how does gain scale with ensemble size
print("\nEnsemble size sweep, Taylor/3acc/ZCP-OFF (all subsets of each size, mean RMSE):")
import itertools
P=np.array([cond[(3,'off','taylor')][s][1] for s in sorted(cond[(3,'off','taylor')])])
for k in range(1,7):
    v=[np.sqrt(((P[list(c)].mean(0)-g)**2).mean()) for c in itertools.combinations(range(6),k)]
    print(f"  m={k}: RMSE={np.mean(v):.4f} (min {np.min(v):.4f}, max {np.max(v):.4f}, {len(v)} subsets)")
json.dump(res,open(paths.artifact("ensemble_results.json"),'w'),indent=1)
