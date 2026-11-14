import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cond, five
import numpy as np
from scipy import stats
key,_,g=cond[(3,'off','taylor')][42]; g=np.array(g)
DS=np.array([d for _,d in key]); m100=DS=='cifar-100'
def ens(K):
    return np.array([cond[K][s][1] for s in sorted(cond[K])]).mean(0)
mu=ens((3,'off','taylor')); mu0=ens((3,'off','no-layer'))
print("=== Stratified evaluation, Taylor/3acc/ZCP-OFF, 6-seed ensemble ===")
def line(lbl,sel,p=mu):
    f=five(p[sel],g[sel])
    print(f"  {lbl:34s} n={sel.sum():3d} RMSE={f['RMSE']:6.3f} MAE={f['MAE']:6.3f} r={f['Pearson']:.4f} rho={f['Spearman']:.4f} R2={f['R2']:7.4f}")
allsel=np.ones(len(g),bool)
line("all 115 records",allsel)
line("collapsed runs (acc < 20%)",g<20)
line("non-collapsed (acc >= 20%)",g>=20)
line("CIFAR-100 only",m100)
line("CIFAR-100, non-collapsed",m100&(g>=20))
print()
print("=== Collapse detection (acc < 20%) from the prediction alone ===")
for thr in (20,30):
    y=(g<thr).astype(int); n1=y.sum(); n0=len(y)-n1
    U=stats.mannwhitneyu(-mu[y==1],-mu[y==0]).statistic; auc=U/(n1*n0)
    # bootstrap CI
    rng=np.random.default_rng(0); vals=[]
    for _ in range(5000):
        i1=rng.choice(np.flatnonzero(y==1),n1); i0=rng.choice(np.flatnonzero(y==0),n0)
        vals.append(stats.mannwhitneyu(-mu[i1],-mu[i0]).statistic/(n1*n0))
    lo,hi=np.percentile(vals,[2.5,97.5])
    print(f"  threshold {thr}%: positives={n1}  AUROC={auc:.3f} [{lo:.3f},{hi:.3f}]")
print()
print("=== Variance accounting for pooled R^2 (Taylor/3acc/OFF ensemble) ===")
tot=((g-g.mean())**2).sum()
bet=sum(((g[DS==d].mean()-g.mean())**2)*np.sum(DS==d) for d in set(DS))
print(f"  between-dataset share of target variance        {bet/tot:.3f}")
coll=(g<20).astype(float)
betc=sum(((g[coll==c].mean()-g.mean())**2)*np.sum(coll==c) for c in (0,1))
print(f"  collapsed-vs-converged share of target variance {betc/tot:.3f}")
both=np.array([f"{d}|{int(c)}" for d,c in zip(DS,coll)])
betb=sum(((g[both==k].mean()-g.mean())**2)*np.sum(both==k) for k in set(both))
print(f"  dataset x collapse share                        {betb/tot:.3f}")
for lbl,pred in [("dataset-mean predictor",np.array([g[DS==d].mean() for d in DS])),
                 ("dataset x collapse mean predictor",np.array([g[both==k].mean() for k in both]))]:
    f=five(pred,g); print(f"  {lbl:34s} RMSE={f['RMSE']:6.3f} r={f['Pearson']:.4f} R2={f['R2']:.4f}")
print()
print("=== Taylor's advantage over the no-layer baseline, by stratum (ensembles) ===")
for lbl,sel in [("all 115",allsel),("non-collapsed",g>=20),("CIFAR-100",m100),("CIFAR-100 non-collapsed",m100&(g>=20))]:
    a=five(mu0[sel],g[sel]); b=five(mu[sel],g[sel])
    print(f"  {lbl:26s} dRMSE={a['RMSE']-b['RMSE']:+.3f}  dr={b['Pearson']-a['Pearson']:+.4f}  drho={b['Spearman']-a['Spearman']:+.4f}")
