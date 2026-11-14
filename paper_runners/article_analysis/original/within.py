import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cond, five
import numpy as np
from scipy import stats
key,_,g0=cond[(3,'off','taylor')][42]
DS=np.array([d for _,d in key])
m=DS=='cifar-100'
print("Pooled (115) vs within-CIFAR-100 (88), 3acc conditions, means over seeds")
print(f"{'condition':26s} {'RMSE_all':>9s} {'r_all':>7s} {'RMSE_c100':>10s} {'r_c100':>7s} {'rho_c100':>8s}")
for zcp in ('off','on'):
    for f in ['no-layer','taylor','sensitivity','gradient-flow','weightwatcher']:
        k=(3,zcp,f)
        if k not in cond: continue
        ss=sorted(cond[k]); A=[];B=[]
        for s in ss:
            _,p,g=cond[k][s]; A.append(five(p,g)); B.append(five(p[m],g[m]))
        print(f"{f+'/'+zcp:26s} {np.mean([x['RMSE'] for x in A]):9.4f} {np.mean([x['Pearson'] for x in A]):7.4f} {np.mean([x['RMSE'] for x in B]):10.4f} {np.mean([x['Pearson'] for x in B]):7.4f} {np.mean([x['Spearman'] for x in B]):8.4f}")
# how much of pooled variance is between-dataset
g=g0; ds=DS
grand=g.mean(); tot=((g-grand)**2).sum()
bet=sum(((g[ds==d].mean()-grand)**2)*np.sum(ds==d) for d in set(ds))
print(f"\nTarget variance decomposition: between-dataset share = {bet/tot:.3f} ({bet/tot*100:.1f}%)")
# dataset-mean-only predictor
pred_dm=np.array([g[ds==d].mean() for d in ds])
print("dataset-mean-only predictor: RMSE=%.4f r=%.4f R2=%.4f"%(np.sqrt(((pred_dm-g)**2).mean()), stats.pearsonr(pred_dm,g)[0], 1-((pred_dm-g)**2).sum()/tot))
