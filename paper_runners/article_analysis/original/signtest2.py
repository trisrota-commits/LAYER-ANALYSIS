import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cond, five
import numpy as np
from scipy import stats
LOWER={'RMSE','MAE'}; METR=['RMSE','MAE','Pearson','Spearman','R2']
def means(k,seeds):
    ms=[five(*cond[k][s][1:]) for s in seeds]
    return {m:float(np.mean([x[m] for x in ms])) for m in METR}
feats=['no-layer','weightwatcher','cka','effective-rank','taylor','gradient-flow','sensitivity','activation','all-seven']
rows=[]
for nacc in (2,3):
    for f in feats:
        a,b=(nacc,'off',f),(nacc,'on',f)
        if a in cond and b in cond:
            seeds=sorted(set(cond[a])&set(cond[b]))
            A=means(a,seeds); B=means(b,seeds)
            for m in METR:
                rows.append((nacc,f,m,(A[m]<B[m]) if m in LOWER else (A[m]>B[m]),A[m],B[m],len(seeds)))
k=sum(1 for r in rows if r[3]); n=len(rows)
print(f"Common-seed sign test: ZCP-OFF better in {k}/{n}; p={stats.binomtest(k,n,0.5).pvalue:.3e}")
for nacc in (2,3):
    sub=[r for r in rows if r[0]==nacc]; kk=sum(1 for r in sub if r[3])
    print(f"  {nacc}acc: {kk}/{len(sub)} p={stats.binomtest(kk,len(sub),0.5).pvalue:.4g}")
print("\nper feature condition (OFF better / 5 metrics), with RMSE delta (OFF-ON):")
for nacc in (2,3):
    for f in feats:
        sub=[r for r in rows if r[0]==nacc and r[1]==f]
        if not sub: continue
        kk=sum(1 for r in sub if r[3]); d=[r for r in sub if r[2]=='RMSE'][0]
        print(f"  {nacc}acc {f:16s} OFF-better {kk}/5  dRMSE={d[4]-d[5]:+.4f} (seeds={d[6]})")
