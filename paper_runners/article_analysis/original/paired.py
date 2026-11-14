import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from metrics import cond
import numpy as np
from scipy.stats import rankdata
B=10000
rng=np.random.default_rng(0)
key,_,g0=cond[(3,'off','taylor')][42]
n=len(g0); I=rng.integers(0,n,size=(B,n))
DS=np.array([d for _,d in key]); M100=DS=='cifar-100'

def boot_metrics(p,g,sub=None):
    if sub is None: idx=I
    else:
        pos=np.flatnonzero(sub); m=len(pos)
        r=np.random.default_rng(1).integers(0,m,size=(B,m)); idx=pos[r]
    P=p[idx]; G=g[idx]; E=P-G
    rmse=np.sqrt((E**2).mean(1)); mae=np.abs(E).mean(1)
    def pear(X,Y):
        Xc=X-X.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True)
        return (Xc*Yc).sum(1)/np.sqrt((Xc**2).sum(1)*(Yc**2).sum(1))
    pe=pear(P,G)
    sp=pear(rankdata(P,axis=1),rankdata(G,axis=1))
    r2=1-(E**2).sum(1)/((G-G.mean(1,keepdims=True))**2).sum(1)
    return dict(RMSE=rmse,MAE=mae,Pearson=pe,Spearman=sp,R2=r2)

CACHE={}
def cond_boot(k,sub=None):
    ck=(k,sub is not None)
    if ck in CACHE: return CACHE[ck]
    seeds=sorted(cond[k]); acc=None
    for s in seeds:
        _,p,g=cond[k][s]; m=boot_metrics(p,g,sub)
        acc={n2:(m[n2] if acc is None else acc[n2]+m[n2]) for n2 in m}
    out={n2:acc[n2]/len(seeds) for n2 in acc}
    CACHE[ck]=out; return out

LOWER={'RMSE','MAE'}
def compare(ref,cand,sub=None):
    A=cond_boot(ref,sub); C=cond_boot(cand,sub)
    res={}
    for mname in ['RMSE','MAE','Pearson','Spearman','R2']:
        d = (A[mname]-C[mname]) if mname in LOWER else (C[mname]-A[mname])
        delta=float(d.mean()); lo,hi=np.percentile(d,[2.5,97.5])
        pr=min((d<=0).mean(),(d>=0).mean()); p=min(1.0,max(2*pr,1.0/B))
        res[mname]=(delta,float(lo),float(hi),float(p))
    return res

def holm(pvals):
    idx=np.argsort(pvals); adj=np.empty(len(pvals)); run=0.0
    for rank,i in enumerate(idx):
        v=(len(pvals)-rank)*pvals[i]; run=max(run,v); adj[i]=min(1.0,run)
    return adj
def bh(pvals):
    p=np.asarray(pvals); m=len(p); idx=np.argsort(p); adj=np.empty(m); prev=1.0
    for rank in range(m-1,-1,-1):
        i=idx[rank]; v=min(prev, p[i]*m/(rank+1)); adj[i]=v; prev=v
    return adj

FAMILIES={
 "A. Taylor vs no-layer baseline (ZCP OFF)":[((3,'off','no-layer'),(3,'off','taylor'),"no-layer -> Taylor")],
 "B. Enabling ZCPs":[((3,'off','taylor'),(3,'on','taylor'),"Taylor: OFF -> ON"),
                     ((3,'off','taylor-sensitivity'),(3,'on','taylor-sensitivity'),"Taylor+Sens: OFF -> ON")],
 "C. Adding features to Taylor (ZCP OFF)":[((3,'off','taylor'),(3,'off',c),f"Taylor -> {c}") for c in
    ['taylor-sensitivity','taylor-gradient-flow','taylor-activation','taylor-cka','taylor-effective-rank',
     'taylor-sensitivity-activation','taylor-sensitivity-cka','taylor-sensitivity-effective-rank','taylor-sensitivity-gradient-flow']],
 "D. Taylor vs other single features (ZCP OFF)":[((3,'off',f),(3,'off','taylor'),f"{f} -> Taylor") for f in
    ['weightwatcher','cka','effective-rank','gradient-flow','sensitivity','activation']],
}
allp=[]; rowsall=[]
for fam,comps in FAMILIES.items():
    print("="*96); print(fam)
    rows=[]
    for ref,cand,label in comps:
        r=compare(ref,cand)
        for mname,(d,lo,hi,p) in r.items(): rows.append((label,mname,d,lo,hi,p))
    ps=[x[5] for x in rows]; ha=holm(ps)
    print(f"{'comparison':36s} {'metric':9s} {'delta':>10s} {'95% CI':>22s} {'p':>8s} {'p_Holm':>8s}")
    for (label,mname,d,lo,hi,p),a in zip(rows,ha):
        star='*' if a<0.05 else ''
        print(f"{label:36s} {mname:9s} {d:10.4f} [{lo:9.4f},{hi:9.4f}] {p:8.4f} {a:8.4f}{star}")
        rowsall.append((fam,label,mname,d,lo,hi,p,a))
    print()
ps=[r[6] for r in rowsall]; b=bh(ps)
print("="*96)
print(f"Global family: {len(ps)} tests. Survive Holm-within-family at 0.05: {sum(1 for r in rowsall if r[7]<0.05)}; survive BH-global at 0.05: {int((b<0.05).sum())}")
for r,bb in zip(rowsall,b):
    if bb<0.05: print(f"  BH-sig: {r[1]:36s} {r[2]:9s} delta={r[3]:8.4f} p={r[6]:.4f} q={bb:.4f}")
import json; json.dump([list(r)+[float(bb)] for r,bb in zip(rowsall,b)],open(paths.artifact("paired_results.json"),'w'),indent=0)
