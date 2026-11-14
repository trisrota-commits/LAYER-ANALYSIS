import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cond, five
import numpy as np, collections
from scipy import stats

key,_,g0 = cond[(3,'off','taylor')][42]
DS=np.array([d for _,d in key])
mask_c100 = DS=='cifar-100'

def sel_metrics(p,g,mask):
    p,g=p[mask],g[mask]
    tau=float(stats.kendalltau(p,g)[0])
    rho=float(stats.spearmanr(p,g)[0])
    out=dict(tau=tau,rho=rho)
    order=np.argsort(-p, kind='stable')
    best=g.max()
    for k in (1,5,10):
        topk=order[:k]
        out[f'regret@{k}']=float(best-g[topk].max())
        if k>1:
            true_top=set(np.argsort(-g,kind='stable')[:k].tolist())
            out[f'P@{k}']=len(true_top & set(topk.tolist()))/k
    return out

def agg(condkey, mask):
    seeds=sorted(cond[condkey])
    ms=[sel_metrics(cond[condkey][s][1], cond[condkey][s][2], mask) for s in seeds]
    names=ms[0].keys()
    return {n:(float(np.mean([m[n] for m in ms])), float(np.std([m[n] for m in ms],ddof=1)/np.sqrt(len(ms)))) for n in names}, seeds

# random-selection reference on cifar-100
rng=np.random.default_rng(0)
g=g0[mask_c100]; n=len(g); best=g.max()
rand={}
for k in (1,5,10):
    vals=[best-g[rng.choice(n,k,replace=False)].max() for _ in range(20000)]
    rand[f'regret@{k}']=float(np.mean(vals))
print("CIFAR-100 subset n=",n," best gt=",round(best,2))
print("random selection expected regret:", {k:round(v,2) for k,v in rand.items()})
print()
order=['no-layer','weightwatcher','cka','effective-rank','taylor','gradient-flow','sensitivity','activation','taylor-sensitivity']
hdr=f"{'condition':34s} {'tau':>7s} {'rho':>7s} {'P@5':>6s} {'P@10':>6s} {'reg@1':>7s} {'reg@5':>7s} {'reg@10':>7s}"
for zcp in ('off','on'):
    print(f"--- 3acc, ZCP {zcp.upper()} (CIFAR-100 subset, n={n}) ---"); print(hdr)
    for f in order:
        k=(3,zcp,f)
        if k not in cond: continue
        m,seeds=agg(k,mask_c100)
        print(f"{f:34s} {m['tau'][0]:7.4f} {m['rho'][0]:7.4f} {m['P@5'][0]:6.3f} {m['P@10'][0]:6.3f} {m['regret@1'][0]:7.2f} {m['regret@5'][0]:7.2f} {m['regret@10'][0]:7.2f}")
    print()
