import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cond, five
import numpy as np
from scipy import stats
key,_,g0=cond[(3,'off','taylor')][42]
DS=np.array([d for _,d in key]); m=DS=='cifar-100'
EPOCH=139.53; TAYLOR=0.49; TOTAL_EP=50
def pool(K):
    seeds=sorted(cond[K]); return np.array([cond[K][s][1] for s in seeds]).mean(0)
g=g0[m]; n=len(g)
print("=== IDEA 1: screening-budget simulation on the 88 CIFAR-100 records ===")
print(f"best true accuracy in pool: {g.max():.2f}; mean {g.mean():.2f}")
print("budget = seconds of candidate training to find the returned architecture\n")
def cost_screen(k, obs=3): return n*obs*EPOCH + n*obs*TAYLOR + k*TOTAL_EP*EPOCH
def cost_full():          return n*TOTAL_EP*EPOCH
rng=np.random.default_rng(0)
print(f"{'strategy':34s} {'k':>3s} {'found acc':>10s} {'regret':>7s} {'GPU-h':>8s} {'speedup':>8s}")
print(f"{'train all 88 to 50 epochs':34s} {'-':>3s} {g.max():10.2f} {0.0:7.2f} {cost_full()/3600:8.1f} {1.0:8.2f}x")
for K,label in [((3,'off','taylor'),'3-acc + Taylor screen'),((3,'off','no-layer'),'3-acc only screen'),((3,'on','taylor'),'3-acc + Taylor + ZCP screen')]:
    p=pool(K)[m]
    for k in (3,5,10):
        top=np.argsort(-p)[:k]; found=g[top].max(); c=cost_screen(k)
        print(f"{label:34s} {k:3d} {found:10.2f} {g.max()-found:7.2f} {c/3600:8.1f} {cost_full()/c:8.2f}x")
# random search with equal budget
for k in (3,5,10):
    c=cost_screen(k); eq=int(c/(TOTAL_EP*EPOCH))
    vals=[g[rng.choice(n,min(eq,n),replace=False)].max() for _ in range(5000)]
    print(f"{'random search, equal budget':34s} {eq:3d} {np.mean(vals):10.2f} {g.max()-np.mean(vals):7.2f} {c/3600:8.1f} {cost_full()/c:8.2f}x")

print("\n=== Substitution test: does the ZCP gain depend on how strong the layer block is? ===")
feats=['no-layer','weightwatcher','cka','effective-rank','taylor','gradient-flow','sensitivity','activation']
x=[];y=[]
for f in feats:
    a,b=(3,'off',f),(3,'on',f)
    if a in cond and b in cond:
        ss=sorted(set(cond[a])&set(cond[b]))
        ro=np.mean([five(*cond[a][s][1:])['RMSE'] for s in ss])
        rn=np.mean([five(*cond[b][s][1:])['RMSE'] for s in ss])
        base=np.mean([five(*cond[(3,'off','no-layer')][s][1:])['RMSE'] for s in sorted(cond[(3,'off','no-layer')])])
        strength=base-ro   # how much the layer statistic helps on its own (ZCP OFF)
        gain_from_zcp=ro-rn
        x.append(strength); y.append(gain_from_zcp)
        print(f"  {f:16s} standalone gain={strength:+.3f}  ZCP gain={gain_from_zcp:+.3f}")
r=stats.pearsonr(x,y); rs=stats.spearmanr(x,y)
print(f"  correlation(standalone strength, benefit of adding ZCPs): r={r[0]:.3f} p={r[1]:.3f}; rho={rs[0]:.3f} p={rs[1]:.3f}")
