import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import json, os, re, csv, collections, numpy as np
from scipy import stats
ROOT=paths.experiment_logs()
rows=json.load(open(paths.artifact("registry.json")))

FEATS=["no-layer","all-seven","weightwatcher","cka","effective-rank","taylor","gradient-flow","sensitivity","activation"]
def parse(exp, run):
    m=re.match(r"^(?:(\d)acc-)?(?:zcp-(on|off)-)?(.+?)-s(\d\d)$", run)
    if not m: return None
    nacc, zcp, feat, seed = m.groups()
    # experiment-level defaults
    if nacc is None:
        nacc = "2" if "2acc" in exp else "3"
    if zcp is None:
        if "zcp-off" in exp or "3acc-off" in exp or "off-v1" in exp: zcp="off"
        elif "zcp-on" in exp: zcp="on"
        elif "layer-ablation" in exp: zcp="on"
        else: zcp="?"
    return int(nacc), zcp, feat, int(seed)

def load(path):
    d=list(csv.DictReader(open(path)))
    key=[(int(r["idx"]), r["dataset"]) for r in d]
    p=np.array([float(r["pred_best_accuracy"]) for r in d])
    g=np.array([float(r["gt_best_accuracy"]) for r in d])
    return key,p,g

cond=collections.defaultdict(dict)   # (nacc,zcp,feat) -> seed -> (key,p,g)
for r in rows:
    pr=parse(r["exp"], r["run"])
    if pr is None: continue
    nacc,zcp,feat,seed=pr
    if r["n"]!=115: continue
    cond[(nacc,zcp,feat)][seed]=load(r["path"])

def five(p,g):
    e=p-g
    rmse=float(np.sqrt((e**2).mean())); mae=float(np.abs(e).mean())
    pe=float(stats.pearsonr(p,g)[0]); sp=float(stats.spearmanr(p,g)[0])
    r2=1-float((e**2).sum()/((g-g.mean())**2).sum())
    return dict(RMSE=rmse,MAE=mae,Pearson=pe,Spearman=sp,R2=r2)

if __name__=="__main__":
    for k in sorted(cond):
        seeds=sorted(cond[k])
        ms=[five(*cond[k][s][1:]) for s in seeds]
        m={n:np.mean([x[n] for x in ms]) for n in ms[0]}
        print(f"{str(k):40s} seeds={seeds} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} r={m['Pearson']:.4f} rho={m['Spearman']:.4f} R2={m['R2']:.4f}")
