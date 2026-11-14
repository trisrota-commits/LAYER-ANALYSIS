import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import json, os, glob, csv, sys
ROOT=paths.experiment_logs()
rows=[]
for contract in glob.glob(ROOT+"/*/*/contract.json"):
    base=os.path.dirname(contract)
    try: c=json.load(open(contract))
    except Exception as e: continue
    runs=c.get("runs",{})
    for name,meta in runs.items():
        pred=os.path.join(base,"results",name,"test_predictions.csv")
        if not os.path.exists(pred): continue
        n=sum(1 for _ in open(pred))-1
        rows.append(dict(exp=os.path.relpath(base,ROOT), run=name,
                         seed=meta.get("seed"), zcp=meta.get("zero_cost_proxies"),
                         data=meta.get("data","").split("/")[-1], n=n, path=pred))
print(len(rows),"runs with predictions")
import collections
by=collections.defaultdict(list)
for r in rows: by[(r["exp"],r["data"],r["zcp"],r["n"])].append(r["seed"])
for k in sorted(by, key=lambda x:(x[0],str(x[1]))):
    print(f"{k[0]:55s} data={str(k[1]):32s} zcp={str(k[2]):5s} n={k[3]:4d} seeds={sorted(s for s in by[k] if s is not None)}")
json.dump(rows, open(paths.artifact("registry.json"),"w"), indent=0)
