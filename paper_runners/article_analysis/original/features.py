import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import json, sys, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SNAP=paths.snapshot()
STATS=['ww_alpha','grad_norm','dead_frac','taylor_imp','cka_redund','rank_ratio','sensitivity']

def parse_layer_stats(s):
    lines=[l for l in s.strip().split("\n") if l.strip()]
    hdr=lines[0].split("\t")
    out={}
    for l in lines[1:]:
        p=l.split("\t"); ep=int(float(p[0]))
        row={}
        for i in range(1,min(len(hdr),len(p))):
            try: row[hdr[i]]=float(p[i])
            except ValueError: row[hdr[i]]=float('nan')
        out[ep]=row
    return out

def load():
    recs=[]
    for line in open(SNAP):
        r=json.loads(line)
        ls=parse_layer_stats(r['layer_stats']) if r.get('layer_stats') else {}
        prm=r.get('prm')
        if isinstance(prm,str): prm=json.loads(prm)
        prm=prm or {}
        recs.append(dict(id=r['id'], dataset=r['dataset'], nn=r['nn'],
            a1=r['accuracy_epoch_1'], a2=r['accuracy_epoch_2'], a3=r['accuracy_epoch_3'],
            y=float(r['best_accuracy']), ls=ls,
            lr=prm.get('lr'), momentum=prm.get('momentum'), batch=prm.get('batch'),
            total_epochs=r.get('total_epochs'), epochs_to_best=r.get('epochs_to_best')))
    return recs

def featurize(recs, stats=('taylor_imp',), epochs=(1,2,3)):
    X=[]; names=[]
    for i,r in enumerate(recs):
        row=[r['a1'],r['a2'],r['a3'], r['a3']-r['a2'], r['a2']-r['a1'],
             np.log10(max(r['lr'],1e-8)) if r['lr'] else 0.0,
             r['momentum'] or 0.0, np.log2(r['batch'] or 1)]
        if i==0: names=['acc1','acc2','acc3','d32','d21','log_lr','momentum','log_batch']
        for s in stats:
            for e in epochs:
                v=r['ls'].get(e,{}).get(s, np.nan)
                row.append(v)
                if i==0: names.append(f"{s}_e{e}")
        X.append(row)
    X=np.array(X,dtype=float)
    return X, names

if __name__=="__main__":
    recs=load()
    print("snapshot records:",len(recs))
    print("with layer stats:",sum(1 for r in recs if r['ls']))
    print("datasets:",collections.Counter(r['dataset'] for r in recs))
    X,names=featurize(recs)
    print("feature names:",names)
    print("nan rate per column:", np.round(np.isnan(X).mean(0),3))
