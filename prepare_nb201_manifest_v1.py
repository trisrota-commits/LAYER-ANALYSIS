"""Freeze an NB201/CIFAR100 cohort and benchmark targets, without training."""
import argparse
import bz2
import collections
import csv
import hashlib
import io
import json
import math
import pickle
import random
import tarfile
from pathlib import Path
import numpy as np

ARCHIVE_SHA='580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6'
SEEDS=(777,888,999)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class RestrictedReader(pickle.Unpickler):
    def find_class(self,module,name):
        multi=np._core.multiarray if hasattr(np,'_core') else np.core.multiarray
        allowed={('collections','OrderedDict'):collections.OrderedDict,('numpy','dtype'):np.dtype,
                 ('numpy.core.multiarray','scalar'):multi.scalar,('numpy._core.multiarray','scalar'):multi.scalar}
        if (module,name) not in allowed:raise ValueError(f'Unexpected pickle type: {module}.{name}')
        return allowed[module,name]

def decode(content):
    return RestrictedReader(io.BytesIO(bz2.decompress(content))).load()

def plan():
    selected=sorted(random.Random(2026).sample(range(15625),1100))
    order=selected.copy();random.Random(20261025).shuffle(order)
    parts={'train':sorted(order[:770]),'val':sorted(order[770:825]),'test':sorted(order[825:])}
    assert len(set(order))==1100 and set(order)==set(selected)
    return selected,parts

def target(full,index):
    assert full['arch_index']==index
    assert set(full['dataset_seed']['cifar100'])==set(SEEDS),f'{index}: benchmark seeds differ'
    curves=[];details={}
    for seed in SEEDS:
        r=full['all_results'][('cifar100',seed)]
        assert r['name']=='cifar100' and int(r['seed'])==seed and r['epochs']==200
        cfg=r['arch_config']
        assert cfg['channel']==16 and cfg['num_cells']==5 and cfg['class_num']==100
        assert cfg['arch_str']==full['arch_str']
        values=[float(r['eval_acc1es'][f'x-valid@{e}']) for e in range(200)]
        assert all(math.isfinite(v) and 0<=v<=100 for v in values)
        curves.append(values)
        details[str(seed)]=dict(best_accuracy=max(values),best_epoch=values.index(max(values))+1,
                               final_validation_accuracy=values[-1])
    mean=np.mean(curves,axis=0)
    best_epoch=int(np.argmax(mean))+1
    return dict(best_accuracy=float(mean[best_epoch-1]),best_epoch=best_epoch,
                final_validation_accuracy=float(mean[-1]),per_seed=details)

def put(p,obj):
    with p.open('x',encoding='utf-8',newline='') as f:json.dump(obj,f,indent=2,allow_nan=False)

def build(archive,out):
    assert not out.exists(),f'Refusing existing output: {out}'
    assert sha(archive)==ARCHIVE_SHA,'Archive bytes differ from inspected download'
    selected,splits=plan()
    out.mkdir()
    # Membership is persisted BEFORE reading any architecture outcome values.
    split_manifest=dict(sample_seed=2026,split_seed=20261025,selected_ids=selected,splits=splits,
                        selection='Uniform sample of 1100 IDs from range(15625), without outcome filtering',
                        grouping='Architecture-ID disjoint; NOT operation-multiset or graph-isomorphism disjoint')
    put(out/'split_manifest.json',split_manifest)
    membership={i:s for s,ids in splits.items() for i in ids}
    rows=[];details={};architecture_inputs=[];target_hashes={}
    with tarfile.open(archive,'r:') as tar:
        for index in selected:
            name=f'NATS-tss-v1_0-3ffb9-simple/{index:06d}.pickle.pbz2'
            payload=tar.extractfile(name).read()
            full=decode(payload)['200'];result=target(full,index)
            target_hashes[name]=hashlib.sha256(payload).hexdigest()
            rid=f'nb201-cifar100-{index:05d}'
            rows.append(dict(record_id=rid,architecture_id=index,dataset='cifar100',split=membership[index],
                             target=result['best_accuracy'],best_epoch=result['best_epoch'],
                             final_validation_accuracy=result['final_validation_accuracy']))
            architecture_inputs.append(dict(record_id=rid,architecture_id=index,architecture_string=full['arch_str'],
                                           dataset='cifar100',split=membership[index],C=16,N=5,num_classes=100,candidate_seed=777))
            details[str(index)]=result
    with (out/'targets.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    put(out/'architectures.json',architecture_inputs)
    put(out/'target_details.json',details)
    protocol=dict(dataset='cifar100',benchmark='NATS-Bench topology / NAS-Bench-201 search space',
        n_architectures=1100,split_counts={s:len(ids) for s,ids in splits.items()},candidate_training_seed=777,
        early_features='Fresh local validation accuracies and Taylor from same run, epochs 1,2,3; first 3 epochs of 200-epoch schedule',
        benchmark_target_seeds=list(SEEDS),benchmark_target='max over epochs 1..200 of mean x-valid accuracy across seeds 777,888,999',
        target_distinction='Maximum of the mean validation curve; NOT mean of per-seed maxima and NOT final-epoch accuracy',
        trajectory_distinction='Local early features and historical benchmark labels are separate executions; not exact historical replay',
        benchmark_archive_sha256=ARCHIVE_SHA,source_members_sha256=target_hashes,
        primary_endpoint_proposal='Taylor vs no-layer MAE on held-out records with target >=20; fit six predictor seeds42..47; analysis code must be finalized before predictor evaluation',
        sampling_note='No filtering by accuracy, collapse, operation mix, or later extraction success',
        pilot_overlap=sorted(set(selected)&{0,9,123}),
        status='Cohort and targets prepared; feature acquisition and predictor training NOT started')
    put(out/'protocol.json',protocol)
    with (out/'targets.csv').open(newline='') as f:check=list(csv.DictReader(f))
    assert len(check)==1100 and len({r['record_id'] for r in check})==1100
    for a,b in zip(rows,check):
        assert a['record_id']==b['record_id'] and a['target']==float(b['target'])
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    put(out/'complete.json',dict(status='manifest_complete',files_sha256=hashes,
        script_sha256=sha(Path(__file__)),source_archive_sha256=ARCHIVE_SHA))
    print('MANIFEST COMPLETE:',out)
    print('SPLIT COUNTS:',protocol['split_counts'])
    print('TARGET:',protocol['benchmark_target'])
    print('PILOT OVERLAP:',protocol['pilot_overlap'])
    print('No candidate or Qwen training launched.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args.archive,args.output)
