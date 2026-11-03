"""Full frozen CIFAR10-valid cohort acquisition using the verified pilot worker.
Requires the existing immutable pilot bundles. Does not train any predictor.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

NAME='nb201-v1-cifar10-full'
BASE='nb201-v1-cifar10-pilot'
BASE_SHA='85b62fd53d25c55a18a094e8eb4463419fc7d2ddf1eb58fac459c8f0c4d327e9'
NS='b-t-bose'
SHARDS=4
TIMEOUT=3600

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))

def base_module(root):
    p=root/BASE/'runner.py'
    require(sha(p)==BASE_SHA,'Validated pilot worker changed')
    spec=importlib.util.spec_from_file_location('verified_cifar10_pilot',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    contract,guard=m.verify(root)
    return m,contract,guard

def inputs(root):
    b=root/BASE;cohort=js(b/'cohort.json')
    require(cohort['dataset']=='cifar10-valid' and cohort['n']==1100,'Cohort mismatch')
    require(cohort['counts']=={'train':772,'val':53,'test':275},'Split changed')
    members={int(r['arch_index']):r for r in cohort['architectures']}
    with (b/'benchmark_targets.csv').open(newline='') as f:targets={int(r['architecture_id']):r for r in csv.DictReader(f)}
    require(len(members)==len(targets)==1100 and members.keys()==targets.keys(),'Cohort identity mismatch')
    return cohort,members,targets

def check_record(record,index,members,targets):
    r=members[index];t=targets[index]
    require(record['arch_index']==index and record['id']==r['id'],'Record identity mismatch')
    require(record['arch_str']==r['arch_str'] and record['arch_family']==r['arch_family'],'Architecture mismatch')
    require(record['dataset']=='cifar10-valid' and record['candidate_seed']==777,'Dataset/seed mismatch')
    require(record['total_epochs']==200 and record['weights_source']=='train','Measurement protocol mismatch')
    for e in (1,2,3):
        require(record[f'accuracy_epoch_{e}']==float(t[f'accuracy_epoch_{e}']),'Benchmark early-accuracy mismatch')
        require(math.isfinite(record['measured_early_accuracy'][str(e)]),'Missing local validation accuracy')
    require(record['best_accuracy']==float(t['best_accuracy']) and record['epochs_to_best']==int(t['epochs_to_best']),'Target mismatch')
    require(record['benchmark_seeds']==t['benchmark_seeds'],'Benchmark seed policy mismatch')
    table=[line.split('\t') for line in record['layer_stats'].splitlines() if line.strip()]
    require(table[0]==['epoch','taylor_imp'] and [r[0] for r in table[1:]]==['1','2','3'],'Taylor table mismatch')
    require(all(math.isfinite(float(r[1])) and float(r[1])>=0 for r in table[1:]),'Invalid Taylor values')

def pilot_records(root,members,targets):
    out=root/BASE/'results';complete=js(out/'complete.json')
    require(complete['status']=='complete' and complete['architectures']==[3,32,41],'Pilot incomplete')
    for name,h in complete['sha256'].items():require(sha(out/name)==h,'Pilot result drift: '+name)
    records={}
    for index in (3,32,41):
        rec=js(out/f'arch-{index}-complete.json');check_record(rec,index,members,targets);records[index]=rec
    return records

def prepare(root):
    b=root/NAME;require(not b.exists(),'Existing full bundle preserved: '+str(b))
    module,bc,guard=base_module(root);cohort,members,targets=inputs(root)
    reused=pilot_records(root,members,targets)
    todo=[i for i in sorted(members) if i not in reused]
    shards=[todo[k::SHARDS] for k in range(SHARDS)]
    require(len(todo)==1097 and len(set(sum(shards,[])))==1097,'Shard partition failure')
    require(set(sum(shards,[]))|set(reused)==set(members),'Incomplete cohort')
    template=js(root/BASE/'job.json');jobs=[]
    for k in range(SHARDS):
        job=copy.deepcopy(template);labels={'experiment-batch':NAME,'shard':str(k)}
        job['metadata']={'name':f'{NAME}-s{k}','namespace':NS,'labels':labels}
        job['spec']['activeDeadlineSeconds']=259200
        pod=job['spec']['template']['spec'];job['spec']['template']['metadata']={'labels':labels.copy()}
        require(pod['nodeSelector']=={'kubernetes.io/hostname':'w2-mi'},'Pilot placement changed')
        c=pod['containers'][0];c['name']='collector'
        c['args']=[f'''set -euo pipefail
unset PYTHONPATH
exec python3 -u /a/project/{NAME}/runner.py shard --root /a/project --shard {k} --output /a/output
''']
        require(c['workingDir']=='/a/output','Unexpected working directory')
        for vol in pod['volumes']:
            if vol['name']=='output':vol['hostPath']['path']=str(b/'results')
        jobs.append(job)
    dependencies={str((root/BASE/'contract.json').relative_to(root)):sha(root/BASE/'contract.json'),
                  str((root/BASE/'results/complete.json').relative_to(root)):sha(root/BASE/'results/complete.json')}
    b.mkdir();(b/'results').mkdir()
    put(b/'runner.py',Path(__file__).read_text())
    for name in ['cohort.json','benchmark_targets.csv']:
        with (b/name).open('xb') as f:f.write((root/BASE/name).read_bytes())
    put(b/'reused_pilot_records.json',list(reused.values()))
    put(b/'manifest.json',dict(architectures=sorted(members),reused_ids=sorted(reused),shards=shards))
    for k,j in enumerate(jobs):put(b/'jobs'/f's{k}.json',j)
    hashes={p.relative_to(b).as_posix():sha(p) for p in b.rglob('*') if p.is_file()}
    put(b/'contract.json',dict(name=NAME,n_architectures=1100,n_reused=3,n_new=1097,shards=SHARDS,
       candidate_timeout_seconds=TIMEOUT,sha256=hashes,dependencies_sha256=dependencies,
       input_policy=bc['input_policy'],implementation=bc['implementation'],aggregation=bc['aggregation'],
       failure_policy='Preserve failures; continue remaining candidates; never silently drop/replace IDs; collection requires all1100',
       execution='Four shard Jobs, one fresh process per architecture; same verified pilot worker and tested node/data mount'))
    verify(root)
    print('PREPARED:',b,'; reused=3; new=1097; shard sizes='+str(list(map(len,shards)))+'; NONE submitted.',flush=True)

def verify(root):
    b=root/NAME;c=js(b/'contract.json')
    for n,h in c['sha256'].items():require(sha(b/n)==h,'Prepared input drift: '+n)
    for n,h in c['dependencies_sha256'].items():require(sha(root/n)==h,'Dependency drift: '+n)
    base_module(root)
    return c

def candidate_result(out,index,members,targets):
    c=js(out/'complete.json')
    require(c['status']=='complete' and c['architectures']==[index] and c['epochs']==[1,2,3],'Candidate completion mismatch')
    for name,h in c['sha256'].items():require(sha(out/name)==h,'Candidate output drift: '+name)
    record=js(out/f'arch-{index}-complete.json');check_record(record,index,members,targets)
    for e in (1,2,3):
        r=js(out/f'arch-{index}-epoch-{e}.json')
        require(r['architecture_id']==index and r['epoch']==e,'Epoch mismatch')
        require(math.isfinite(r['taylor']) and r['taylor']>=0,'Invalid Taylor')
        table=record['layer_stats'].splitlines()[e].split('\t')
        require(float(table[1])==float(format(r['taylor'],'.6g')),'Snapshot/epoch Taylor mismatch')
        require(record['measured_early_accuracy'][str(e)]==r['local_validation_accuracy'],'Local accuracy mismatch')
    return record

def child(root,index,output):
    verify(root);_,members,targets=inputs(root)
    manifest=js(root/NAME/'manifest.json')
    require(index in sum(manifest['shards'],[]),'Index not assigned to new acquisition')
    out=output/'candidates'/f'{index:05d}'
    require(not out.exists(),'Candidate output already exists; preserve and inspect')
    out.mkdir(parents=True)
    m,c,guard=base_module(root);original=m.verify
    c=copy.deepcopy(c);c['pilot_ids']=[index]
    # In-memory dispatch only: never alter the verified pilot's files.
    m.verify=lambda requested_root:(c,guard)
    try:
        m.worker(root,Path('/a/data'),out)
        candidate_result(out,index,members,targets)
    except Exception:
        if not (out/'failure.json').exists():put(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
    finally:m.verify=original

def shard(root,k,output):
    c=verify(root);manifest=js(root/NAME/'manifest.json')
    require(0<=k<SHARDS,'Invalid shard')
    state=output/'shards'/f's{k}';require(not state.exists(),'Shard output exists; no automatic retry')
    state.mkdir(parents=True)
    _,members,targets=inputs(root)
    ids=manifest['shards'][k];put(state/'started.json',dict(shard=k,indices=ids))
    outcomes=[]
    for position,index in enumerate(ids,1):
        log=state/f'arch-{index}.log';start=time.monotonic()
        try:
            with log.open('xb') as handle:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'child','--root',str(root),
                    '--index',str(index),'--output',str(output)],stdout=handle,stderr=subprocess.STDOUT,
                    timeout=c['candidate_timeout_seconds'])
            require(result.returncode==0,f'Child exited {result.returncode}; see {log.name}')
            candidate_result(output/'candidates'/f'{index:05d}',index,members,targets)
            row=dict(architecture_id=index,status='complete',elapsed_seconds=time.monotonic()-start)
        except Exception as e:
            row=dict(architecture_id=index,status='failed',elapsed_seconds=time.monotonic()-start,error=str(e))
        outcomes.append(row)
        with (state/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('SHARD',k,f'{position}/{len(ids)}','ARCH',index,row['status'],flush=True)
    failures=[r for r in outcomes if r['status']!='complete']
    put(state/'report.json',dict(shard=k,attempted=len(ids),complete=len(ids)-len(failures),failures=failures))
    require(not failures,f'Shard {k}: {len(failures)} failures preserved; do not restart blindly')
    print('SHARD_COMPLETE',k,len(ids),flush=True)

def submit(root):
    verify(root);b=root/NAME
    require(not any((b/'results').iterdir()),'Existing results; inspect rather than relaunch')
    paths=[b/'jobs'/f's{k}.json' for k in range(SHARDS)]
    for k in range(SHARDS):
        r=subprocess.run(['kubectl','-n',NS,'get','job',f'{NAME}-s{k}','--ignore-not-found','-o','name'],capture_output=True,text=True)
        require(r.returncode==0 and not r.stdout.strip(),'Existing job or query failure: '+r.stdout+r.stderr)
    for p in paths:subprocess.run(['kubectl','-n',NS,'create','--dry-run=server','-f',str(p)],check=True)
    receipts=[]
    for p in paths:
        r=subprocess.run(['kubectl','-n',NS,'create','-f',str(p)],capture_output=True,text=True)
        receipts.append(dict(job=js(p)['metadata']['name'],returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout or r.stderr,flush=True)
    put(b/'submission.json',receipts)
    require(all(r['returncode']==0 for r in receipts),'Some submissions failed; inspect submission.json')

def show(root):
    verify(root);b=root/NAME;manifest=js(b/'manifest.json')
    print('Reused pilot architectures:',manifest['reused_ids'])
    for k in range(SHARDS):
        path=b/'results/shards'/f's{k}'/'progress.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
        print('SHARD',k,'reported',len(rows),'of',len(manifest['shards'][k]),'complete',sum(r['status']=='complete' for r in rows),'failed',sum(r['status']!='complete' for r in rows))

def collect(root):
    verify(root);b=root/NAME
    require(not (b/'snapshot.jsonl').exists() and not (b/'collection_complete.json').exists(),
            'Existing final snapshot/completion preserved')
    _,members,targets=inputs(root)
    # Revalidate original pilot artifacts before incorporating their records.
    all_records=pilot_records(root,members,targets)
    manifest=js(b/'manifest.json')
    for k,ids in enumerate(manifest['shards']):
        report=js(b/'results/shards'/f's{k}'/'report.json')
        require(report['attempted']==report['complete']==len(ids) and not report['failures'],'Incomplete shard')
        for index in ids:
            require(index not in all_records,'Duplicate architecture')
            all_records[index]=candidate_result(b/'results/candidates'/f'{index:05d}',index,members,targets)
    require(set(all_records)==set(manifest['architectures']) and len(all_records)==1100,'Incomplete cohort')
    records=[all_records[i] for i in manifest['architectures']]
    with (b/'snapshot.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
    loaded=[json.loads(l) for l in (b/'snapshot.jsonl').read_text().splitlines()]
    require(loaded==records,'Snapshot readback failure')
    put(b/'collection_complete.json',dict(status='complete',n=1100,reused=3,new=1097,
       snapshot_sha256=sha(b/'snapshot.jsonl'),cohort_sha256=sha(b/'cohort.json'),
       scope='Feature acquisition complete; predictor fitting/statistical comparison not yet performed'))
    print('COLLECTION_COMPLETE:',b/'snapshot.jsonl',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','shard','child','show','collect'])
    p.add_argument('--root',type=Path,default=Path('/shared/ssd/home/b-t-bose/lmur'));p.add_argument('--output',type=Path,default=Path('/a/output'))
    p.add_argument('--shard',type=int);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='shard':shard(a.root,a.shard,a.output)
    elif a.action=='child':child(a.root,a.index,a.output)
    else:{'prepare':prepare,'submit':submit,'show':show,'collect':collect}[a.action](a.root)

if __name__=='__main__':main()
