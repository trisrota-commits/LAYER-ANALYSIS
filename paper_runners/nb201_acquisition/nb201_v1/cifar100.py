"""Independent CIFAR100 acquisition on the frozen supervisor NB201 cohort.
Target: mean final x-valid accuracy at epoch200 over benchmark seeds777/888.
Local early validation accuracies and Taylor come from the same fresh run.
"""
import argparse,bz2,collections,copy,csv,hashlib,importlib.util,io,json,math,pickle,subprocess,sys,tarfile,time,traceback
from pathlib import Path

NAME='nb201-v1-cifar100-full'
PARENT='nas201-taylor-pilot-v1'
PARENT_SHA='87b2357f87038bd6206243962dbef149a0c55557ffcf2f05b1996c0badf54e5f'
EXTRACTOR_SHA='f19dcb88702679de1ada97dde6766bbbdb44019d4b05c04e1557fa1bdde9ad9f'
ARCHIVE_SHA='580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6'
SHARDS=4

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def js(p):return json.loads(Path(p).read_text())
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))

def load_module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class RestrictedReader(pickle.Unpickler):
    def find_class(self,module,name):
        import numpy as np
        multi=np._core.multiarray if hasattr(np,'_core') else np.core.multiarray
        allowed={('collections','OrderedDict'):collections.OrderedDict,('numpy','dtype'):np.dtype,
            ('numpy.core.multiarray','scalar'):multi.scalar,('numpy._core.multiarray','scalar'):multi.scalar}
        require((module,name) in allowed,'Unexpected pickle type: '+module+'.'+name)
        return allowed[module,name]

def build_targets(source,archive,out):
    require(not out.exists(),'Existing CIFAR100 metadata preserved')
    require(sha(archive)==ARCHIVE_SHA,'Benchmark archive mismatch')
    original=js(source);cohort=copy.deepcopy(original)
    require(original['n']==1100 and original['counts']=={'train':772,'val':53,'test':275},'Wrong source cohort')
    ids=[r['arch_index'] for r in original['architectures']]
    require(len(ids)==len(set(ids))==1100,'Duplicate architecture IDs')
    members={r['arch_index']:r for r in cohort['architectures']}
    membership={i:s for s,group in original['splits'].items() for i in group}
    rows=[];payload_hashes={}
    with tarfile.open(archive,'r:') as tar:
        for i in ids:
            member=f'NATS-tss-v1_0-3ffb9-simple/{i:06d}.pickle.pbz2'
            payload=tar.extractfile(member).read()
            data=RestrictedReader(io.BytesIO(bz2.decompress(payload))).load()['200']
            require(data['arch_index']==i and data['arch_str']==members[i]['arch_str'],'Architecture mismatch')
            values=[]
            for seed in (777,888):
                trial=data['all_results'][('cifar100',seed)]
                require(trial['epochs']==200 and int(trial['seed'])==seed,'Wrong benchmark run')
                cfg=trial['arch_config']
                require(cfg['channel']==16 and cfg['num_cells']==5 and cfg['class_num']==100,'Wrong benchmark network')
                value=float(trial['eval_acc1es']['x-valid@199'])
                require(math.isfinite(value) and 0<=value<=100,'Invalid benchmark target')
                values.append(value)
            rid=hashlib.md5(f'nb201:cifar100:{i}'.encode()).hexdigest()
            members[i]['id']=rid
            rows.append(dict(record_id=rid,architecture_id=i,dataset='cifar100',split=membership[i],
                target_final_validation_accuracy=math.fsum(values)/2,target_epoch=200,
                benchmark_seed_777_accuracy=values[0],benchmark_seed_888_accuracy=values[1]))
            payload_hashes[member]=hashlib.sha256(payload).hexdigest()
    cohort.update(dataset='cifar100',target_policy='Mean final x-valid accuracy at epoch200 over benchmark seeds777 and888',
        target_seed_policy='Fixed seeds777/888, both available for every sampled architecture; seed999 never used',
        early_accuracy_policy='Fresh local x-valid accuracies from the same run as Taylor, not benchmark ori-test values',
        source_cifar10_cohort_sha256=sha(source),status='Metadata prepared; CIFAR100 full acquisition not launched')
    out.mkdir();put(out/'cohort.json',cohort)
    with (out/'targets.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'targets.csv').open(newline='') as f:back=list(csv.DictReader(f))
    require(len(back)==1100,'Target readback count mismatch')
    for a,b in zip(rows,back):require(a['record_id']==b['record_id'] and a['target_final_validation_accuracy']==float(b['target_final_validation_accuracy']),'Target readback mismatch')
    put(out/'metadata.json',dict(dataset='cifar100',n=1100,source_archive_sha256=ARCHIVE_SHA,source_cifar10_cohort_sha256=sha(source),
        source_members_sha256=payload_hashes,files_sha256={n:sha(out/n) for n in ['cohort.json','targets.csv']},
        target_policy=cohort['target_policy'],early_accuracy_policy=cohort['early_accuracy_policy']))
    print('CIFAR100 METADATA VERIFIED:',out,'; 1100 targets, same772/53/275 grouped split; no training.')

def prepare(root):
    b=root/NAME;src=root/'nb201_v1/cifar100';parent=root/PARENT
    require(not b.exists(),'Existing full bundle preserved: '+str(b))
    meta=js(src/'metadata.json')
    for n,h in meta['files_sha256'].items():require(sha(src/n)==h,'Metadata changed: '+n)
    require(sha(root/'nb201_v1/cohort.json')==meta['source_cifar10_cohort_sha256'],'Source CIFAR10 cohort changed')
    require(sha(parent/'runner.py')==PARENT_SHA,'Verified CIFAR100 pilot source changed')
    pilot=load_module(parent/'runner.py','verified_cifar100_pilot');pc=pilot.verify(parent)
    require(sha(parent/'LayerAnalysis.py')==EXTRACTOR_SHA,'Taylor implementation changed')
    done=js(parent/'results/complete.json');require(done['status']=='complete','CIFAR100 pilot incomplete')
    for n,h in done['files_sha256'].items():require(sha(parent/'results'/n)==h,'Pilot output drift: '+n)
    cohort=js(src/'cohort.json');ids=[r['arch_index'] for r in cohort['architectures']]
    require(len(ids)==len(set(ids))==1100 and not set(ids)&{0,9,123},'Unexpected pilot overlap or cohort')
    groups=[ids[k::SHARDS] for k in range(SHARDS)]
    engine=(parent/'runner.py').read_text()
    old="require(len(list(csv.DictReader(f)))==3,'CSV readback failed')"
    require(engine.count(old)==1,'Verified worker CSV-count anchor changed')
    engine=engine.replace(old,"require(len(list(csv.DictReader(f)))==len(ARCHS),'CSV readback failed')",1)
    compile(engine,'engine.py','exec')
    template=js(parent/'job.json');jobs=[]
    for k in range(SHARDS):
        job=copy.deepcopy(template);labels={'experiment-batch':NAME,'shard':str(k)}
        job['metadata']={'name':f'{NAME}-s{k}','namespace':'b-t-bose','labels':labels}
        job['spec']['activeDeadlineSeconds']=259200;job['spec']['template']['metadata']={'labels':labels.copy()}
        pod=job['spec']['template']['spec'];require(pod['nodeSelector']=={'kubernetes.io/hostname':'w2-mi'},'Pilot node changed')
        container=pod['containers'][0];container['name']='collector'
        require(container['workingDir']=='/a/output','Unexpected work directory')
        container['args']=[f'''set -euo pipefail
unset PYTHONPATH
exec python3 -u /a/project/{NAME}/runner.py shard --root /a/project --shard {k} --output /a/output
''']
        for v in pod['volumes']:
            if v['name']=='output':v['hostPath']['path']=str(b/'results')
        jobs.append(job)
    b.mkdir();(b/'results').mkdir()
    put(b/'runner.py',Path(__file__).read_text());put(b/'engine.py',engine)
    for name in ['cohort.json','targets.csv','metadata.json']:
        with (b/name).open('xb') as f:f.write((src/name).read_bytes())
    # Small immutable source copy; no datasets, model checkpoints, or old outputs.
    for name in pc['sha256']:
        if name=='LayerAnalysis.py' or name.startswith('autodl/'):
            path=b/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write((parent/name).read_bytes())
    put(b/'manifest.json',dict(architectures=ids,shards=groups,reused_ids=[]))
    for k,j in enumerate(jobs):put(b/'jobs'/f's{k}.json',j)
    contract={k:copy.deepcopy(v) for k,v in pc.items() if k!='sha256'}
    contract.update(experiment=NAME,architectures={str(r['arch_index']):r['arch_str'] for r in cohort['architectures']},
        purpose='Full frozen CIFAR100 cohort feature acquisition; not predictor evaluation or exact historical replay',
        candidate_timeout_seconds=3600,shards=SHARDS,benchmark_target_seeds=[777,888],target_epoch=200,
        target_kind='final_validation_accuracy',input_policy=meta['early_accuracy_policy'],target_policy=meta['target_policy'],
        code_change='Derived worker changes only the singleton CSV row-count check; NAME and ARCHS bound in each isolated child process',
        parent_pilot_contract_sha256=sha(parent/'contract.json'),parent_pilot_completion_sha256=sha(parent/'results/complete.json'))
    contract['sha256']={p.relative_to(b).as_posix():sha(p) for p in b.rglob('*') if p.is_file()}
    put(b/'contract.json',contract);verify(root)
    print('PREPARED:',b,';1100 new architectures;shards='+str(list(map(len,groups)))+';NONE submitted.',flush=True)

def verify(root):
    b=root/NAME;c=js(b/'contract.json')
    for n,h in c['sha256'].items():require(sha(b/n)==h,'Prepared artifact drift: '+n)
    require(c['target_kind']=='final_validation_accuracy' and c['benchmark_target_seeds']==[777,888],'Target policy drift')
    return c

def input_maps(root):
    b=root/NAME;members={r['arch_index']:r for r in js(b/'cohort.json')['architectures']}
    with (b/'targets.csv').open(newline='') as f:targets={int(r['architecture_id']):r for r in csv.DictReader(f)}
    require(len(members)==len(targets)==1100 and members.keys()==targets.keys(),'Identity mismatch')
    return members,targets

def validate_record(record,index,members,targets):
    r=members[index];t=targets[index]
    require(record['record_id']==r['id'] and record['architecture_id']==index and record['architecture_string']==r['arch_str'],'Record identity mismatch')
    require(record['dataset']=='cifar100' and record['candidate_seed']==777,'Dataset/seed mismatch')
    require(record['split']==t['split'] and record['target_kind']=='final_validation_accuracy' and record['target_epoch']==200,'Split/target semantics mismatch')
    require(record['target_final_validation_accuracy']==float(t['target_final_validation_accuracy']),'Target value mismatch')
    require(record['benchmark_target_seeds']==[777,888],'Target seeds mismatch')
    for e in (1,2,3):
        require(math.isfinite(record[f'accuracy_epoch_{e}']) and 0<=record[f'accuracy_epoch_{e}']<=100,'Invalid local early accuracy')
        require(math.isfinite(record[f'taylor_epoch_{e}']) and record[f'taylor_epoch_{e}']>=0,'Invalid Taylor')

def validate_candidate(out,index,members,targets):
    done=js(out/'candidate_complete.json');require(done['architecture_id']==index and done['status']=='complete','Candidate completion mismatch')
    require(sha(out/'record.json')==done['record_sha256'] and sha(out/'complete.json')==done['engine_completion_sha256'],'Candidate hash mismatch')
    engine=js(out/'complete.json');require(engine['architectures']==[index] and engine['epochs']==[1,2,3],'Engine scope mismatch')
    for n,h in engine['files_sha256'].items():require(sha(out/n)==h,'Acquisition output drift')
    rec=js(out/'record.json');validate_record(rec,index,members,targets)
    for e in (1,2,3):
        epoch=js(out/f'arch-{index}-epoch-{e}.json')
        require(epoch['architecture_id']==index and epoch['epoch']==e,'Wrong epoch identity')
        require(rec[f'accuracy_epoch_{e}']==epoch['validation_accuracy'] and rec[f'taylor_epoch_{e}']==epoch['taylor'],'Snapshot/epoch value mismatch')
    return rec

def child(root,index,output):
    verify(root);b=root/NAME;members,targets=input_maps(root)
    require(index in members,'Unselected architecture')
    out=output/'candidates'/f'{index:05d}';require(not out.exists(),'Existing candidate output preserved');out.mkdir(parents=True)
    engine=load_module(b/'engine.py','cifar100_acquisition_engine')
    engine.NAME=NAME;engine.ARCHS={index:members[index]['arch_str']}
    try:
        engine.worker(root,Path('/a/data'),out)
        acquired=js(out/f'arch-{index}-complete.json');target=targets[index]
        require(acquired['architecture_id']==index and acquired['architecture_string']==members[index]['arch_str'],'Acquisition identity mismatch')
        require(acquired['dataset']=='cifar100' and acquired['candidate_seed']==777,'Acquisition dataset/seed mismatch')
        record=dict(record_id=members[index]['id'],architecture_id=index,architecture_string=members[index]['arch_str'],
            arch_family=members[index]['arch_family'],dataset='cifar100',split=target['split'],candidate_seed=777,
            log_params=acquired['log_params'],depth=acquired['depth'],
            target_kind='final_validation_accuracy',target_epoch=200,benchmark_target_seeds=[777,888],
            target_final_validation_accuracy=float(target['target_final_validation_accuracy']))
        require([r['epoch'] for r in acquired['epochs']]==[1,2,3],'Incomplete early epochs')
        for r in acquired['epochs']:
            record[f"accuracy_epoch_{r['epoch']}"]=r['validation_accuracy'];record[f"taylor_epoch_{r['epoch']}"]=r['taylor']
        validate_record(record,index,members,targets);put(out/'record.json',record)
        put(out/'candidate_complete.json',dict(status='complete',architecture_id=index,record_sha256=sha(out/'record.json'),engine_completion_sha256=sha(out/'complete.json')))
        validate_candidate(out,index,members,targets)
    except Exception:
        if not (out/'failure.json').exists():put(out/'failure.json',{'traceback':traceback.format_exc()})
        raise

def shard(root,k,output):
    c=verify(root);manifest=js(root/NAME/'manifest.json');require(0<=k<SHARDS,'Invalid shard')
    state=output/'shards'/f's{k}';require(not state.exists(),'Existing shard output preserved');state.mkdir(parents=True)
    members,targets=input_maps(root);ids=manifest['shards'][k];put(state/'started.json',{'indices':ids,'shard':k});rows=[]
    for pos,index in enumerate(ids,1):
        start=time.monotonic()
        try:
            with (state/f'arch-{index}.log').open('xb') as f:
                r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'child','--root',str(root),'--index',str(index),'--output',str(output)],stdout=f,stderr=subprocess.STDOUT,timeout=c['candidate_timeout_seconds'])
            require(r.returncode==0,f'Child exited {r.returncode}')
            validate_candidate(output/'candidates'/f'{index:05d}',index,members,targets)
            row=dict(architecture_id=index,status='complete',elapsed_seconds=time.monotonic()-start)
        except Exception as e:row=dict(architecture_id=index,status='failed',elapsed_seconds=time.monotonic()-start,error=str(e))
        rows.append(row)
        with (state/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('CIFAR100 SHARD',k,f'{pos}/{len(ids)}','ARCH',index,row['status'],flush=True)
    failed=[r for r in rows if r['status']!='complete'];put(state/'report.json',dict(shard=k,attempted=len(ids),complete=len(ids)-len(failed),failures=failed))
    require(not failed,f'Shard {k} has failures; preserve outputs and inspect')
    print('SHARD_COMPLETE',k,flush=True)

def submit(root):
    verify(root);b=root/NAME;require(not any((b/'results').iterdir()),'Existing results; no automatic retry')
    paths=[b/'jobs'/f's{k}.json' for k in range(SHARDS)]
    for k in range(SHARDS):
        r=subprocess.run(['kubectl','-n','b-t-bose','get','job',f'{NAME}-s{k}','--ignore-not-found','-o','name'],capture_output=True,text=True)
        require(r.returncode==0 and not r.stdout.strip(),'Existing Job or failed query: '+r.stdout+r.stderr)
    for p in paths:subprocess.run(['kubectl','-n','b-t-bose','create','--dry-run=server','-f',str(p)],check=True)
    rows=[]
    for p in paths:
        r=subprocess.run(['kubectl','-n','b-t-bose','create','-f',str(p)],capture_output=True,text=True)
        rows.append(dict(job=js(p)['metadata']['name'],returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout or r.stderr,flush=True)
    put(b/'submission.json',rows);require(all(r['returncode']==0 for r in rows),'Partial submission; inspect receipt, do not resubmit blindly')

def show(root):
    verify(root);b=root/NAME;manifest=js(b/'manifest.json')
    for k,ids in enumerate(manifest['shards']):
        p=b/'results/shards'/f's{k}'/'progress.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        print('SHARD',k,'reported',len(rows),'of',len(ids),'complete',sum(r['status']=='complete' for r in rows),'failed',sum(r['status']=='failed' for r in rows))

def collect(root):
    verify(root);b=root/NAME;require(not (b/'snapshot.jsonl').exists() and not (b/'collection_complete.json').exists(),'Existing collection preserved')
    members,targets=input_maps(root);manifest=js(b/'manifest.json');records={}
    for k,ids in enumerate(manifest['shards']):
        report=js(b/'results/shards'/f's{k}'/'report.json')
        require(report['attempted']==report['complete']==len(ids) and not report['failures'],'Incomplete shard; no records dropped')
        for i in ids:require(i not in records,'Duplicate ID');records[i]=validate_candidate(b/'results/candidates'/f'{i:05d}',i,members,targets)
    require(set(records)==set(members) and len(records)==1100,'Incomplete cohort')
    ordered=[records[i] for i in manifest['architectures']]
    with (b/'snapshot.jsonl').open('x') as f:
        for r in ordered:f.write(json.dumps(r,allow_nan=False)+'\n')
    require([json.loads(l) for l in (b/'snapshot.jsonl').read_text().splitlines()]==ordered,'Snapshot readback failure')
    put(b/'collection_complete.json',dict(status='complete',n=1100,dataset='cifar100',target_kind='final_validation_accuracy',target_epoch=200,
        snapshot_sha256=sha(b/'snapshot.jsonl'),scope='Feature acquisition only; no predictor fitted'))
    print('COLLECTION_COMPLETE:',b/'snapshot.jsonl')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['targets','prepare','submit','child','shard','show','collect'])
    p.add_argument('--root',type=Path,default=Path('/shared/ssd/home/b-t-bose/lmur'));p.add_argument('--output',type=Path,default=Path('/a/output'))
    p.add_argument('--archive',type=Path);p.add_argument('--cohort',type=Path);p.add_argument('--shard',type=int);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='targets':build_targets(a.cohort,a.archive,a.output)
    elif a.action=='child':child(a.root,a.index,a.output)
    elif a.action=='shard':shard(a.root,a.shard,a.output)
    else:{'prepare':prepare,'submit':submit,'show':show,'collect':collect}[a.action](a.root)

if __name__=='__main__':main()
