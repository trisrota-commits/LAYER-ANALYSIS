"""Isolated ImageNet16-120 acquisition on the frozen NB201 cohort.

Reuse the verified CIFAR100 collection functions and candidate-training engine
as frozen, adapted copies. Never edit CIFAR bundles or Train.py.
"""
import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

NAME = 'nb201-v1-imagenet16-full'
DATASET = 'ImageNet16-120'
HOST = Path('/shared/ssd/home/b-t-bose/lmur')
PARENT = 'nas201-taylor-pilot-v1'
PARENT_SHA = '87b2357f87038bd6206243962dbef149a0c55557ffcf2f05b1996c0badf54e5f'
CIFAR_SHA = '809d929e74f414a704cf751894c0b999d0a331bdb95535b5fb659f751b0b0095'
EXTRACTOR_SHA = 'f19dcb88702679de1ada97dde6766bbbdb44019d4b05c04e1557fa1bdde9ad9f'
ARCHIVE_SHA = '580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6'
COMMIT = 'f46486e21b71ae6459a700be720d7648b5429569'
SPLIT = 'configs/nas-benchmark/imagenet-16-120-test-split.txt'
IMAGE = 'docker.io/abrainone/ai-linux@sha256:af6bc18dcfcc9cdf73deef2ef916d0d8408318cddceedd0208e6b88208e05e14'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False))


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def partition(cohort):
    ids=sorted(r['arch_index'] for r in cohort['architectures'])
    require(len(ids)==len(set(ids))==1100,'Wrong frozen cohort')
    pilot=ids[:3];remaining=ids[3:]
    return dict(architectures=ids,pilot_ids=pilot,shards=[pilot]+[remaining[k::4] for k in range(4)])


def build_targets(cohort_path,archive,out):
    require(not out.exists(),'Existing metadata preserved')
    source=Path(__file__).with_name('cifar100.py')
    require(sha(source)==CIFAR_SHA,'CIFAR reader source changed')
    reader=module(source,'verified_benchmark_reader')
    require(sha(archive)==ARCHIVE_SHA,'Benchmark archive mismatch')
    cohort=copy.deepcopy(js(cohort_path))
    require(cohort['n']==1100 and cohort['counts']==dict(train=772,val=53,test=275),'Wrong source split')
    members={r['arch_index']:r for r in cohort['architectures']}
    membership={i:s for s,ids in cohort['splits'].items() for i in ids}
    require(len(members)==len(membership)==1100,'Duplicate or missing architecture')
    url=f'https://raw.githubusercontent.com/D-X-Y/AutoDL-Projects/{COMMIT}/{SPLIT}'
    split_text=urllib.request.urlopen(url,timeout=45).read().decode()
    image_split=json.loads(split_text)
    a=set(map(int,image_split['xvalid'][1]));b=set(map(int,image_split['xtest'][1]))
    require(len(a)==len(b)==3000 and not a&b and a|b==set(range(6000)),'Wrong image evaluation split')
    rows=[];hashes={}
    import bz2
    with tarfile.open(archive,'r:') as tar:
        for index,rec in members.items():
            name=f'NATS-tss-v1_0-3ffb9-simple/{index:06d}.pickle.pbz2'
            payload=tar.extractfile(name).read()
            data=reader.RestrictedReader(io.BytesIO(bz2.decompress(payload))).load()['200']
            require(data['arch_index']==index and data['arch_str']==rec['arch_str'],'Benchmark architecture mismatch')
            values=[]
            for seed in (777,888):
                trial=data['all_results'][(DATASET,seed)]
                require(trial['epochs']==200 and int(trial['seed'])==seed,'Wrong benchmark seed/budget')
                cfg=trial['arch_config']
                require(cfg['channel']==16 and cfg['num_cells']==5 and cfg['class_num']==120,'Wrong benchmark network')
                value=float(trial['eval_acc1es']['x-valid@199'])
                require(math.isfinite(value) and 0<=value<=100,'Invalid benchmark target')
                values.append(value)
            rec['id']=hashlib.md5(f'nb201:{DATASET}:{index}'.encode()).hexdigest()
            rows.append(dict(record_id=rec['id'],architecture_id=index,dataset=DATASET,split=membership[index],
                target_final_validation_accuracy=math.fsum(values)/2,target_epoch=200,
                benchmark_seed_777_accuracy=values[0],benchmark_seed_888_accuracy=values[1]))
            hashes[name]=hashlib.sha256(payload).hexdigest()
    cohort.update(dataset=DATASET,target_policy='Mean epoch200 x-valid over fixed seeds777/888',
        target_seed_policy='Fixed benchmark seeds777/888 for every architecture; seed999 is not used',
        early_accuracy_policy='Local x-valid accuracies from the same fresh run as Taylor',
        status='ImageNet16 target metadata prepared; candidate acquisition and predictor fitting not yet executed',
        source_cifar10_cohort_sha256=sha(cohort_path))
    out.mkdir(parents=True);put(out/'cohort.json',cohort)
    with (out/'targets.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    put(out/'image-test-split.txt',split_text)
    with (out/'targets.csv').open() as f:
        back=list(csv.DictReader(f))
    require(len(back)==1100,'Target readback count mismatch')
    for expected,actual in zip(rows,back):
        require(expected['record_id']==actual['record_id'] and expected['target_final_validation_accuracy']==float(actual['target_final_validation_accuracy']),'Target readback mismatch')
    put(out/'metadata.json',dict(dataset=DATASET,n=1100,target_kind='final_validation_accuracy',target_epoch=200,
        benchmark_target_seeds=[777,888],source_archive_sha256=ARCHIVE_SHA,source_members_sha256=hashes,
        source_cifar10_cohort_sha256=sha(cohort_path),official_split_url=url,
        files_sha256={n:sha(out/n) for n in ('cohort.json','targets.csv','image-test-split.txt')}))
    print('TARGETS_VERIFIED',out,'1100 architectures; no training',flush=True)


def replace_once(text,old,new):
    require(text.count(old)==1,'Source anchor changed: '+old[:100])
    return text.replace(old,new,1)


def engine_source(original):
    source=replace_once(original,"NAME='nas201-taylor-pilot-v1'",f'NAME={NAME!r}')
    source=replace_once(source,"SPLIT='configs/nas-benchmark/cifar100-test-split.txt'",f'SPLIT={SPLIT!r}')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='find_data')
    source=replace_once(source,ast.get_source_segment(source,node),'''def find_data(root):
    require((root/'train_data_batch_1').is_file() and (root/'val_data').is_file(), 'ImageNet16 batches missing from pod mount')
    return root''')
    changes=[
        ("require(total==5000,'Wrong xvalid image count')","require(total==3000,'Wrong xvalid image count')"),
        ('from torchvision import transforms,datasets','from torchvision import transforms'),
        ('    from xautodl.models import get_cell_based_tiny_net,CellStructure,get_search_spaces',
         '    from xautodl.models import get_cell_based_tiny_net,CellStructure,get_search_spaces\n    from xautodl.datasets.DownsampledImageNet import ImageNet16'),
        ('len(valid_ids)==len(test_ids)==5000','len(valid_ids)==len(test_ids)==3000'),
        ('mean=[v/255 for v in [129.3,124.1,112.4]];std=[v/255 for v in [68.2,65.4,70.4]]',
         'mean=[v/255 for v in [122.68,116.66,104.01]];std=[v/255 for v in [63.22,61.26,65.09]]'),
        ('transforms.RandomCrop(32,padding=4)','transforms.RandomCrop(16,padding=2)'),
        ('train=datasets.CIFAR100(str(data_root),train=True,transform=train_transform,download=False)',
         'train=ImageNet16(str(data_root),train=True,transform=train_transform,use_num_of_class_only=120)'),
        ('validation=datasets.CIFAR100(str(data_root),train=False,transform=eval_transform,download=False)',
         'validation=ImageNet16(str(data_root),train=False,transform=eval_transform,use_num_of_class_only=120)'),
        ("require(len(train)==50000 and len(validation)==10000,'Unexpected CIFAR100 size')",
         "require(len(train)==151700 and len(validation)==6000,'Unexpected ImageNet16-120 size')"),
        ('num_classes=100','num_classes=120'),
        ("dataset='cifar100'",f'dataset={DATASET!r}'),
        ("require(len(list(csv.DictReader(f)))==3,'CSV readback failed')",
         "require(len(list(csv.DictReader(f)))==len(ARCHS),'CSV readback failed')"),
    ]
    for old,new in changes:
        source=replace_once(source,old,new)
    compile(source,'engine.py','exec')
    return source


def common_source(original):
    names={'require','sha','js','put','load_module','verify','input_maps','validate_record',
           'validate_candidate','child','shard','show','collect'}
    chunks=[]
    found=set()
    for node in ast.parse(original).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            chunks.append(ast.get_source_segment(original,node))
        elif isinstance(node,ast.FunctionDef) and node.name in names:
            chunks.append(ast.get_source_segment(original,node));found.add(node.name)
    require(found==names,'Missing collection function')
    source=f'NAME={NAME!r}\nSHARDS=5\n'+ '\n\n'.join(chunks)
    source=source.replace("'cifar100'",repr(DATASET)).replace('CIFAR100 SHARD','IMAGENET16 SHARD')
    source+='''

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['child','shard'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('/a/output'))
    p.add_argument('--shard',type=int)
    p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='child': child(a.root,a.index,a.output)
    else: shard(a.root,a.shard,a.output)
'''
    compile(source,'common.py','exec')
    return source


def image_checksums(path):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ImageNet16')
    result={}
    for node in cls.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('train_list','valid_list'):
            result.update(dict(ast.literal_eval(node.value)))
    require(set(result)=={f'train_data_batch_{i}' for i in range(1,11)}|{'val_data'},'Wrong ImageNet16 batch list')
    return result


def verify_images(data,checks):
    problems=[]
    for name,expected in checks.items():
        path=data/name
        if not path.is_file():
            problems.append(name+': MISSING');continue
        h=hashlib.md5()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected:problems.append(name+': MD5 MISMATCH')
    require(not problems,'Dataset not ready at '+str(data)+'; files preserved:\n'+'\n'.join(problems))
    print('IMAGE_FILES_VERIFIED',data,len(checks),'official batches',flush=True)


def validate_metadata(metadata,original):
    meta=js(metadata/'metadata.json')
    require(meta['dataset']==DATASET and meta['n']==1100 and meta['source_archive_sha256']==ARCHIVE_SHA,'Wrong metadata source/scope')
    require(meta['target_kind']=='final_validation_accuracy' and meta['target_epoch']==200 and meta['benchmark_target_seeds']==[777,888],'Wrong target policy')
    require(meta['source_cifar10_cohort_sha256']==sha(original),'Frozen architecture cohort changed')
    for name,h in meta['files_sha256'].items():require(sha(metadata/name)==h,'Metadata drift: '+name)
    cohort=js(metadata/'cohort.json');old=js(original)
    require(cohort['splits']==old['splits'] and cohort['counts']==old['counts'],'Architecture split drift')
    members={r['arch_index']:r for r in cohort['architectures']}
    for r in old['architectures']:
        new=members[r['arch_index']]
        require(new['arch_str']==r['arch_str'] and new['arch_family']==r['arch_family'],'Architecture identity changed')
    with (metadata/'targets.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==len({int(r['architecture_id']) for r in rows})==1100,'Wrong target coverage')
    membership={i:s for s,ids in cohort['splits'].items() for i in ids}
    for r in rows:
        i=int(r['architecture_id']);values=[float(r[f'benchmark_seed_{s}_accuracy']) for s in (777,888)]
        require(r['record_id']==members[i]['id']==hashlib.md5(f'nb201:{DATASET}:{i}'.encode()).hexdigest(),'Wrong record ID')
        require(r['dataset']==DATASET and r['split']==membership[i] and int(r['target_epoch'])==200,'Target metadata mismatch')
        require(all(math.isfinite(v) and 0<=v<=100 for v in values) and float(r['target_final_validation_accuracy'])==math.fsum(values)/2,'Target value mismatch')
    split=js(metadata/'image-test-split.txt');a=set(map(int,split['xvalid'][1]));b=set(map(int,split['xtest'][1]))
    require(len(a)==len(b)==3000 and not a&b and a|b==set(range(6000)),'Image split mismatch')
    return cohort


def prepare(root,data,metadata):
    b=root/NAME;require(not b.exists(),'Existing experiment preserved: '+str(b))
    data=data.resolve();require(data.is_absolute() and not str(data).startswith('/shared/local/'),'Use an explicit shared dataset path, not worker-local storage')
    cohort=validate_metadata(metadata,root/'nb201_v1/cohort.json')
    parent=root/PARENT
    require(sha(parent/'runner.py')==PARENT_SHA,'Verified parent runner changed')
    require(sha(parent/'LayerAnalysis.py')==EXTRACTOR_SHA,'Verified Taylor extractor changed')
    parent_module=module(parent/'runner.py','verified_acquisition_parent');pc=parent_module.verify(parent)
    require(pc['official_commit']==COMMIT,'Official source revision changed')
    checks=image_checksums(parent/'autodl/xautodl/datasets/DownsampledImageNet.py')
    verify_images(data,checks)
    source=Path(__file__).with_name('cifar100.py');require(sha(source)==CIFAR_SHA,'Collection source changed')
    engine=engine_source((parent/'runner.py').read_text());common=common_source(source.read_text())
    template=js(parent/'job.json');plan=partition(cohort)
    b.mkdir();(b/'results').mkdir()
    put(b/'runner.py',Path(__file__).read_text());put(b/'engine.py',engine);put(b/'common.py',common)
    for name in ('cohort.json','targets.csv','metadata.json','image-test-split.txt'):
        shutil.copy2(metadata/name,b/name)
    shutil.copy2(parent/'LayerAnalysis.py',b/'LayerAnalysis.py')
    for name in pc['sha256']:
        if name.startswith('autodl/'):
            target=b/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,target)
    target=b/'autodl'/SPLIT;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(metadata/'image-test-split.txt',target)
    put(b/'manifest.json',plan)
    for k,ids in enumerate(plan['shards']):
        job=copy.deepcopy(template);name=NAME+('-pilot' if k==0 else f'-s{k-1}')
        labels={'experiment-batch':NAME,'stage':'acquisition','shard':str(k),'pilot':str(k==0).lower()}
        job['metadata']=dict(name=name,namespace='b-t-bose',labels=labels)
        job['spec']['activeDeadlineSeconds']=14400 if k==0 else 259200
        job['spec']['template']['metadata']=dict(labels=labels)
        pod=job['spec']['template']['spec'];pod.pop('nodeSelector',None)
        c=pod['containers'][0];c['name']='collector';c['image']=IMAGE;c['workingDir']='/a/output'
        c['args']=[f'''set -euo pipefail
unset PYTHONPATH
python3 -c 'import torch, torchvision, numpy; print("DEPENDENCIES", torch.__version__, torchvision.__version__, numpy.__version__)'
exec python3 -u /a/project/{NAME}/common.py shard --root /a/project --shard {k} --output /a/output
''']
        mounts={v['name']:v for v in c['volumeMounts']}
        require({'project','data','output'}<=set(mounts),'Unexpected acquisition mount template')
        require(mounts['project']['mountPath']=='/a/project' and mounts['data']['mountPath']=='/a/data' and mounts['output']['mountPath']=='/a/output','Unexpected mount paths')
        mounts['project']['readOnly']=True;mounts['data']['readOnly']=True
        volumes={v['name']:v for v in pod['volumes']}
        volumes['project']['hostPath']['path']=str(root)
        volumes['data']['hostPath']['path']=str(data)
        volumes['output']['hostPath']['path']=str(b/'results')
        put(b/'jobs'/f's{k}.json',job)
    hashes={p.relative_to(b).as_posix():sha(p) for p in b.rglob('*') if p.is_file()}
    put(b/'contract.json',dict(name=NAME,dataset=DATASET,n_architectures=1100,pilot_ids=plan['pilot_ids'],
        candidate_seed=777,taylor_seed=20261025,taylor_batch_size=256,candidate_timeout_seconds=3600,
        epochs=[1,2,3],schedule_epochs=200,C=16,N=5,target_kind='final_validation_accuracy',target_epoch=200,
        benchmark_target_seeds=[777,888],official_commit=COMMIT,sha256=hashes,
        data_path=str(data),data_md5=checks,source_parent_sha256=PARENT_SHA,extractor_sha256=EXTRACTOR_SHA,
        validation='151700 training images; official 3000-image x-valid subset; x-test never evaluated',
        stage='Feature acquisition only; no predictors fitted',pilot_policy='First three frozen architectures reused in full collection; remaining IDs only in four other shards'))
    print('PREPARED',b,'pilot',plan['pilot_ids'],'remaining shards',[len(x) for x in plan['shards'][1:]],'NO JOBS SUBMITTED',flush=True)


def prepared(root,images=False):
    b=root/NAME;contract=js(b/'contract.json')
    # Verify before importing the prepared executable source.
    for name,h in contract['sha256'].items():require(sha(b/name)==h,'Prepared artifact drift: '+name)
    common=module(b/'common.py','verified_imagenet_collection')
    common.verify(root)
    if images:verify_images(Path(contract['data_path']),contract['data_md5'])
    return common,contract


def submit(root,stage):
    require(stage in ('pilot','full'),'Invalid submission stage')
    common,c=prepared(root,images=True);b=root/NAME;plan=js(b/'manifest.json')
    receipt=b/f'submission-{stage}.jsonl'
    require(not receipt.exists(),'Existing submission receipt preserved; inspect Jobs before any retry')
    if stage=='full':
        report=js(b/'results/shards/s0/report.json')
        require(report['attempted']==report['complete']==3 and not report['failures'],'Pilot artifacts incomplete')
        members,targets=common.input_maps(root)
        for index in plan['pilot_ids']:common.validate_candidate(b/'results/candidates'/f'{index:05d}',index,members,targets)
        result=subprocess.run(['kubectl','-n','b-t-bose','get','job',NAME+'-pilot','-o','json'],capture_output=True,text=True)
        require(result.returncode==0,'Cannot inspect pilot Job: '+result.stderr)
        require(json.loads(result.stdout).get('status',{}).get('succeeded')==1,'Pilot Job has not succeeded')
    indices=[0] if stage=='pilot' else list(range(1,5))
    jobs=[]
    for k in indices:
        require(not (b/'results/shards'/f's{k}').exists(),'Existing shard artifacts preserved')
        require(not any((b/'results/candidates'/f'{i:05d}').exists() for i in plan['shards'][k]),'Candidate outputs already exist; no retraining')
        path=b/'jobs'/f's{k}.json';job=js(path);name=job['metadata']['name']
        r=subprocess.run(['kubectl','-n','b-t-bose','get','job',name,'--ignore-not-found','-o','name'],capture_output=True,text=True)
        require(r.returncode==0 and not r.stdout.strip(),'Existing Job or query failure: '+r.stdout+r.stderr)
        subprocess.run(['kubectl','-n','b-t-bose','create','--dry-run=server','-f',str(path)],check=True)
        jobs.append((name,path))
    with receipt.open('x',encoding='utf-8') as f:
        for name,path in jobs:
            r=subprocess.run(['kubectl','-n','b-t-bose','create','-f',str(path)],capture_output=True,text=True)
            f.write(json.dumps(dict(job=name,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n');f.flush()
            print(r.stdout or r.stderr,flush=True)
            require(r.returncode==0,'Partial submission; preserve receipt and existing Jobs')
    print('SUBMITTED',stage,len(jobs),'Job(s)',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['targets','prepare','check','pilot','full','show','collect'])
    p.add_argument('--root',type=Path,default=HOST)
    p.add_argument('--data',type=Path)
    p.add_argument('--metadata',type=Path,default=Path(__file__).parent/'imagenet16')
    p.add_argument('--cohort',type=Path)
    p.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='targets':
        require(a.cohort and a.archive and a.output,'targets requires --cohort --archive --output')
        build_targets(a.cohort,a.archive,a.output)
    elif a.action=='prepare':
        prepare(a.root,a.data or a.root/'nn-dataset/data/ImageNet16',a.metadata)
    elif a.action in ('pilot','full'):
        submit(a.root,a.action)
    else:
        common,c=prepared(a.root,images=a.action=='check')
        if a.action=='check':print('PREFLIGHT_PASS',NAME,'free GiB',round(shutil.disk_usage(a.root).free/2**30,2))
        elif a.action=='show':
            print('Internal shard0 is the three-architecture pilot; shards1-4 are the remaining cohort.')
            common.show(a.root)
        else:common.collect(a.root)


if __name__=='__main__':
    main()
