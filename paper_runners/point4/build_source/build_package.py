import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).parent
OUT = HERE / 'point4-all3-no-checkpoints-v1'
BASE = 'lemur-taylor-robustness-full-v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(data if isinstance(data,str) else json.dumps(data,indent=2,allow_nan=False))
def replace(s,a,b):
    assert s.count(a)==1, (a[:100],s.count(a))
    return s.replace(a,b)

original=read(HERE/'contract.json')
assert sha(HERE/'worker.py') == original['files_sha256']['worker.py']
records=[json.loads(x) for x in (HERE/'snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
assert len({r['id'] for r in records})==len(records)
byid={r['id']:r for r in records}
fold=next(f for f in read(HERE/'folds.json') if f['fold']==0)
allowed=set(fold['inner_train_ids'])
groups=read(HERE/'groups.json'); groupof={i:g for g,ids in groups.items() for i in ids}
old_by_group={x['group']:x for x in original['selected']}
assert len(old_by_group)==20
OUT.mkdir(exist_ok=False)
selection={}; manifests=[]

for dataset, suffix, classes, norm, ctor, length in (
 ('cifar-10','cifar10',10, '([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])', 'datasets.CIFAR10(str(data), train=True,',50000),
 ('cifar-100','cifar100',100,'([v / 255 for v in (129.3, 124.1, 112.4)], [v / 255 for v in (68.2, 65.4, 70.4)])','datasets.CIFAR100(str(data), train=True,',50000),
 ('imagenette','imagenette',10,'([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])','datasets.Imagenette(str(data), split="train",',9469),
):
    name='lemur-robustness-all3-v1-'+suffix
    directory=OUT/name
    candidates={}
    for r in records:
        if r['id'] in allowed and r['dataset']==dataset and r['task']=='img-classification' and r['metric']=='acc':
            candidates.setdefault(groupof[r['id']],[]).append(r)
    selected=[]
    for group, choices in sorted(candidates.items()):
        old=old_by_group.get(group) if dataset=='cifar-100' else None
        r=byid[old['record_id']] if old else min(choices,key=lambda x:x['id'])
        assert r in choices
        i=len(selected)
        item=dict(index=i,record_id=r['id'],original_nn=r['nn'],group=group,dataset=dataset,
                  model_file=f'models/candidate_{i}.py',transform_file=f'transforms/candidate_{i}.py',record_file=f'records/candidate_{i}.json')
        if old: item['reuse']={'bundle':BASE,'index':old['index'],'record_id':old['record_id']}
        selected.append(item)
        put(directory/item['model_file'],r['nn_code'])
        put(directory/item['transform_file'],r['transform_code'])
        put(directory/item['record_file'],r)
    selection[dataset]=selected
    s=(HERE/'worker.py').read_text(encoding='utf-8')
    s=replace(s,"NAME = '"+BASE+"'", "NAME = '"+name+"'")
    s=replace(s,"(output.shape[1] == 100)",f"(output.shape[1] == {classes})")
    s=replace(s,"[batch,100] classification tensor",f"[batch,{classes}] classification tensor")
    a=s.index('    norm = (',s.index('def build(')); b=s.index('    transform_module',a)
    s=s[:a]+'    norm = '+norm+'\n'+s[b:]
    a=s.index('    raw_data = datasets.CIFAR100',s.index('def build(')); b=s.index('\n    # Validate shape',a)
    s=s[:a]+f'    raw_data = {ctor} download=False)\n    training = {ctor} transform=recorded, download=False)\n    require(len(training) == len(raw_data) == {length}, "Unexpected dataset length")\n'+s[b:]
    s=replace(s,'model = module.Net((batch, *input_shape), (100,), prm, device).to(device)',f'model = module.Net((batch, *input_shape), ({classes},), prm, device).to(device)')
    a=s.index('    checkpoint_bytes = sum(',s.index('def build(')); b=s.index('    extractor = load_module',a)
    s=s[:a]+'    require(shutil.disk_usage(output).free > 256 * 1024**2, "Less than 256 MiB free; preserving existing output")\n\n'+s[b:]
    a=s.index('        path = dest / f"epoch-{epoch}.pt"',s.index('def candidate(')); b=s.index('        probe_checkpoint(',a)
    s=s[:a]+'''        put(dest / f"epoch-{epoch}.json", {
            'epoch': epoch, 'record_id': record['id'],
            'state_sha256': state_hash(model),
            'persistent_checkpoint': False,
            'timing': 'Immediately after learn; before evaluation/extraction',
            'scope': 'In-memory extraction point; not replayable after process exit',
        })
'''+s[b:]
    # New candidates use the existing build validation, not a second disposable fit.
    s=replace(s,'            for phase in ("preflight", "candidate"):', '''            item = c['selected'][index]
            if item.get('reuse'):
                require((output / f'candidate-{index}' / 'reuse_provenance.json').is_file(), 'Missing prepared reuse evidence')
            for phase in (() if item.get('reuse') else ('candidate',)):''')
    # Checksums still cover all copied metadata for reused runs.
    s=replace(s,'"Controlled new CIFAR100 subset, not historical trajectory replay.",',json.dumps('All eligible fold0 inner-training '+dataset+' architecture groups; deterministic representative per group, not historical trajectory replay.')+',')
    compile(s,name+'/worker.py','exec')
    put(directory/'worker.py',s)
    c=copy.deepcopy(original)
    c.update(name=name,selected=selected,shards=[list(range(k,len(selected),2)) for k in range(2)],dataset=dataset,
             persistent_checkpoints=False, selection='All eligible fold0 inner-training architecture groups for this dataset; preserve existing selected CIFAR100 records; otherwise lexicographically smallest record ID per group; no performance selection or failure replacement.',
             preprocessing=original['preprocessing'],scope='New runs do not persist checkpoints; exact existing completed records reused via verified metadata copies.')
    c['files_sha256']={str(p.relative_to(directory)).replace('\\','/'):sha(p) for p in directory.rglob('*') if p.is_file()}
    put(directory/'contract.json',c)
    manifests.append({'name':name,'dataset':dataset,'count':len(selected)})
    for k in range(2):
        job=read(HERE/'jobs'/f's{k}.json')
        jobname=f'{name}-s{k}'
        job['metadata']['name']=jobname
        job['metadata']['labels']['experiment-batch']='lemur-robustness-all3-v1'
        job['metadata']['labels']['dataset']=suffix
        job['spec']['activeDeadlineSeconds']=604800
        pod=job['spec']['template']['spec']
        job['spec']['template']['metadata']['labels']=copy.deepcopy(job['metadata']['labels'])
        pod['nodeSelector']={'kubernetes.io/hostname':'w2-mi'}
        container=pod['containers'][0]
        container['args']=[container['args'][0].replace('/a/project/'+BASE+'/worker.py','/a/project/point4-all3-no-checkpoints-v1/'+name+'/worker.py').replace('--root /a/project','--root /a/project/point4-all3-no-checkpoints-v1')]
        # Keep parent project dependencies accessible through package symlinks prepared on cluster.
        init=pod['initContainers'][0]
        init['args']=[f'''import torch, torchvision, numpy
from torchvision import datasets
assert torch.__version__ == '2.7.1+cu126'
assert torchvision.__version__ == '0.22.1'
assert numpy.__version__ == '2.2.5'
data = {ctor.replace('str(data)',repr('/a/data'))} download=False)
assert len(data)=={length}
assert len(data.classes)=={classes}
for i in (0,len(data)-1):
    image,label=data[i]
    assert 0<=label<{classes}
print('DATA_READY', {dataset!r}, len(data), flush=True)
''']
        for vol in pod['volumes']:
            if vol['name']=='output': vol['hostPath']['path']=f'/shared/ssd/home/b-t-bose/lmur/point4-all3-no-checkpoints-v1/{name}/results'
            if vol['name']=='data' and dataset!='cifar-100': vol['hostPath']['path']='/shared/local/data/a-group-automl/data'
        put(OUT/'jobs'/f'{suffix}-s{k}.json',job)

put(OUT/'selection.json',selection)
put(OUT/'manifest.json',{'bundles':manifests,'source_hashes':{n:sha(HERE/n) for n in ('snapshot.jsonl','folds.json','groups.json')},'old_contract_sha256':sha(HERE/'contract.json')})
print(json.dumps({d:{'groups':len(v),'reuse':sum('reuse' in x for x in v)} for d,v in selection.items()},indent=2))
