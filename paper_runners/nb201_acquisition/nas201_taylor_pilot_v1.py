"""Three-architecture NAS-Bench-201/CIFAR-100 Taylor acquisition pilot.
No benchmark targets, Qwen fitting, or full-dataset feature extraction are performed.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

NAME='nas201-taylor-pilot-v1'
COMMIT='f46486e21b71ae6459a700be720d7648b5429569'
HOST=Path('/shared/ssd/home/b-t-bose/lmur')
CONFIG='configs/nas-benchmark/hyper-opts/200E.config'
SPLIT='configs/nas-benchmark/cifar100-test-split.txt'
ARCHS={
 0:'|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|',
 9:'|avg_pool_3x3~0|+|none~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|',
 123:'|avg_pool_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|none~0|avg_pool_3x3~1|nor_conv_3x3~2|',
}

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def put(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,indent=2,allow_nan=False))

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def verify(bundle):
    c=json.loads((bundle/'contract.json').read_text())
    for rel,digest in c['sha256'].items():require(sha(bundle/rel)==digest,'Source drift: '+rel)
    return c

def prepare(root,node):
    bundle=root/NAME
    require(not bundle.exists(),'Refusing existing bundle: '+str(bundle))
    extractor=root/'nn-dataset/ab/nn/util/LayerAnalysis.py'
    require(extractor.is_file(),'Missing existing LayerAnalysis.py')
    source=extractor.read_text();compile(source,str(extractor),'exec')
    url=f'https://codeload.github.com/D-X-Y/AutoDL-Projects/zip/{COMMIT}'
    print('Downloading pinned official source (not benchmark weights):',url,flush=True)
    with urllib.request.urlopen(url,timeout=120) as response:archive=response.read()
    selected={}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for item in z.infolist():
            if item.is_dir():continue
            rel=Path(*Path(item.filename).parts[1:])
            require('..' not in rel.parts,'Unsafe archive path')
            name=rel.as_posix()
            if (name.startswith('xautodl/') and name.endswith('.py')) or name in [CONFIG,SPLIT,'LICENSE.md','LICENSE']:
                selected[name]=z.read(item)
    require(CONFIG in selected and SPLIT in selected,'Official configuration files missing')
    bundle.mkdir()
    (bundle/'results').mkdir()
    put(bundle/'runner.py',Path(__file__).read_text())
    put(bundle/'LayerAnalysis.py',source)
    for rel,content in selected.items():
        dest=bundle/'autodl'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(content)
    job={
      'apiVersion':'batch/v1','kind':'Job','metadata':{'name':NAME,'namespace':'b-t-bose','labels':{'experiment-batch':NAME}},
      'spec':{'backoffLimit':0,'activeDeadlineSeconds':14400,'template':{
        'metadata':{'labels':{'experiment-batch':NAME}},'spec':{
          'restartPolicy':'Never','nodeSelector':{'kubernetes.io/hostname':node},
          'securityContext':{'runAsNonRoot':True,'runAsUser':1060,'runAsGroup':1376,'supplementalGroups':[1060,1376]},
          'containers':[{'name':'pilot','image':'abrainone/ai-linux:cu12.6.3-latest','workingDir':'/a/output',
            'command':['/bin/bash','-lc'],'args':[f'''set -euo pipefail
unset PYTHONPATH
python3 -c 'import torch, torchvision, numpy; print("DEPENDENCIES", torch.__version__, torchvision.__version__, numpy.__version__)'
exec python3 -u /a/project/{NAME}/runner.py worker --root /a/project --data /a/data --output /a/output
'''],
            'env':[{'name':'PYTHONDONTWRITEBYTECODE','value':'1'},{'name':'OMP_NUM_THREADS','value':'4'},
                   {'name':'NODE_NAME','valueFrom':{'fieldRef':{'fieldPath':'spec.nodeName'}}}],
            'resources':{'requests':{'cpu':'4','memory':'12Gi','nvidia.com/gpu':'1'},'limits':{'cpu':'8','memory':'24Gi','nvidia.com/gpu':'1'}},
            'volumeMounts':[{'name':'project','mountPath':'/a/project','readOnly':True},
                           {'name':'data','mountPath':'/a/data','readOnly':True},
                           {'name':'output','mountPath':'/a/output'},
                           {'name':'shm','mountPath':'/dev/shm'}]}],
          'volumes':[{'name':'project','hostPath':{'path':str(root),'type':'Directory'}},
                     {'name':'data','hostPath':{'path':'/shared/local/data/a-group-automl/data','type':'Directory'}},
                     {'name':'output','hostPath':{'path':str(bundle/'results'),'type':'Directory'}},
                     {'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'2Gi'}}]}}}}
    put(bundle/'job.json',job)
    hashes={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    put(bundle/'contract.json',dict(experiment=NAME,official_commit=COMMIT,source_archive_sha256=hashlib.sha256(archive).hexdigest(),
        architectures=ARCHS,candidate_seed=777,epochs_observed=[1,2,3],schedule_epochs=200,C=16,N=5,
        taylor_batch_size=256,taylor_batches=1,taylor_seed=20261025,
        measurement='Fixed augmented training batch, real labels; existing Taylor method on a deep copy; RNG restored',
        validation='Official CIFAR100 xvalid subset; xtest is never evaluated',
        purpose='Extraction/runtime pilot only; NOT a benchmark performance comparison or exact replay of historical training',
        sha256=hashes))
    verify(bundle)
    print('PREPARED:',bundle,'; no training submitted.',flush=True)

def find_data(root):
    candidates=[root,root/'cifar-100',root/'cifar100',root/'CIFAR100',root/'cifar.python']
    matches=[p for p in candidates if (p/'cifar-100-python/train').is_file() and (p/'cifar-100-python/test').is_file()]
    require(len(matches)==1,'Need one CIFAR100 torchvision-format root. Checked: '+', '.join(map(str,candidates))+'; matches: '+str(matches))
    return matches[0]

def state_hash(model):
    import torch
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        tensor=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def sync(device):
    import torch
    if device.type=='cuda':torch.cuda.synchronize(device)

def measure_taylor(model,batch,analyzer_class,device):
    import torch
    import numpy as np
    original=state_hash(model)
    flags=[m.training for m in model.modules()]
    gradients=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    python_rng=random.getstate();numpy_rng=np.random.get_state();torch_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    clone=None
    try:
        clone=copy.deepcopy(model)
        analyzer=analyzer_class(clone,device)
        loss=torch.nn.CrossEntropyLoss()
        # Official tiny networks return (features, logits); no module wrapping or renaming.
        def loss_fn(output,labels):return loss(output[1],labels)
        sync(device);start=time.perf_counter()
        raw=analyzer.taylor_importance([batch],loss_fn,num_batches=1)
        sync(device);seconds=time.perf_counter()-start
        table=analyzer.build_layer_table({'taylor_importance':raw})
        values={name:row['taylor_imp'] for name,row in table.items()}
        require(values and all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in values.values()),
                'Incomplete/nonfinite Taylor coverage; no zero filling')
        score=math.fsum(values.values())/len(values)
        return dict(taylor=score,layers=values,layer_count=len(values),extraction_seconds=seconds)
    finally:
        del clone
        random.setstate(python_rng);np.random.set_state(numpy_rng);torch.set_rng_state(torch_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        require(state_hash(model)==original,'Taylor measurement changed original weights/buffers')
        require([m.training for m in model.modules()]==flags,'Taylor measurement changed original training mode')
        for p,g in zip(model.parameters(),gradients):
            require((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad,g)),
                    'Taylor measurement changed original gradients')

def train_epoch(net,loader,optimizer,scheduler,criterion,epoch,device):
    net.train();scheduler.update(epoch-1,0.0);sync(device);start=time.perf_counter()
    total=correct=0;loss_sum=0.
    for step,(x,y) in enumerate(loader):
        scheduler.update(None,float(step)/len(loader))
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        optimizer.zero_grad();_,logits=net(x);loss=criterion(logits,y)
        require(bool(loss.isfinite()),'Nonfinite training loss')
        loss.backward();optimizer.step()
        total+=len(y);correct+=int((logits.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
    sync(device)
    return dict(train_accuracy=100*correct/total,train_loss=loss_sum/total,training_seconds=time.perf_counter()-start,learning_rate=min(scheduler.get_lr()))

def evaluate(net,loader,device):
    import torch
    net.eval();total=correct=0
    with torch.no_grad():
        for x,y in loader:
            _,logits=net(x.to(device));require(bool(logits.isfinite().all()),'Nonfinite validation logits')
            correct+=int((logits.argmax(1).cpu()==y).sum());total+=len(y)
    require(total==5000,'Wrong xvalid image count')
    return 100*correct/total

def worker(root,data_root,output):
    bundle=root/NAME;c=verify(bundle)
    require(not any(output.iterdir()),'Outputs exist; refusing another run')
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms,datasets
    require(torch.cuda.is_available(),'GPU required for real pilot')
    device=torch.device('cuda:0');torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data_root=find_data(data_root)
    sys.path.insert(0,str(bundle/'autodl'))
    from xautodl.models import get_cell_based_tiny_net,CellStructure,get_search_spaces
    optim_module=module(bundle/'autodl/xautodl/procedures/optimizers.py','official_optimizer')
    layer_module=module(bundle/'LayerAnalysis.py','frozen_layer_analysis')
    architectures=CellStructure.gen_all(get_search_spaces('cell','nats-bench'),4,False)
    random.Random(88).shuffle(architectures)
    require(len(architectures)==15625,'Architecture enumeration mismatch')
    for index,arch in ARCHS.items():require(architectures[index].tostr()==arch,'Official architecture-index mapping changed')
    cfg_raw=json.loads((bundle/'autodl'/CONFIG).read_text())
    convert={'int':int,'float':float,'str':str,'bool':lambda v:bool(int(v))}
    cfg=SimpleNamespace(**{key:convert[kind](value) for key,(kind,value) in cfg_raw.items()})
    require(cfg.epochs==200 and cfg.batch_size==256 and cfg.LR==.1,'Official training config changed')
    split=json.loads((bundle/'autodl'/SPLIT).read_text())
    valid_ids=list(map(int,split['xvalid'][1]));test_ids=list(map(int,split['xtest'][1]))
    require(len(valid_ids)==len(test_ids)==5000 and not set(valid_ids)&set(test_ids),'Bad official image split')
    mean=[v/255 for v in [129.3,124.1,112.4]];std=[v/255 for v in [68.2,65.4,70.4]]
    train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4),transforms.ToTensor(),transforms.Normalize(mean,std)])
    eval_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    train=datasets.CIFAR100(str(data_root),train=True,transform=train_transform,download=False)
    validation=datasets.CIFAR100(str(data_root),train=False,transform=eval_transform,download=False)
    require(len(train)==50000 and len(validation)==10000,'Unexpected CIFAR100 size')
    # Dedicated RNGs prevent loading/evaluation from perturbing the training sampler.
    random.seed(c['taylor_seed']);np.random.seed(c['taylor_seed']);torch.manual_seed(c['taylor_seed'])
    batch_ids=random.Random(c['taylor_seed']).sample(range(len(train)),c['taylor_batch_size'])
    batch=torch.utils.data.default_collate([train[i] for i in batch_ids])
    valid_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(validation,valid_ids),batch_size=cfg.batch_size,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(777))
    put(output/'runtime.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0),node=os.environ.get('NODE_NAME'),data_root=str(data_root),
        config=vars(cfg),official_commit=COMMIT,extractor_sha256=sha(bundle/'LayerAnalysis.py'),measurement_train_indices=batch_ids,
        note='Official architecture/config/transforms/xvalid; fresh local training, not exact historical RNG replay. No benchmark target queried.'))
    records=[]
    for index,arch in ARCHS.items():
        random.seed(c['candidate_seed']);np.random.seed(c['candidate_seed']);torch.manual_seed(c['candidate_seed']);torch.cuda.manual_seed_all(c['candidate_seed'])
        net=get_cell_based_tiny_net(dict(name='infer.tiny',C=16,N=5,arch_str=arch,num_classes=100)).to(device)
        optimizer,scheduler,criterion=optim_module.get_optim_scheduler(net.parameters(),cfg)
        loader=torch.utils.data.DataLoader(train,batch_size=cfg.batch_size,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(c['candidate_seed']))
        # Real tensor compatibility check before the first epoch; does not update the trained model.
        probe=measure_taylor(net,batch,layer_module.LayerAnalyzer,device)
        print('ARCHITECTURE_PREFLIGHT_PASS',index,'layers',probe['layer_count'],flush=True)
        record=dict(architecture_id=index,architecture_string=arch,dataset='cifar100',candidate_seed=c['candidate_seed'],
                    log_params=math.log1p(sum(p.numel() for p in net.parameters())),depth=len(list(net.modules())),epochs=[])
        for epoch in (1,2,3):
            timings=train_epoch(net,loader,optimizer,scheduler,criterion,epoch,device)
            accuracy=evaluate(net,valid_loader,device)
            measurement=measure_taylor(net,batch,layer_module.LayerAnalyzer,device)
            one=dict(epoch=epoch,validation_accuracy=accuracy,**timings,**measurement)
            record['epochs'].append(one)
            put(output/f'arch-{index}-epoch-{epoch}.json',dict(architecture_id=index,architecture_string=arch,candidate_seed=c['candidate_seed'],**one))
            print('EPOCH_COMPLETE',index,epoch,'validation_accuracy',accuracy,'taylor',measurement['taylor'],flush=True)
        records.append(record);put(output/f'arch-{index}-complete.json',record)
        del net,optimizer,scheduler;torch.cuda.empty_cache()
    with (output/'features.csv').open('x',newline='') as f:
        fields=['architecture_id','dataset','candidate_seed','log_params','depth']+[f'{name}_epoch_{e}' for e in (1,2,3) for name in ('accuracy','taylor')]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            row={k:r[k] for k in fields if k in r}
            for e in r['epochs']:
                row[f"accuracy_epoch_{e['epoch']}"]=e['validation_accuracy'];row[f"taylor_epoch_{e['epoch']}"]=e['taylor']
            w.writerow(row)
    with (output/'features.csv').open(newline='') as f:require(len(list(csv.DictReader(f)))==3,'CSV readback failed')
    put(output/'complete.json',dict(status='complete',architectures=list(ARCHS),epochs=[1,2,3],benchmark_targets_attached=False,
        training_seconds_total=sum(e['training_seconds'] for r in records for e in r['epochs']),
        extraction_seconds_total=sum(e['extraction_seconds'] for r in records for e in r['epochs']),
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('PILOT_COMPLETE:',output,'; feature-extraction pilot only, not full benchmark replication.',flush=True)

def submit(root):
    bundle=root/NAME;verify(bundle)
    require(not any((bundle/'results').iterdir()),'Outputs exist; inspect before rerunning')
    r=subprocess.run(['kubectl','-n','b-t-bose','get','job',NAME,'--ignore-not-found','-o','name'],capture_output=True,text=True)
    require(r.returncode==0 and not r.stdout.strip(),'Job exists or query failed: '+r.stdout+r.stderr)
    for flags in [['--dry-run=server'],[]]:
        subprocess.run(['kubectl','-n','b-t-bose','create',*flags,'-f',str(bundle/'job.json')],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','worker','show'])
    p.add_argument('--root',type=Path,default=HOST);p.add_argument('--node',default='w2-mi')
    p.add_argument('--data',type=Path,default=Path('/a/data'));p.add_argument('--output',type=Path,default=Path('/a/output'))
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.node)
    elif a.action=='submit':submit(a.root)
    elif a.action=='show':
        out=a.root/NAME/'results';done=json.loads((out/'complete.json').read_text())
        for name,digest in done['files_sha256'].items():require(sha(out/name)==digest,'Result drift: '+name)
        print(json.dumps(done,indent=2));print((out/'features.csv').read_text())
    else:
        require(a.output.is_dir() and not any(a.output.iterdir()),
                'Outputs exist or output directory is unavailable; refusing worker')
        try:worker(a.root,a.data,a.output)
        except Exception:
            if a.output.is_dir() and not (a.output/'failure.json').exists():
                put(a.output/'failure.json',dict(error=traceback.format_exc()))
            raise

if __name__=='__main__':main()
