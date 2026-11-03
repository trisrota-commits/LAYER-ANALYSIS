"""CIFAR10-valid acquisition pilot for the supervisor's frozen NB201 workflow.
Original attachment scripts remain unchanged. No Qwen training or full cohort run.
"""
import argparse,copy,csv,hashlib,importlib.util,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace

NAME='nb201-v1-cifar10-pilot'
PARENT='nas201-taylor-pilot-v1'
PARENT_SHA='87b2357f87038bd6206243962dbef149a0c55557ffcf2f05b1996c0badf54e5f'
EXTRACTOR_SHA='f19dcb88702679de1ada97dde6766bbbdb44019d4b05c04e1557fa1bdde9ad9f'
SPLIT_SHA='986da6174822048d4c5a25ee8ebdb7e754cd5372b51c633ffd4898ace8db9e32'

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def put(p,x):
    with Path(p).open('x',encoding='utf-8',newline='') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))

def js(p):return json.loads(Path(p).read_text())

def mod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(root):
    src=root/'nb201_v1';out=root/NAME;parent=root/PARENT
    require(not out.exists(),'Existing pilot preserved: '+str(out))
    require(sha(parent/'runner.py')==PARENT_SHA,'Verified helper changed')
    guard=mod(parent/'runner.py','verified_pilot');guard.verify(parent)
    require(sha(parent/'LayerAnalysis.py')==EXTRACTOR_SHA,'Extractor changed')
    require(sha(src/'cifar-split.txt')==SPLIT_SHA,'Official split changed')
    cohort=js(src/'cohort.json')
    require(cohort['dataset']=='cifar10-valid' and cohort['n']==1100,'Wrong cohort')
    require(cohort['counts']=={'train':772,'val':53,'test':275},'Unexpected grouped split')
    chosen=cohort['architectures'][:3]
    require([r['arch_index'] for r in chosen]==[3,32,41],'Pilot identities changed')
    require(js(parent/'results/complete.json')['status']=='complete','Earlier GPU pilot not complete')
    shell=f'''set -euo pipefail
unset PYTHONPATH
python3 -c 'import torch, torchvision, numpy; print("DEPENDENCIES", torch.__version__, torchvision.__version__, numpy.__version__)'
exec python3 -u /a/project/{NAME}/runner.py worker --root /a/project --data /a/data --output /a/output
'''
    job={'apiVersion':'batch/v1','kind':'Job','metadata':{'name':NAME,'namespace':'b-t-bose','labels':{'experiment-batch':NAME}},
      'spec':{'backoffLimit':0,'activeDeadlineSeconds':14400,'template':{'metadata':{'labels':{'experiment-batch':NAME}},'spec':{
        'restartPolicy':'Never','nodeSelector':{'kubernetes.io/hostname':'w2-mi'},
        'securityContext':{'runAsNonRoot':True,'runAsUser':1060,'runAsGroup':1376,'supplementalGroups':[1060,1376]},
        'containers':[{'name':'pilot','image':'abrainone/ai-linux:cu12.6.3-latest','workingDir':'/a/output','command':['/bin/bash','-lc'],'args':[shell],
          'env':[{'name':'PYTHONDONTWRITEBYTECODE','value':'1'},{'name':'OMP_NUM_THREADS','value':'4'},{'name':'NODE_NAME','valueFrom':{'fieldRef':{'fieldPath':'spec.nodeName'}}}],
          'resources':{'requests':{'cpu':'4','memory':'12Gi','nvidia.com/gpu':'1'},'limits':{'cpu':'8','memory':'24Gi','nvidia.com/gpu':'1'}},
          'volumeMounts':[{'name':'project','mountPath':'/a/project','readOnly':True},{'name':'data','mountPath':'/a/data','readOnly':True},
                          {'name':'output','mountPath':'/a/output'},{'name':'shm','mountPath':'/dev/shm'}]}],
        'volumes':[{'name':'project','hostPath':{'path':str(root),'type':'Directory'}},
                   {'name':'data','hostPath':{'path':'/shared/local/data/a-group-automl/data','type':'Directory'}},
                   {'name':'output','hostPath':{'path':str(out/'results'),'type':'Directory'}},
                   {'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'2Gi'}}]}}}}
    out.mkdir();(out/'results').mkdir()
    for name in ['cohort.json','benchmark_targets.csv','cifar-split.txt']:
        with (out/name).open('xb') as f:f.write((src/name).read_bytes())
    put(out/'runner.py',Path(__file__).read_text());put(out/'job.json',job)
    put(out/'contract.json',dict(name=NAME,pilot_ids=[3,32,41],candidate_seed=777,epochs=[1,2,3],schedule_epochs=200,
      full_cohort_size=1100,sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
      parent_runner_sha256=PARENT_SHA,extractor_sha256=EXTRACTOR_SHA,
      input_policy='Benchmark mean early accuracies and peak of mean validation curve, as in attached collector; local validation accuracies separately retained',
      implementation='Official CIFAR10 train/valid split and transforms; no gradient clipping. Existing Taylor extractor on a clone in train mode. Taylor only, not seven reimplemented statistics.',
      aggregation='Mean over layers for which existing extractor returns a finite Taylor value; layers not returned are explicitly listed, never zero-filled',
      scope='Three-architecture collection pilot only; no primary-endpoint analysis or Qwen training'))
    verify(root);print('PREPARED:',out,'; NOT submitted.',flush=True)

def verify(root):
    b=root/NAME;c=js(b/'contract.json')
    for name,h in c['sha256'].items():require(sha(b/name)==h,'Prepared artifact drift: '+name)
    require(sha(root/PARENT/'runner.py')==PARENT_SHA,'Helper source drift')
    guard=mod(root/PARENT/'runner.py','verified_helper');guard.verify(root/PARENT)
    require(sha(root/PARENT/'LayerAnalysis.py')==EXTRACTOR_SHA,'Extractor source drift')
    return c,guard

def data_path(root):
    options=[root,root/'cifar10',root/'cifar-10',root/'CIFAR10',root/'cifar.python']
    found=[p for p in options if (p/'cifar-10-batches-py/data_batch_1').is_file() and (p/'cifar-10-batches-py/test_batch').is_file()]
    require(len(found)==1,'Need one torchvision CIFAR10 root; checked '+str(options)+'; found '+str(found))
    return found[0]

def taylor(net,batch,LayerAnalyzer,guard,device):
    coverage={}
    class ReportCoverage(LayerAnalyzer):
        def build_layer_table(self,analysis):
            table=super().build_layer_table(analysis);raw=analysis['taylor_importance']
            coverage['total_weight_layers']=len(table)
            coverage['unmeasured_layers']=[n for n in table if n not in raw]
            return {n:row for n,row in table.items() if n in raw}
    result=guard.measure_taylor(net,batch,ReportCoverage,device)
    result.update(coverage)
    return result

def accuracy(net,loader,device):
    import torch
    net.eval();correct=total=0
    with torch.no_grad():
        for x,y in loader:
            _,logits=net(x.to(device));require(bool(logits.isfinite().all()),'Nonfinite validation logits')
            correct+=int((logits.argmax(1).cpu()==y).sum());total+=len(y)
    require(total==25000,'Wrong CIFAR10-valid count')
    return 100*correct/total

def worker(root,data,output):
    c,guard=verify(root);b=root/NAME;parent=root/PARENT
    require(output.is_dir() and not any(output.iterdir()),'Output not empty')
    import numpy as np
    import torch,torchvision
    from torchvision import datasets,transforms
    require(torch.cuda.is_available(),'CUDA GPU required')
    device=torch.device('cuda:0');torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    sys.path.insert(0,str(parent/'autodl'))
    from xautodl.models import get_cell_based_tiny_net
    opt=mod(parent/'autodl/xautodl/procedures/optimizers.py','official_optimizer')
    layer=mod(parent/'LayerAnalysis.py','frozen_extractor')
    raw=js(parent/'autodl/configs/nas-benchmark/hyper-opts/200E.config')
    cv={'int':int,'float':float,'str':str,'bool':lambda v:bool(int(v))}
    cfg=SimpleNamespace(**{k:cv[t](v) for k,(t,v) in raw.items()})
    require(cfg.epochs==200 and cfg.batch_size==256 and cfg.LR==.1,'Wrong recipe')
    split=js(b/'cifar-split.txt');train_ids=list(map(int,split['train'][1]));valid_ids=list(map(int,split['valid'][1]))
    require(len(train_ids)==len(valid_ids)==25000 and not set(train_ids)&set(valid_ids),'Image split failure')
    droot=data_path(data)
    mean=[v/255 for v in [125.3,123.,113.9]];std=[v/255 for v in [63.,62.1,66.7]]
    tr=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4),transforms.ToTensor(),transforms.Normalize(mean,std)])
    te=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    training=datasets.CIFAR10(str(droot),train=True,transform=tr,download=False)
    validation=datasets.CIFAR10(str(droot),train=True,transform=te,download=False)
    require(len(training)==len(validation)==50000,'Unexpected CIFAR10 training dataset')
    random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025)
    measure_ids=random.Random(20261025).sample(train_ids,256)
    batch=torch.utils.data.default_collate([training[i] for i in measure_ids])
    val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(validation,valid_ids),batch_size=256,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(777))
    with (b/'benchmark_targets.csv').open(newline='') as f:targets={int(r['architecture_id']):r for r in csv.DictReader(f)}
    members={r['arch_index']:r for r in js(b/'cohort.json')['architectures']}
    put(output/'runtime.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
      gpu=torch.cuda.get_device_name(0),node=os.environ.get('NODE_NAME'),data_root=str(droot),config=vars(cfg),
      train_images=25000,validation_images=25000,original_test_images_evaluated=0,measurement_train_indices=measure_ids,
      input_policy=c['input_policy'],implementation=c['implementation'],extractor_sha256=EXTRACTOR_SHA))
    snapshots=[]
    for index in c['pilot_ids']:
        spec=members[index];target=targets[index]
        random.seed(777);np.random.seed(777);torch.manual_seed(777);torch.cuda.manual_seed_all(777)
        net=get_cell_based_tiny_net(dict(name='infer.tiny',C=16,N=5,arch_str=spec['arch_str'],num_classes=10)).to(device)
        optimizer,scheduler,criterion=opt.get_optim_scheduler(net.parameters(),cfg)
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(training,train_ids),batch_size=256,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(777))
        probe=taylor(net,batch,layer.LayerAnalyzer,guard,device)
        print('PREFLIGHT_PASS',index,'measured/total',probe['layer_count'],probe['total_weight_layers'],flush=True)
        per_epoch=[]
        for epoch in (1,2,3):
            tm=guard.train_epoch(net,loader,optimizer,scheduler,criterion,epoch,device)
            measured=accuracy(net,val_loader,device)
            score=taylor(net,batch,layer.LayerAnalyzer,guard,device)
            row=dict(architecture_id=index,epoch=epoch,local_validation_accuracy=measured,**score,**tm)
            per_epoch.append(row);put(output/f'arch-{index}-epoch-{epoch}.json',row)
            print('EPOCH_COMPLETE',index,epoch,'local_valid',measured,'Taylor',score['taylor'],flush=True)
        rec=dict(id=spec['id'],arch_index=index,arch_str=spec['arch_str'],arch_family=spec['arch_family'],
          task='img-classification',dataset='cifar10-valid',metric='acc',nn=f'nb201-{index:05d}',candidate_seed=777,
          prm=json.dumps(dict(batch=256,lr=.1,momentum=.9,weight_decay=.0005,epoch_max=200)),
          accuracy_epoch_1=float(target['accuracy_epoch_1']),accuracy_epoch_2=float(target['accuracy_epoch_2']),accuracy_epoch_3=float(target['accuracy_epoch_3']),
          measured_early_accuracy={str(r['epoch']):r['local_validation_accuracy'] for r in per_epoch},
          best_accuracy=float(target['best_accuracy']),epochs_to_best=int(target['epochs_to_best']),total_epochs=200,
          layer_stats='epoch\ttaylor_imp\n'+'\n'.join(str(r['epoch'])+'\t'+format(r['taylor'],'.6g') for r in per_epoch),
          weights_source='train',benchmark_seeds=target['benchmark_seeds'],implementation=c['implementation'])
        snapshots.append(rec);put(output/f'arch-{index}-complete.json',rec)
        del net,optimizer,scheduler;torch.cuda.empty_cache()
    with (output/'snapshot.jsonl').open('x') as f:
        for r in snapshots:f.write(json.dumps(r,allow_nan=False)+'\n')
    loaded=[json.loads(l) for l in (output/'snapshot.jsonl').read_text().splitlines()]
    require(loaded==snapshots,'Snapshot readback mismatch')
    put(output/'complete.json',dict(status='complete',architectures=c['pilot_ids'],epochs=[1,2,3],scope='CIFAR10-valid acquisition pilot, not a predictor result',
      sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('CIFAR10_PILOT_COMPLETE',output,flush=True)

def submit(root):
    verify(root);b=root/NAME
    require(not any((b/'results').iterdir()),'Outputs already exist')
    r=subprocess.run(['kubectl','-n','b-t-bose','get','job',NAME,'--ignore-not-found','-o','name'],capture_output=True,text=True)
    require(r.returncode==0 and not r.stdout.strip(),'Existing job or failed query: '+r.stdout+r.stderr)
    for flags in [['--dry-run=server'],[]]:subprocess.run(['kubectl','-n','b-t-bose','create',*flags,'-f',str(b/'job.json')],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','worker','show']);p.add_argument('--root',type=Path,default=Path('/shared/ssd/home/b-t-bose/lmur'));p.add_argument('--data',type=Path,default=Path('/a/data'));p.add_argument('--output',type=Path,default=Path('/a/output'));a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='submit':submit(a.root)
    elif a.action=='show':
        out=a.root/NAME/'results';c=js(out/'complete.json')
        for name,h in c['sha256'].items():require(sha(out/name)==h,'Result hash mismatch')
        print(json.dumps(c,indent=2));print((out/'snapshot.jsonl').read_text())
    else:
        require(a.output.is_dir() and not any(a.output.iterdir()),'Existing output preserved')
        try:worker(a.root,a.data,a.output)
        except Exception:
            if not (a.output/'failure.json').exists():put(a.output/'failure.json',{'traceback':traceback.format_exc()})
            raise

if __name__=='__main__':main()
