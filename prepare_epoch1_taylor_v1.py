"""Prepare/submit isolated epoch-1 Taylor Qwen refits; never modify reference artifacts."""
import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

NAME = 'taylor-epoch1-3acc-off-v1'
NS = 'b-t-bose'
SEEDS = tuple(range(42, 48))
PINS = {
 'zcp-off-toggle-refit7-v1/contract.json': 'c84e26fe94c4f51a17518adb3c98dbad33426267179e60219d6bc2dcf5d95fed',
 'zcp-off-toggle-refit7-v1/jobs/3acc-taylor-s42.json': 'bc43fdcf13cb9caddb5afc21b3836fb83e8f80b93145b4da160a62648c056e4f',
 'taylor-3acc-zcp-off-seeds45-47-v1/contract.json': '2891f36e8e38020510e0edaa425f0edc530c34845ba8383f20fa82c60a7bd975',
 'taylor-3acc-zcp-off-seeds45-47-v1/runner.py': 'ad4c81abc45e413e1b87d12a806f77004654444262020f432e6655962f4479d6',
 'taylor-3acc-zcp-off-seeds45-47-v1/jobs.json': '7712162fc9a61508669bf8062a31d8ec371ee1a4e3b1481527831ffe4c5805cc',
}

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_json(p):
    return json.loads(p.read_text())

def read_rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def host(root, p):
    return root / Path(p).relative_to('/a/mm')

def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='') as f:
        f.write(data if isinstance(data, str) else json.dumps(data, indent=2, allow_nan=False))

def replace_once(text, old, new):
    check(text.count(old)==1, 'Reference runner anchor changed: '+old[:80])
    return text.replace(old,new,1)

TABLE = re.compile(r'(?m)^epoch\ttaylor_imp\r?\n(?P<rows>(?:[123]\t[^\r\n]+(?:\r?\n|$)){3})')

def project(record):
    result=copy.deepcopy(record)
    users=[i for i,m in enumerate(record['messages']) if m['role']=='user']
    check(len(users)==1, 'Expected one user message')
    i=users[0]; text=record['messages'][i]['content']
    matches=list(TABLE.finditer(text))
    check(len(matches)==1, 'Expected exactly one three-epoch Taylor table')
    match=matches[0]
    lines=match.group('rows').splitlines(keepends=True)
    check([x.split('\t')[0] for x in lines]==['1','2','3'], 'Taylor epochs out of order')
    for e in (1,2,3):
        check(len(re.findall(rf'(?m)^epoch_{e}_accuracy:',text))==1, 'Missing/duplicate early accuracy')
    check(not re.search(r'(?m)^ZERO_COST_PROXIES',text), 'Unexpected proxy input')
    new=text[:match.start('rows')]+lines[0]+text[match.end('rows'):]
    # Exact restoration is stronger than count/target checks alone.
    boundary=match.start('rows')+len(lines[0])
    check(new[:boundary]+''.join(lines[1:])+new[boundary:]==text, 'Non-Taylor prompt drift')
    result['messages'][i]['content']=new
    restored=copy.deepcopy(result); restored['messages'][i]['content']=text
    check(restored==record, 'Non-user record drift')
    return result

EXPORT = '''

def export_predictions(out, root, key, seed):
    import csv
    import math
    registry = json.loads((root.parent / 'test_registry.json').read_text())
    with (out / 'test_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(registry) == 115
    exported = []
    for i, (row, item) in enumerate(zip(rows, registry)):
        assert int(row['idx']) == i
        assert row['dataset'] == item['dataset']
        assert float(row['gt_best_accuracy']) == item['target']
        value = float(row['pred_best_accuracy'])
        assert math.isfinite(value), ('Invalid prediction', key, i)
        exported.append(dict(record_id=item['record_id'], prediction=value,
                             target=item['target'], seed=seed))
    path = out / 'per_record_predictions.csv'
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['record_id','prediction','target','seed'])
        writer.writeheader()
        writer.writerows(exported)
    with path.open(newline='') as f:
        saved = list(csv.DictReader(f))
    assert [r['record_id'] for r in saved] == [r['record_id'] for r in registry]
    for original, saved_row in zip(exported, saved):
        assert original['prediction'] == float(saved_row['prediction'])
        assert original['target'] == float(saved_row['target'])
        assert original['seed'] == int(saved_row['seed'])
'''

AUDIT = '''    # Audit actual training-form text and effective cap before expensive fitting.
    import importlib.metadata
    cap = min(int(v) for v in saved_lengths.values())
    lengths = []
    for sample in data:
        encoded = tokenizer(sample['text'], truncation=False, add_special_tokens=True)
        lengths.append(len(encoded['input_ids']))
    audit = {
        'run': key, 'records': len(lengths), 'effective_cap': cap,
        'max_training_tokens': max(lengths),
        'over_cap_records': sum(n > cap for n in lengths),
        'gpu': t.torch.cuda.get_device_name(0),
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch','transformers','trl','unsloth','unsloth-zoo','datasets','peft')},
    }
    (out / 'runtime_preflight.json').write_text(json.dumps(audit, indent=2))
    assert audit['over_cap_records'] == 0, 'Training prompts exceed historical cap; stop before fitting'
    print('EPOCH1_RUNTIME_PREFLIGHT_PASS', json.dumps(audit), flush=True)
'''

def prepare(root):
    bundle=root/NAME
    check(not bundle.exists(), f'Refusing existing bundle: {bundle}')
    dependencies={}
    for rel,digest in PINS.items():
        p=root/rel
        check(p.is_file() and sha(p)==digest, 'Reference drift/missing: '+str(p))
        dependencies[rel]=digest
    old=read_json(root/'zcp-off-toggle-refit7-v1/contract.json')
    extended=read_json(root/'taylor-3acc-zcp-off-seeds45-47-v1/contract.json')
    check(old['base']==extended['base'], 'Different base models')
    for c in (old,extended):
        for path,digest in c['source_sha256'].items():
            p=host(root,path)
            check(p.is_file() and sha(p)==digest, 'Historical source drift: '+str(p))
            dependencies[str(p.relative_to(root))]=digest
    reference=old['runs']['3acc-taylor-s42']
    source=host(root,reference['data'])
    projected={}; counts={}; original={}
    for split,count in [('train',978),('val',58),('test',115)]:
        name=f'{split}_llm_dataset.jsonl'; p=source/name
        check(sha(p)==reference['input_sha256'][name], 'Prompt hash mismatch: '+name)
        dependencies[str(p.relative_to(root))]=sha(p)
        rows=read_rows(p); check(len(rows)==count, 'Split count mismatch')
        original[split]=rows
        projected[name]=[project(r) for r in rows]
        counts[split]=len(rows)
    # Reuse the independently verified raw-ID registry from the classical run.
    completed=root/'classical-eight-features-v1'
    marker=read_json(completed/'complete.json')
    check(marker['status']=='complete','Classical registry not completed')
    p=completed/'records.csv'
    check(sha(p)==marker['files_sha256']['records.csv'],'Registry hash mismatch')
    dependencies[str(p.relative_to(root))]=sha(p)
    with p.open(newline='') as f:
        rr=[r for r in csv.DictReader(f) if r['split']=='test']
    check(len(rr)==115 and len({r['record_id'] for r in rr})==115,'Invalid test IDs')
    registry=[]
    for i,(r,chat) in enumerate(zip(rr,original['test'])):
        check(int(r['position'])==i and r['architecture_id']==chat['architecture_id'],'Registry identity/order mismatch')
        answer=json.loads(next(m['content'] for m in chat['messages'] if m['role']=='assistant'))
        user=next(m['content'] for m in chat['messages'] if m['role']=='user')
        dataset=re.findall(r'(?m)^dataset:[ \t]*(.+)$',user)
        check(len(dataset)==1 and dataset[0].strip()==r['dataset'],'Registry dataset mismatch')
        check(float(r['target'])==float(answer['best_accuracy']),'Registry target mismatch')
        registry.append(dict(record_id=r['record_id'],dataset=r['dataset'],target=float(r['target']),raw_target=float(r['raw_target'])))
    contract=copy.deepcopy(extended)
    contract['runs']={}
    texts={name:''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows) for name,rows in projected.items()}
    digests={name:hashlib.sha256(text.encode()).hexdigest() for name,text in texts.items()}
    for seed in SEEDS:
        parent=old if seed<=44 else extended
        run=copy.deepcopy(parent['runs'][f'3acc-taylor-s{seed}'])
        check(run['input_sha256']==reference['input_sha256'],'Reference inputs differ across seeds')
        check(run['zero_cost_proxies'] is False,'Reference proxy mismatch')
        for name,digest in run['selection_sha256'].items():
            p=host(root,run['selected'])/name
            check(p.is_file() and sha(p)==digest,'Saved settings drift: '+str(p))
            dependencies[str(p.relative_to(root))]=digest
        run.update(data=f'/a/mm/{NAME}/prompts', output=f'/a/mm/{NAME}/results/s{seed}',
                   settings_seed=run.get('settings_seed',seed),zcp_enabled=False,
                   input_sha256=digests,layer_epochs=[1],reference_run=f'3acc-taylor-s{seed}')
        contract['runs'][f'3acc-taylor-epoch1-s{seed}']=run
    runner=(root/'taylor-3acc-zcp-off-seeds45-47-v1/runner.py').read_text()
    runner=replace_once(runner,"    trainer.train()  # Fresh run: no resume checkpoint or old optimizer/scheduler.",
                         AUDIT+"    trainer.train()  # Fresh run: no resume checkpoint or old optimizer/scheduler.")
    runner=replace_once(runner,"        # Coverage is reported honestly, not used to retune or rerun a model.",
                         "        export_predictions(out, root, key, run['seed'])\n        # Coverage is reported honestly, not used to retune or rerun a model.")
    runner=replace_once(runner,"if __name__ == '__main__':",EXPORT+"\nif __name__ == '__main__':")
    compile(runner,'runner.py','exec')
    contract['source_sha256'][f'/a/mm/{NAME}/runner.py']=hashlib.sha256(runner.encode()).hexdigest()
    template=read_json(root/'zcp-off-toggle-refit7-v1/jobs/3acc-taylor-s42.json')
    jobs=[]
    for seed in SEEDS:
        job=copy.deepcopy(template)
        labels={'experiment-batch':NAME,'run-seed':str(seed),'layer-horizon':'1','accuracy-horizon':'3'}
        job['metadata']={'name':f'{NAME}-s{seed}','namespace':NS,'labels':labels}
        job['spec']['template']['metadata']={'labels':labels.copy()}
        pod=job['spec']['template']['spec']; c=pod['containers'][0]
        check(c['workingDir']=='/a/mm','Unexpected reference working directory')
        check(any(v.get('hostPath',{}).get('path')==str(root) for v in pod['volumes']), 'Unexpected root mount')
        env={item['name']:item['value'] for item in c['env']}
        env.update(DATA_ROOT=f'/a/mm/{NAME}/prompts',RUN_SEED=str(seed),OUT_DIR=f'/a/mm/{NAME}/results/s{seed}',
                   XDG_CACHE_HOME=f'/tmp/cache-{NAME}-s{seed}',TORCHINDUCTOR_CACHE_DIR=f'/tmp/torchinductor-{NAME}-s{seed}')
        c['env']=[{'name':k,'value':v} for k,v in env.items()]
        shell=c['args'][0]
        check(shell.count('export PYTHONPATH=')==1,'Unexpected reference shell')
        setup=shell.split('export PYTHONPATH=')[0]
        c['args']=[setup+f'''export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset
cd /a/mm
python -u /a/mm/{NAME}/launcher.py verify --root /a/mm
python -u /a/mm/{NAME}/runner.py train /a/mm/{NAME}/contract.json 3acc-taylor-epoch1-s{seed}
python -u /a/mm/{NAME}/runner.py test /a/mm/{NAME}/contract.json 3acc-taylor-epoch1-s{seed}
''']
        jobs.append(job)
    # Every dependency and transform passes before creating the new directory.
    bundle.mkdir(exist_ok=False)
    put(bundle/'runner.py',runner)
    put(bundle/'launcher.py',Path(__file__).read_text())
    put(bundle/'contract.json',contract)
    put(bundle/'test_registry.json',registry)
    for name,text in texts.items():
        put(bundle/'prompts'/name,text)
        check(read_rows(bundle/'prompts'/name)==projected[name], 'Written prompt mismatch')
    for job in jobs: put(bundle/'jobs'/(job['metadata']['name']+'.json'),job)
    files={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
    put(bundle/'prepared.json',dict(status='prepared',counts=counts,seeds=SEEDS,epoch_change='Taylor rows 2 and 3 removed ONLY',
                                  files_sha256=files,dependencies_sha256=dependencies,
                                  runtime='Inherited image tag and reference package pins; transitive dependencies and actual GPU runtime require recorded preflight',
                                  storage='User accepted low-space risk; no old artifacts deleted; checkpoint policy unchanged'))
    verify(root)
    print('PREPARED:',bundle,'; six Jobs generated, NONE submitted.',flush=True)

def verify(root):
    bundle=root/NAME; receipt=read_json(bundle/'prepared.json')
    for rel,digest in receipt['files_sha256'].items():
        check(sha(bundle/rel)==digest,'Prepared artifact drift: '+rel)
    for rel,digest in receipt['dependencies_sha256'].items():
        check(sha(root/rel)==digest,'Reference dependency drift: '+rel)
    print('VERIFIED: prepared inputs, runner, Jobs, and historical dependencies',flush=True)

def submit(root):
    verify(root)
    bundle=root/NAME
    paths=[bundle/'jobs'/f'{NAME}-s{s}.json' for s in SEEDS]
    for seed in SEEDS:
        check(not (bundle/'results'/f's{seed}').exists(),f'Output exists for seed {seed}; inspect, do not retrain')
        result=subprocess.run(['kubectl','-n',NS,'get','job',f'{NAME}-s{seed}','--ignore-not-found','-o','name'],capture_output=True,text=True)
        check(result.returncode==0, result.stderr)
        check(not result.stdout.strip(),f'Job exists for seed {seed}; inspect, do not duplicate')
    for p in paths:
        subprocess.run(['kubectl','-n',NS,'create','--dry-run=server','-f',str(p)],check=True)
    results=[]
    for p in paths:
        result=subprocess.run(['kubectl','-n',NS,'create','-f',str(p)],capture_output=True,text=True)
        results.append(dict(job=p.stem,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        print(result.stdout or result.stderr,flush=True)
    import datetime
    receipt=bundle/('submission-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
    put(receipt,results)
    check(all(r['returncode']==0 for r in results),'Some submissions failed; inspect receipt, do not rerun blindly')
    print('SUBMITTED six Jobs. Scheduling/training/completion remain unverified.',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','verify','submit'])
    p.add_argument('--root',type=Path,default=Path('/shared/ssd/home/b-t-bose/lmur'))
    a=p.parse_args()
    {'prepare':prepare,'verify':verify,'submit':submit}[a.action](a.root)

if __name__=='__main__':
    main()
