"""Lock verified inner selections; launch one seed's six matched outer fits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE = 'lemur-nested-qwen-full-v1'
LOCK = 'lemur-nested-qwen-seeds43-47-selection-v1'
PINS = {
    'control.py': 'c6d9733240e8bd4930abf03a25476102ee085c26ff3d434a0aba9d3bc03908dc',
    'worker.py': 'fe6309053b6b88f46f744515c8238031705751159a093659da789fd6bc745dad',
    'storage-safe-continuation-v1/worker_no_intermediate.py': 'cea1d1a216ef90840bbaeee484a6253aebc93387f7513511c2d64493aa00980a',
}
EXPECTED_WINNERS = {43:['c07','c06','c04'],44:['c01','c01','c01'],45:['c07','c06','c00'],46:['c05','c06','c00'],47:['c07','c03','c04']}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def js(p):
    return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def put(p, value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def load_controller(root):
    base=root/BASE
    for name,digest in PINS.items():
        require(sha(base/name)==digest,'Original source changed: '+name)
    source=(base/'control.py').read_text()
    anchor='args["seed"] == 42'
    require(source.count(anchor)==1,'Unexpected verifier source')
    c={'__name__':'outer_helpers','__file__':str(base/'control.py')}
    exec(compile(source.replace(anchor,'args["seed"] == run["seed"]'),str(base/'control.py'),'exec'),c)
    return c


def select(scores, trials):
    require(len(scores)==24 and len({s['run'] for s in scores})==24,'Incomplete/duplicate inner search')
    folds={}
    for fold in range(3):
        candidates=[r for r in scores if r['fold']==fold]
        require(len(candidates)==8 and {r['trial_id'] for r in candidates}==set(trials),'Incomplete fold trials')
        best=min(candidates,key=lambda r:(r['RMSE'],r['trial_id']))
        folds[str(fold)]={'trial':trials[best['trial_id']],'selection_score':best}
    return {'rule':'Inner classification RMSE; ascending trial_id breaks ties','all_inner_scores':scores,'folds':folds,'outer_results_used':False}


def lock_all(root,c):
    c['verify']()
    existing=root/LOCK/'all_selections.json'
    if existing.exists():
        locked=js(existing)
        require(set(locked['seeds'])=={str(s) for s in range(43,48)} and locked['outer_results_used'] is False,'Invalid selection lock')
        for name,digest in locked['input_sha256'].items():
            p=(root/name).resolve()
            require(p.is_relative_to(root.resolve()) and sha(p)==digest,'Locked selection evidence changed: '+name)
        # Later adapter-only cleanup does not invalidate the preserved selection evidence.
        print('EXISTING_ALL_SEED_SELECTION_LOCK_VERIFIED',sha(existing),flush=True)
        return locked
    original=js(root/BASE/'inner_contract.json')
    trials={t['trial_id']:t for t in js(root/'lemur-nested-qwen-v1/protocol.json')['search_space']}
    environment=js(root/BASE/'environment_lock.json')
    payload={'seeds':{},'input_sha256':{},'outer_results_used':False}
    for seed in range(43,48):
        folder=root/f'lemur-nested-qwen-s{seed}-inner-p12-v1'
        contract=js(folder/'inner_contract.json')
        require(set(contract['runs'])==set(original['runs']),'Incomplete inner configuration set')
        recovery=root/'lemur-nested-qwen-s44-inner-recovery-v1'
        recovered=js(recovery/'inner_contract.json') if seed==44 else None
        scores=[]
        for key,oldrun in sorted(contract['runs'].items()):
            owner=folder; run=oldrun
            if recovered is not None and key in recovered['runs']:
                owner=recovery;run=recovered['runs'][key]
            expected=copy.deepcopy(original['runs'][key]);expected.update(seed=seed,output=run['output'])
            require(run==expected,'Unexpected inner configuration change')
            require(run['output']=='/a/mm/'+owner.name+'/results/'+key,'Wrong inner output scope')
            out=c['host'](run['output']);receipt_path=owner/(key+'-verified.json');receipt=js(receipt_path)
            require(receipt['seed']==seed and receipt['run']==key,'Receipt identity mismatch')
            require(receipt['contract_sha256']==sha(owner/'inner_contract.json'),'Inner contract drift')
            require(receipt['prediction_sha256']==sha(out/'test_predictions.csv'),'Inner prediction drift')
            rows=c['read_predictions'](key,run)
            require(js(out/'nested_runtime.json')['packages']==environment,'Inner runtime differs')
            scores.append({'run':key,'fold':run['fold'],'trial_id':run['trial_id'],**c['score'](rows),'predictions_sha256':sha(out/'test_predictions.csv')})
            for p in [receipt_path,owner/'inner_contract.json',out/'test_predictions.csv',out/'test_complete.json',out/'refit_complete.json',out/'effective_trainer_args.json',out/'nested_runtime.json']:
                payload['input_sha256'][p.relative_to(root).as_posix()]=sha(p)
        choice=select(scores,trials)
        require([choice['folds'][str(f)]['trial']['trial_id'] for f in range(3)]==EXPECTED_WINNERS[seed],'Winners differ from verified preflight; stop for inspection')
        payload['seeds'][str(seed)]=choice
        print('INNER_VERIFIED',seed,EXPECTED_WINNERS[seed],flush=True)
    for name in ['protocol.json','folds.json','groups.json','prompt_templates.json','snapshot.jsonl']:
        p=root/'lemur-nested-qwen-v1'/name;payload['input_sha256'][p.relative_to(root).as_posix()]=sha(p)
    directory=root/LOCK;path=directory/'all_selections.json'
    if path.exists():require(js(path)==payload,'Previously locked selections differ')
    else:
        require(not directory.exists(),'Incomplete lock directory preserved')
        directory.mkdir();put(path,payload)
    print('ALL_FIVE_SEEDS_SELECTIONS_LOCKED',sha(path),flush=True)
    return payload


def stage_controller(c,root,here):
    c.update(HERE=here,ROOT=root,NAME=here.name,POD=Path('/a/mm')/here.name)
    return c


def build_job(c,key,run,seed,bootstrap):
    job=c['job_object'](key,run,'outer_contract.json')
    trainer=job['spec']['template']['spec']['containers'][0]
    for entry in trainer['env']:
        if entry['name']=='RUN_SEED':entry['value']=str(seed)
    p='/a/mm/'+c['NAME']
    trainer['args']=[bootstrap+'\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\n'
        +f'python -B -u {p}/storage-safe-continuation-v1/worker_no_intermediate.py train {p}/outer_contract.json {key}\n'
        +f'python -B -u {p}/storage-safe-continuation-v1/worker_no_intermediate.py test {p}/outer_contract.json {key}\n'
        +f'python -B -u {p}/outer_launcher.py --seed {seed} --verify {key}\n']
    require(job['spec']['backoffLimit']==0,'Unexpected retry policy')
    return job


def verify_outer(seed,key):
    here=Path(__file__).resolve().parent;root=here.parent
    require(here.name==f'lemur-nested-qwen-s{seed}-outer-v1','Wrong outer directory')
    c=stage_controller(load_controller(root),root,here);c['verify']()
    for name,digest in js(here/'outer_manifest.json').items():
        require(sha(here/name)==digest,'Outer input drift')
    contract=js(here/'outer_contract.json');run=contract['runs'][key]
    require(run['seed']==seed and run['phase']=='outer','Wrong seed/phase')
    require(sha(here/'selections.json')==contract['selection_lock_sha256'],'Selection lock changed')
    rows=c['read_predictions'](key,run);out=c['host'](run['output'])
    require(js(out/'nested_runtime.json')['packages']==js(here/'environment_lock.json'),'Environment drift')
    require(js(out/'effective_trainer_args.json')['save_strategy']=='no','Checkpoint policy changed')
    require(not list((out/'tuned_model').glob('checkpoint-*')),'Unexpected intermediate checkpoints')
    score=c['score'](rows)
    put(here/(key+'-verified.json'),{'seed':seed,'run':key,'metrics':score,'prediction_sha256':sha(out/'test_predictions.csv'),'contract_sha256':sha(here/'outer_contract.json'),'selection_lock_sha256':sha(here/'selections.json')})
    print('OUTER_VERIFIED',seed,key,json.dumps(score),flush=True)


def launch(seed):
    root=Path('/shared/ssd/home/b-t-bose/lmur');base=root/BASE
    here=root/f'lemur-nested-qwen-s{seed}-outer-v1'
    require(not here.exists(),'Existing outer directory preserved; do not rerun blindly')
    c=load_controller(root)
    locked=lock_all(root,c)
    # Every seed's choices are locked before any new outer prompts are prepared.
    here.mkdir()
    inner=copy.deepcopy(js(root/f'lemur-nested-qwen-s{seed}-inner-p12-v1/inner_contract.json'))
    put(here/'inner_contract.json',inner)
    put(here/'selections.json',locked['seeds'][str(seed)])
    for name in ['worker.py','settings.json','pod_template.json','environment_lock.json']:
        shutil.copyfile(base/name,here/name)
    (here/'storage-safe-continuation-v1').mkdir()
    wrapper='storage-safe-continuation-v1/worker_no_intermediate.py'
    shutil.copyfile(base/wrapper,here/wrapper)
    shutil.copyfile(Path(__file__).resolve(),here/'outer_launcher.py')
    c=stage_controller(c,root,here)
    c['prepare_outer'](locked['seeds'][str(seed)])
    contract=js(here/'outer_contract.json')
    require(len(contract['runs'])==6,'Expected six matched outer refits')
    for key,run in contract['runs'].items():
        require(run['seed']==seed and run['settings_seed']==42 and run['phase']=='outer','Wrong seed/settings/phase')
        require(run['output']=='/a/mm/'+here.name+'/results/'+key,'Wrong outer output scope')
    files=['worker.py',wrapper,'settings.json','pod_template.json','environment_lock.json','inner_contract.json','selections.json','outer_launcher.py']
    static={name:sha(here/name) for name in files}
    static['../'+LOCK+'/all_selections.json']=sha(root/LOCK/'all_selections.json')
    put(here/'static_manifest.json',static)
    bootstrap=js(here/'settings.json')['bootstrap']
    jobs=[build_job(c,key,run,seed,bootstrap) for key,run in sorted(contract['runs'].items())]
    payload={'apiVersion':'v1','kind':'List','items':jobs};put(here/'jobs.json',payload)
    subprocess.run(['kubectl','-n','b-t-bose','create','--dry-run=server','-f',str(here/'jobs.json')],check=True)
    subprocess.run(['kubectl','-n','b-t-bose','create','-f',str(here/'jobs.json')],check=True)
    print('SUBMITTED',here.name,'six independent outer Jobs; no cleanup; other seeds not launched',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=range(43,48),required=True)
    parser.add_argument('--verify')
    args=parser.parse_args()
    if args.verify:verify_outer(args.seed,args.verify)
    else:launch(args.seed)
