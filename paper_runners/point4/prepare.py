"""Prepare only new package outputs and adopt verified metadata; never modify old runs."""
import hashlib
import json
import shutil
from pathlib import Path

P = Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def put(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,allow_nan=False)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def reuse_plan(old, target_bundle, item):
    source=old/'results'/f"candidate-{item['reuse']['index']}"
    old_contract=read(old/'contract.json')
    old_item=old_contract['selected'][item['reuse']['index']]
    require(old_item['record_id']==item['record_id'], 'Reuse record ID mismatch')
    require(old_item['group']==item['group'],'Reuse group mismatch')
    for field in ('model_file','transform_file'):
        require(sha(old/old_item[field])==sha(target_bundle/item[field]),'Reuse source mismatch: '+field)
    require(read(old/old_item['record_file'])==read(target_bundle/item['record_file']), 'Reuse record differs')
    done=read(source/'complete.json')
    require(done['status']=='complete' and done['record_id']==item['record_id'], 'Reuse not complete')
    required=['runtime.json']+[f'{stem}{e}{ext}' for e in (1,2,3) for stem,ext in (
        ('epoch-','.json'),('measurements-e','.jsonl'),('state-check-e','.json'),('scaling-checks-e','.json'))]
    for name in required:
        require(name in done['files_sha256'],'Receipt missing '+name)
        require(sha(source/name)==done['files_sha256'][name], 'Reuse checksum mismatch '+str(source/name))
    for e in (1,2,3):
        state=read(source/f'state-check-e{e}.json')
        require(all(state.get(k) is True for k in ('original_state_preserved','original_modes_preserved','original_gradients_preserved','rng_restored')), 'Reuse preservation failure')
        rows=[json.loads(x) for x in (source/f'measurements-e{e}.jsonl').read_text().splitlines()]
        require(len(rows)==state['measurements'], 'Reuse row count mismatch')
        require(sum(r['status']!='ok' for r in rows)==state['failed_measurements'], 'Reuse failure count mismatch')
        require(all(r['checkpoint_state_sha256']==state['checkpoint_state_sha256'] and r['epoch']==e for r in rows), 'Reuse checkpoint identity mismatch')
    return source,required,done

def main():
    parent=P.parent
    manifest=read(P/'manifest.json')
    require(not (P/'prepared.json').exists(),'Already prepared: use existing Jobs; do not overwrite')
    for name,expected in read(P/'package_hashes.json').items():
        require(sha(P/name)==expected,'Package file changed: '+name)
    for name,expected in manifest['source_hashes'].items():
        require(sha(parent/'lemur-nested-qwen-v1'/name)==expected,'Frozen nested input changed: '+name)
    old=parent/'lemur-taylor-robustness-full-v1'
    require(sha(old/'contract.json')==manifest['old_contract_sha256'],'Original contract changed')
    plans=[]
    for bundle in manifest['bundles']:
        d=P/bundle['name']; c=read(d/'contract.json')
        require(sha(parent/'nn-dataset/ab/nn/util/LayerAnalysis.py')==c['extractor_sha256'],'Extractor changed')
        require(not (d/'results').exists(),'Output exists; preserving it: '+str(d/'results'))
        for item in c['selected']:
            if item.get('reuse'):
                source,names,done=reuse_plan(old,d,item)
                plans.append((d,item,source,names,done))
    needed=sum((s/n).stat().st_size for _,_,s,names,_ in plans for n in names)
    require(shutil.disk_usage(P).free > needed+256*1024**2,'Insufficient space for metadata plus 256 MiB reserve')
    link=P/'nn-dataset'
    require(not link.exists() and not link.is_symlink(),'Dependency link already exists')
    link.symlink_to('../nn-dataset',target_is_directory=True)
    for bundle in manifest['bundles']: (P/bundle['name']/'results').mkdir()
    for d,item,source,names,done in plans:
        dest=d/'results'/f"candidate-{item['index']}"; dest.mkdir()
        for name in names: shutil.copyfile(source/name,dest/name)
        put(dest/'reuse_provenance.json',{
            'source':str(source),'source_complete_sha256':sha(source/'complete.json'),
            'record_id':item['record_id'],'source_candidate':item['reuse']['index'],
            'scope':'Verified measurement/provenance files copied only; original checkpoints preserved in original run; no checkpoint files copied.',
        })
        put(dest/'complete.json',{'status':'complete','record_id':item['record_id'],'epochs':[1,2,3],
            'files_sha256':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
    counts={b['dataset']:b['count'] for b in manifest['bundles']}
    put(P/'prepared.json',{'cohort':counts,'reused_candidates':len(plans),'persistent_new_checkpoints':False})
    print('PREPARED',json.dumps(counts),'REUSED',len(plans),'NO_JOBS_SUBMITTED',flush=True)

if __name__=='__main__': main()
