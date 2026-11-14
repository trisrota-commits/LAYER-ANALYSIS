"""Safe, full-precision implementation of the uploaded seed-intervals patch."""
import csv
import json
from pathlib import Path
from analysis import revision_descriptive as source
from analysis.thesis_common import require, sha

TABLES=('S2','S3','S10','S11','S12','S13')
LABELS={'no-layer':'No layer','all-seven':'All seven','weightwatcher':'WeightWatcher','cka':'CKA',
        'effective-rank':'Effective rank','taylor':'Taylor importance','gradient-flow':'Gradient flow',
        'sensitivity':'Sensitivity','activation':'Dead-act. fraction'}


def run(repo,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    source.run(repo,out/'verified_descriptive')
    rows=[]
    for table in TABLES:
        long=json.loads((out/'verified_descriptive'/f'table_{table}.json').read_text())
        groups={}
        for r in long:
            group=r['group']
            if group not in groups:
                if table in ('S2','S3'):
                    arm='no layer' if r['feature']=='no-layer' else ('layer' if table=='S2' else 'all seven')
                    label=f"{r['observations']} accuracy + {arm}"
                else:label=LABELS[r['feature']]
                groups[group]=dict(table=table,row=label,n=len(r['seeds']),seeds=r['seeds'],sources=r['sources'])
            groups[group].update({r['metric']:r['mean'],r['metric']+'_lo':r['low'],r['metric']+'_hi':r['high']})
        rows.extend(groups.values())
    require(len(rows)==44,'Incomplete seed-interval patch coverage')
    with (out/'seed_intervals.csv').open('x',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    with (out/'seed_intervals.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2,allow_nan=False)
    text=['# Seed-level 95% t-intervals','',
          'Patch functionality integrated using verified predictions and full precision, not rounded metric logs.',
          'Pilot intervals describe three nominal-seed repetitions; other intervals use seeds 42-47.','']
    for table in TABLES:
        text += ['## '+table,'','| Row | '+' | '.join(source.lemur.METRICS)+' |','|---|'+'---|'*5]
        for r in rows:
            if r['table']==table:text.append('| '+r['row']+' | '+' | '.join(f"{r[m]:.6f} [{r[m+'_lo']:.6f}, {r[m+'_hi']:.6f}]" for m in source.lemur.METRICS)+' |')
    (out/'seed_intervals.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',rows=len(rows),tables=TABLES,files_sha256=hashes),indent=2)+'\n',encoding='utf-8')
    print('COMPLETE: 44 seed-interval rows;',out,flush=True)
    return rows
