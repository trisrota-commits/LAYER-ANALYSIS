"""One entry point exporting every numerical table in the audited paper/supplement.

S1 and S46 are contextual synthesis/literature tables, explicitly not calculations.
No network training, predictor fitting, acquisition or calibration refitting.
"""
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
from collections import defaultdict
from analysis.thesis_common import require, sha


def native(value):
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native(v) for v in value]
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def inventory():
    groups = {
        'context': ['S1','S46'], 'core': ['main_I','S8','S9'],
        'descriptive': ['main_IV','S2','S3','S4','S10','S11','S12','S13','S14','S15','S16','S17','S18','S20','S22'],
        'classical-saved': ['main_VII','S6','S7'],
        'nb201-tables': ['main_III','S30','S31','S33'], 'lemur-paired': ['S5'],
        'article': ['main_II','main_V','main_VI','S24','S26','S27','S28','S29','S34','S35'],
        'paired-tables': ['S19','S21','S23','S36','S37','S38','S39','S40'],
        'initialization': ['S41','S42','S43','S44','S45'], 'collapse':['S25'],
        'loss-confidence-per-arm':['S48'], 'point4-tables':['S47','S53','S54','S55','S56','S57','S58'],
        'loss-confidence-bonferroni40':['S49','S50','S51','S52'], 'nb201-qwen':['S32'],
        'nested-outer':['S59','S60','S61'],
    }
    result={table:mode for mode,tables in groups.items() for table in tables}
    expected={f'S{i}' for i in range(1,62)}|{f'main_{i}' for i in ('I','II','III','IV','V','VI','VII')}
    require(set(result)==expected and sum(map(len,groups.values()))==len(result),'Table inventory mismatch')
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


EXPECTED_ROWS = {
    'main_I':8,'main_II':5,'main_III':3,'main_IV':45,'main_V':5,'main_VI':6,'main_VII':20,
    'S2':20,'S3':20,'S4':20,'S5':20,'S6':15,'S7':15,'S8':7,'S9':7,
    'S10':45,'S11':45,'S12':45,'S13':45,'S14':10,'S15':15,'S16':13,
    'S17':25,'S18':10,'S19':10,'S20':20,'S21':35,'S22':30,'S23':20,
    'S24':10,'S25':5,'S26':5,'S27':19,'S28':8,'S29':7,'S30':6,'S31':21,
    'S32':12,'S33':4,'S34':90,'S35':24,'S36':70,'S37':35,'S38':50,'S39':25,
    'S40':5,'S41':15,'S42':15,'S43':10,'S44':25,'S45':20,'S47':3,'S48':60,
    'S49':15,'S50':15,'S51':24,'S52':30,'S53':3,'S54':72,'S55':9,'S56':9,
    'S57':6,'S58':3,'S59':18,'S60':18,'S61':5,
}


def validate_table(name, rows):
    require(len(rows)==EXPECTED_ROWS[name], f'Incomplete {name}: {len(rows)} rows, expected {EXPECTED_ROWS[name]}')
    required = {'S6':{'model','metric','without_taylor','with_taylor','change'},
                'S28':{'feature','tau','rho','P@10','regret@1','regret@5','regret@10'},
                'S35':{'feature','zcp','tau','rho','P@5','P@10','regret@1','regret@10'},
                'S61':{'metric','no_layer','taylor','selected_procedure','improvement',
                       'ci95_low','ci95_high','bonferroni5_low','bonferroni5_high'},
                'main_VI':{'population','condition','n','RMSE','Pearson','Spearman','R2'}}
    for row in rows:
        require(isinstance(row,dict) and required.get(name,set()).issubset(row),'Missing fields in '+name)
    if name in ('S2','S3','S4','S10','S11','S12','S13','S14','S17','S22'):
        require(all({'mean','low','high','metric'}.issubset(r) and r['low'] is not None and r['high'] is not None for r in rows),
                'Missing intervals in '+name)


def _nested_s61(nested):
    """Join descriptive selected-procedure means without changing paired inference."""
    metrics = {'RMSE', 'MAE', 'Pearson', 'Spearman', 'R2'}
    selected = [r for r in nested['mean_metrics'] if r.get('method') == 'selected-procedure']
    require(len(selected) == 1, 'S61 requires exactly one selected-procedure mean row')
    selected = selected[0]
    require(metrics.issubset(selected) and all(np.isfinite(selected[m]) for m in metrics),
            'S61 selected-procedure metrics missing or nonfinite')
    paired = nested['paired']
    require(len(paired) == len(metrics) and {r['metric'] for r in paired} == metrics,
            'S61 requires each of the five paired metrics exactly once')
    return [dict(r, selected_procedure=selected[r['metric']]) for r in paired]


def _article_tables(article):
    primary=article['primary_endpoint']; support=primary['supporting']; means=primary['means']
    main=[]
    for label,mean_key,rec,seed,n in [
        ('MAE target>=20','mae_non_collapsed',primary['primary_by_record'],primary['primary_by_seed'],primary['n_non_collapsed']),
        ('MAE all','mae_all',support['mae_all_by_record'],support['mae_all_by_seed'],primary['n_records']),
        ('RMSE target>=20','rmse_non_collapsed',None,support['rmse_non_collapsed_by_seed'],primary['n_non_collapsed']),
        ('RMSE all','rmse_all',None,support['rmse_all_by_seed'],primary['n_records']),
        ('MAE target<20','mae_collapsed',support['mae_collapsed_by_record'],None,primary['n_collapsed'])]:
        v=means[mean_key]
        main.append(dict(comparison=label,n=n,no_layer=v['no_layer'],taylor=v['taylor'],
                         improvement=v['no_layer']-v['taylor'],record_p=rec['p'] if rec else None,
                         seed_p=seed['p'] if seed else None))
    robust=primary['robustness_non_collapsed']; rec=primary['primary_by_record']; seed=primary['primary_by_seed']
    s24=[dict(test='record-paired t',unit='records',estimate=rec['gain'],p=rec['p']),
         dict(test='seed-paired t',unit='seeds',estimate=seed['gain'],p=seed['p']),
         dict(test='crossed mixed model',unit='records/seeds',estimate=robust['crossed_mixed_model'].get('gain'),**robust['crossed_mixed_model']),
         dict(test='two-way cluster bootstrap',unit='records/seeds',estimate=robust['two_way_cluster_bootstrap']['gain'],**robust['two_way_cluster_bootstrap']),
         dict(test='record bootstrap',unit='records',estimate=rec['gain'],p=rec['p_bootstrap'],ci=rec['boot_ci']),
         dict(test='Wilcoxon signed-rank',unit='record-averaged error differences',p=rec['p_wilcoxon']),
         dict(test='sign-flip permutation',unit='cells',estimate=rec['gain'],**robust['sign_flip_permutation'])]
    s24.extend(dict(test=label,unit='pooled error cells',**robust[key]) for label,key in
               [('mean absolute error','location_mean'),('median absolute error','location_median'),('20% trimmed mean absolute error','location_trimmed_20pct')])
    families=defaultdict(list)
    for row in article['multiplicity']:families[row['family']].append(row)
    counts=[dict(family=f,tests=len(rows),raw_significant=sum(r['p']<.05 for r in rows),
                 holm_significant=sum(r['p_holm']<.05 for r in rows),bh_significant=sum(r['q_bh']<.05 for r in rows),
                 smallest_p=min(r['p'] for r in rows)) for f,rows in families.items()]
    total=dict(family='Total',**{k:sum(r[k] for r in counts) for k in ('tests','raw_significant','holm_significant','bh_significant')},
               smallest_p=min(r['smallest_p'] for r in counts))
    counts.append(total)
    zcp=defaultdict(list)
    for row in article['zcp_sign_test']['rows']:zcp[tuple(row[:2])].append(row)
    s27=[]
    for (obs,feature),rows in zcp.items():
        rmse=next(r for r in rows if r[2]=='RMSE')
        s27.append(dict(observations=obs,feature=feature,off_better=sum(r[3] for r in rows),metrics=len(rows),
                        delta_RMSE=rmse[4]-rmse[5],seeds=rmse[6]))
    s27.append(dict(observations=None,feature='Total',off_better=sum(r['off_better'] for r in s27),
                    metrics=sum(r['metrics'] for r in s27),delta_RMSE=None,seeds=None))
    s29=[dict(configuration=k,single_RMSE=v['single']['RMSE'],ensemble_RMSE=v['ens']['RMSE'],gain=v['gain'],p=v['p'])
         for k,v in article['ensemble'].items()]
    return {'main_II':main,'main_V':counts,'main_VI':article['main_table_VI'],'S24':s24,'S26':counts,'S27':s27,
            'S28':article['selection']['S28'],'S29':s29,'S34':article['multiplicity'],'S35':article['selection']['S35']}


def run(repo,out):
    repo,out=Path(repo).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    require(not any(out.is_relative_to(repo/p) for p in ('evidence','analysis','tests','paper_runners')),'Output overlaps source evidence')
    plan=inventory()
    frozen=_read(repo/'INPUT_MANIFEST.json')['files']
    before={n:sha(repo/n) for n in frozen}
    require(all(before[n]==h for n,h in frozen.items()),'Frozen input manifest mismatch')
    extras={'evidence/thesis-new-logs-20261112-084434.zip':'2b233d441517d5fca8c4d6d564abce824706c8d2828aae0dbf48d867c895e249',
            'evidence/paper-missing-evidence-v1.tar.gz':'1b354d5d45e65e287052f14e141169e1f23e11427d0be0f128b028344a105884'}
    require(all(sha(repo/n)==h for n,h in extras.items()),'Recovered evidence checksum mismatch')
    from analysis import (revision_descriptive,revision_paired_tables,revision_classical_saved,revision_nb201_tables,
                          revision_article,revision_collapse,revision_initialization,revision_point4_tables,
                          revision_loss_confidence_per_arm,revision_loss_confidence,revision_lemur,revision_nb201,
                          revision_nested,thesis_metadata,thesis_measurements)
    out.mkdir(parents=True,exist_ok=False)
    work=out/'workflows';work.mkdir()

    def execute(name, fn, *args):
        print('PAPER WORKFLOW:',name,flush=True)
        return fn(*args,work/name)
    # Explicit calls preserve existing public API argument order.
    execute('descriptive',revision_descriptive.run,repo)
    paired=execute('paired-tables',revision_paired_tables.run,repo)
    classical=execute('classical-saved',revision_classical_saved.run,repo)
    nb=execute('nb201-tables',revision_nb201_tables.run,repo)
    article=execute('article',revision_article.run,repo)
    execute('collapse',revision_collapse.run,repo)
    execute('initialization',revision_initialization.run,repo)
    point4=execute('point4-tables',revision_point4_tables.run,repo/'evidence/paper-missing-evidence-v1.tar.gz')
    execute('loss-confidence-per-arm',revision_loss_confidence_per_arm.run,repo/'evidence/loss-confidence-all-metrics-inputs-v1.tar.gz')
    execute('loss-confidence-bonferroni40',revision_loss_confidence.run,repo/'evidence/loss-confidence-all-metrics-inputs-v1.tar.gz')
    print('PAPER WORKFLOW: lemur-paired',flush=True)
    revision_lemur.run(repo,work/'lemur-paired',repo/'evidence/two-experiments-cpu-inputs-v1.zip')
    execute('nb201-qwen',revision_nb201.run,repo)
    print('PAPER WORKFLOW: nested-outer',flush=True)
    revision_nested.run(repo,repo/'evidence/nested-outer-six-seeds-v1.tar.gz',work/'nested-outer')
    return export_tables(repo,work,out,before,extras)


def export_tables(repo,work,out,before,extras):
    """Export only completed, hash-verified calculator results; never overwrite."""
    from analysis import thesis_metadata,thesis_measurements
    repo,work,out=Path(repo),Path(work),Path(out)
    require(work.parent.resolve()==out.resolve() and work.name=='workflows','Unexpected workflow directory')
    table_dir=out/'tables'
    if table_dir.exists() or (out/'complete.json').exists():
        raise FileExistsError('Existing table export preserved: '+str(out))
    plan=inventory()
    expected=set(plan.values())-{'context','core'}
    require({p.name for p in work.iterdir() if p.is_dir()}==expected,'Incomplete workflow inventory')
    for name in expected:
        folder=work/name
        path=folder/('receipt.json' if name in ('article','descriptive') else 'complete.json')
        receipt=_read(path)
        if name=='descriptive':
            require(receipt.get('input_preservation_verified') is True,'Unverified descriptive inputs')
            hashes=receipt['output_sha256']
        else:
            require(receipt.get('status')=='complete','Incomplete workflow: '+name)
            hashes=receipt['files_sha256']
        require(bool(hashes),'Empty workflow receipt: '+name)
        for relative,digest in hashes.items():
            payload=(folder/relative).resolve()
            require(payload.is_relative_to(folder.resolve()) and payload.is_file(),'Unsafe/missing workflow payload')
            require(sha(payload)==digest,'Changed workflow payload: '+name+'/'+relative)
    table_dir.mkdir()
    # Consume the serialized producer contracts, not raw NumPy-bearing returns.
    # In particular, article diagnostics deliberately serialize invalid standard
    # errors as null while explicitly withholding invalid mixed-model inference.
    paired=_read(work/'paired-tables/results.json')
    classical=_read(work/'classical-saved/results.json')
    nb=_read(work/'nb201-tables/results.json')
    article=_read(work/'article/results.json')
    point4=_read(work/'point4-tables/full_results.json')
    tables={}
    for p in (work/'descriptive').glob('table_*.json'):
        name=p.stem.removeprefix('table_');name={'MainIV':'main_IV','MainVII':'main_VII'}.get(name,name)
        tables[name]=_read(p)
    for name in plan:
        if plan[name]=='paired-tables':tables[name]=paired['tables'][name]
    tables.update({'S6':classical['s6'],'S7':classical['s7'],'main_VII':classical['main_vii']})
    tables.update(nb['tables']);tables.update(_article_tables(article))
    tables['main_I']=thesis_metadata.build_tables(repo)['4.1']
    measured,_=thesis_measurements.build_tables(repo)
    for name,stream in [('S8','early_to_epoch50'),('S9','same_epoch')]:
        rows=sorted([r for r in measured['6.1'] if r['stream']==stream],key=lambda r:-r['MIC'])
        tables[name]=[dict(rank=1+sum(x['MIC']>r['MIC'] for x in rows),**r) for r in rows]
    tables.update(point4['tables'])
    cohorts=defaultdict(dict)
    for row in tables['S47']:
        cohorts[row['source'],row['dataset']][row['split']]=row['n']
    tables['S47']=[dict(source=source,dataset=dataset,total=sum(counts.values()),train=counts['train'],
                       validation=counts['val'],test=counts['test'],metadata_only=True)
                   for (source,dataset),counts in cohorts.items()]
    # Include S58's classification counts as well as the all-task split counts.
    with tarfile.open(repo/'evidence/paper-missing-evidence-v1.tar.gz','r:gz') as archive:
        summary=json.load(archive.extractfile('lemur-nested-qwen-v1/fold_summary.json'))
    for row,s in zip(tables['S58'],summary):
        require(row['fold']==s['fold'] and row['inner_train_n']==s['inner_train'],'Fold summary mismatch')
        row.update(inner_val_classification=s['inner_val_classification'],outer_test_classification=s['outer_test_classification'])
    tables['S48']=_read(work/'loss-confidence-per-arm/results.json')['results']
    lc=_read(work/'loss-confidence-bonferroni40/results.json')['results']
    tables['S49']=[r for r in lc if r['population']=='target_ge20' and r['contrast']=='A -> A_T']
    tables['S50']=[r for r in lc if r['population']=='target_ge20' and r['contrast']=='A_LC -> A_LC_T']
    tables['S52']=[r for r in lc if r['population']=='target_ge20' and r['contrast'] in ('A -> A_LC','A_T -> A_LC_T')]
    lc_groups=defaultdict(list)
    for row in lc:lc_groups[row['dataset'],row['population'],row['contrast']].append(row)
    tables['S51']=[dict(dataset=ds,population=pop,contrast=contrast,
                        supported=[r['metric'] for r in rows if r['bonferroni40_low']>0],
                        worsened=[r['metric'] for r in rows if r['bonferroni40_high']<0],
                        inconclusive=[r['metric'] for r in rows if r['bonferroni40_low']<=0<=r['bonferroni40_high']])
                    for (ds,pop,contrast),rows in lc_groups.items()]
    tables['S5']=_read(work/'lemur-paired/results.json')['results']
    tables['S32']=_read(work/'nb201-qwen/results.json')['results']
    collapse=_read(work/'collapse/results.json')
    tables['S25']=[dict(kind='score',condition=arm,estimate=r['auc'],low=r['ci95'][0],high=r['ci95'][1]) for arm,r in collapse['scores'].items()]
    tables['S25'] += [dict(kind='paired_vs_epoch3',condition=arm,estimate=r['auc_difference'],low=r['ci95'][0],high=r['ci95'][1]) for arm,r in collapse['paired'].items()]
    init=_read(work/'initialization/results.json')
    for table in ('S41','S42','S43','S44','S45'):
        tables[table]=[dict(kind='descriptive',**r) for r in init['descriptive'] if r['table']==table]+[dict(kind='paired',**r) for r in init['paired'] if r['table']==table]
    nested=_read(work/'nested-outer/results.json')
    tables['S59']=[dict(seed=int(seed),fold=int(fold),**value['trial'],selection_score=value['selection_score'])
                   for seed,folds in nested['selections'].items() for fold,value in folds.items()]
    tables['S60']=nested['per_seed'];tables['S61']=_nested_s61(nested)
    tables=native(tables)
    numerical={k for k,v in plan.items() if v!='context'}
    require(set(EXPECTED_ROWS)==numerical,'Numerical coverage contract missing a table')
    require(set(tables)==numerical,'Missing or unexpected table exports: '+str(set(tables)^numerical))
    index=[]
    for name,mode in plan.items():
        if mode=='context':
            index.append(dict(table=name,kind='contextual synthesis/literature',rows=0,workflow=mode,path=None))
            continue
        rows=tables[name];require(bool(rows) and all(isinstance(r,dict) for r in rows),'Empty/malformed table: '+name)
        validate_table(name,rows)
        path=table_dir/(name+'.json');path.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.with_suffix('.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
        index.append(dict(table=name,kind='numerical export',rows=len(rows),workflow=mode,path=str(path.relative_to(out))))
    flags=[article['s24_reconciliation'],*point4['flags'],*classical.get('paper_discrepancies',[]),*nb['mixed_model_flags']]
    printed_checks=[
        ('S2',next(r for r in tables['S2'] if r['observations']==2 and r['feature']=='no-layer' and r['metric']=='RMSE'),'mean',11.4716,4),
        ('S5',next(r for r in tables['S5'] if r['reference']=='Calibrated SynFlow' and r['metric']=='Pearson'),'bonferroni20_high',1.1765,4),
        ('S54',next(r for r in tables['S54'] if r['dataset']=='cifar10' and r['epoch']==3 and r['batch_size']==32 and r['mode']=='train' and r['augmentation']=='off'),'spearman',.939,3)]
    for table,row,field,printed,decimals in printed_checks:
        if format(row[field],f'.{decimals}f')!=format(printed,f'.{decimals}f'):
            flags.append(dict(table=table,field=field,computed=row[field],printed=printed,
                              note='Displayed precision discrepancy in the audited supplement; inputs unchanged.'))
    require(before=={n:sha(repo/n) for n in before} and all(sha(repo/n)==h for n,h in extras.items()),'Original inputs changed')
    coverage=dict(status='complete',inventory=index,numerical_tables=len(numerical),contextual_tables=2,
                  flags=native(flags),notes=['Every numerical table has an export; this does not mean every printed cell is correct.',
                                    'S24 invalid mixed-model inference is explicitly withheld, never substituted.',
                                    'S1 is a synthesis of other results; S46 is a literature comparison. Neither is a new numerical computation.',
                                    'No original training/acquisition/calibration reruns. Inspect each workflow provenance and uncertainty scope.'])
    (out/'TABLE_INDEX.json').write_text(json.dumps(coverage,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Paper table exports','',*coverage['notes'],'','| Table | Kind | Rows | Workflow |','|---|---|---:|---|']
    lines += [f"| {r['table']} | {r['kind']} | {r['rows']} | {r['workflow']} |" for r in index]
    (out/'TABLE_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256=hashes,input_sha256=dict(before,**extras)),indent=2)+'\n',encoding='utf-8')
    print('COMPLETE PAPER TABLE EXPORT:',len(numerical),'numerical tables; 2 contextual tables;',out,flush=True)
    return coverage
