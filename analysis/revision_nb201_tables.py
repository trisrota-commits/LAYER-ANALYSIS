"""Complete NB201 paper table projections from the existing verified calculations."""
import csv
import json
from pathlib import Path
import numpy as np
from analysis import thesis_nb201 as source
from analysis.thesis_common import sha, require

SPECS = {'cifar10':'nb201-qwen-primary-v1/cifar10', 'cifar100':'nb201-qwen-primary-v1/cifar100',
         'imagenet16':'nb201-imagenet16-primary-v1'}


def project_primary(reports, groups):
    p = np.array([reports[d]['primary'][u]['p'] for d in SPECS for u in ('record','seed')])
    order = np.argsort(p); adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p),0,-1)))
    main, descriptive, supporting = [], [], []
    for i, ds in enumerate(SPECS):
        r = reports[ds]; desc = r['descriptive']; tests = r['primary']
        row = dict(dataset=ds, n=r['counts']['noncollapsed'], operation_groups=groups[ds],
                   no_layer_MAE=desc['noncollapsed']['no-layer']['mean']['MAE'],
                   taylor_MAE=desc['noncollapsed']['taylor']['mean']['MAE'],
                   improvement=tests['record']['improvement'],
                   relative_reduction_percent=100*tests['record']['improvement']/desc['noncollapsed']['no-layer']['mean']['MAE'],
                   no_layer_Spearman_all=desc['all']['no-layer']['mean']['Spearman'],
                   taylor_Spearman_all=desc['all']['taylor']['mean']['Spearman'])
        for j, unit in enumerate(('record','seed')):
            test = tests[unit]
            row.update({unit+'_p':test['p'], unit+'_holm2_p':test['holm_p'],
                        unit+'_holm6_p':float(adjusted[2*i+j]), unit+'_t':test['t'],
                        unit+'_ci95_low':test['ci95'][0], unit+'_ci95_high':test['ci95'][1]})
        main.append(row)
        for arm in ('no-layer','taylor'):
            descriptive.append(dict(dataset=ds, condition=arm, n=r['counts']['all'], **desc['all'][arm]['mean']))
        for name in ('record_bootstrap','two_way_bootstrap','family_bootstrap','wilcoxon','cell_signflip','crossed_mixed_model'):
            result = r['robustness'][name]; ci = result.get('ci95') or [None,None]
            supporting.append(dict(dataset=ds, method=name, p=result.get('p'), low=ci[0], high=ci[1],
                                   status=result.get('status','computed'), details=result))
    return main, descriptive, supporting


def run(repo, out):
    repo, out = Path(repo), Path(out)
    if out.exists(): raise FileExistsError('Existing output preserved: '+str(out))
    frozen=json.loads((repo/'INPUT_MANIFEST.json').read_text())['files']
    before={n:sha(repo/n) for n in frozen if n.startswith('evidence/nb201')}
    require(all(h==frozen[n] for n,h in before.items()),'Frozen NB201 inputs changed')
    out.mkdir(parents=True,exist_ok=False)
    raw=source.build_tables(repo,out/'work')
    require(all(r['verification_status']=='verified' for r in raw['6.14']+raw['6.15']+raw['6.16']),
            'NB201 core numerical mismatch; diagnostics preserved')
    reports={r['dataset']:json.loads(Path(r['report_path']).read_text()) for r in raw['6.15']}
    groups={ds:len({r['family'] for r in json.loads((repo/'evidence'/path/'inputs/registry.json').read_text()) if r['target']>=20})
            for ds,path in SPECS.items()}
    main, descriptive, supporting=project_primary(reports,groups)
    classical=[]
    for r in raw['6.16']:
        report=json.loads(Path(r['report_path']).read_text())['results']['all']
        classical.append(dict(dataset=r['dataset'],model=r['model'],n=r['n'],
                              no_layer_MAE=report['mean_metrics']['no-layer']['MAE'],
                              taylor_MAE=report['mean_metrics']['taylor']['MAE'],improvement=r['improvement'],
                              bonferroni8_low=r['family_bonferroni8_low'],bonferroni8_high=r['family_bonferroni8_high']))
    tables={'main_III':main,'S30':descriptive,'S31':main+supporting,'S33':classical}
    flags=[dict(dataset=r['dataset'],mixed_status=r['mixed_verification_status']) for r in raw['6.15'] if r['mixed_verification_status']!='verified']
    for name, rows in tables.items():
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/(name+'.csv')).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    require(before=={n:sha(repo/n) for n in before},'NB201 inputs changed during calculation')
    result=dict(tables=tables,primary_reports=reports,mixed_model_flags=flags,input_sha256=before,
                notes=['Joint Holm6 is recomputed across three datasets and both primary tests.',
                       'Supporting bootstrap/mixed calculations use the original published implementation; invalid or mismatched mixed fits are flagged.',
                       'S30 uses all test records; primary/S31 use target>=20; S33 uses all records and the original Bonferroni8 family.'])
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (out/'complete.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='complete',files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}),f,indent=2)
    print('COMPLETE: main III, S30, S31 and S33; flags:',flags,flush=True)
    return result
