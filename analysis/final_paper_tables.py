"""Stable-label exports for the corrected final paper and restored supplement.

Consumes completed calculations only. Values retain full precision and the
original uncertainty columns; extra columns/metrics are labelled source detail.
"""
import argparse
import csv
import json
from pathlib import Path
from analysis.thesis_common import require, sha

LEGACY_LABELS = {
    'tab:corpus':'main_I','tab:nb201-main':'main_III','tab:primary':'main_II',
    'tab:singlefeat':'main_IV','tab:strata':'main_VI','tab:classical':'main_VII',
    'tab:sm-lemur-extra':'S5','tab:sm-classical-within':'S6',
    'tab:sm-classical-notaylor':'S7','tab:sm-checkpoints':'S15',
    'tab:sm-nb201-supporting':'S32','tab:sm-lc-at':'S49',
    'tab:sm-lc-increment':'S50','tab:sm-lc-summary':'S51',
    'tab:accpred-metrics-ci':'S2','tab:main-85-5-10-results':'S3',
    'tab:baseline-proxies-85-5-10':'S4','tab:early-mic':'S8','tab:same-epoch-mic':'S9',
    'tab:layer-features-2acc-zcp-on':'S10','tab:layer-features-2acc-zcp-off':'S11',
    'tab:layer-statistic-cost-performance':'S12','tab:layer-features-3acc-zcp-off':'S13',
    'tab:taylor-zcp-on-off':'S14','tab:layer-extraction-time':'S16',
    'tab:taylor-round2-pairs':'S17','tab:taylor-sensitivity-zcp-on-off':'S18',
    'tab:zcp-paired-five-metrics':'S19','tab:taylor-round3-triples':'S20',
    'tab:taylor-paired-bootstrap-rmse':'S21','tab:taylor-readmission-features':'S22',
    'tab:readmission-paired-five-metrics':'S23','smtab:robust':'S24',
    'smtab:collapse-baseline':'S25','tab:multiplicity-families':'S26',
    'tab:zcp-aggregate':'S27','tab:selection-quality':'S28','smtab:ens':'S29',
    'smtab:nb201-descriptive':'S30','smtab:nb201-primary':'S31',
    'smtab:nb201-global-holm':'S31','smtab:nb201-bootstrap':'S31',
    'smtab:nb201-supporting-tests':'S31','smtab:nb201-mixed':'S31',
    'smtab:nb201-classical':'S33','tab:supp-multiplicity':'S34','tab:supp-selection':'S35',
    'tab:lodo-taylor-zcp-off':'S36','tab:lodo-taylor-record-bootstrap':'S37',
    'tab:lofo-taylor-zcp-off-comparison':'S38','tab:lofo-taylor-grouped-bootstrap':'S39',
    'tab:per-layer-versus-averaged-taylor':'S40','tab:init-taylor-proxy-comparison':'S41',
    'tab:init-proxies-matched-86':'S42','tab:init-proxies-paired-86':'S43',
    'tab:init-proxy-combinations':'S44','tab:init-proxy-combinations-paired':'S45',
    'tab:matched-cohort-scope':'S47','tab:matched-four-arm':'S48',
    'smtab:acq-coverage':'S53','smtab:acq-sensitivity':'S54','smtab:acq-cv':'S55',
    'smtab:acq-rescale':'S56','smtab:acq-timing':'S57','smtab:nested-partitions':'S58',
    'smtab:nested-selections':'S59','smtab:nested-seeds':'S60','smtab:nested-pooled':'S61',
}
ALIASES = {label:'smtab:nb201-primary' for label in
           ['smtab:nb201-global-holm','smtab:nb201-bootstrap',
            'smtab:nb201-supporting-tests','smtab:nb201-mixed']}
CONTEXT_LABEL = 'tab:closest-related'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified(folder):
    folder=Path(folder).resolve()
    receipt=read(folder/'complete.json')
    require(receipt.get('status')=='complete','Incomplete calculation: '+str(folder))
    require(bool(receipt.get('files_sha256')),'Empty receipt')
    for name,h in receipt['files_sha256'].items():
        p=(folder/name).resolve()
        require(p.is_relative_to(folder) and p.is_file() and sha(p)==h,'Changed/unsafe output: '+name)
    return receipt


def fourarm(rows):
    values={}
    for r in rows:
        if r['population']!='target_ge20':continue
        arms=r['contrast'].split(' -> ')
        for arm,column in zip(arms,['reference','candidate']):
            key=(r['dataset'],arm,r['metric'])
            value=dict(dataset=r['dataset'],population=r['population'],arm=arm,
                       metric=r['metric'],mean=r[column],n=r['n'],groups=r['groups'])
            require(key not in values or values[key]==value,'Inconsistent four-arm projection')
            values[key]=value
    require(len(values)==60 and {k[1] for k in values}=={'A','A_T','A_LC','A_LC_T'},
            'Incomplete four-arm means')
    return list(values.values())


def export(tables,xgboost,lc,nested,out):
    roots=[Path(p).resolve() for p in [tables,xgboost,lc,nested]]
    tables,xgboost,lc,nested=roots;out=Path(out).resolve()
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    require(not any(out.is_relative_to(p) for p in roots),'Output overlaps completed inputs')
    receipts=[verified(p) for p in roots]
    bound=[dict(role=role,receipt_sha256=sha(p/'complete.json'))
           for role,p in zip(['legacy_tables','xgboost_bca','lc_sensitivity','nested_bca'],roots)]
    result={label:read(tables/'tables'/(old+'.json')) for label,old in LEGACY_LABELS.items()}
    lc_result=read(lc/'results.json')
    result['tab:fourarm']=fourarm(read(tables/'workflows/loss-confidence-bonferroni40/results.json')['results'])
    result['tab:lc-bca']=[r for r in lc_result['results'] if r['metric'] in ('MAE','RMSE')]
    result['tab:sm-lc-bca']=lc_result['results']
    result['tab:sm-lc-zou']=lc_result['zou']
    result['tab:xgboost-bca']=read(xgboost/'results.json')['results']
    require(len(result)==73,'Corrected table-label inventory mismatch')
    require({r['group'] for r in result['tab:sm-checkpoints']}=={'3/off/no-layer','3/off/taylor','epoch1'},
            'Checkpoint reference row missing')
    require(len(result['tab:lc-bca'])==6 and len(result['tab:sm-lc-bca'])==15
            and len(result['tab:sm-lc-zou'])==3 and len(result['tab:xgboost-bca'])==5,
            'Incomplete sensitivity tables')
    out.mkdir(parents=True,exist_ok=False);folder=out/'tables';folder.mkdir()
    index=[]
    for label,rows in result.items():
        name=label.replace(':','-')
        path=folder/(name+'.json');path.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.with_suffix('.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
        index.append(dict(label=label,rows=len(rows),legacy_table=LEGACY_LABELS.get(label),
                          alias_of=ALIASES.get(label),path=path.relative_to(out).as_posix()))
    (out/'nested_bca_prose.json').write_text(json.dumps(read(nested/'results.json')['results'],indent=2)+'\n',encoding='utf-8')
    report=dict(status='complete',scope='Corrected final paper: numerical table source exports, not exact LaTeX rendering',
                numerical_tables=len(result)-len(ALIASES),numerical_labels=len(result),
                contextual_tables=[dict(label=CONTEXT_LABEL,kind='literature comparison; no numerical calculation')],
                manuscript_archive_sha256='b2377ca73b90377d4dde1e0a64c7665f860a27de1e8d0dad781e06fe3e036043',
                tables=index,input_receipts=bound,source_script_sha256=sha(Path(__file__)),
                limitations=['Four NB labels are aliases of one merged primary table, not independent results.',
                    'Legacy exports retain additional source rows/columns and historical discrepancy flags; do not infer final numbering.',
                    'Four-arm means are target>=20; BCa and Zou use all available test records.',
                    'No new calculations; uncertainty families preserved. No original model training or selection replay.'])
    (out/'TABLE_INDEX.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for p,receipt,item in zip(roots,receipts,bound):
        require(sha(p/'complete.json')==item['receipt_sha256'],'Receipt changed during export')
        verified(p)
    (out/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256={
        p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}),indent=2)+'\n',encoding='utf-8')
    print('FINAL TABLE LABELS: 69 numerical tables / 73 labels; one contextual table; nested BCa prose retained.',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['tables','xgboost','lc','nested','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();export(args.tables,args.xgboost,args.lc,args.nested,args.out)
