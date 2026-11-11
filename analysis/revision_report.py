"""Consolidate all recent paired analyses; preserve originals and correction scopes."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pymupdf
from analysis import revision_common as m

METRICS=['RMSE','MAE','Pearson','Spearman','R2']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(folder):
    receipt=json.loads((folder/'complete.json').read_text())
    assert receipt['status']=='complete'
    for name,digest in receipt['files_sha256'].items():
        assert Path(name).name == name and not Path(name).is_absolute(), 'Unsafe receipt path'
        assert sha(folder/name)==digest,name


def status(low,high):
    return 'Favors candidate' if low>0 else 'Favors reference' if high<0 else 'Inconclusive'


def run(inputs, out):
    HOME, OUT = Path(inputs), Path(out)
    assert not OUT.exists(),'Existing report preserved'
    rows=[];sources={};groups=defaultdict(list)
    def add(family,dataset,population,contrast,metric,reference,candidate,gain,low,high,adjlow,adjhigh,correction,n,clusters=''):
        row=dict(family=family,dataset=dataset,population=population,contrast=contrast,metric=metric,n=n,clusters=clusters,
                 reference=float(reference),candidate=float(candidate),improvement=float(gain),ci95_low=float(low),ci95_high=float(high),
                 adjusted_low=float(adjlow),adjusted_high=float(adjhigh),correction=correction,
                 adjusted_result=status(adjlow,adjhigh))
        rows.append(row);groups[family,dataset,population,contrast,correction].append(row)
    for name in ['loss-confidence-all-metrics-v1','nb201-qwen-extra-metrics-v1','lemur-extra-paired-intervals-v1','classical-no-taylor-v1']:
        folder=HOME/name;verified(folder);sources[name]=sha(folder/'complete.json')
    lc=json.loads((HOME/'loss-confidence-all-metrics-v1/results.json').read_text())
    for r in lc['results']:
        add('HistGB four-arm study',r['dataset'],r['population'],r['contrast'],r['metric'],r['reference'],r['candidate'],r['improvement'],r['ci95_low'],r['ci95_high'],r['bonferroni40_low'],r['bonferroni40_high'],'Bonferroni40 within dataset',r['n'],r['groups'])
    nb=json.loads((HOME/'nb201-qwen-extra-metrics-v1/results.json').read_text())
    for r in nb['results']:
        add('NB201 Qwen supporting metrics',r['dataset'],'all','No layer -> Taylor',r['metric'],r['no_layer'],r['taylor'],r['improvement'],r['ci95_low'],r['ci95_high'],r['bonferroni12_low'],r['bonferroni12_high'],'Bonferroni12 across datasets',r['n'],r['groups'])
    le=json.loads((HOME/'lemur-extra-paired-intervals-v1/results.json').read_text())
    for r in le['results']:
        add('LEMUR extra comparisons','LEMUR','all',r['reference']+' -> Qwen+Taylor',r['metric'],r['reference_value'],r['qwen_taylor'],r['improvement'],r['ci95_low'],r['ci95_high'],r['bonferroni20_low'],r['bonferroni20_high'],'Bonferroni20 across four contrasts',r['n'])
    # Persist the previously run classical comparison using the same resampling algorithm.
    folder=HOME/'classical-no-taylor-v1'
    methods=['qwen_reference','ridge','random_forest','hist_gradient_boosting'];pred=[];identity=None;y=None
    for method in methods:
        vectors=[]
        for seed in range(42,48):
            with (folder/f'{method}-s{seed}-predictions.csv').open() as f:rr=list(csv.DictReader(f))
            ids=[r['record_id'] for r in rr];target=np.array([float(r['target']) for r in rr])
            if identity is None:identity=ids;y=target
            assert ids==identity and len(set(ids))==115 and np.array_equal(target,y)
            assert all(int(r['seed'])==seed for r in rr)
            vectors.append([float(r['prediction']) for r in rr])
        pred.append(vectors)
    pred=np.array(pred);assert np.isfinite(pred).all()
    helper=Path(m.__file__)
    point=m.metrics(y,pred).mean(axis=1);sign=np.array([-1,-1,1,1,1]);gain=(point[0]-point[1:])*sign
    rng=np.random.default_rng(0);draws=np.empty((3,10000,5))
    for start in range(0,10000,250):
        idx=rng.integers(115,size=(250,115));values=m.metrics(y[idx],pred[:,:,idx]).mean(axis=1)
        draws[:,start:start+250]=(values[0][None,:,:]-values[1:])*sign
    assert np.isfinite(draws).all()
    low,high=np.quantile(draws,[.025,.975],axis=1);al,ah=np.quantile(draws,[.025/15,1-.025/15],axis=1)
    for i,method in enumerate(methods[1:]):
        for k,metric in enumerate(METRICS):
            add('Qwen vs no-Taylor classical','LEMUR','all',method+' without Taylor -> Qwen+Taylor',metric,point[i+1,k],point[0,k],gain[i,k],low[i,k],high[i,k],al[i,k],ah[i,k],'Bonferroni15 across three contrasts',115)
    assert len(rows)==167
    assert len({(r['family'],r['dataset'],r['population'],r['contrast'],r['metric']) for r in rows})==len(rows)
    OUT.mkdir(parents=True)
    with (OUT/'all_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'classical_no_taylor_bootstrap.npz',differences=draws,predictions=pred,targets=y)
    notes=[
        'Scope: all 167 metric-level intervals from the four recent paired-analysis families discussed in this chat. This is not an inventory of every historical test in the thesis.',
        'Positive improvement always favors the candidate named after the arrow. Error metrics: reference minus candidate; correlations and R2: candidate minus reference.',
        'Every bootstrap uses 10000 draws and seed 0, with fitted predictors held fixed. Predictor-seed uncertainty is not resampled.',
        'NB201 Qwen and HistGB use operation-family resampling, preserving every record in each sampled group. LEMUR comparisons resample paired test records.',
        'Correction families are separate: NB201 Qwen Bonferroni12; HistGB Bonferroni40 within each dataset; LEMUR extra contrasts Bonferroni20; Qwen/classical Bonferroni15. There is no claimed joint correction across all 167 rows.',
        'HistGB all and target_ge20 populations are both included. ImageNet16-120 HistGB uses a partial cohort; the Qwen study uses all 275 test records per dataset.',
        'All HistGB seed-labelled prediction vectors were identical within each arm. This is not six independent confirmations of fitting variability.',
        'LEMUR XGBoost and proxy predictions are rescored against the same two-decimal targets as Qwen. Minor last-digit differences from old full-precision descriptive tables are expected.',
        'The SynFlow/NASWOT comparisons contrast complete approaches with unequal information budgets. They do not isolate Taylor or measurement time.',
        'These extensions are post-hoc supporting analyses. Corrections do not remove earlier feature-selection bias. Percentile intervals are approximate, and the most extreme adjusted quantiles use relatively few tail draws.',
        'Not newly computed here: original primary Holm-adjusted MAE tests, older ablations/holdouts, paired Taylor effects within classical LEMUR models, or nested outer results for seeds43-47.',
    ]
    report=dict(notes=notes,source_receipt_sha256=sources,results=rows,script_sha256=sha(Path(__file__)),metric_helper_sha256=sha(helper))
    (OUT/'all_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Full recent paired-test results','']+['- '+n for n in notes]+['']
    doc=pymupdf.open()
    def text(page,x,y,value,size=10,font='helv'):
        page.insert_text((x,y),value,fontsize=size,fontname=font)
    cover=doc.new_page(width=842,height=595)
    text(cover,30,35,'Full recent paired-test results',18)
    import textwrap
    yy=65
    for note in notes:
        for line in textwrap.wrap('- '+note,width=142):text(cover,30,yy,line,8);yy+=12
        yy+=4
    summary={}
    for family in dict.fromkeys(r['family'] for r in rows):
        subset=[r for r in rows if r['family']==family]
        summary[family]={s:sum(r['adjusted_result']==s for r in subset) for s in ['Favors candidate','Favors reference','Inconclusive']}
    for key,subset in groups.items():
        family,dataset,population,contrast,correction=key
        subset.sort(key=lambda r:METRICS.index(r['metric']))
        title=f'{family} | {dataset} | {population}'
        lines+=['## '+title,contrast,'',f"n={subset[0]['n']}; groups={subset[0]['clusters'] or 'not grouped'}; {correction}",'',
                '| Metric | Reference | Candidate | Improvement | Unadjusted 95% CI | Adjusted CI | Adjusted result |',
                '|---|---:|---:|---:|---|---|---|']
        page=doc.new_page(width=842,height=595)
        text(page,30,35,title,13);text(page,30,57,contrast,11)
        text(page,30,79,f"n={subset[0]['n']}; groups={subset[0]['clusters'] or 'not grouped'}; {correction}",10)
        text(page,30,100,'Positive differences favor the candidate. Intervals concern differences, not individual method scores.',9)
        header=f"{'Metric':<10}{'Reference':>12}{'Candidate':>12}{'Gain':>12}  {'Unadjusted 95% interval':<27}{'Adjusted interval':<27}Result"
        text(page,30,135,header,8.8,'cour')
        y0=160
        for r in subset:
            ci=f"[{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]";adj=f"[{r['adjusted_low']:.6f}, {r['adjusted_high']:.6f}]"
            lines.append(f"| {r['metric']} | {r['reference']:.6f} | {r['candidate']:.6f} | {r['improvement']:+.6f} | {ci} | {adj} | {r['adjusted_result']} |")
            display=f"{r['metric']:<10}{r['reference']:>12.6f}{r['candidate']:>12.6f}{r['improvement']:>+12.6f}  {ci:<27}{adj:<27}{r['adjusted_result']}"
            text(page,30,y0,display,8.8,'cour');y0+=25
        lines+=['']
        text(page,30,550,'See first page for scope, resampling and limitations. Unrounded values are retained in CSV/JSON.',9)
    for i,page in enumerate(doc):text(page,760,578,f'{i+1} / {len(doc)}',8)
    doc.save(OUT/'full_results.pdf',garbage=4,deflate=True)
    (OUT/'full_results.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'complete.json').write_text(json.dumps(dict(status='complete',metric_rows=len(rows),tables=len(groups),pdf_pages=len(doc),files_sha256=files),indent=2)+'\n')
    print('COMPLETE',len(rows),'metric rows;',len(groups),'complete tables;',len(doc),'PDF pages')
    print(json.dumps(summary,indent=2));print(OUT)
