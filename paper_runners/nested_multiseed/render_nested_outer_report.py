"""Render the verified six-seed nested analysis; do not change numerical results."""
from pathlib import Path
import csv
import hashlib
import json
import textwrap
import pymupdf

ROOT=Path('C:/Users/trisr/Downloads/nested-outer-six-seed-analysis-v1')
r=json.loads((ROOT/'results.json').read_text())
receipt=json.loads((ROOT/'complete.json').read_text())
for name,h in receipt['files_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
assert not (ROOT/'nested_results.pdf').exists()
metrics=['RMSE','MAE','Pearson','Spearman','R2']
doc=pymupdf.open()

def page(title):
    p=doc.new_page(width=842,height=595)
    p.insert_text((30,35),title,fontsize=16)
    return p

def table(p,start,head,lines):
    p.insert_text((30,start),head,fontsize=9,fontname='cour')
    for i,line in enumerate(lines):p.insert_text((30,start+24*(i+1)),line,fontsize=9,fontname='cour')

p=page('Nested outer evaluation: six seeds, all five metrics')
notes=[
    'Seeds42-47;1139 classification records per condition and seed; all target accuracies.',
    'Architecture groups per fold: '+str(r['groups_per_fold'])+'; classification records per fold: '+str(r['records_per_fold'])+'.',
    'Metrics are pooled across outer folds within each seed, then averaged across seeds. Predictions are not ensembled.',
    'Paired bootstrap:10000 architecture-group resamples stratified within each original outer fold. The same resample is used for both conditions and every seed.',
    'The original number of groups is sampled with replacement per fold; all records in a sampled group are retained, including repeated groups.',
    'Bonferroni5 corrects the five Taylor-versus-no-layer metric comparisons. Positive differences always favor Taylor.',
    'Inference conditions on the observed folds, fitted models and inner selections; it does not repeat the nested training/search procedure or resample predictor seeds.',
    'Selected-procedure values are descriptive and follow each fold\'s locked inner feature choice; they are not an ensemble or an outer-test-selected model.',
]
y=65
for note in notes:
    for line in textwrap.wrap('- '+note,130):p.insert_text((30,y),line,fontsize=9);y+=14
    y+=5
head=f"{'Method':<23}"+''.join(f'{m:>14}' for m in metrics)
lines=[f"{v['method']:<23}"+''.join(f'{v[m]:>14.6f}' for m in metrics) for v in r['mean_metrics']]
table(p,y+15,head,lines)

p=page('Paired bootstrap: Taylor versus no-layer')
p.insert_text((30,62),'Reference - Taylor for RMSE/MAE; Taylor - reference for correlations and R2.',fontsize=10)
head=f"{'Metric':<10}{'Improvement':>13}  {'Unadjusted 95% CI':<27}{'Bonferroni5 CI':<27}{'Seeds +':>8}  Result"
lines=[]
for v in r['paired']:
    ci=f"[{v['ci95_low']:.6f}, {v['ci95_high']:.6f}]";adj=f"[{v['bonferroni5_low']:.6f}, {v['bonferroni5_high']:.6f}]"
    lines.append(f"{v['metric']:<10}{v['improvement']:>+13.6f}  {ci:<27}{adj:<27}{str(v['seeds_favoring_taylor'])+'/6':>8}  {v['adjusted_result']}")
table(p,95,head,lines)
p.insert_text((30,280),'MAE has a positive unadjusted interval. All five adjusted intervals include zero.',fontsize=11)
p.insert_text((30,301),'Seed agreement is descriptive: the seeds share the same records and folds.',fontsize=10)

for part in [r['per_seed'][:9],r['per_seed'][9:]]:
    p=page('Per-seed pooled outer metrics')
    head=f"{'Seed':<7}{'Method':<23}"+''.join(f'{m:>14}' for m in metrics)
    lines=[f"{v['seed']:<7}{v['method']:<23}"+''.join(f'{v[m]:>14.6f}' for m in metrics) for v in part]
    table(p,75,head,lines)

for i,p in enumerate(doc):p.insert_text((760,575),f'{i+1}/{len(doc)}',fontsize=8)
doc.save(ROOT/'nested_results.pdf',garbage=4,deflate=True)
lines=['# Nested outer results: seeds42-47','']+notes+['','## Mean metrics','',
    '| Method | RMSE | MAE | Pearson | Spearman | R2 |','|---|---:|---:|---:|---:|---:|']
for v in r['mean_metrics']:lines.append('| '+v['method']+' | '+' | '.join(f'{v[m]:.6f}' for m in metrics)+' |')
lines+=['','## Paired bootstrap','', '| Metric | Improvement | 95% CI | Bonferroni5 CI | Seeds favoring Taylor |','|---|---:|---|---|---:|']
for v in r['paired']:lines.append(f"| {v['metric']} | {v['improvement']:+.6f} | [{v['ci95_low']:.6f}, {v['ci95_high']:.6f}] | [{v['bonferroni5_low']:.6f}, {v['bonferroni5_high']:.6f}] | {v['seeds_favoring_taylor']}/6 |")
lines+=['','## Per-seed metrics','','| Seed | Method | RMSE | MAE | Pearson | Spearman | R2 |','|---|---|---:|---:|---:|---:|---:|']
for v in r['per_seed']:lines.append('| '+str(v['seed'])+' | '+v['method']+' | '+' | '.join(f'{v[m]:.6f}' for m in metrics)+' |')
(ROOT/'nested_results.md').write_text('\n'.join(lines)+'\n')
receipt['files_sha256'].update({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['nested_results.pdf','nested_results.md']})
(ROOT/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PDF/Markdown written; numerical result files preserved.')
print('Total classification architecture groups:',sum(r['groups_per_fold']))
for v in r['paired']:
    print(f"{v['metric']} | {v['no_layer']:.4f} | {v['taylor']:.4f} | {v['improvement']:+.4f} | [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}] | [{v['bonferroni5_low']:.4f}, {v['bonferroni5_high']:.4f}] | {v['seeds_favoring_taylor']}/6")
