"""Regenerate the NB201 forest panel only, using verified saved statistics."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pymupdf

REPO = Path(__file__).resolve().parents[2]
EXTRA = Path(__file__).resolve().parent / 'inputs'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
OUT = parser.parse_args().out.resolve()
assert not OUT.exists(), 'Existing output preserved'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((REPO/'INPUT_MANIFEST.json').read_text())['files']
assert sha(EXTRA/'intervals.csv') == '2e8c1f7afcdf1199756de7a94bee4a7e251de3ee08375d221b58fa5924e97b74', 'Supporting intervals changed'
with (EXTRA/'intervals.csv').open() as f:
    additional = list(csv.DictReader(f))
datasets = ['CIFAR-10-valid','CIFAR-100','ImageNet16-120']
paths = ['evidence/nb201-qwen-primary-v1/cifar10/analysis/results.json',
         'evidence/nb201-qwen-primary-v1/cifar100/analysis/results.json',
         'evidence/nb201-imagenet16-primary-v1/analysis/results.json']
points=[]
for dataset,relative in zip(datasets,paths):
    assert sha(REPO/relative)==manifest[relative],relative
    result=json.loads((REPO/relative).read_text())['primary']['record']
    points.append(dict(dataset=dataset,metric='MAE*',estimate=result['improvement'],low=result['ci95'][0],high=result['ci95'][1],adjusted_low=None,adjusted_high=None,population='target_ge20',interval='Record-paired t interval, unadjusted',source=relative))
    for metric in ['RMSE','Pearson','Spearman','R2']:
        rows=[r for r in additional if r['dataset']==dataset and r['metric']==metric]
        assert len(rows)==1 and rows[0]['population']=='all' and int(rows[0]['n'])==275
        r=rows[0]
        points.append(dict(dataset=dataset,metric=metric,estimate=float(r['improvement']),low=float(r['ci95_low']),high=float(r['ci95_high']),adjusted_low=float(r['bonferroni12_low']),adjusted_high=float(r['bonferroni12_high']),population='all275',interval='Operation-family percentile bootstrap; Bonferroni12',source='nb201-qwen-extra-metrics-v1/intervals.csv'))
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.linewidth':0.6,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,5,figsize=(7.16,1.65),sharey=True)
fig.subplots_adjust(left=.18,right=.995,top=.84,bottom=.36,wspace=.38)
limits=[(-.10,1.65),(-.18,3.25),(-.018,.34),(-.02,.36),(-.04,.83)]
for ax,metric,lim in zip(axes,['MAE*','RMSE','Pearson','Spearman','R2'],limits):
    ax.axvline(0,color='#777777',linestyle='--',linewidth=.7,zorder=0)
    for i,dataset in enumerate(datasets):
        row=next(r for r in points if r['metric']==metric and r['dataset']==dataset)
        if row['adjusted_low'] is not None:
            ax.plot([row['adjusted_low'],row['adjusted_high']],[i,i],color='#9aafc2',linewidth=1.3,zorder=1)
        ax.plot([row['low'],row['high']],[i,i],color='#2876b7',linewidth=3,zorder=2,solid_capstyle='butt')
        ax.plot(row['estimate'],i,'o',color='#171717',markersize=3.1,zorder=3)
    ax.set_xlim(*lim);ax.set_ylim(2.55,-.55)
    ax.set_title(r'$R^2$' if metric=='R2' else metric,pad=7,fontsize=9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,labelsize=8)
    ax.tick_params(axis='x',labelsize=7,pad=2,length=3)
    ax.set_yticks(range(3))
axes[0].set_yticklabels(datasets)
fig.text(.58,.20,'Improvement from Taylor (positive favours Taylor)',ha='center',fontsize=8)
handles=[Line2D([0],[0],color='#2876b7',lw=3,label='95% paired interval'),
         Line2D([0],[0],color='#9aafc2',lw=1.3,label='Bonferroni-adjusted (12; non-MAE metrics)')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,-.005),ncol=2,frameon=False,fontsize=7,handlelength=2.3,columnspacing=1.8)
# Verify exact plotted estimates and endpoints before saving.
for ax,metric in zip(axes,['MAE*','RMSE','Pearson','Spearman','R2'],):
    lines=[line for line in ax.lines if line.get_marker()=='o']
    assert len(lines)==3
    for i,line in enumerate(lines):
        expected=next(r for r in points if r['metric']==metric and r['dataset']==datasets[i])
        np.testing.assert_array_equal(line.get_xdata(),[expected['estimate']])
        np.testing.assert_array_equal(line.get_ydata(),[i])
OUT.mkdir(parents=True)
fig.savefig(OUT/'fig_nb201.pdf')
fig.savefig(OUT/'fig_nb201.png',dpi=250)
plt.close(fig)
with (OUT/'plot_data.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
with pymupdf.open(OUT/'fig_nb201.pdf') as doc:
    assert len(doc)==1
    text=doc[0].get_text()
    for label in datasets+['MAE*','RMSE','Pearson','Spearman']:assert label in text,label
    assert '(a)' not in text and '(b)' not in text
    for block in doc[0].get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox'];assert x0>=0 and y0>=0 and x1<=doc[0].rect.width and y1<=doc[0].rect.height,span['text']
(OUT/'provenance.json').write_text(json.dumps({'points':points,'note':'Forest panel regenerated, not cropped. MAE uses primary target>=20 record-paired t intervals; remaining metrics use all275 records and Bonferroni12 operation-family bootstrap intervals. No new fitting or inference.','script_sha256':sha(Path(__file__))},indent=2)+'\n')
print('PASS:15 plotted estimates verified; one-page PDF; no panel letters; labels within page bounds.')
print(OUT/'fig_nb201.pdf')
