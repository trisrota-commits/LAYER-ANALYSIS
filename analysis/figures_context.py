"""Data-grounded reconstructions of the overview and contextual comparison tally."""
from collections import Counter
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from analysis.thesis_common import require, sha


def _save(fig,out,stem):
    files={}
    for ext in ('pdf','png'):
        p=out/(stem+'.'+ext)
        require(not p.exists(),'Existing figure preserved: '+str(p))
        fig.savefig(p,dpi=240,bbox_inches='tight',pad_inches=.035)
        files[ext]=dict(path=p.name,sha256=sha(p))
    plt.close(fig)
    return files


def load_data(repo,tables_root):
    paths={n:Path(tables_root)/'tables'/f'{n}.json' for n in ('main_IV','main_III','S16')}
    tables={n:json.loads(p.read_text()) for n,p in paths.items()}
    rmse={}
    for feature in ('no-layer','taylor'):
        rows=[r for r in tables['main_IV'] if r.get('feature')==feature and r['metric']=='RMSE'
              and r['observations']==3 and r['zcp']=='off']
        require(len(rows)==1 and rows[0]['seeds']==list(range(42,48)) and rows[0]['n_test']==115,'Overview condition mismatch')
        rmse[feature]=rows[0]['mean']
    require(len(tables['main_III'])==3,'Overview requires three NB201 primary datasets')
    reductions=[r['relative_reduction_percent'] for r in tables['main_III']]
    costs=[r for r in tables['S16'] if r.get('operation')=='taylor_importance']
    require(len(costs)==1 and costs[0]['boundary']=='one checkpoint','Overview cost boundary mismatch')
    overview=dict(lemur_rmse=rmse,nb201_primary_relative_mae_reductions=reductions,
                  taylor_seconds_per_checkpoint=costs[0]['seconds'],
                  populations='LEMUR all115, six-seed mean metrics; NB201 target>=20 primary MAE; separate96-candidate cost cohort',
                  target='LEMUR full-run best_accuracy; NB201 dataset-specific benchmark target',
                  source_sha256={n:sha(p) for n,p in paths.items()})
    path=Path(repo)/'figures/current_paper/comparison_inventory.json'
    inventory=json.loads(path.read_text())
    rows=inventory['comparisons'];require(sorted(r['id'] for r in rows)==list(range(1,27)),'Incomplete/duplicate comparison inventory')
    require(all(r['category'] in inventory['categories'] and r['group'] in inventory['groups'] for r in rows),'Unknown tally category')
    tally=[]
    for group in [*inventory['groups'],'All comparisons']:
        subset=rows if group=='All comparisons' else [r for r in rows if r['group']==group]
        counts=Counter(r['category'] for r in subset)
        tally.append(dict(group=group,counts={k:counts[k] for k in inventory['categories']},
                          better=counts['supported']+counts['observed'],total=len(subset)))
    require(tally[-1]['counts']==dict(supported=10,observed=7,mixed=8,worse=1),'Table S1 category count changed')
    return overview,dict(rows=tally,inventory=inventory,source_sha256=sha(path))


def render(repo,tables_root,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    overview,tally=load_data(repo,tables_root)
    out.mkdir(parents=True,exist_ok=False)
    inventory={}
    with plt.rc_context({'font.family':'serif','font.size':7,'axes.linewidth':.5,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(7.16,1.5));fig.subplots_adjust(left=.005,right=.995,bottom=.02,top=.98)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        boxes=[(.01,.12),(.17,.14),(.35,.21),(.60,.15),(.79,.20)]
        for i,(x,w) in enumerate(boxes):
            ax.add_patch(FancyBboxPatch((x,.26),w,.61,boxstyle='round,pad=0.006,rounding_size=.02',
                                       linewidth=0,facecolor='#192b43' if i==4 else '#edf2f6'))
            if i<4:
                ax.add_patch(FancyArrowPatch((x+w+.012,.565),(boxes[i+1][0]-.012,.565),
                                            arrowstyle='-|>',mutation_scale=8,lw=.7,color='#555555'))
        def text(x,y,s,**kw):ax.text(x,y,s,ha='center',va='center',**kw)
        text(.07,.68,'Candidate',fontweight='bold');text(.07,.49,'architecture',fontsize=6.3,color='.3')
        text(.24,.75,'Train',fontweight='bold')
        for i in range(3):
            x=.191+i*.036
            ax.add_patch(FancyBboxPatch((x,.50),.027,.13,boxstyle='round,pad=0.002,rounding_size=.008',facecolor='#287bd6',linewidth=0))
            text(x+.0135,.565,str(i+1),color='white',fontsize=6.5)
        text(.24,.38,'epochs only',fontsize=6.3,color='.3')
        text(.455,.75,'Measure per epoch',fontweight='bold')
        text(.455,.60,r'accuracy $a_1,a_2,a_3$',fontsize=6.3)
        ax.add_patch(FancyBboxPatch((.372,.35),.166,.15,boxstyle='round,pad=.004,rounding_size=.012',facecolor='#e8a027',linewidth=0))
        text(.455,.425,r'Taylor $T_1,T_2,T_3$',fontweight='bold',color='white',fontsize=6.5)
        text(.675,.68,'Predictor',fontweight='bold');text(.675,.53,'Qwen3-8B (QLoRA)',fontsize=6.1,color='.3')
        text(.675,.41,'or HistGB',fontsize=6.3,color='.3')
        text(.89,.75,'Final accuracy',fontweight='bold',color='white')
        r=overview['lemur_rmse'];text(.89,.59,rf"LEMUR RMSE {r['no-layer']:.2f} $\to$ {r['taylor']:.2f}",fontsize=6.1,color='white')
        values=overview['nb201_primary_relative_mae_reductions']
        text(.89,.46,f'NB201 MAE {min(values):.0f}% to {max(values):.0f}% lower',fontsize=5.8,color='white')
        text(.89,.33,f"{overview['taylor_seconds_per_checkpoint']:.2f} s per checkpoint",fontsize=6.1,color='#e8a027')
        text(.5,.13,'matched comparison: the same predictor with and without the Taylor block',fontsize=6.1,fontstyle='italic',color='.35')
        inventory['main_1']=dict(stem='fig_overview',files=_save(fig,out,'fig_overview'),data='overview_data.json',
                                 implementation='New vector reconstruction of the supplied overview; numerical cards calculated from table exports.')
        fig,ax=plt.subplots(figsize=(3.45,2.05));fig.subplots_adjust(left=.36,right=.99,top=.81,bottom=.06)
        positions=[0,1,2,3,4.45];colors={'supported':'#245582','observed':'#8eb8d6','mixed':'#cecdc5','worse':'#b7563c'}
        for y,row in zip(positions,tally['rows']):
            left=0
            for category in tally['inventory']['categories']:
                n=row['counts'][category]
                if n:
                    bars=ax.barh(y,n,left=left,height=.63,color=colors[category],edgecolor='white',linewidth=.5)
                    require(bars[0].get_width()==n,'Tally artist width mismatch')
                    ax.text(left+n/2,y,str(n),ha='center',va='center',fontsize=5.5,
                            color='white' if category in ('supported','worse') else '#252525')
                left+=n
            ax.text(30.5,y,f"{row['better']}/{row['total']}",ha='right',va='center',fontsize=6,color='.3')
        ax.axhline(3.72,color='.75',lw=.5,xmin=0,xmax=.86)
        ax.set_yticks(positions,[r['group'] for r in tally['rows']],fontsize=6.1,color='.35')
        ax.set_xlim(0,31);ax.set_ylim(5.05,-.65);ax.set_xticks([]);ax.tick_params(axis='y',length=0,pad=3)
        for spine in ax.spines.values():spine.set_visible(False)
        handles=[plt.Rectangle((0,0),1,1,color=colors[k]) for k in tally['inventory']['categories']]
        ax.legend(handles,tally['inventory']['categories'],loc='lower left',bbox_to_anchor=(-.02,1.02),ncol=4,
                  frameon=False,fontsize=5.3,handlelength=.9,handletextpad=.35,columnspacing=.8,borderpad=0)
        inventory['main_7']=dict(stem='fig_taylor_tally',files=_save(fig,out,'fig_taylor_tally'),data='tally_data.json',
                                 implementation='Contextual Table S1 classifications transcribed with source hash; bar counts computed, not independent significance tests.')
    for name,data in [('overview_data.json',overview),('tally_data.json',tally)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest=dict(figures=inventory,source_sha256=sha(Path(__file__)),
                  notes=['Layout reconstructed from the supplied PDF, not a claim of pixel-identical historical rendering.',
                         'No original experiment reruns; overview combines explicitly different populations/cost studies.',
                         'The tally preserves the paper classifications; it does not automatically reassess their scientific support.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return inventory
