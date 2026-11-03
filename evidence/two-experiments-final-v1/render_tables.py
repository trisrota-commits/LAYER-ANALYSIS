import json
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text())
d={(x['model'],x['metric']):x for x in r['descriptive']}
b={(x['candidate'],x['reference'],x['metric']):x for x in r['paired']}
metrics=['RMSE','MAE','Pearson','Spearman','R2']
labels={'RMSE':r'RMSE $\downarrow$','MAE':r'MAE $\downarrow$','Pearson':r'Pearson $\uparrow$','Spearman':r'Spearman $\uparrow$','R2':r'$R^2$ $\uparrow$'}
text=[r'''% Add these packages to the PREAMBLE if not already loaded:
% \usepackage{booktabs,graphicx}

% Define once before the tables. Each cell is estimate [lower, upper].
\providecommand{\expci}[3]{}
\renewcommand{\expci}[3]{%
  \raisebox{-0.5\height}{%
    \shortstack{\ensuremath{#1}\\{\tiny\ensuremath{[#2,}}\\{\tiny\ensuremath{#3]}}}}%
}
''']

def table(caption,label,heads,cols,paired=False,candidate=None,note=''):
    text.extend([r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.15}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{@{}l'+'c'*len(cols)+'@{}}',r'\toprule','Metric & '+' & '.join(heads)+r' \\',r'\midrule'])
    for k in metrics:
        best=None if paired else (min if k in ['RMSE','MAE'] else max)(d[x,k]['mean'] for x in cols)
        cells=[]
        for col in cols:
            row=b[candidate,col,k] if paired else d[col,k]
            value=row['improvement'] if paired else row['mean']
            s=f'{value:.4f}'
            if not paired and round(value,4)==round(best,4):s=r'\mathbf{'+s+'}'
            cells.append(r'\expci{'+s+'}{'+f'{row["low"]:.4f}'+'}{'+f'{row["high"]:.4f}'+'}')
        text.append((k if paired and k!='R2' else r'$R^2$' if paired else labels[k])+' & '+' & '.join(cells)+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\par\smallskip',r'\begin{minipage}{\linewidth}',r'\scriptsize '+note,r'\end{minipage}',r'\end{table}',''])

table('Classical regressors and the Qwen reference on the same 115 test records. Cells show means and 95\% seed-level Student-$t$ intervals across seeds 42--47 ($df=5$). Bold indicates the best displayed mean per metric.',
      'tab:classical-eight-feature-results',['Ridge','Random forest','HistGB','Qwen'],['ridge','random_forest','hist_gradient_boosting','qwen_reference'],
      note=r'HistGB denotes histogram gradient boosting. Classical inputs are accuracies and layer-averaged Taylor values from epochs 1--3, log parameter count, and depth. The existing Qwen Taylor/ZCP-OFF reference omits these two static fields but includes task/dataset/metric and training-budget metadata; the feature sets are therefore not strictly identical. All models use the 978/58/115 split and a final refit on 1,036 training-plus-validation records. Classical preprocessing is fitted on training records only; hyperparameters are selected using validation RMSE. Ridge and HistGB produced identical predictions across seeds, yielding degenerate seed intervals.')

table('Paired-bootstrap improvements of the Qwen reference over classical regressors. Cells show the mean per-seed difference and its unadjusted 95\% percentile interval from 20,000 paired test-record resamples.',
      'tab:classical-eight-feature-paired',['vs. Ridge','vs. Random forest','vs. HistGB'],['ridge','random_forest','hist_gradient_boosting'],True,'qwen_reference',
      note=r'Positive values favor Qwen: reference minus Qwen for RMSE/MAE, and Qwen minus reference for Pearson/Spearman/$R^2$. The same 115-record resample is applied to every model and seed; metrics are computed per seed and then averaged. Seeds 42--47 remain fixed; predictions are not ensembled. These are test-record intervals, not seed-level intervals. All intervals include zero after a separate Bonferroni adjustment across the 15 contrasts (adjusted intervals not shown). Input differences are described in Table~\ref{tab:classical-eight-feature-results}.')

table('Taylor checkpoint ablation: six-seed means and 95\% seed-level Student-$t$ intervals ($df=5$) on the same 115 test records. All conditions retain accuracy observations from epochs 1--3 with ZCPs OFF. Bold indicates the best displayed mean per metric.',
      'tab:taylor-epoch1-results',[r'No layer',r'Taylor epoch 1',r'Taylor epochs 1--3'],['no_layer','epoch1','qwen_reference'],
      note=r'All conditions use seeds 42--47, the fixed 978/58/115 split, and seven-epoch final Qwen refits on 1,036 training-plus-validation records. Epoch 1 refers only to the Taylor observation checkpoint, not the Qwen training budget. The epoch-1 prompts remove only Taylor rows 2 and 3, preserving other prompt content, targets, record order, and missing values. No-layer and three-checkpoint results reuse existing matched reference predictions.')

table('Paired-bootstrap improvements of epoch-1 Taylor relative to the three-checkpoint Taylor and no-layer references. Cells show the mean per-seed difference and its unadjusted 95\% percentile interval from 20,000 paired test-record resamples.',
      'tab:taylor-epoch1-paired',[r'vs. Taylor epochs 1--3',r'vs. No layer'],['qwen_reference','no_layer'],True,'epoch1',
      note=r'Positive values favor epoch-1 Taylor: reference minus epoch-1 Taylor for RMSE/MAE, and epoch-1 Taylor minus reference for Pearson/Spearman/$R^2$. All conditions retain three early accuracies and ZCPs OFF. Resampling uses the same 115 record IDs across conditions and six fixed fitted seeds; per-seed metrics are averaged without ensembling. The RMSE and $R^2$ intervals favor three-checkpoint Taylor before correction; every interval includes zero after a separate Bonferroni adjustment across the 10 contrasts (adjusted intervals not shown). Inconclusive differences do not establish equivalence.')

content='\n'.join(text)
assert content.count(r'\begin{table}')==content.count(r'\end{table}')==4
assert len(set(__import__('re').findall(r'\\label\{([^}]+)\}',content)))==4
with (p/'four_tables.tex').open('x',encoding='utf-8',newline='') as f:f.write(content+'\n')
print(content)
