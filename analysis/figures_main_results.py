"""Current article figures 2–6 and supplement S7–S9, prediction-only.

render(repo, tables_root, out) writes an exclusively new directory. tables_root
is the verified paper export (the directory containing tables/ and workflows/).
Original functions are AST-extracted, never imported with their stale registry.
"""
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b1'
FEATURES = ['no-layer', 'weightwatcher', 'cka', 'effective-rank', 'taylor',
            'gradient-flow', 'sensitivity', 'activation', 'all-seven']
NICE = dict(zip(FEATURES, ['No layer', 'WeightWatcher', 'CKA', 'Effective rank',
                         'Taylor', 'Gradient flow', 'Sensitivity', 'Dead-act. fraction', 'All seven']))
DATASETS = {'cifar-100': 'CIFAR-100', 'cifar-10': 'CIFAR-10', 'svhn': 'SVHN',
            'mnist': 'MNIST', 'imagenette': 'ImageNette',
            'celeba-gender': 'CelebA-Gender', 'places365': 'Places365'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensemble_points(predictions, target, feature):
    """The original figure's sqrt(E_subset MSE), NOT E_subset RMSE."""
    p, g = np.asarray(predictions), np.asarray(target)
    rows = []
    for m in range(1, len(p) + 1):
        v = [float(np.mean((p[list(c)].mean(0) - g) ** 2))
             for c in itertools.combinations(range(len(p)), m)]
        rows.append(dict(kind='point', feature=feature, x=m, y=float(np.sqrt(np.mean(v))),
                         mean_mse=float(np.mean(v)), mean_subset_rmse=float(np.mean(np.sqrt(v))),
                         subsets=len(v), n=len(g)))
    return rows


def ranked_regret(prediction, target, ks):
    order = np.argsort(-np.asarray(prediction), kind='stable')
    return [float(np.max(target) - np.max(np.asarray(target)[order[:k]])) for k in ks]


def original_functions(repo, filename, names, namespace):
    """Allowlisted definitions only; discard original save/close and stale arrows."""
    path = Path(repo) / 'paper_runners/article_analysis/original' / filename
    tree = ast.parse(path.read_bytes())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in tree.body} != set(names):
        raise ValueError('Missing original figure function')
    for node in tree.body:
        node.body = [s for s in node.body if not (
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and ast.unparse(s.value.func) in {'fig.savefig', 'plt.close'})
            and not (node.name == 'ensemble' and 106 <= s.lineno <= 113)]
        if node.name in {'ensemble', 'strata'}:
            node.body.append(ast.Return(value=ast.Name(id='fig', ctx=ast.Load())))
    # Keep offset-map keys and visible labels in sync with current terminology.
    for item in ast.walk(tree):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            item.value = {'Activation': 'Dead-act. fraction', 'Eff. rank': 'Effective rank',
                          'Grad. flow': 'Gradient flow'}.get(item.value, item.value)
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def check_pdf(path, labels):
    import pymupdf
    with pymupdf.open(path) as doc:
        if len(doc) != 1:
            raise ValueError('Figure must be one page')
        page = doc[0]
        text = page.get_text()
        missing = [label for label in labels if label not in text]
        clipped = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if min(x0, y0) < -0.1 or x1 > page.rect.width + .1 or y1 > page.rect.height + .1:
                        clipped.append(span['text'])
        if missing or clipped:
            raise ValueError(f'PDF validation failed: missing={missing}, clipped={clipped}')
    return dict(missing_labels=missing, clipped_text=clipped, pages=1)


def render(repo, tables_root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from analysis import revision_article as article

    repo, root, out = Path(repo).resolve(), Path(tables_root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Existing output preserved: {out}')
    # Canonical loaders validate targets and ordered prediction identities.
    inputs = article.verify_inputs(repo)
    article.verify_sources(repo)
    cond, prediction_sources, keys = article.load_conditions(repo)
    source_hashes = {str(repo / p): digest for p, digest in inputs.items()}
    for name in ['figures.py', 'figures_panel.py', 'primary_endpoint.py']:
        p = repo / 'paper_runners/article_analysis/original' / name
        source_hashes[str(p)] = sha(p)
    source_hashes[str(Path(__file__).resolve())] = sha(__file__)

    def table(name):
        p = root / 'tables' / (name + '.json')
        source_hashes[str(p)] = sha(p)
        return json.loads(p.read_text())

    descriptive = table('main_IV')
    timings = table('S16')
    toggle = table('S27')
    heldout = table('S36')
    paired = table('S37')
    nb_primary, nb_extra = table('main_III'), table('S32')
    primary_file = root / 'workflows/article/results.json'
    source_hashes[str(primary_file)] = sha(primary_file)
    primary = json.loads(primary_file.read_text())['primary_endpoint']
    out.mkdir(parents=True, exist_ok=False)
    inventory = dict(figures={}, source_sha256=source_hashes, prediction_sources=prediction_sources,
                     reference_documents=['project_final_4_-20.pdf (main pp.5–7)',
                                          'project_final_4_-21.pdf (supplement pp.21,27)'],
                     notes='Reconstructed vector figures except audited original-function reuse noted per figure. No training, acquisition or stale registry execution.')

    def arrays(feature):
        d = cond[3, 'off', feature]
        seeds = sorted(d)
        return np.array([d[s][1] for s in seeds]), np.asarray(d[seeds[0]][2])

    def rmse(k, seeds=None):
        d = cond[k]
        return float(np.mean([np.sqrt(np.mean((np.asarray(d[s][1]) - d[s][2]) ** 2))
                              for s in (sorted(d) if seeds is None else seeds)]))

    def tidy(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.25, linewidth=.5)
        ax.set_axisbelow(True)

    def save(fig, identity, stem, points, notes, labels):
        fig.canvas.draw()
        paths = [stem + ext for ext in ('.pdf', '.png', '.json', '.csv')]
        fig.savefig(out / paths[0], bbox_inches='tight', pad_inches=.12)
        fig.savefig(out / paths[1], bbox_inches='tight', pad_inches=.12, dpi=220)
        plt.close(fig)
        receipt = check_pdf(out / paths[0], labels)
        with (out / paths[2]).open('x', encoding='utf-8') as f:
            json.dump(dict(figure=identity, points=points, notes=notes,
                           source_sha256=source_hashes), f, indent=2, allow_nan=False)
        fields = list(dict.fromkeys(k for row in points for k in row))
        with (out / paths[3]).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in points)
        inventory['figures'][identity] = dict(files=paths, pdf=paths[0], png=paths[1],
                                             data=paths[2:4], notes=notes, pdf_check=receipt)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'axes.linewidth': .6, 'savefig.bbox': 'tight'}):
        # Main 2: record-paired t intervals separately in each seed and pooled.
        pn, y = arrays('no-layer')
        pt, _ = arrays('taylor')
        mask = y >= 20
        gains = (abs(pn - y) - abs(pt - y))[:, mask]
        if gains.shape != (6, 106):
            raise ValueError('Primary population differs from 6 seeds x 106 records')
        rows = []
        for label, g in list(zip([f'Seed {s}' for s in range(42, 48)], gains)) + [('Pooled (106 rec.)', gains.mean(0))]:
            low, high = stats.t.interval(.95, len(g)-1, loc=g.mean(), scale=stats.sem(g))
            rows.append(dict(label=label, estimate=float(g.mean()), low=float(low), high=float(high),
                             n=len(g), population='target>=20', interval='record-paired t95, unadjusted'))
        np.testing.assert_allclose([rows[-1]['estimate'], rows[-1]['low'], rows[-1]['high']],
                                   [primary['primary_by_record']['gain'], *primary['primary_by_record']['ci']], atol=1e-12)
        fig, ax = plt.subplots(figsize=(5.5, 3.4), layout='constrained')
        for i, r in enumerate(rows):
            ax.plot([r['low'], r['high']], [i, i], color=BLUE, lw=2)
            ax.plot(r['estimate'], i, 'D' if i == 6 else 'o', color=ORANGE if i == 6 else BLUE)
        ax.axvline(0, color=MUTED, ls='--')
        ax.set_yticks(range(7), [r['label'] for r in rows]); ax.set_ylim(7.1, -.5)
        pooled=rows[-1]
        ax.text(pooled['estimate'], 6.65,
                f"{pooled['estimate']:.3f} [{pooled['low']:.3f}, {pooled['high']:.3f}], p = {primary['primary_by_record']['p']:.4f}",
                ha='center', fontsize=8)
        ax.set_xlabel('MAE reduction (accuracy points)'); ax.set_title('Taylor better →', loc='right')
        tidy(ax)
        save(fig, 'main2', 'fig_primary', rows,
             'Reconstruction of current Fig.2: per-seed and pooled record-paired t95 intervals. Holm correction applies to the primary two-test p-value family, not these displayed intervals.',
             ['Seed 42', 'Seed 47', 'Pooled (106 rec.)', 'MAE reduction'])

        # Main 3: distinct primary/supporting populations and both interval layers.
        rows = []
        names = {'cifar10': 'CIFAR-10-valid', 'cifar100': 'CIFAR-100', 'imagenet16': 'ImageNet16-120'}
        datasets = list(names.values())
        for r in nb_primary:
            rows.append(dict(dataset=names[r['dataset']], metric='MAE*', estimate=r['improvement'],
                             low=r['record_ci95_low'], high=r['record_ci95_high'], adjusted_low=None,
                             adjusted_high=None, n=r['n'], population='target>=20', interval='record-paired t95'))
        for r in nb_extra:
            if r['n'] != 275 or r['population'] != 'all':
                raise ValueError('NB201 supporting population changed')
            rows.append(dict(dataset=r['dataset'], metric=r['metric'], estimate=r['improvement'],
                             low=r['ci95_low'], high=r['ci95_high'], adjusted_low=r['bonferroni12_low'],
                             adjusted_high=r['bonferroni12_high'], n=r['n'], population='all275',
                             interval='operation-group bootstrap 10000; Bonferroni12'))
        fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.7), sharey=True)
        fig.subplots_adjust(left=.17, bottom=.35, right=.98, top=.85, wspace=.35)
        for ax, metric in zip(axes, ['MAE*', 'RMSE', 'Pearson', 'Spearman', 'R2']):
            ax.axvline(0, color=MUTED, ls='--', lw=.7)
            for i, dataset in enumerate(datasets):
                r = next(r for r in rows if r['metric'] == metric and r['dataset'] == dataset)
                if r['adjusted_low'] is not None:
                    ax.plot([r['adjusted_low'], r['adjusted_high']], [i, i], color='#9aafc2', lw=1.5)
                ax.plot([r['low'], r['high']], [i, i], color=BLUE, lw=3)
                point, = ax.plot(r['estimate'], i, 'o', color=INK, ms=4)
                np.testing.assert_array_equal(point.get_xdata(), [r['estimate']])
            ax.set_title(metric); ax.set_yticks(range(3)); ax.set_ylim(2.5, -.5); tidy(ax)
        axes[0].set_yticklabels(datasets)
        fig.text(.59, .22, 'Improvement from Taylor (positive favours Taylor)', ha='center')
        fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=3, label='95% paired interval'),
                            Line2D([0], [0], color='#9aafc2', lw=1.5, label='Bonferroni-adjusted (12; non-MAE metrics)')],
                   loc='lower center', ncol=2, frameon=False)
        save(fig, 'main3', 'fig_nb201_improve', rows,
             'Reconstruction of checked 15-point forest; current main_III and S32 values. MAE primary record t95; four other metrics all275 operation-group bootstrap and Bonferroni12. No substitution of Holm p corrections for CI corrections.',
             datasets + ['MAE*', 'RMSE', 'Pearson', 'Spearman', 'Bonferroni-adjusted'])

        operations = dict(zip(FEATURES, ['no-layer', 'weightwatcher', 'cka_similarity', 'effective_rank',
                                        'taylor_importance', 'gradient_flow', 'sensitivity', 'activation_statistics', 'all_seven']))
        costs = {f: next(r['seconds'] for r in timings if r['operation'] == op) for f, op in operations.items()}
        means = {r['feature']: r['mean'] for r in descriptive if r['metric'] == 'RMSE'}
        for f in FEATURES:
            np.testing.assert_allclose(means[f], rmse((3, 'off', f)), atol=1e-12)
        rows = [dict(feature=f, label=NICE[f], x=costs[f], y=means[f], seconds=costs[f],
                     n_prediction=115, n_timing=96, seeds=list(range(42,48))) for f in FEATURES[1:]]
        fig, ax = plt.subplots(figsize=(6.6, 3.8), layout='constrained')
        ax.axhline(means['no-layer'], color=MUTED, ls='--', label='No layer')
        offsets = {'activation': (0, -19), 'gradient-flow': (4, 12), 'effective-rank': (6, -14),
                   'taylor': (8, -13), 'weightwatcher': (5, -15), 'cka': (-5, -15),
                   'sensitivity': (0, 12), 'all-seven': (-8, 12)}
        for r in rows:
            ax.scatter(r['x'], r['y'], color=ORANGE if r['feature']=='taylor' else BLUE, zorder=3)
            dx, dy = offsets[r['feature']]
            ax.annotate(r['label'], (r['x'], r['y']), xytext=(dx,dy), textcoords='offset points',
                        fontsize=8, ha='right' if dx < 0 else 'left')
        ax.set_xscale('log'); ax.set_xlim(.26, 40); ax.margins(y=.22)
        ax.set_xlabel('Extraction time per checkpoint (s, log scale)'); ax.set_ylabel('RMSE (accuracy points)')
        ax.legend(frameon=False); tidy(ax)
        rows.append(dict(feature='no-layer', label='No layer', x=None, y=means['no-layer'], seconds=0., kind='horizontal reference'))
        save(fig, 'main4', 'fig_cost', rows, 'Reconstruction. Measured full-precision S16 per-checkpoint costs (96 candidates); main_IV mean per-seed RMSE (115 records). Zero-cost baseline is a horizontal reference, never log(0).', ['Taylor', 'Dead-act. fraction', 'No layer', 'Extraction time'])

        rows = []
        for f in FEATURES[:-1]:
            off = rmse((3, 'off', f)); on = rmse((3, 'on', f))
            rows.append(dict(feature=f, label=NICE[f], x=means['no-layer']-off, y=off-on,
                             off_rmse=off, on_rmse=on, seeds=list(range(42,48)), n=115))
        fig, ax = plt.subplots(figsize=(6.5, 4.), layout='constrained')
        x, v = np.array([r['x'] for r in rows]), np.array([r['y'] for r in rows])
        m, b = np.polyfit(x, v, 1); xx = np.array([-.2, .72]); ax.plot(xx, m*xx+b, color=MUTED)
        ax.axhline(0, color=MUTED, lw=.7); ax.axvline(0, color=MUTED, lw=.7)
        offsets = {'no-layer': (-10, 8), 'cka': (-7, -13), 'effective-rank': (-8,-14),
                   'taylor': (0, 10), 'gradient-flow': (7,0), 'weightwatcher': (7,5),
                   'sensitivity': (7,-10), 'activation': (7,4)}
        for r in rows:
            ax.scatter(r['x'], r['y'], color=ORANGE if r['y']>0 else BLUE, marker='o' if r['y']>0 else 's')
            dx,dy=offsets[r['feature']]
            ax.annotate(r['label'], (r['x'], r['y']), xytext=(dx,dy), textcoords='offset points',
                        ha='right' if dx<0 else 'left', fontsize=8)
        ax.set_xlim(-.22,.76); ax.set_ylim(-.70,.50)
        ax.set_xlabel('Standalone benefit of the layer statistic (RMSE points)'); ax.set_ylabel('Benefit of adding ZCPs')
        ax.legend(handles=[Line2D([0],[0],marker='s',ls='',color=BLUE,label='ZCPs hurt'),
                           Line2D([0],[0],marker='o',ls='',color=ORANGE,label='ZCPs help')], frameon=False,loc='lower left')
        tidy(ax)
        save(fig, 'main5', 'fig_zcp', rows, 'Reconstruction using canonical six-seed predictions, not rounded make_fig2.py constants. Descriptive fitted line only; no r/p annotation. Corrected Dead-act. fraction label.', ['Dead-act. fraction', 'Taylor', 'Benefit of adding ZCPs'])

        ns = dict(np=np, itertools=itertools, plt=plt, stats=stats, cond=cond, COL=5.0,
                  BLUE=BLUE, ORANGE=ORANGE, AQUA=AQUA, YELLOW=YELLOW, MAGENTA=MAGENTA,
                  INK=INK, INK2=INK2, MUTED=MUTED)
        original_functions(repo, 'figures.py', ['tidy', 'strata', 'ensemble'], ns)
        fig = ns['strata']()
        mu = pt.mean(0)
        rows = [dict(idx=int(k[0]), dataset=k[1], x=float(g), y=float(p), collapsed=bool(g<20), seeds=6)
                for k,g,p in zip(keys,y,mu)]
        for collection, d in zip(fig.axes[0].collections[:5], ['cifar-100','svhn','mnist','imagenette','celeba-gender']):
            expected=np.array([[r['x'],r['y']] for r in rows if r['dataset']==d])
            np.testing.assert_allclose(collection.get_offsets(), expected)
        save(fig, 'main6', 'fig_strata', rows, 'Original figures.py strata() reused via AST-only extraction, canonical validated conditions injected; same six-seed ensemble and 9 ringed target<20 records.', ['CIFAR-100 (88)', 'SVHN (13)', 'True best accuracy', 'Predicted best accuracy'])

        # S7 original panel functions; only cost/LODO constants replaced with tables.
        order = ['svhn','imagenette','cifar-10','cifar-100','celeba-gender','mnist','places365']
        lodo=[]
        for d in order:
            a=next(r for r in heldout if r['dataset']==d and r['metric']=='RMSE' and r['condition']=='no-layer')
            b=next(r for r in heldout if r['dataset']==d and r['metric']=='RMSE' and r['condition']=='taylor')
            lodo.append((DATASETS[d],a['records'],a['mean'],b['mean']))
        ns.update(COST=costs, NICE=NICE, LODO=lodo)
        def five(p,g):
            return {'RMSE': float(np.sqrt(np.mean((np.asarray(p)-g)**2)))}
        ns['five']=five
        original_functions(repo, 'figures_panel.py', ['tidy','rmse','panel_cost','panel_zcp','panel_lodo'], ns)
        fig, axes=plt.subplots(2,2,figsize=(11.8,7.4))
        fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.94,wspace=.47,hspace=.53)
        ns['panel_cost'](axes[0,0]); ns['panel_zcp'](axes[0,1]); ns['panel_lodo'](axes[1,1])
        # Replace the original rounded training reference with the measured mean.
        training=next(r['seconds'] for r in timings if r['operation']=='training')
        axes[0,0].lines[0].set_xdata([training,training])
        axes[0,0].texts[-1].set_position((training,max(means.values())))
        rows=[dict(panel='a',feature=f,label=NICE[f],x=max(costs[f],.12),y=means[f],seconds=costs[f],
                   zero_display_surrogate=f=='no-layer') for f in FEATURES]
        rows.append(dict(panel='a',kind='training_reference',x=training,y=None,seconds=training))
        for n in (2,3):
            for i,f in enumerate(FEATURES):
                ss=sorted(set(cond[n,'off',f]) & set(cond[n,'on',f]))
                delta=rmse((n,'off',f),ss)-rmse((n,'on',f),ss)
                expected=next(r['delta_RMSE'] for r in toggle if r['observations']==n and r['feature']==f)
                np.testing.assert_allclose(delta,expected,atol=1e-12)
                rows.append(dict(panel='b',feature=f,observations=n,x=delta,y=i+(-.19 if n==2 else .19),seeds=ss,n=115))
        ds=np.array([k[1] for k in keys]); mask=ds=='cifar-100'; gg=y[mask]; ks=np.arange(1,21)
        rng=np.random.default_rng(0)
        random=[float(np.mean([gg.max()-gg[rng.choice(len(gg),int(k),replace=False)].max() for _ in range(4000)])) for k in ks]
        ax=axes[1,0]; tidy(ax)
        for f, vals, col in [('random choice',random,MUTED),('taylor',ranked_regret(pt.mean(0)[mask],gg,ks),BLUE),('no-layer',ranked_regret(pn.mean(0)[mask],gg,ks),ORANGE)]:
            ax.plot(ks,vals,color=col,label=NICE.get(f,f))
            rows.extend(dict(panel='c',feature=f,x=int(k),y=v,n=len(gg),seeds=6,
                             random_draws=4000 if f=='random choice' else None,ranking='stable descending ensemble prediction') for k,v in zip(ks,vals))
        ax.set_xlabel('Architectures trained to completion, k'); ax.set_ylabel('Regret (accuracy points)')
        ax.set_title('(c) Architecture selection on CIFAR-100',loc='left'); ax.set_ylim(bottom=0); ax.legend(frameon=False)
        rows.extend(dict(panel='d',dataset=d,label=lab,n=n,x=a-b,y=i,no_layer=a,taylor=b)
                    for i,(d,(lab,n,a,b)) in enumerate(zip(order,lodo)))
        save(fig,'S7','fig_panels',rows,'Original panel_cost, panel_zcp and panel_lodo reused with full-precision S16/S36 injected. The no-layer cost is zero, displayed at 0.12 on the original log axis (explicit surrogate). Panel c reconstructed with stable ties, six-seed ensemble rankings, RNG default_rng(0), 4000 no-replacement random draws per k; not the per-seed selection-table estimand.', ['Dead-act. fraction','CIFAR-100','Places365','Architecture selection'])

        # S8: original subset-MSE curves; separate mean-RMSE comparison arrows.
        rows=[]
        for f in ['taylor','no-layer']:
            p,g=arrays(f); points=ensemble_points(p,g,f); rows.extend(points)
            ms=np.array([r['x'] for r in points]); mse=np.array([r['mean_mse'] for r in points])
            a,b=np.linalg.lstsq(np.vstack([np.ones_like(ms,dtype=float),1./ms]).T,mse,rcond=None)[0]
            rows.extend(dict(kind='curve',feature=f,x=float(v),y=float(np.sqrt(a+b/v)),a=float(a),b=float(b)) for v in np.linspace(1,6,100))
        fig=ns['ensemble'](); ax=fig.axes[0]
        for line,f in zip([line for line in ax.lines if line.get_marker()=='o'], ['taylor','no-layer']):
            expected=[r['y'] for r in rows if r['feature']==f and r['kind']=='point']
            np.testing.assert_allclose(line.get_ydata(),expected,atol=1e-12)
        t1=next(r['mean_subset_rmse'] for r in rows if r['kind']=='point' and r['feature']=='taylor' and r['x']==1)
        t6=next(r['y'] for r in rows if r['kind']=='point' and r['feature']=='taylor' and r['x']==6)
        n1=next(r['mean_subset_rmse'] for r in rows if r['kind']=='point' and r['feature']=='no-layer' and r['x']==1)
        for label,xpos,low,high in [('input effect',1.25,t1,n1),('seed effect',6.35,t6,t1)]:
            ax.annotate('',xy=(xpos,low),xytext=(xpos,high),arrowprops=dict(arrowstyle='<->',color=INK2,lw=.8))
            ax.text(xpos+.12,(low+high)/2,label.replace(' ','\n')+f'\n{high-low:.2f}',fontsize=7,va='center')
            rows.append(dict(kind='effect',feature=label,x=xpos,y=low,high=high,effect=high-low,
                             estimand='Arithmetic mean per-seed RMSE comparison; seed effect versus full ensemble RMSE'))
        ax.set_xlim(.7,7.35)
        ax.set_title('Points: square root of mean subset MSE\nArrows: mean-RMSE comparisons',fontsize=8)
        save(fig,'S8','fig_ensemble',rows,'Original ensemble() subset enumeration and sqrt(a+b/m) fit reused. Arrow effects retain the paper arithmetic mean per-seed RMSE comparison (input effect 0.62, seed effect 0.43), computed from predictions rather than hardcoded. Curve and arrow estimands are explicitly distinguished and both exported.', ['Taylor','No layer','Number of seeded predictors averaged','RMSE'])

        # S9: paper display limits with explicit arrows; full intervals in data.
        order=['imagenette','svhn','cifar-10','cifar-100','celeba-gender','mnist','places365']
        rows=[]
        fig,axes=plt.subplots(1,2,figsize=(9,4.2),sharey=True,layout='constrained')
        for ax,metric in zip(axes,['MAE','RMSE']):
            for i,d in enumerate(order):
                r=next(r for r in paired if r['dataset']==d and r['metric']==metric)
                row=dict(dataset=d,label=DATASETS[d],metric=metric,estimate=r['improvement'],low=r['low'],high=r['high'],
                         n=r['records'],groups=r['architecture_groups'],interval='95% architecture-grouped paired bootstrap, unadjusted',
                         display_xlim=[-5.,3.],interval_truncated=r['low'] < -5 or r['high'] > 3)
                rows.append(row)
                supported=r['low']>0 or r['high']<0
                col=BLUE if r['low']>0 else ORANGE if r['high']<0 else MUTED
                ax.plot([r['low'],r['high']],[i,i],color=col,lw=1.5)
                ax.plot(r['improvement'],i,'o',markerfacecolor=col if supported else 'white',markeredgecolor=col)
                if r['low'] < -5:
                    ax.annotate('',xy=(-4.96,i),xytext=(-4.45,i),arrowprops=dict(arrowstyle='->',color=col,lw=1.2))
                if r['high'] > 3:
                    ax.annotate('',xy=(2.96,i),xytext=(2.45,i),arrowprops=dict(arrowstyle='->',color=col,lw=1.2))
                if not -5 <= r['improvement'] <= 3:
                    ax.text(-3.5,i-.22,f"{r['improvement']:.1f}",fontsize=7,color=INK2)
            ax.axvline(0,color=MUTED,ls='--'); ax.set_title(metric); ax.set_xlabel('Improvement (accuracy points)')
            ax.set_yticks(range(7)); ax.set_ylim(6.5,-.5); ax.set_xlim(-5,3); tidy(ax)
        axes[0].set_yticklabels([f"{DATASETS[d]} ({next(r['n'] for r in rows if r['dataset']==d)})" for d in order])
        save(fig,'S9','fig_lodo',rows,'Reconstruction from S37 paired intervals (not descriptive S36 seed intervals). Filled markers exclude zero. Paper display range [-5,3] retained; arrows show interval truncation and off-axis estimates are labelled. Full untruncated intervals remain in plot data.', ['MAE','RMSE','Places365 (5)','Improvement'])

    for p,digest in source_hashes.items():
        if sha(p)!=digest:
            raise ValueError('Source changed during rendering: '+p)
    inventory['outputs_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    with (out/'manifest.json').open('x',encoding='utf-8') as f:
        json.dump(inventory,f,indent=2,allow_nan=False)
    return inventory


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tables-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=render(args.repo,args.tables_root,args.out)
    print(f"Rendered {len(result['figures'])} figures into {args.out}")
