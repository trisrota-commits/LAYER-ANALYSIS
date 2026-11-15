"""Generate all current article/supplement figures from verified table evidence."""
import json
from pathlib import Path
import re
import shutil
import pymupdf
from analysis.thesis_common import require, sha

FIGURES = {
    'main_1':'fig_overview','main_2':'fig_primary','main_3':'fig_nb201_improve',
    'main_4':'fig_cost','main_5':'fig_zcp','main_6':'fig_strata','main_7':'fig_taylor_tally',
    'S1':'mean_dcor_all_datasets_epochs','S2':'layer_dcor_epochs123_heatmaps_with_places365',
    'S3':'layer_dcor_epoch50_heatmap_with_places365','S4':'supp_pareto_frontier',
    'S5':'fig_panels','S6':'fig_ensemble','S7':'fig_lodo','S8':'fig_lc_arms',
    'S9':'fig_lc_increment','S10':'fig_robust',
}


def checked_path(root, relative):
    p=(Path(root)/relative).resolve()
    require(p.is_relative_to(Path(root).resolve()) and p.is_file(),'Unsafe/missing artifact: '+str(relative))
    return p


def validate_tables(repo, root):
    root=Path(root)
    index=json.loads((root/'TABLE_INDEX.json').read_text(encoding='utf-8'))
    receipt=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    require(index.get('status')==receipt.get('status')=='complete','Incomplete paper table export')
    require(index.get('numerical_tables')==66 and len(index.get('inventory',[]))==68,'Unexpected table coverage')
    before={}
    for name,digest in receipt['files_sha256'].items():
        p=checked_path(root,name); require(sha(p)==digest,'Changed table artifact: '+name);before[name]=digest
    for name,digest in receipt['input_sha256'].items():
        require(sha(checked_path(repo,name))==digest,'Repository inputs differ from table export: '+name)
    before['complete.json']=sha(root/'complete.json')
    return before,index


def normalized_id(identity):
    # Producers retain archival figure IDs for provenance; the paper index uses
    # current numbering after removing the two historical pilot figures.
    supplement=re.fullmatch(r'S(\d+)',identity)
    if supplement:
        number=int(supplement[1])
        require(3<=number<=12,'Removed or unknown producer figure: '+identity)
        return f'S{number-2}'
    m=re.fullmatch(r'main_?(\d+)',identity)
    return 'main_'+m[1] if m else identity


def data_paths(entry):
    values=entry.get('data') or entry.get('datafiles')
    if isinstance(values,str):return [values]
    if isinstance(values,list):return values
    values=[entry[k] for k in ('json','csv') if k in entry]
    if values:return values
    info=entry.get('files',{}).get('data') if isinstance(entry.get('files'),dict) else None
    return [info['path']] if info else []


def run(repo, tables_root, out):
    repo,tables_root,out=Path(repo).resolve(),Path(tables_root).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing figure output preserved: '+str(out))
    require(not out.is_relative_to(tables_root),'Figure output must be separate from immutable table input')
    require(not any(out.is_relative_to(repo/p) for p in ('images','figures','analysis','evidence','tests','paper_runners')),
            'Figure output overlaps canonical source')
    before,table_index=validate_tables(repo,tables_root)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    # Register shipped fonts in this process; never delete the user's font cache.
    for font in (Path(matplotlib.get_data_path())/'fonts'/'ttf').glob('*.ttf'):
        font_manager.fontManager.addfont(str(font))
    from analysis import figures_context,figures_main_results,figures_supplement_legacy,figures_supplement_new
    modules=[('context',figures_context),('main_results',figures_main_results),
             ('supplement_legacy',figures_supplement_legacy),('supplement_new',figures_supplement_new)]
    source_hashes={str(Path(m.__file__).relative_to(repo)):sha(Path(m.__file__)) for _,m in modules}
    source_hashes['analysis/revision_paper_figures.py']=sha(Path(__file__))
    source_hashes['figures/current_paper/comparison_inventory.json']=sha(repo/'figures/current_paper/comparison_inventory.json')
    out.mkdir(parents=True,exist_ok=False);parts=out/'parts';parts.mkdir();images=out/'images';images.mkdir()
    entries={};blockers=[]
    for group,module in modules:
        print('FIGURE GROUP:',group,flush=True)
        folder=parts/group
        with matplotlib.rc_context({'text.usetex':False}):
            module.render(repo,tables_root,folder)
        manifests=[p for p in (folder/'manifest.json',folder/'inventory.json') if p.is_file()]
        require(len(manifests)==1,'Ambiguous/missing figure manifest: '+group)
        manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
        records=manifest.get('figures',manifest)
        for original_id,entry in records.items():
            identity=normalized_id(original_id)
            require(identity in FIGURES and identity not in entries,'Unexpected/duplicate figure: '+identity)
            if entry.get('status')=='blocked':
                blockers.append(dict(figure=identity,group=group,reason=entry.get('blocker')));continue
            files={}
            for ext in ('pdf','png'):
                relative=entry.get(ext)
                if relative is None:relative=entry['files'][ext]['path']
                source=checked_path(folder,relative)
                target=images/(FIGURES[identity]+'.'+ext)
                require(not target.exists(),'Duplicate canonical figure filename')
                shutil.copyfile(source,target)
                require(sha(source)==sha(target),'Figure copy changed bytes')
                files[ext]=dict(path=target.relative_to(out).as_posix(),sha256=sha(target))
            with pymupdf.open(out/files['pdf']['path']) as doc:
                require(len(doc)==1 and bool(doc[0].get_text().strip()),'Empty/multipage figure: '+identity)
            data=data_paths(entry)
            require(bool(data),'Missing plot-coordinate artifact: '+identity)
            coords=[]
            for relative in data:
                p=checked_path(folder,relative)
                if p.suffix=='.json':require(bool(json.loads(p.read_text(encoding='utf-8'))),'Empty plot data')
                coords.append(dict(path=p.relative_to(out).as_posix(),sha256=sha(p)))
            entries[identity]=dict(stem=FIGURES[identity],files=files,plot_data=coords,
                                   group=group,producer_figure_id=original_id,
                                   producer_manifest=manifests[0].relative_to(out).as_posix(),
                                   notes=entry.get('notes',entry.get('implementation','See producer manifest/provenance.')))
    if blockers:
        (out/'BLOCKED.json').write_text(json.dumps(blockers,indent=2)+'\n',encoding='utf-8')
        raise ValueError('Figure coverage blocked: '+json.dumps(blockers))
    require(set(entries)==set(FIGURES),'Incomplete current figure coverage: '+str(set(FIGURES)-set(entries)))
    entries={identity:entries[identity] for identity in FIGURES}
    for name,digest in before.items():require(sha(checked_path(tables_root,name))==digest,'Table input changed during rendering')
    for name,digest in source_hashes.items():require(sha(repo/name)==digest,'Figure source changed during rendering')
    report=dict(status='complete',figures=entries,figure_count=len(entries),
                table_index_sha256=sha(tables_root/'TABLE_INDEX.json'),source_sha256=source_hashes,
                source_table_flags=table_index.get('flags',[]),
                reference_documents={'main':'project_final_4_-20.pdf','supplement':'project_final_4_-21.pdf'},
                notes=['Current seven article and ten supplement figures; 17 figures total.',
                       'Original/reconstructed rendering and exact plotted data are recorded per producer. No claim of pixel-identical historic PDFs.',
                       'Historical pilot figures S1/S2 are removed and not executed. Original supplement S3-S12 map to current S1-S10.',
                       'Producer manifests retain archival IDs for traceability; this index is the current paper numbering. Table numbering is unchanged.',
                       'Main 7 uses the paper Table S1 qualitative classifications, not independent statistical-test counts.',
                       'No training, acquisition, fitting or input modification by this figure command.'])
    (out/'FIGURE_INDEX.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Current paper figures','',*report['notes'],'','| Figure | PDF | PNG | Producer |','|---|---|---|---|']
    lines += [f"| {i} | {e['files']['pdf']['path']} | {e['files']['png']['path']} | {e['group']} |" for i,e in entries.items()]
    (out/'FIGURE_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',figures=len(entries),files_sha256=hashes),indent=2)+'\n',encoding='utf-8')
    print('COMPLETE CURRENT FIGURES:',len(entries),out,flush=True)
    return report
