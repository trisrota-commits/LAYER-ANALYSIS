"""Single-command numerical tables plus current paper figures; never manuscript editing."""
import json
from pathlib import Path
from analysis.thesis_common import sha, require


def run(repo,out):
    repo,out=Path(repo).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing complete-paper output preserved: '+str(out))
    from analysis.revision_paper_tables import run as table_run
    from analysis.revision_paper_figures import run as figure_run
    out.mkdir(parents=True,exist_ok=False)
    tables=table_run(repo,out/'tables')
    figures=figure_run(repo,out/'tables',out/'figures')
    require(tables['numerical_tables']==66 and tables['contextual_tables']==2,'Incomplete numerical paper coverage')
    require(figures['figure_count']==19,'Incomplete current figure coverage')
    index=dict(status='complete',numerical_tables=66,contextual_tables=2,figures=19,
               table_index='tables/TABLE_INDEX.json',figure_index='figures/FIGURE_INDEX.json',
               images='figures/images',notes=[
                   'One entry point runs all numerical table workflows, then all current article/supplement figure renderers.',
                   'This builds reproducible paper artifacts, not the final LaTeX manuscript PDF.',
                   'Inspect table discrepancy flags and figure reconstruction/provenance notes before replacing manuscript assets.',
                   'No original training, predictor refitting, feature acquisition or calibration refitting.'])
    (out/'PAPER_INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    (out/'PAPER_INDEX.md').write_text('# Complete paper artifacts\n\n66 numerical tables, two contextual tables, 19 figures.\n\n'
                                    '- Tables: `tables/TABLE_INDEX.md`\n- Figures: `figures/FIGURE_INDEX.md`\n'
                                    '- Canonical image filenames: `figures/images/`\n\n'+'\n'.join(index['notes'])+'\n',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256=hashes),indent=2)+'\n',encoding='utf-8')
    print('COMPLETE PAPER ARTIFACTS: 66 numerical tables, 2 contextual tables, 19 figures;',out,flush=True)
    return index
