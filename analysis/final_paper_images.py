"""Map verified rendered figures to the supplied final manuscript's asset names.

No numerical recalculation or plot changes; the legacy image export stays intact.
"""
import argparse
import json
from pathlib import Path
import shutil
from analysis.thesis_common import require, sha

# (final placement, legacy placement, required TeX filename stem).
PLACEMENTS = [
    ('main_1','main_1','fig_overview'),('main_2','main_3','fig_nb201_improve'),
    ('main_3','main_2','fig_primary'),('main_4','main_4','fig_cost'),
    ('main_5','main_5','fig_zcp'),('main_6','main_6','fig_strata'),
    ('S1','S1','fig_dcor'),('S2','S2','fig_dcor_heat_epochs123'),
    ('S3','S3','fig_dcor_heat_epoch50'),('S4','S4','supp_pareto_frontier'),
    ('S5','main_6','fig_strata'),('S6','S5','fig_panels'),('S7','S6','fig_ensemble'),
    ('S8','S7','fig_lodo'),('S9','S8','fig_lc_arms'),('S10','S9','fig_lc_increment'),
    ('S11','S10','fig_robust'),
]


def checked(root, name, digest):
    p=(root/name).resolve()
    require(p.is_relative_to(root) and p.is_file() and sha(p)==digest,
            'Missing/unsafe/changed rendered artifact: '+str(name))
    return p


def export(inputs, out):
    inputs,out=Path(inputs).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing output preserved: '+str(out))
    require(not out.is_relative_to(inputs),'Output must be separate from rendered inputs')
    receipt=json.loads((inputs/'complete.json').read_text(encoding='utf-8'))
    index=json.loads((inputs/'FIGURE_INDEX.json').read_text(encoding='utf-8'))
    require(receipt.get('status')==index.get('status')=='complete','Incomplete image export')
    hashes=receipt['files_sha256']
    require('FIGURE_INDEX.json' in hashes,'Figure index not bound to receipt')
    for name,h in hashes.items():checked(inputs,name,h)
    figures=index['figures'];sources={}
    for placement,old,stem in PLACEMENTS:
        require(old in figures,'Missing original figure '+old)
        for ext in ('pdf','png'):
            item=figures[old]['files'][ext]
            require(hashes.get(item['path'])==item['sha256'],'Figure hash not bound to receipt')
            sources[stem+'.'+ext]=checked(inputs,item['path'],item['sha256'])
    require(len(sources)==32,'Unexpected final asset inventory')
    out.mkdir(parents=True,exist_ok=False);images=out/'images';images.mkdir()
    for name,p in sources.items():
        shutil.copyfile(p,images/name)
        require(sha(p)==sha(images/name),'Image copy changed bytes')
    records={placement:dict(stem=stem,legacy_figure=old,
                           files={ext:dict(path='images/'+stem+'.'+ext,
                                          sha256=sha(images/(stem+'.'+ext))) for ext in ('pdf','png')})
             for placement,old,stem in PLACEMENTS}
    # Retain complete original coordinate/provenance inventory beside the mapping.
    shutil.copyfile(inputs/'FIGURE_INDEX.json',out/'LEGACY_FIGURE_INDEX.json')
    report=dict(status='complete',figures=records,unique_assets=16,figure_placements=17,
                manuscript_archive_sha256='6cde5df47e0f716e28b843b0e737fadd8f306b5ee35de38574e3c0a2552f97ce',
                source_image_receipt_sha256=sha(inputs/'complete.json'),
                source_index_sha256=sha(inputs/'FIGURE_INDEX.json'),
                source_script_sha256=sha(Path(__file__)),
                notes=['Six main placements, eleven supplement placements; fig_strata is shared.',
                       'Legacy tally omitted from final asset directory; original inputs untouched.',
                       'Filename/order mapping only, not pixel-identical reconstruction of supplied PDFs.',
                       'Keep the original image output for plot coordinates referenced by LEGACY_FIGURE_INDEX.json.'])
    (out/'FIGURE_INDEX.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for name,h in hashes.items():checked(inputs,name,h)
    (out/'complete.json').write_text(json.dumps(dict(status='complete',files_sha256={
        p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}),indent=2)+'\n',encoding='utf-8')
    print('FINAL LAYOUT: 16 unique PDF/PNG assets; 17 placements; original inputs preserved.',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='Completed run_images.py output, NOT tables')
    parser.add_argument('--out',type=Path,required=True,help='New final-layout output')
    args=parser.parse_args();export(args.inputs,args.out)
