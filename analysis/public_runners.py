"""Complete public table/image pipelines; historical calculators remain internal."""
import contextlib
import json
from pathlib import Path
import shutil
import traceback
from analysis.thesis_common import require, sha
from analysis.final_paper_tables import verified


def new_output(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Existing output preserved: '+str(out))
    require(not any(out.is_relative_to(repo/p) for p in
                    ('analysis','evidence','tests','paper_runners','figures','images','docs','.git')),
            'Output overlaps repository source/evidence')
    return repo, out


def stage(work, name, function, *args):
    print('Running '+name+' ...', flush=True)
    log = work/(name+'.log')
    try:
        with log.open('x', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                return function(*args)
    except Exception:
        with log.open('a', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        print('Failed '+name+'; preserved diagnostics: '+str(log), flush=True)
        raise


def publish(source, out, data_folder):
    """Flatten only this pipeline's new final artifacts; preserve stage receipts."""
    for item in source.iterdir():
        if item.name == 'complete.json':
            continue
        files = list(item.iterdir()) if item.name == data_folder else [item]
        for path in files:
            require(path.is_file(), 'Unexpected final artifact directory')
            target = out/path.name
            require(not target.exists(), 'Duplicate published artifact: '+target.name)
            shutil.copyfile(path, target)
            require(sha(path) == sha(target), 'Published artifact changed bytes')


def finish(repo, out, kind, **metadata):
    command = 'run_tables.py' if kind == 'paper_tables' else 'run_images.py'
    report = dict(status='complete', kind=kind, **metadata,
                  source_sha256={n:sha(repo/n) for n in ['analysis/public_runners.py', command]})
    (out/'RUN.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    hashes = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(report, files_sha256=hashes), indent=2)+'\n', encoding='utf-8')
    print('Complete: '+str(out), flush=True)
    return report


def table_inputs(inputs):
    """Accept the complete public table output or a verified legacy table output."""
    inputs = Path(inputs).resolve()
    receipt = verified(inputs)
    require(receipt.get('kind') in (None, 'paper_tables'), 'Expected a completed table output')
    if receipt.get('kind') == 'paper_tables':
        relative = receipt['legacy_tables']
        source = (inputs/relative).resolve()
        require(source != inputs and source.is_relative_to(inputs), 'Unsafe legacy table location')
        require(relative+'/complete.json' in receipt['files_sha256'], 'Unbound legacy table receipt')
        return source
    return inputs


def tables(repo, out):
    repo, out = new_output(repo, out)
    from analysis import revision_paper_tables, revision_xgboost_bca, revision_lc_sensitivity
    from analysis import revision_nested_bca, final_paper_tables
    out.mkdir(parents=True, exist_ok=False)
    work = out/'workflows'; work.mkdir()
    base, xgb, lc, nested = [work/n for n in ['base','xgboost-bca','lc-sensitivity','nested-bca']]
    stage(work, 'base', revision_paper_tables.run, repo, base)
    stage(work, 'xgboost-bca', revision_xgboost_bca.run, repo, xgb)
    stage(work, 'lc-sensitivity', revision_lc_sensitivity.run, repo, lc)
    stage(work, 'nested-bca', revision_nested_bca.run, repo, nested)
    labelled = work/'labelled'
    report = stage(work, 'labelled', final_paper_tables.export, base, xgb, lc, nested, labelled)
    require(report['numerical_tables'] == 69 and report['numerical_labels'] == 73,
            'Incomplete final table coverage')
    publish(labelled, out, 'tables')
    index = json.loads((out/'TABLE_INDEX.json').read_text(encoding='utf-8'))
    for entry in index['tables']:
        entry['path'] = Path(entry['path']).name
    (out/'TABLE_INDEX.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    return finish(repo, out, 'paper_tables', numerical_tables=69, numerical_labels=73,
                  contextual_tables=1, legacy_tables='workflows/base',
                  notes=['All final-paper numerical tables, including BCa/Zou; no model training.',
                         'CSV/JSON files are in this directory; full stage evidence/logs are in workflows/.',
                         'Export coverage does not certify every manuscript value or original training provenance.'])


def images(repo, inputs, out):
    repo, out = new_output(repo, out)
    inputs = Path(inputs).resolve()
    require(not out.is_relative_to(inputs), 'Image output must be separate from table inputs')
    source = table_inputs(inputs)
    input_hash = sha(inputs/'complete.json')
    from analysis import revision_paper_figures, final_paper_images
    # Preflight before creating outputs; this checks original frozen evidence too.
    revision_paper_figures.validate_tables(repo, source)
    out.mkdir(parents=True, exist_ok=False)
    work = out/'workflows'; work.mkdir()
    rendered, final = work/'rendered', work/'final'
    stage(work, 'rendered', revision_paper_figures.run, repo, source, rendered)
    report = stage(work, 'final', final_paper_images.export, rendered, final)
    require(report['unique_assets'] == 16 and report['figure_placements'] == 17,
            'Incomplete final image coverage')
    publish(final, out, 'images')
    index = json.loads((out/'FIGURE_INDEX.json').read_text(encoding='utf-8'))
    legacy = json.loads((rendered/'FIGURE_INDEX.json').read_text(encoding='utf-8'))
    for entry in index['figures'].values():
        for item in entry['files'].values():
            item['path'] = Path(item['path']).name
        original = legacy['figures'][entry['legacy_figure']]
        entry['plot_data'] = [dict(item, path='workflows/rendered/'+item['path'])
                              for item in original['plot_data']]
        entry['producer_manifest'] = 'workflows/rendered/'+original['producer_manifest']
    index['legacy_provenance_root'] = 'workflows/rendered'
    index['notes'].append('All original rendered coordinates and provenance are retained in workflows/rendered/.')
    (out/'FIGURE_INDEX.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    verified(inputs)
    require(sha(inputs/'complete.json') == input_hash, 'Table input receipt changed')
    return finish(repo, out, 'paper_images', unique_assets=16, figure_placements=17,
                  table_receipt_sha256=input_hash,
                  notes=['PDF/PNG files are in this directory; plot coordinates/provenance are in workflows/.',
                         'Completed tables consumed without rerunning table or bootstrap workflows.',
                         'Figures are documented reconstructions, not pixel-identical original PDFs.'])
