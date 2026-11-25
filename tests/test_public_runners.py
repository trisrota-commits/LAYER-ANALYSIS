"""Two public commands: tables calculate; images consume completed tables only."""
from pathlib import Path
import hashlib
import importlib.util
import json
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]


def receipt(folder, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob('*') if p.is_file() and p.name != 'complete.json'}
    (folder/'complete.json').write_text(json.dumps(dict(status='complete', files_sha256=hashes, **extra)))


def test_tables_cli_dispatches_only_table_pipeline(tmp_path):
    path = ROOT/'run_tables.py'
    assert path.is_file(), 'Missing single public table runner'
    spec = importlib.util.spec_from_file_location('public_tables_cli', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    out = tmp_path/'tables'
    with patch('analysis.public_runners.tables') as run:
        module.main(['--out', str(out)])
    run.assert_called_once_with(ROOT, out.resolve())


def test_completed_tables_resolve_legacy_inputs_and_reject_changed_files(tmp_path):
    path = ROOT/'analysis/public_runners.py'
    assert path.is_file(), 'Missing shared public-runner integration'
    from analysis.public_runners import table_inputs
    root = tmp_path/'tables'; legacy = root/'workflows/base'
    legacy.mkdir(parents=True)
    (legacy/'data.json').write_text('{}')
    receipt(legacy)
    (root/'TABLE_INDEX.json').write_text('{}')
    # The aggregate receipt must bind the child receipt as well as its data.
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    (root/'complete.json').write_text(json.dumps(dict(status='complete', kind='paper_tables',
        legacy_tables='workflows/base', files_sha256=hashes)))
    assert table_inputs(root) == legacy.resolve()
    (legacy/'data.json').write_text('changed')
    with pytest.raises(ValueError): table_inputs(root)


def test_all_table_stages_and_flat_output(tmp_path):
    from analysis.public_runners import tables
    calls=[]
    def calculator(name):
        def run(repo,out):
            calls.append(name);out.mkdir(parents=True)
            (out/'result.json').write_text('{}');receipt(out)
        return run
    def labels(base,xgb,lc,nested,out):
        calls.append('labels');(out/'tables').mkdir(parents=True)
        (out/'tables/tab-example.json').write_text('[]')
        (out/'tables/tab-example.csv').write_text('metric,value\n')
        index=dict(numerical_tables=69,numerical_labels=73,
                   tables=[dict(label='tab:example',path='tables/tab-example.json')])
        (out/'TABLE_INDEX.json').write_text(json.dumps(index));receipt(out)
        return index
    with patch('analysis.revision_paper_tables.run',calculator('base')), \
         patch('analysis.revision_xgboost_bca.run',calculator('xgb')), \
         patch('analysis.revision_lc_sensitivity.run',calculator('lc')), \
         patch('analysis.revision_nested_bca.run',calculator('nested')), \
         patch('analysis.final_paper_tables.export',labels), \
         patch('analysis.revision_paper_figures.run',side_effect=AssertionError('No images in table runner')):
        out=tmp_path/'out';tables(ROOT,out)
        assert calls==['base','xgb','lc','nested','labels']
        assert (out/'tab-example.csv').is_file()
        index=json.loads((out/'TABLE_INDEX.json').read_text())
        assert index['tables'][0]['path']=='tab-example.json'
        result=json.loads((out/'complete.json').read_text())
        assert result['kind']=='paper_tables' and result['legacy_tables']=='workflows/base'
        for name,digest in result['files_sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
        with pytest.raises(FileExistsError):tables(ROOT,out)


def test_failed_stage_preserves_diagnostics_without_completion(tmp_path):
    from analysis.public_runners import tables
    out=tmp_path/'failed'
    with patch('analysis.revision_paper_tables.run',side_effect=RuntimeError('fixture failure')):
        with pytest.raises(RuntimeError,match='fixture failure'):tables(ROOT,out)
    assert (out/'workflows/base.log').is_file()
    assert 'fixture failure' in (out/'workflows/base.log').read_text()
    assert not (out/'complete.json').exists()
