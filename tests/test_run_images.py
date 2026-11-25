"""Separate image CLI dispatch and preservation contracts."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import run_images

ROOT = Path(__file__).resolve().parents[1]


def test_list_actual_cli():
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_images.py'), '--list'],
                            capture_output=True, text=True, check=True)
    inventory = json.loads(result.stdout)
    assert set(inventory) == {f'main_{i}' for i in range(1, 7)} | {f'S{i}' for i in range(1, 12)}
    assert inventory['S1'] == 'fig_dcor'
    assert inventory['S11'] == 'fig_robust'
    assert len(set(inventory.values())) == 16
    assert not any('accpred' in stem for stem in inventory.values())


def test_separate_renderer_dispatch(tmp_path):
    inputs = tmp_path / 'tables'
    inputs.mkdir()
    out = tmp_path / 'images'
    with patch('analysis.public_runners.images') as render:
        with patch('analysis.revision_paper_tables.run', side_effect=AssertionError('Must not rerun tables')):
            run_images.main(['--inputs', str(inputs), '--out', str(out)])
    render.assert_called_once_with(ROOT, inputs.resolve(), out.resolve())


def test_default_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path/'recomputed/tables'; inputs.mkdir(parents=True)
    with patch('analysis.public_runners.images') as render:
        run_images.main([])
    render.assert_called_once_with(ROOT, inputs.resolve(), (tmp_path/'recomputed/images').resolve())


def test_existing_output_preserved(tmp_path):
    sentinel = tmp_path / 'sentinel'
    sentinel.write_bytes(b'preserve')
    with pytest.raises(SystemExit) as exc:
        run_images.main(['--inputs', str(tmp_path), '--out', str(tmp_path)])
    assert exc.value.code == 2
    assert sentinel.read_bytes() == b'preserve'


@pytest.mark.parametrize('args', [['--list', '--out', 'unused'], ['--inputs', 'missing-input-directory', '--out', 'unused']])
def test_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        run_images.main(args)
    assert exc.value.code == 2
