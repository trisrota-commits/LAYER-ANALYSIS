"""Focused real-evidence/artist/integrity tests; never train or alter sources."""
from pathlib import Path
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
import pytest

from analysis import figures_supplement_legacy as m

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def evidence(paper_tables):
    if not (REPO / m.PACKET).exists():
        pytest.skip('Frozen published evidence and verified paper tables are required')
    return m.build_tables(REPO)


def test_dominance_ties():
    points = [dict(feature=str(i),seconds=x,rmse=y) for i,(x,y) in enumerate([(0,3),(1,2),(1,2),(2,4)])]
    assert [p['feature'] for p in m.nondominated(points)] == ['0','1','2']


@pytest.mark.parametrize('observations', [2,3])
def test_pilot_sources_ci_and_artists(evidence, observations, paper_tables):
    data = m.pilot_data(REPO, paper_tables, observations)
    fig = m.pilot_figure(data)
    try:
        for ax, metrics in zip(fig.axes, [['runtime'],['RMSE','MAE'],['Pearson','Spearman'],['R2']]):
            expected = []
            for feature in ['no-layer','all-seven']:
                for metric in metrics:
                    rows = data['runtimes'] if metric == 'runtime' else data['metrics']
                    r = next(r for r in rows if r['feature'] == feature and (metric == 'runtime' or r['metric'] == metric))
                    expected.append(r['mean'])
            np.testing.assert_allclose([p.get_height() for p in ax.patches], expected)
        assert len(data['runtimes']) == 2
        assert all(len(r['values']) == len(r['sources']) == 3 for r in data['runtimes'])
        assert 'not original workstation' in data['cohort']
        bad = copy.deepcopy(data['metrics']); bad[0]['mean'] += .1
        with pytest.raises(ValueError, match='CI disagrees'):
            m.validate_metrics(bad, [42]*3, 22)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('epochs', [None, [1,2,3], [50]])
def test_dependence_artist_values_and_global_bh(evidence, epochs):
    _, cells = evidence
    fig, data = m.dependence_figure(cells, epochs)
    try:
        if epochs is None:
            np.testing.assert_allclose([p.get_width() for p in fig.axes[0].patches], [p['mean'] for p in data['points']])
            assert all(p['cells'] == 28 for p in data['points'])
        else:
            assert 'places365' in data['datasets']
            for ax, grid in zip(fig.axes, data['grids']):
                np.testing.assert_array_equal(ax.images[0].get_array(), grid['values'])
                assert len(ax.texts) == 49
                for text, value, q in zip(ax.texts, np.ravel(grid['values']), np.ravel(grid['q'])):
                    assert text.get_text() == f'{value:.3f}' + ('*' if q <= .05 else '')
        bad = copy.deepcopy(cells); bad[0]['global_bh_fdr_q_value'] = '0.987654'
        with pytest.raises(ValueError, match='BH'):
            m.checked_cells(bad)
        with pytest.raises(ValueError, match='196'):
            m.checked_cells(cells[:-1])
    finally:
        plt.close(fig)


def test_six_seed_pareto_artist_coordinates(evidence, paper_tables):
    tables, _ = evidence
    data = m.pareto_data(paper_tables, tables['6.13'])
    assert set(data['frontier']) == {'no-layer','activation','taylor','gradient-flow'}
    assert next(p for p in data['points'] if p['feature'] == 'no-layer')['seconds'] == 0
    fig = m.pareto_figure(data)
    try:
        for ax in fig.axes:
            np.testing.assert_allclose(np.concatenate([c.get_offsets() for c in ax.collections]), [[p['seconds'],p['rmse']] for p in data['points']])
        assert fig.axes[0].get_xlim()[1] > 24
        assert fig.axes[1].get_xlim()[0] < 0 < fig.axes[1].get_xlim()[1] < 1
    finally:
        plt.close(fig)


def test_real_render_manifest_integrity_and_overwrite(evidence, tmp_path, paper_tables):
    out = tmp_path / 'figures'
    before = m.sha(REPO / m.PACKET)
    result = m.render(REPO, paper_tables, out)
    assert set(result) == {f'S{i}' for i in range(1,7)}
    assert all(r['status'] == 'rendered' for r in result.values()), result
    for row in result.values():
        for f in row['files'].values():
            assert m.sha(out / f['path']) == f['sha256']
        with pymupdf.open(out / row['files']['pdf']['path']) as doc:
            assert len(doc) == 1 and len(doc[0].get_text()) > 40
    saved = {p.name:m.sha(p) for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        m.render(REPO, paper_tables, out)
    assert saved == {p.name:m.sha(p) for p in out.iterdir()}
    assert m.sha(REPO / m.PACKET) == before
    assert json.loads((out/'manifest.json').read_text())['input_integrity'] == 'unchanged'


def test_pilot_blocker_does_not_silently_skip_other_figures(evidence, tmp_path, monkeypatch, paper_tables):
    def missing(*args):
        raise KeyError('archive/member/full.log missing train_runtime')
    monkeypatch.setattr(m, 'pilot_data', missing)
    result = m.render(REPO, paper_tables, tmp_path/'partial')
    assert all(result[k]['status'] == 'blocked' and 'full.log' in result[k]['blocker'] for k in ['S1','S2'])
    assert all(result[k]['status'] == 'rendered' for k in ['S3','S4','S5','S6'])
