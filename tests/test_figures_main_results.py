"""Focused scientific-coordinate and publication-export checks."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from analysis import figures_main_results as figures

REPO = Path(__file__).resolve().parents[1]


def test_ensemble_estimand():
    p = np.array([[0., 0.], [4., 0.]])
    rows = figures.ensemble_points(p, np.zeros(2), 'test')
    assert rows[0]['y'] == pytest.approx(2.)
    assert rows[0]['mean_subset_rmse'] == pytest.approx(np.sqrt(2))
    assert rows[1]['y'] == pytest.approx(np.sqrt(2))
    assert rows[0]['subsets'] == 2


def test_stable_selection_ties():
    assert figures.ranked_regret(np.array([1., 1., 0.]), np.array([2., 8., 9.]), [1, 2]) == [7., 1.]


def test_overwrite_refusal_before_inputs(tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    keep = target / 'keep'
    keep.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        figures.render(tmp_path / 'absent', tmp_path / 'absent', target)
    assert keep.read_bytes() == b'preserve'
    assert list(target.iterdir()) == [keep]


@pytest.fixture(scope='module')
def rendered(tmp_path_factory, paper_tables):
    out = Path(os.environ['FIGURE_TEST_OUT']) if 'FIGURE_TEST_OUT' in os.environ else tmp_path_factory.mktemp('figures-parent') / 'results'
    inventory = figures.render(REPO, paper_tables, out)
    return out, inventory


def test_inventory_and_receipts(rendered, paper_tables):
    out, inventory = rendered
    assert set(inventory['figures']) == {'main2', 'main3', 'main4', 'main5', 'main6', 'S7', 'S8', 'S9'}
    for record in inventory['figures'].values():
        assert len(record['files']) == 4
        for relative in record['files']:
            assert (out / relative).is_file()
        assert record['pdf_check']['clipped_text'] == []
        assert record['pdf_check']['missing_labels'] == []
    for relative, digest in inventory['outputs_sha256'].items():
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == digest
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        figures.render(REPO, paper_tables, out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


def test_real_coordinates_and_populations(rendered, paper_tables):
    out, _ = rendered
    load = lambda name: json.loads((out / (name + '.json')).read_text())['points']
    primary = load('fig_primary')
    assert len(primary) == 7 and all(r['n'] == 106 for r in primary)
    assert primary[-1]['estimate'] == pytest.approx(0.4956466540880506)
    assert primary[-1]['low'] == pytest.approx(0.15844744470752548)
    nb = load('fig_nb201_improve')
    assert len(nb) == 15
    assert {r['n'] for r in nb if r['metric'] == 'MAE*'} == {273, 272, 258}
    assert all(r['n'] == 275 and r['adjusted_low'] is not None for r in nb if r['metric'] != 'MAE*')
    zcp = load('fig_zcp')
    assert len(zcp) == 8
    assert next(r for r in zcp if r['feature'] == 'no-layer')['x'] == 0
    assert next(r for r in zcp if r['feature'] == 'taylor')['x'] == pytest.approx(0.6156858619524996)
    import pymupdf
    with pymupdf.open(out / 'fig_zcp.pdf') as document:
        text = document[0].get_text()
        assert 'p=' not in text and 'r=' not in text
        assert 'Dead-act. fraction' in text
    strata = load('fig_strata')
    assert len(strata) == 115 and sum(r['collapsed'] for r in strata) == 9
    assert sum(r['dataset'] == 'cifar-100' for r in strata) == 88
    panels = load('fig_panels')
    assert len([r for r in panels if r['panel'] == 'b']) == 18
    assert len([r for r in panels if r['panel'] == 'c']) == 60
    costs = json.loads((paper_tables / 'tables/S16.json').read_text())
    expected = next(r['seconds'] for r in costs if r['operation'] == 'taylor_importance')
    assert next(r for r in panels if r['panel'] == 'a' and r['feature'] == 'taylor')['seconds'] == expected
    ensemble = load('fig_ensemble')
    taylor = [r for r in ensemble if r['feature'] == 'taylor' and r['kind'] == 'point']
    assert len(taylor) == 6 and taylor[-1]['y'] == pytest.approx(8.0498, abs=0.00005)
    assert taylor[0]['y'] != pytest.approx(taylor[0]['mean_subset_rmse'], abs=1e-6)
    lodo = load('fig_lodo')
    assert len(lodo) == 14
    source = json.loads((paper_tables / 'tables/S37.json').read_text())
    for row in lodo:
        original = next(r for r in source if r['dataset'] == row['dataset'] and r['metric'] == row['metric'])
        assert (row['estimate'], row['low'], row['high']) == (original['improvement'], original['low'], original['high'])


def test_reference_and_render_contact_sheets(rendered):
    """Produce retained visual evidence, never write into manuscript directories."""
    import pymupdf
    from PIL import Image, ImageDraw
    out, inventory = rendered
    views = out / 'visual_checks'
    views.mkdir(exist_ok=False)
    figures_list = list(inventory['figures'].items())
    for start in (0, 4):
        canvas = Image.new('RGB', (1500, 1100), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (identity, record) in enumerate(figures_list[start:start+4]):
            with Image.open(out / record['png']) as original:
                image = original.convert('RGB')
                image.thumbnail((730, 510))
                x, y = (j % 2) * 750, (j // 2) * 550
                canvas.paste(image, (x, y + 25))
                draw.text((x+10, y+5), identity, fill='black')
        canvas.save(views / f'rendered-{start+1}-{start+4}.png')
    attachments = Path('C:/Users/trisr/Desktop/thesis/.hermes/desktop-attachments')
    references = [('project_final_4_-20.pdf', [4, 5, 6]), ('project_final_4_-21.pdf', [20, 26])]
    reference_hashes = {}
    for name, pages in references:
        path = attachments / name
        if not path.exists():
            continue
        reference_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with pymupdf.open(path) as document:
            for index in pages:
                document[index].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(views / f'{path.stem}-page{index+1}.png')
    (views / 'reference_sha256.json').write_text(json.dumps(reference_hashes, indent=2))
