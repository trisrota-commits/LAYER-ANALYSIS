"""Real verified-evidence integration and plot representation regression tests."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from analysis import figures_supplement_new as sut

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / 'recomputed/paper-complete-verification-final/paper'


@pytest.fixture(scope='module')
def prepared():
    return sut.prepare(REPO, TABLES)


@pytest.fixture(scope='module')
def figures(prepared):
    figures = sut.build_figures(prepared[0])
    yield figures
    import matplotlib.pyplot as plt
    for figure in figures.values():
        plt.close(figure)


def test_complete_coordinates_and_source_precision(prepared):
    coords, provenance = prepared
    assert {k: len(v) for k, v in coords.items()} == {'S10': 36, 'S11': 15, 'S12': 54}
    for r in coords['S10']:
        source = sut._unique(json.loads((TABLES/'tables/S48.json').read_text()),
                             dataset=r['dataset'], arm=r['arm'], metric=r['metric'])
        assert all(r[k] == v for k, v in source.items())
        assert r['y'] == source['estimate']
        assert r['n'] == (198 if 'partial' in r['dataset'] else 275)
    for name, source in provenance['sources'].items():
        assert source['sha256'] == hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()
    assert 'not original' in provenance['renderer']


def test_all_intervals_and_marker_semantics(prepared, figures):
    rows = prepared[0]['S11']
    for m, metric in enumerate(sut.METRICS):
        chosen = [r for r in rows if r['metric'] == metric]
        ax = figures['S11'].axes[m]
        points = [line for line in ax.lines if line.get_marker() == 'o']
        assert len(points) == 3
        assert len(ax.collections) == 6
        for i, (r, point) in enumerate(zip(chosen, points)):
            assert r['filled'] == (r['ci95_low'] > 0 or r['ci95_high'] < 0)
            assert point.get_markerfacecolor() == ('black' if r['filled'] else 'white')
            assert list(point.get_xdata()) == [r['improvement']]
            for j, prefix in enumerate(('bonferroni40','ci95')):
                segment = ax.collections[2*i+j].get_segments()[0]
                assert segment[:,0].tolist() == [r[prefix+'_low'],r[prefix+'_high']]
            assert ax.collections[2*i].get_linewidths()[0] < ax.collections[2*i+1].get_linewidths()[0]
            assert not r['corrected_excludes_zero']
    assert {r['metric'] for r in rows if r['filled']} == {'RMSE','Spearman','R2'}
    assert {r['dataset'] for r in rows if r['filled']} == {'CIFAR-100'}
    assert not sut._excludes(0,1)
    assert not sut._excludes(-1,0)
    assert not sut._excludes(-1,.000002)


def test_s12_all_matching_cells_counts_and_precision(prepared, figures):
    rows = prepared[0]['S12']
    rank = [r for r in rows if r['panel'] == 'rank']
    magnitude = [r for r in rows if r['panel'] == 'magnitude']
    assert len(rank) == 36 and len(magnitude) == 18
    assert {r['reference'] for r in rank} == {'batch32/eval/augmentation-off/mean-loss'}
    for r in rank:
        assert len(set(r['candidate_indices'])) == r['n_matched_architectures']
        assert (r['batch_size'],r['mode'],r['augmentation']) == sut.ALTERNATIVES[r['y']]
        assert r['loss_reduction'] == 'mean'
    for d, ax in zip(sut.ROBUST_DATASETS, figures['S12'].axes[1:]):
        assert ax.images[0].get_array().shape == (4,3)
        for r in [r for r in rank if r['dataset'] == d]:
            assert ax.images[0].get_array()[r['y'],r['x']] == r['spearman']
    r = sut._unique(rank, dataset='cifar10', epoch=3, batch_size=32, mode='train')
    assert f"{r['spearman']:.3f}" == '0.938'
    r = sut._unique(magnitude, dataset='cifar100', epoch=2, kind='raw')
    assert f"{r['x']:.3f}" == '1.978'
    for r in magnitude:
        assert r['x'] == r[r['kind']+'_median']
        assert 0 <= r['closer'] <= r['n']


def test_figure_shapes_labels_and_bold_best(prepared, figures):
    assert [len(figures[n].axes) for n in ('S10','S11','S12')] == [3,5,4]
    for ax, metric in zip(figures['S10'].axes, ('MAE','RMSE','Spearman')):
        assert len(ax.patches) == 12
        assert ax.get_title().startswith(metric)
        assert [t.get_text() for t in ax.get_xticklabels()] == list(sut.LABELS)
        rows = [r for arm in sut.ARMS for r in prepared[0]['S10'] if r['arm'] == arm and r['metric'] == metric]
        for bar, text, r in zip(ax.patches, ax.texts, rows):
            assert bar.get_height() == r['estimate']
            assert (text.get_fontweight() == 'bold') == r['best']
    assert [ax.get_title() for ax in figures['S11'].axes] == ['MAE','RMSE','Pearson','Spearman','R²']
    assert figures['S12'].axes[0].get_xlabel() == 'median |log₂(T₁₂₈/T₃₂)|'
    for figure in figures.values():
        figure.canvas.draw()
        bounds = figure.bbox
        # Locators retain off-view tick artists which Matplotlib does not draw.
        undrawn = set()
        for ax in figure.axes:
            for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                low, high = sorted(limits)
                for tick in axis.get_major_ticks():
                    if not low <= tick.get_loc() <= high:
                        undrawn.update((tick.label1, tick.label2))
        for text in figure.findobj(lambda obj: hasattr(obj,'get_text') and bool(obj.get_text())):
            if not text.get_visible() or text in undrawn:
                continue
            box = text.get_window_extent(figure.canvas.get_renderer())
            assert bounds.x0-1 <= box.x0 and box.x1 <= bounds.x1+1
            assert bounds.y0-1 <= box.y0 and box.y1 <= bounds.y1+1


def test_real_exports_one_page_and_existing_output_untouched(tmp_path):
    import fitz
    out = tmp_path/'supplement'
    inventory = sut.render(REPO,TABLES,out)
    for name, files in inventory['figures'].items():
        with fitz.open(out/files['pdf']) as pdf:
            assert len(pdf) == 1
            assert pdf[0].get_text().strip()
            assert pdf[0].get_pixmap().width > 500
        assert (out/files['png']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        data = json.loads((out/files['json']).read_text())
        with (out/files['csv']).open(newline='',encoding='utf-8') as f:
            assert len(list(csv.DictReader(f))) == len(data) == files['coordinates']
    before = {p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        sut.render(REPO,TABLES,out)
    assert before == {p.name:p.read_bytes() for p in out.iterdir()}
    provenance = json.loads((out/'provenance.json').read_text())
    for name, digest in provenance['output_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('table,field,value', [
    ('S48','n',274), ('S50','ci95_low',9),
    ('S54','spearman',.5), ('S56','raw_median',.5),
])
def test_corrupt_table_inputs_fail_before_creating_output(tmp_path, table, field, value):
    root = tmp_path/'tables'
    root.mkdir()
    for name in ('S48','S50','S54','S56'):
        data = json.loads((TABLES/'tables'/f'{name}.json').read_text())
        if name == table:
            data[0][field] = value
        (root/f'{name}.json').write_text(json.dumps(data))
    out = tmp_path/'output'
    with pytest.raises(ValueError):
        sut.render(REPO,root,out)
    assert not out.exists()


def test_missing_and_duplicate_rows_rejected():
    with pytest.raises(ValueError):
        sut._unique([],dataset='missing')
    with pytest.raises(ValueError):
        sut._unique([{'x':1},{'x':1}],x=1)
    with pytest.raises(ValueError):
        sut._interval({'ci95_low':2,'ci95_high':1},'ci95')
