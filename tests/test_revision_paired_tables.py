"""Regression coverage for complete standalone table preservation."""
import importlib
import json
import os
from pathlib import Path
import zipfile
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_published_packet_schema():
    with zipfile.ZipFile(REPO / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip') as z:
        manifest = json.loads(z.read('reproduction_packet_manifest.json'))
        assert set(z.namelist()) - set(manifest['files']) == {'README.txt', 'reproduction_packet_manifest.json'}
        assert not set(manifest['files']) - set(z.namelist())


@pytest.mark.skipif(not os.environ.get('PAIRED_PDF_AUDIT'), reason='optional local PDF comparison')
def test_pdf_table_identity():
    import fitz
    import re
    adapter = importlib.import_module('analysis.revision_paired_tables')
    with zipfile.ZipFile(REPO / adapter.PACKET) as z:
        tables = adapter.parse_reports(*(z.read('experiment_logs/' + n).decode('utf-8-sig') for n in adapter.REPORTS))
    pdf = fitz.open(os.environ.get('PAIRED_PDF_PATH', 'C:/Users/trisr/Desktop/thesis/.hermes/desktop-attachments/project_final_4_-21.pdf'))
    pages = {'S19': 15, 'S21': 16, 'S22': 17, 'S23': 17, 'S36': 26, 'S37': 27, 'S38': 28, 'S39': 28, 'S40': 29}
    checked = 0
    mismatches = []
    for table, page in pages.items():
        section = pdf[page].get_text().split('TABLE ' + table + '\n', 1)[1].split('\nTABLE ', 1)[0]
        numbers = set(re.findall(r'-?\d+\.\d{4}', section.replace('\u2212', '-')))
        for row in tables[table]:
            for field in ('mean', 'candidate_mean', 'reference_mean', 'improvement', 'low', 'high'):
                if field in row:
                    rounded = f"{row[field]:.4f}"
                    if rounded == '-0.0000':
                        rounded = '0.0000'
                    if rounded not in numbers:
                        mismatches.append((table, row, field, rounded))
                    checked += 1
    print('PDF_NUMERIC_FIELDS_CHECKED', checked, 'DISPLAY_DISCREPANCIES', mismatches)
    # Preserve source calculations rather than changing them to fit manuscript
    # rounding. These observed last-digit differences are an explicit audit
    # baseline, not permission to tolerate arbitrary new discrepancies.
    signatures = [(t, r.get('candidate', r.get('condition')), r.get('family'), r['metric'], f, v)
                  for t, r, f, v in mismatches]
    assert signatures == [
        ('S23', 'T+S+E', None, 'Pearson', 'high', '0.0028'),  # PDF 0.0029
        ('S38', 'TaylorOFF', 'detector', 'MAE', 'mean', '8.9987'),  # PDF 8.9988 (tie)
        ('S39', 'TaylorOFF', 'other', 'Pearson', 'improvement', '-0.0151'),  # PDF -0.0152
    ]
    assert checked == 850


def test_main_readmission_keeps_positive_near_zero_endpoint():
    adapter = importlib.import_module('analysis.revision_paired_tables')
    text = ('tab:readmission-paired-five-metrics (ALL FIVE METRICS)\n'
            'candidate=T+S+G; reference=T+S\n'
            '  Spearman: delta=-0.001 CI95=[-0.002, 0.00000178530432] finite_draws=9999/10000\n')
    row = adapter.parse_main(text)['S23'][0]
    assert row['high'] == 0.00000178530432
    assert row['valid_draws'] == 9999
    assert row['candidate'] == 'T+S+G'


def test_archived_reports_have_every_paper_field():
    adapter = importlib.import_module('analysis.revision_paired_tables')
    with zipfile.ZipFile(REPO / 'evidence/overleaf-table-evidence-reproducible-20261019-v2.zip') as z:
        tables = adapter.parse_reports(*(z.read('experiment_logs/' + n).decode('utf-8-sig') for n in
                    ('paired_main_tables.log', 'lodo_tables_reproduced.log', 'lofo_tables_reproduced.log')))
    expected = {'S19': 10, 'S21': 35, 'S23': 20, 'S36': 70, 'S37': 35,
                'S38': 50, 'S39': 25, 'S40': 5, 'main_descriptive': 65,
                'lofo_all_descriptive': 75}
    for name, count in expected.items():
        assert len(tables[name]) == count
    assert tables['S40'][0]['candidate_mean'] == pytest.approx(9.0467, abs=0.00005)
    assert tables['S40'][0]['reference_mean'] == pytest.approx(8.4749, abs=0.00005)
    places = {r['metric']: r for r in tables['S37'] if r['dataset'] == 'places365'}
    assert places['Pearson']['valid_draws'] == 9588
    assert places['R2']['valid_draws'] == 9984
    detector = {r['metric']: r for r in tables['S39'] if r['family'] == 'detector'}
    assert detector['Pearson']['valid_draws'] == 9283
    assert detector['R2']['valid_draws'] == 9838
    assert detector['R2']['records'] == 4
    assert detector['R2']['architecture_groups'] == 4


def test_parser_rejects_dropped_readmission_metric():
    adapter = importlib.import_module('analysis.revision_paired_tables')
    with zipfile.ZipFile(REPO / adapter.PACKET) as z:
        main, lodo, lofo = [z.read('experiment_logs/' + n).decode('utf-8-sig') for n in adapter.REPORTS]
    start = main.index('tab:readmission-paired-five-metrics')
    prefix, section = main[:start], main[start:]
    import re
    section = re.sub(r'  Spearman: delta=[^\n]+\n', '', section, count=1)
    with pytest.raises(ValueError, match='Incomplete S23'):
        adapter.parse_reports(prefix + section, lodo, lofo)


def test_reference_comparison_rejects_endpoint_sign_loss():
    adapter = importlib.import_module('analysis.revision_paired_tables')
    with pytest.raises(ValueError, match='Reference mismatch'):
        adapter.compare_tables({'S23': [{'high': 0.0}]}, {'S23': [{'high': 0.00000178530432}]})


def test_packet_symlink_refused(tmp_path):
    adapter = importlib.import_module('analysis.revision_paired_tables')
    archive = tmp_path / 'link.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        z.writestr(info, '../outside')
    with pytest.raises(ValueError, match='Unsafe archive link'):
        adapter.stage_packet(archive, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_existing_output_is_refused_before_inputs(tmp_path):
    adapter = importlib.import_module('analysis.revision_paired_tables')
    with pytest.raises(FileExistsError):
        adapter.run(tmp_path / 'missing', tmp_path)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/drive', 'dir\\escape', 'A/../escape'])
def test_unsafe_packet_members_refused(tmp_path, name):
    adapter = importlib.import_module('analysis.revision_paired_tables')
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        info = zipfile.ZipInfo('placeholder')
        info.filename = name  # ZipInfo constructor normalizes backslashes on Windows.
        z.writestr(info, b'no')
    with pytest.raises(ValueError, match='Unsafe'):
        adapter.stage_packet(archive, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_hash_failure_does_not_create_output(tmp_path):
    adapter = importlib.import_module('analysis.revision_paired_tables')
    (tmp_path / 'INPUT_MANIFEST.json').write_text(json.dumps({'files': {n: '0' * 64 for n in adapter.INPUTS}}))
    for name in adapter.INPUTS:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_text('changed')
    with pytest.raises(ValueError, match='SHA256'):
        adapter.run(tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


@pytest.mark.skipif(not os.environ.get('PAIRED_REAL_RUN'), reason='full frozen-archive bootstrap opt-in')
def test_full_real_run(tmp_path):
    adapter = importlib.import_module('analysis.revision_paired_tables')
    out = Path(os.environ.get('PAIRED_REAL_OUT', str(tmp_path / 'paired')))
    result = adapter.run(REPO, out)
    assert (out / 'complete.json').is_file()
    assert result['reference_comparison'] == 'PASS'
    row = next(r for r in result['tables']['S23'] if r['candidate'] == 'T+S+G' and r['metric'] == 'Spearman')
    assert row['high'] > 0
    assert row['high'] == pytest.approx(0.00000178530432, abs=0.0000000000005)
    receipt = json.loads((out / 'complete.json').read_text())
    for name, digest in receipt['files_sha256'].items():
        assert adapter.sha(out / name) == digest
    with pytest.raises(FileExistsError):
        adapter.run(REPO, out)
    print('REAL_OUTPUT', out)
