"""Regression against the published S6/S7 and main VII, plus frozen archives."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_complete_saved_export(tmp_path):
    from analysis.revision_classical_saved import run
    out = tmp_path / 'export'
    run(REPO, out)
    report = json.loads((out / 'results.json').read_text())
    assert len(report['s6']) == 15
    assert len(report['main_vii']) == 20
    assert len(report['s7']) == 15
    # PDF -21 p9: without / with / with-minus-without, RMSE then MAE.
    expected = [
        [9.6494,9.5816,-.0678,7.2055,7.1184,-.0871],
        [10.0892,10.0122,-.0770,7.5879,7.3899,-.1980],
        [9.6536,10.0043,.3507,7.2580,7.5542,.2962],
    ]
    for i, values in enumerate(expected):
        rows = report['s6'][5*i:5*i+2]
        actual = [r[k] for r in rows for k in ['without_taylor','with_taylor','change']]
        if i == 2:
            assert actual[2] == pytest.approx(.3506212262803121, abs=2e-12)
            assert report['paper_discrepancies'][0]['printed'] == .3507
            values[2] = .3506  # Paper subtracts rounded arms, not full precision.
        np.testing.assert_allclose(actual, values, atol=.000050001, rtol=0)
    # PDF -20 p8 Table VII, all twenty cells.
    vii = np.array([[8.47,5.90,.9401,.8218,.8595],
                    [9.58,7.12,.9167,.8260,.8207],
                    [10.01,7.39,.9085,.8191,.8042],
                    [10.,7.55,.9046,.8327,.8046]])
    actual = np.array([r['value'] for r in report['main_vii']]).reshape(4,5)
    assert np.all(abs(actual-vii) <= np.array([.005,.005,.00005,.00005,.00005]))
    paper_s7 = np.array([
        [1.1744,-1.0236,3.5708,-1.9036,4.8416],
        [1.3068,.0126,2.6412,-.6133,3.3458],
        [.0234,-.0106,.0703,-.0232,.0978],
        [-.0019,-.0541,.0568,-.0856,.0970],
        [.0413,-.0419,.1243,-.0852,.1732],
        [1.6143,-.6444,3.9350,-1.7083,5.0294],
        [1.6892,.3223,3.0551,-.2591,3.7304],
        [.0329,-.0014,.0783,-.0159,.1080],
        [.0043,-.0502,.0623,-.0794,.0934],
        [.0584,-.0259,.1492,-.0683,.2021],
        [1.1787,-.5903,3.0951,-1.3029,4.0085],
        [1.3594,.2223,2.5108,-.2908,3.0441],
        [.0300,.0034,.0655,-.0072,.0866],
        [-.0198,-.0719,.0288,-.1041,.0512],
        [.0415,-.0231,.1101,-.0571,.1470]])
    keys = ['improvement','ci95_low','ci95_high','bonferroni15_low','bonferroni15_high']
    np.testing.assert_allclose([[r[k] for k in keys] for r in report['s7']], paper_s7, rtol=0, atol=.000050001)
    with zipfile.ZipFile(REPO / 'evidence/thesis-new-logs-20261112-084434.zip') as z:
        with np.load(io.BytesIO(z.read('recent_results/statistical-test-results-full-v1/classical_no_taylor_bootstrap.npz'))) as old:
            with np.load(out / 'bootstrap.npz') as new:
                np.testing.assert_allclose(new['differences'], old['differences'], rtol=0, atol=2e-12)
                archived_bounds = np.quantile(old['differences'], [.025,.975,.025/15,1-.025/15], axis=1)
                exported_bounds = np.array([[r[k] for k in keys[1:]] for r in report['s7']]).reshape(3,5,4)
                np.testing.assert_allclose(exported_bounds, archived_bounds.transpose(1,2,0), rtol=0, atol=2e-12)
                from analysis.revision_common import metrics
                with_scores = metrics(new['targets'], new['with_taylor_predictions']).mean(axis=1)
                without_scores = metrics(new['targets'], new['predictions']).mean(axis=1)
                for i in range(3):
                    for k in range(5):
                        row = report['s6'][5*i+k]
                        assert row['with_taylor'] == with_scores[i+1,k]
                        assert row['without_taylor'] == without_scores[i+1,k]
                        assert row['change'] == with_scores[i+1,k]-without_scores[i+1,k]
                assert new['indices'].shape == (10000,115)
                rng = np.random.default_rng(0)
                np.testing.assert_array_equal(new['indices'], rng.integers(115,size=(10000,115)))
    receipt = json.loads((out / 'complete.json').read_text())
    assert receipt['status'] == 'complete'
    for name, digest in receipt['files_sha256'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run(REPO, out)


@pytest.mark.parametrize('field,value,message', [
    ('record_id','wrong','identity'), ('target','-99','target'),
    ('seed','41','seed'), ('prediction','nan','Nonfinite'),
])
def test_identity_target_seed_and_finite_guards(field, value, message):
    import csv
    from analysis.revision_classical_saved import prediction_csv
    path = REPO / 'evidence/two-experiments-final-v1/ridge-s42-predictions.csv'
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    ids = [r['record_id'] for r in rows]
    targets = np.array([float(r['target']) for r in rows])
    rows[0][field] = value
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        prediction_csv(stream.getvalue().encode(), 42, ids, targets)


@pytest.mark.parametrize('corrupt', ['registry','archive'])
def test_frozen_hash_failure_has_no_completion(tmp_path, corrupt):
    import shutil
    from analysis.revision_classical_saved import run, PREFIX, ARCHIVE
    repo = tmp_path / 'repo'
    (repo / PREFIX).mkdir(parents=True)
    shutil.copyfile(REPO / 'INPUT_MANIFEST.json', repo / 'INPUT_MANIFEST.json')
    for source in (REPO / PREFIX).glob('*.csv'):
        shutil.copyfile(source, repo / PREFIX / source.name)
    shutil.copyfile(REPO / PREFIX / 'test_registry.json', repo / PREFIX / 'test_registry.json')
    victim = repo / (PREFIX + 'test_registry.json' if corrupt == 'registry' else ARCHIVE)
    victim.write_bytes(b'corrupt')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='hash mismatch'):
        run(repo, out)
    assert not out.exists()


def test_existing_output_rejected_before_inputs(tmp_path):
    from analysis.revision_classical_saved import run
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'sentinel').write_text('preserve')
    with pytest.raises(FileExistsError):
        run(tmp_path / 'missing-repo', out)
    assert (out / 'sentinel').read_text() == 'preserve'
