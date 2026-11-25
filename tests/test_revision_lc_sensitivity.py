import importlib
import unittest
import numpy as np


class ZouTests(unittest.TestCase):
    def test_formula_symmetry_reversal_and_invalid_matrix(self):
        m = importlib.import_module('analysis.revision_lc_sensitivity')
        actual = m.zou_overlapping(.9625532108072204, .9420460674004026, .9735208266601371, 275)
        np.testing.assert_allclose(actual, [.0205071434068178, .01275633, .03083306], atol=5e-9, rtol=0)
        reverse = m.zou_overlapping(.9420460674004026, .9625532108072204, .9735208266601371, 275)
        np.testing.assert_allclose(reverse, [-actual[0], -actual[2], -actual[1]], atol=1e-14)
        equal = m.zou_overlapping(.5, .5, .8, 100)
        self.assertAlmostEqual(equal[1], -equal[2])
        for args in [(.9, .9, -.9, 100), (1., .5, .8, 100), (.5, .5, .8, 3), (np.nan, .5, .8, 100)]:
            with self.assertRaises(ValueError):
                m.zou_overlapping(*args)


class ClusterTests(unittest.TestCase):
    def test_whole_family_scipy_draws_intervals_and_reversal(self):
        from scipy.stats import bootstrap
        from analysis.revision_loss_confidence import metrics
        m = importlib.import_module('analysis.revision_lc_sensitivity')
        rng = np.random.default_rng(19)
        y = rng.normal(size=40)
        p = np.array([y + rng.normal(size=40), y + rng.normal(size=40)*.8])
        families = np.repeat(np.arange(10), 4)
        actual = m.cluster_intervals(y, p, families, metrics, n_resamples=499)
        def statistic(groups):
            idx = np.concatenate([np.flatnonzero(families == g) for g in groups])
            values = metrics(y[idx], p[:, idx])
            return (values[1]-values[0])*[-1, -1, 1, 1, 1]
        expected = bootstrap((np.arange(10),), statistic, vectorized=False, paired=False,
                             n_resamples=499, batch=100, method='BCa', rng=np.random.default_rng(0))
        np.testing.assert_array_equal(actual['draws'], expected.bootstrap_distribution)
        np.testing.assert_allclose(actual['BCa95'], expected.confidence_interval)
        np.testing.assert_allclose(actual['jackknife'], [statistic(np.delete(np.arange(10), j)) for j in range(10)])
        reverse = m.cluster_intervals(y, p[::-1], families, metrics, n_resamples=499)
        np.testing.assert_allclose(reverse['BCa_Bonf15'], -actual['BCa_Bonf15'][::-1], atol=1e-12)
        np.testing.assert_allclose(actual['percentile95'], np.quantile(actual['draws'], [.025, .975], axis=1))
        self.assertTrue(np.any(actual['sparse_tail']))
        with self.assertRaises(ValueError):
            m.cluster_intervals(y, p, np.zeros(40), metrics, n_resamples=10)


class RunTests(unittest.TestCase):
    def test_verified_loader_and_new_output_guard_and_receipt(self):
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from unittest.mock import patch
        m = importlib.import_module('analysis.revision_lc_sensitivity')
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'result'
            # Small real bootstrap exercises serialization; production run is fixed at 100000.
            real = m.cluster_intervals
            with patch.object(m, 'cluster_intervals', side_effect=lambda *args: real(*args, n_resamples=199)):
                m.run(repo, out)
            receipt = json.loads((out/'complete.json').read_text())
            for name, digest in receipt['files_sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), digest)
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(len(result['results']), 15)
            self.assertEqual(len(result['zou']), 3)
            self.assertNotIn(str(repo), (out/'results.json').read_text())
            with self.assertRaises(ValueError):
                m.run(repo, out)
            fake = Path(tmp)/'fake'
            (fake/'analysis').mkdir(parents=True)
            (fake/'analysis'/'revision_loss_confidence.py').write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'Loader checksum'):
                m.run(fake, Path(tmp)/'bad')
            self.assertFalse((Path(tmp)/'bad').exists())


if __name__ == '__main__':
    unittest.main()
