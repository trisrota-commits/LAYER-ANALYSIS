"""Focused checks for shared revision metrics; no model training."""
from pathlib import Path
import unittest
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


class RevisionMetricTests(unittest.TestCase):
    def test_metrics_match_independent_implementations_with_ties_and_repeats(self):
        self.assertTrue((ROOT / 'analysis/revision_common.py').is_file())
        from analysis.revision_common import metrics
        y = np.array([20., 25., 25., 50., 80.])
        predictions = np.array([[21., 24., 30., 45., 75.], [22., 22., 22., 44., 70.]])
        for indices in [np.arange(len(y)), np.array([0, 0, 1, 2, 4, 4])]:
            target = y[indices]; pred = predictions[:, indices]
            actual = metrics(target, pred)
            for i, values in enumerate(pred):
                expected = [np.sqrt(mean_squared_error(target, values)), mean_absolute_error(target, values),
                            stats.pearsonr(target, values).statistic, stats.spearmanr(target, values).statistic,
                            r2_score(target, values)]
                np.testing.assert_allclose(actual[i], expected, atol=1e-12)
            np.testing.assert_allclose(metrics(target, pred[None, :, :]), actual[None, :, :])


if __name__ == '__main__':
    unittest.main()
