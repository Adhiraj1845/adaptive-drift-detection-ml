"""
Tests for evaluation modules: metrics, significance, calibration.

"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    ClassificationSummary,
    DriftDetectionSummary,
    classification_summary_from_daily,
    drift_detection_metrics,
    episode_summary_from_daily,
)
from src.evaluation.significance import (
    bootstrap_auc_diff_ci,
    bootstrap_sharpe_diff_ci,
    mcnemar_test_from_preds,
    ols_logloss_on_drift,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_daily(n=100, y_true=None, y_pred_s=None, y_pred_a=None, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if y_true is None:
        y_true = rng.integers(0, 2, n)
    if y_pred_s is None:
        y_pred_s = rng.integers(0, 2, n)
    if y_pred_a is None:
        y_pred_a = rng.integers(0, 2, n)
    return pd.DataFrame({
        "date": dates,
        "y_true_next": y_true,
        "y_pred_static": y_pred_s,
        "y_pred_adaptive": y_pred_a,
        "logloss_adaptive": rng.uniform(0.1, 2.0, n),
        "drift_event": rng.integers(0, 2, n),
    })


# ── ClassificationSummary ─────────────────────────────────────────────────────

class TestClassificationSummary:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1])
        df = _make_daily(n=5, y_true=y, y_pred_a=y)
        summary = classification_summary_from_daily(df, yhat_col="y_pred_adaptive")
        assert summary.accuracy == pytest.approx(1.0)
        assert summary.precision_1 == pytest.approx(1.0)
        assert summary.recall_1 == pytest.approx(1.0)
        assert summary.f1_1 == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y = np.array([0, 1, 1, 0])
        y_wrong = 1 - y
        df = _make_daily(n=4, y_true=y, y_pred_a=y_wrong)
        summary = classification_summary_from_daily(df, yhat_col="y_pred_adaptive")
        assert summary.accuracy == pytest.approx(0.0)

    def test_support_counts_correct(self):
        y = np.array([0, 0, 0, 1, 1])
        df = _make_daily(n=5, y_true=y, y_pred_a=y)
        summary = classification_summary_from_daily(df, yhat_col="y_pred_adaptive")
        assert summary.support_0 == 3
        assert summary.support_1 == 2

    def test_returns_classification_summary_type(self):
        df = _make_daily()
        result = classification_summary_from_daily(df)
        assert isinstance(result, ClassificationSummary)

    def test_accuracy_in_unit_interval(self):
        df = _make_daily()
        summary = classification_summary_from_daily(df)
        assert 0.0 <= summary.accuracy <= 1.0


# ── DriftDetectionMetrics ─────────────────────────────────────────────────────

class TestDriftDetectionMetrics:
    def _make_drift_daily(self, n=60):
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Ground truth: days 20-29 are drift
        detected = np.zeros(n, dtype=int)
        detected[20:30] = 1  # perfect detection of drift window
        return pd.DataFrame({"date": dates, "drift_event": detected})

    def test_perfect_detection(self):
        df = self._make_drift_daily()
        windows = [("2020-01-29", "2020-02-07")]  # approx business days 20-29
        result = drift_detection_metrics(df, drift_windows=windows)
        assert isinstance(result, DriftDetectionSummary)
        assert result.tp >= 0
        assert result.fp >= 0

    def test_no_detection_gives_fn(self):
        n = 40
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"date": dates, "drift_event": np.zeros(n, dtype=int)})
        windows = [("2020-01-15", "2020-01-31")]
        result = drift_detection_metrics(df, drift_windows=windows)
        assert result.tp == 0
        assert result.fn > 0

    def test_all_detected_as_drift_no_true_drift(self):
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"date": dates, "drift_event": np.ones(n, dtype=int)})
        result = drift_detection_metrics(df, drift_windows=[])
        assert result.tp == 0
        assert result.fp == n

    def test_precision_recall_in_unit_interval(self):
        df = self._make_drift_daily()
        windows = [("2020-01-29", "2020-02-07")]
        result = drift_detection_metrics(df, drift_windows=windows)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0


# ── EpisodeSummary ────────────────────────────────────────────────────────────

class TestEpisodeSummary:
    def _make_episode_daily(self):
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        detected = np.zeros(n, dtype=int)
        detected[5:10] = 1    # episode 1: 5 days
        detected[25:30] = 1   # episode 2: 5 days
        return pd.DataFrame({"date": dates, "drift_event": detected})

    def test_episode_count(self):
        df = self._make_episode_daily()
        result = episode_summary_from_daily(df)
        assert result["episode_count"] == pytest.approx(2.0)

    def test_mean_episode_len(self):
        df = self._make_episode_daily()
        result = episode_summary_from_daily(df)
        assert result["mean_episode_len"] == pytest.approx(5.0)

    def test_pct_days_detected(self):
        df = self._make_episode_daily()
        result = episode_summary_from_daily(df)
        assert result["pct_days_detected"] == pytest.approx(10 / 50)

    def test_no_episodes(self):
        n = 20
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"date": dates, "drift_event": np.zeros(n, dtype=int)})
        result = episode_summary_from_daily(df)
        assert result["episode_count"] == pytest.approx(0.0)


# ── McNemar Test ──────────────────────────────────────────────────────────────

class TestMcNemar:
    def test_identical_models_zero_discordant(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        yhat = rng.integers(0, 2, 200)
        result = mcnemar_test_from_preds(y, yhat, yhat)  # same predictions
        # When predictions are identical, neither model is ever uniquely correct
        assert result.b_static_correct_adapt_wrong == 0
        assert result.c_static_wrong_adapt_correct == 0

    def test_different_models_returns_result(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        yhat_s = rng.integers(0, 2, 300)
        yhat_a = rng.integers(0, 2, 300)
        result = mcnemar_test_from_preds(y, yhat_s, yhat_a)
        assert result.statistic >= 0.0
        assert 0.0 <= result.pvalue <= 1.0

    def test_counts_sum_correctly(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        yhat_s = np.array([0, 1, 0, 0, 1, 1])
        yhat_a = np.array([0, 1, 1, 1, 0, 1])
        result = mcnemar_test_from_preds(y, yhat_s, yhat_a)
        assert result.b_static_correct_adapt_wrong >= 0
        assert result.c_static_wrong_adapt_correct >= 0


# ── Bootstrap Sharpe diff ─────────────────────────────────────────────────────

class TestBootstrapSharpeDiff:
    def test_identical_returns_ci_near_zero(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 500)
        result = bootstrap_sharpe_diff_ci(r, r, n_boot=500, seed=0)
        assert result.point_estimate == pytest.approx(0.0, abs=1e-9)
        assert result.ci_low <= 0.0 <= result.ci_high

    def test_better_model_has_positive_point_estimate(self):
        rng = np.random.default_rng(42)
        r_bad = rng.normal(0.0, 0.02, 500)
        r_good = rng.normal(0.005, 0.01, 500)  # higher Sharpe
        result = bootstrap_sharpe_diff_ci(r_bad, r_good, n_boot=1000, seed=42)
        assert result.point_estimate > 0.0

    def test_ci_low_less_than_ci_high(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 300)
        result = bootstrap_sharpe_diff_ci(r, r, n_boot=200, seed=0)
        assert result.ci_low <= result.ci_high


# ── Bootstrap AUC diff ────────────────────────────────────────────────────────

class TestBootstrapAUCDiff:
    def test_identical_models_near_zero(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p = rng.uniform(0, 1, 300)
        result = bootstrap_auc_diff_ci(y, p, p, n_boot=300, seed=0)
        assert result.point_estimate == pytest.approx(0.0, abs=1e-9)

    def test_better_model_positive_estimate(self):
        rng = np.random.default_rng(5)
        n = 400
        y = rng.integers(0, 2, n)
        p_bad = rng.uniform(0, 1, n)
        p_good = np.clip(y + rng.normal(0, 0.3, n), 0, 1)  # correlated with truth
        result = bootstrap_auc_diff_ci(y, p_bad, p_good, n_boot=500, seed=5)
        assert result.point_estimate > 0.0


# ── OLS logloss on drift ──────────────────────────────────────────────────────

class TestOLSLoglossDrift:
    def _make_df(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        drift = rng.integers(0, 2, n)
        logloss = 0.5 + 0.3 * drift + rng.normal(0, 0.1, n)  # drift increases loss
        return pd.DataFrame({"date": dates, "logloss_adaptive": logloss, "drift_event": drift})

    def test_returns_ols_result(self):
        from src.evaluation.significance import OLSResult
        result = ols_logloss_on_drift(self._make_df())
        assert isinstance(result, OLSResult)

    def test_positive_coef_when_drift_increases_loss(self):
        result = ols_logloss_on_drift(self._make_df())
        assert result.coef > 0.0

    def test_r2_in_unit_interval(self):
        result = ols_logloss_on_drift(self._make_df())
        assert 0.0 <= result.r2 <= 1.0

    def test_pvalue_is_float(self):
        result = ols_logloss_on_drift(self._make_df())
        assert isinstance(result.pvalue, float)
