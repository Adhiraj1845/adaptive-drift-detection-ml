"""
Tests for drift controller, calibration, and adaptation strategies.

Covers:
- DriftController: compute_drift_index, decide_action, cooldown, feature/prediction scores
- calibrate_control_limits: correct keys, monotone percentiles
- Adaptation: weighted_update, sliding_window_retrain, ensemble_refresh
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from src.controller.adaptation import ensemble_refresh, sliding_window_retrain, weighted_update
from src.controller.calibration import calibrate_control_limits
from src.controller.drift_controller import DriftController
from src.model.random_forest_model import RandomForestModel


# ── Stub detectors ────────────────────────────────────────────────────────────

class AlwaysHighScore:
    """Feature detector that always returns a fixed high score."""
    def score(self, *_): return 5.0

class AlwaysZeroScore:
    def score(self, *_): return 0.0


# ── DriftController ───────────────────────────────────────────────────────────

class TestDriftController:
    def _ctrl(self, feature_dets=None, pred_det=None, perf_det=None, limits=None):
        return DriftController(
            feature_detectors=feature_dets or [],
            prediction_detector=pred_det,
            performance_detector=perf_det,
            control_limits=limits,
        )

    def test_compute_drift_index_all_zero(self):
        ctrl = self._ctrl()
        idx = ctrl.compute_drift_index(0.0, 0.0, 0.0)
        assert idx == pytest.approx(0.0)

    def test_compute_drift_index_weighted_sum(self):
        ctrl = self._ctrl()
        # Default weights: feature=0.4, prediction=0.3, performance=0.3
        idx = ctrl.compute_drift_index(1.0, 1.0, 1.0)
        assert idx == pytest.approx(1.0)

    def test_decide_action_none_below_low(self):
        ctrl = self._ctrl(limits={"low": 1.0, "moderate": 2.0, "high": 3.0, "severe": 4.0})
        assert ctrl.decide_action(0.5) == "none"

    def test_decide_action_moderate(self):
        ctrl = self._ctrl(limits={"low": 1.0, "moderate": 2.0, "high": 3.0, "severe": 4.0})
        assert ctrl.decide_action(1.5) == "moderate"

    def test_decide_action_high(self):
        ctrl = self._ctrl(limits={"low": 1.0, "moderate": 2.0, "high": 3.0, "severe": 4.0})
        assert ctrl.decide_action(2.5) == "high"

    def test_decide_action_severe(self):
        ctrl = self._ctrl(limits={"low": 1.0, "moderate": 2.0, "high": 3.0, "severe": 4.0})
        assert ctrl.decide_action(3.5) == "severe"

    def test_no_cooldown_initially(self):
        ctrl = self._ctrl()
        today = datetime.date.today()
        assert ctrl.in_cooldown(today) is False

    def test_cooldown_active_after_registration(self):
        ctrl = self._ctrl()
        today = datetime.date.today()
        ctrl.register_adaptation(today)
        assert ctrl.in_cooldown(today) is True

    def test_cooldown_expired_after_window(self):
        ctrl = self._ctrl()
        past = datetime.date.today() - datetime.timedelta(days=10)
        ctrl.register_adaptation(past)
        assert ctrl.in_cooldown(datetime.date.today()) is False

    def test_feature_drift_score_zero_no_detectors(self):
        ctrl = self._ctrl()
        assert ctrl.feature_drift_score([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0)

    def test_feature_drift_score_uses_max(self):
        # KS scores are already in [0,1] — use them to verify max() not mean()
        class KSLow:
            def ks_statistic(self, *_): return 0.3
        class KSHigh:
            def ks_statistic(self, *_): return 0.7
        ctrl = self._ctrl(feature_dets=[KSLow(), KSHigh()])
        assert ctrl.feature_drift_score([], []) == pytest.approx(0.7)

    def test_prediction_drift_score_zero_no_detector(self):
        ctrl = self._ctrl()
        assert ctrl.prediction_drift_score([0.5], [0.5]) == pytest.approx(0.0)


# ── calibrate_control_limits ──────────────────────────────────────────────────

class TestCalibrateControlLimits:
    def _scores(self, n=1000, seed=0):
        return np.random.default_rng(seed).normal(2.5, 0.5, n).tolist()

    def test_returns_required_keys(self):
        limits = calibrate_control_limits(self._scores())
        assert set(limits.keys()) == {"low", "moderate", "high", "severe"}

    def test_limits_are_monotone(self):
        limits = calibrate_control_limits(self._scores())
        assert limits["low"] <= limits["moderate"] <= limits["high"] <= limits["severe"]

    def test_limits_are_floats(self):
        limits = calibrate_control_limits(self._scores())
        for v in limits.values():
            assert isinstance(v, float)

    def test_percentile_values_correct(self):
        # Scores in [0, 0.20] so clamping bounds don't obscure the percentile extraction
        rng = np.random.default_rng(0)
        scores = list(rng.uniform(0.0, 0.20, 1000))
        limits = calibrate_control_limits(scores)
        # calibration uses 60/75/90/95th percentiles with clamp bounds
        expected_low      = float(max(min(np.percentile(scores, 60), 0.30), 0.05))
        expected_moderate = float(max(min(np.percentile(scores, 75), 0.50), 0.10))
        expected_high     = float(max(min(np.percentile(scores, 90), 0.70), 0.20))
        assert limits["low"]      == pytest.approx(expected_low,      rel=1e-6)
        assert limits["moderate"] == pytest.approx(expected_moderate, rel=1e-6)
        assert limits["high"]     == pytest.approx(expected_high,     rel=1e-6)


# ── Adaptation strategies ─────────────────────────────────────────────────────

def _make_training_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int).values)
    return X, y


class TestWeightedUpdate:
    def test_model_can_predict_after_weighted_update(self):
        X, y = _make_training_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X, y)
        weighted_update(model, X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_weighted_update_changes_predictions(self):
        """Predictions after weighted_update should differ from pre-update on at least some rows."""
        X, y = _make_training_data()
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X, y)
        before = model.predict(X).copy()
        # Retrain with flipped labels to force a model change
        y_flipped = 1 - y
        weighted_update(model, X, y_flipped)
        after = model.predict(X)
        assert not np.array_equal(before, after), "weighted_update had no effect"


class TestSlidingWindowRetrain:
    def test_model_can_predict_after_retrain(self):
        X, y = _make_training_data(n=600)
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X, y)
        sliding_window_retrain(model, X, y, window_size=200)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_uses_tail_window(self):
        """Window retrain must not fail when window_size < total rows."""
        X, y = _make_training_data(n=600)
        model = RandomForestModel(n_estimators=5, random_state=0)
        model.train(X, y)
        sliding_window_retrain(model, X, y, window_size=100)
        assert model.predict(X.tail(10)).shape[0] == 10


class TestEnsembleRefresh:
    def _trained_model(self, seed=0):
        X, y = _make_training_data(seed=seed)
        m = RandomForestModel(n_estimators=5, random_state=seed)
        m.train(X, y)
        return m

    def test_appends_to_empty_ensemble(self):
        ensemble = []
        new_model = self._trained_model()
        result = ensemble_refresh(ensemble, new_model)
        assert len(result) == 1
        assert result[0] is new_model

    def test_replaces_oldest_model(self):
        m1 = self._trained_model(0)
        m2 = self._trained_model(1)
        m3 = self._trained_model(2)
        ensemble = [m1, m2]
        result = ensemble_refresh(ensemble, m3)
        assert m1 not in result
        assert m3 in result

    def test_ensemble_size_stays_same(self):
        ensemble = [self._trained_model(i) for i in range(3)]
        new_model = self._trained_model(99)
        result = ensemble_refresh(ensemble, new_model)
        assert len(result) == 3
