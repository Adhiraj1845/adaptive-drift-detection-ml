"""
Tests for all model wrappers: RandomForest, LogisticRegression, GradientBoosting.

Each model is tested for:
- Correct prediction shape
- predict_proba output in [0, 1]
- Predictions are not trivially constant (model learns)
- predict_proba second column sums to 1 per row
"""
import numpy as np
import pandas as pd
import pytest

from src.model.gradient_boosting_model import GradientBoostingModel
from src.model.logistic_regression_model import LogisticRegressionModel
from src.model.random_forest_model import RandomForestModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # linearly separable
    return X, y


ALL_MODELS = [
    RandomForestModel(n_estimators=20, random_state=42),
    LogisticRegressionModel(),
    GradientBoostingModel(random_state=42),
]

MODEL_IDS = ["random_forest", "logistic_regression", "gradient_boosting"]


# ── Shared contract tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ALL_MODELS, ids=MODEL_IDS)
class TestModelContract:
    def test_predict_shape(self, model):
        X, y = _make_data()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == len(X)

    def test_predict_proba_shape(self, model):
        X, y = _make_data()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(X), 2)

    def test_predict_proba_in_unit_interval(self, model):
        X, y = _make_data()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_rows_sum_to_one(self, model):
        X, y = _make_data()
        model.train(X, y)
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predictions_not_all_constant(self, model):
        """Model must predict at least two distinct classes on balanced data."""
        X, y = _make_data()
        model.train(X, y)
        preds = model.predict(X)
        assert len(np.unique(preds)) > 1, "Model predicts only one class — likely not learning"

    def test_accepts_dataframe_input(self, model):
        X, y = _make_data()
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_s = pd.Series(y)
        model.train(X_df, y_s)
        preds = model.predict(X_df)
        assert preds.shape[0] == len(X_df)


# ── RandomForest-specific ─────────────────────────────────────────────────────

class TestRandomForestModel:
    def test_reproducible_with_same_seed(self):
        X, y = _make_data()
        m1 = RandomForestModel(n_estimators=20, random_state=0)
        m2 = RandomForestModel(n_estimators=20, random_state=0)
        m1.train(X, y)
        m2.train(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_different_seeds_may_differ(self):
        X, y = _make_data()
        m1 = RandomForestModel(n_estimators=5, random_state=0)
        m2 = RandomForestModel(n_estimators=5, random_state=99)
        m1.train(X, y)
        m2.train(X, y)
        # Not guaranteed to differ, but with small n_estimators they likely will
        # Just verify both return valid predictions
        assert m1.predict(X).shape == m2.predict(X).shape


# ── LogisticRegression-specific ───────────────────────────────────────────────

class TestLogisticRegressionModel:
    def test_converges_on_separable_data(self):
        X, y = _make_data(n=300)
        model = LogisticRegressionModel()
        model.train(X, y)
        acc = np.mean(model.predict(X) == y)
        assert acc > 0.8, f"Accuracy too low on separable data: {acc:.2f}"


# ── GradientBoosting-specific ─────────────────────────────────────────────────

class TestGradientBoostingModel:
    def test_reproducible_with_same_seed(self):
        X, y = _make_data()
        m1 = GradientBoostingModel(random_state=7)
        m2 = GradientBoostingModel(random_state=7)
        m1.train(X, y)
        m2.train(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
