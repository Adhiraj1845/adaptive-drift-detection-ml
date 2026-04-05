# src/model/logistic_regression_model.py
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, **kwargs):
        # sane defaults for small-medium tabular time series features
        if "max_iter" not in kwargs:
            kwargs["max_iter"] = 2000
        if "solver" not in kwargs:
            kwargs["solver"] = "lbfgs"
        self.model = LogisticRegression(**kwargs)

    def train(self, X_train, y_train, sample_weight=None) -> None:
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None