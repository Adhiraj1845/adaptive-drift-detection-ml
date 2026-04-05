# src/model/gradient_boosting_model.py
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel:
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(**kwargs)

    def train(self, X_train, y_train, sample_weight=None) -> None:
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None