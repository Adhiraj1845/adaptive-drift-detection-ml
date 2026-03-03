# src/model/random_forest_model.py
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
