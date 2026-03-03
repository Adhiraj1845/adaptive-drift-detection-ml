import numpy as np
from src.model.random_forest_model import RandomForestModel

def test_model_train_and_predict():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    model = RandomForestModel(n_estimators=5, random_state=42)
    model.train(X_train, y_train)
    preds = model.predict(X_train)

    assert preds.shape[0] == X_train.shape[0]
