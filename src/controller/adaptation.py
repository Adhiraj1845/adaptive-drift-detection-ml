import numpy as np


def weighted_update(model, X, y):
    n = len(X)
    weights = np.linspace(0.5, 1.5, n)
    model.model.fit(X, y, sample_weight=weights)


def sliding_window_retrain(model, X, y, window_size=500):
    Xw = X.iloc[-window_size:]
    yw = y.iloc[-window_size:]
    model.train(Xw, yw)


def ensemble_refresh(ensemble: list, new_model):
    if not ensemble:
        ensemble.append(new_model)
        return ensemble

    ensemble.pop(0)
    ensemble.append(new_model)
    return ensemble
