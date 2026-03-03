from src.drift_detectors.js_divergence_detector import JSDivergenceDetector


class PredictionDriftDetector:
    """
    Applies JS divergence to prediction probabilities.
    """

    def __init__(self, bins: int = 20):
        self.js = JSDivergenceDetector(bins=bins)

    def score(self, ref_preds, cur_preds):
        return self.js.score(ref_preds, cur_preds)
