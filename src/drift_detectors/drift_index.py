# src/controller/drift_controller.py

class DriftController:
    """
    Controls drift decision-making given a list of detectors.

    compute_drift_index(...) -> float  (0..1)
    check_for_drift(...) -> bool
    """

    def __init__(self, detectors, model, retrain_threshold=0.5):
        self.detectors = detectors or []
        self.model = model
        self.retrain_threshold = float(retrain_threshold)

    def compute_drift_index(self, reference_data, new_data) -> float:
        if len(self.detectors) == 0:
            return 0.0

        fired = 0
        for det in self.detectors:
            if det.detect(reference_data, new_data):
                fired += 1

        return fired / len(self.detectors)

    def check_for_drift(self, reference_data, new_data) -> bool:
        return self.compute_drift_index(reference_data, new_data) >= self.retrain_threshold
