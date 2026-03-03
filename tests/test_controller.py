from src.controller.drift_controller import DriftController
from src.model.random_forest_model import RandomForestModel

class AlwaysDriftDetector:
    def detect(self, reference_data, new_data):
        return True

def test_drift_controller_triggers_update():
    model = RandomForestModel(n_estimators=5, random_state=42)
    controller = DriftController(
        detectors=[AlwaysDriftDetector()],
        model=model,
        retrain_threshold=0.5
    )
    assert controller.check_for_drift([0, 1, 2], [3, 4, 5]) is True
