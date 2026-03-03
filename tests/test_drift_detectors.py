from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.page_hinkley_detector import PageHinkleyDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.drift_index import DriftIndex

def test_ks_detector_no_drift():
    detector = KSTestDetector(p_threshold=0.05)
    data1 = [1, 2, 3, 4, 5]
    data2 = [1, 2, 3, 4, 5]
    assert detector.detect(data1, data2) is False

def test_page_hinkley_no_drift():
    detector = PageHinkleyDetector(threshold=5)
    steady_data = [100, 100, 100, 100, 100]
    assert any(detector.update(x) for x in steady_data) is False

def test_psi_detector_no_drift():
    detector = PSIDetector(threshold=0.1)
    ref = [0] * 50
    new = [0] * 50
    assert detector.detect(ref, new) is False

def test_composite_drift_index_no_drift():
    ks = KSTestDetector(p_threshold=0.05)
    psi = PSIDetector(threshold=0.1)
    composite = DriftIndex([ks, psi])
    assert composite.compute_index([1, 2, 3], [1, 2, 3]) == 0.0
