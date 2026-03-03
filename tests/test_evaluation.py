from src.evaluation import metrics
def test_detection_metrics_return_floats():
    true_drifts = [20, 50, 80]
    detected_drifts = [20, 55, 85]

    prec = metrics.detection_precision(true_drifts, detected_drifts)
    delay = metrics.detection_delay(true_drifts, detected_drifts)
    fpr = metrics.false_positive_rate(true_drifts, detected_drifts, total_time=100)

    assert isinstance(prec, float)
    assert isinstance(delay, float)
    assert isinstance(fpr, float)
