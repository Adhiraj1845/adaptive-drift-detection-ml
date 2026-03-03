import numpy as np


def calibrate_control_limits(baseline_scores: list[float]) -> dict:
    """
    Learn drift index thresholds from baseline distribution.
    """
    return {
        "low": float(np.percentile(baseline_scores, 90)),
        "moderate": float(np.percentile(baseline_scores, 95)),
        "high": float(np.percentile(baseline_scores, 99)),
        "severe": float(np.percentile(baseline_scores, 99.9)),
    }
