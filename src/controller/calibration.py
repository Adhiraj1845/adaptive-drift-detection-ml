import numpy as np


def calibrate_control_limits(baseline_scores: list[float]) -> dict:
    if len(baseline_scores) < 50:
        return {"low": 0.15, "moderate": 0.30, "high": 0.50, "severe": 0.80}
    return {
        "low":      max(min(float(np.percentile(baseline_scores, 60)), 0.30), 0.05),
        "moderate": max(min(float(np.percentile(baseline_scores, 75)), 0.50), 0.10),
        "high":     max(min(float(np.percentile(baseline_scores, 90)), 0.70), 0.20),
        "severe":   max(min(float(np.percentile(baseline_scores, 95)), 1.00), 0.35),
    }
