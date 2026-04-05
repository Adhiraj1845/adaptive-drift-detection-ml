from __future__ import annotations
from typing import Any, Dict, List, Sequence, Optional


class DriftController:

    def __init__(
        self,
        feature_detectors: List[Any],
        prediction_detector: Optional[Any],
        performance_detector: Optional[Any],
        weights: Optional[Dict[str, float]] = None,
        control_limits: Optional[Dict[str, float]] = None,
        cooldown_days: int = 5,
    ):
        self.feature_detectors = feature_detectors or []
        self.prediction_detector = prediction_detector
        self.performance_detector = performance_detector

        self.weights = weights or {
            "feature": 0.4,
            "prediction": 0.3,
            "performance": 0.3,
        }

        self.control_limits = control_limits or {
            "low": 0.6,
            "moderate": 1.0,
            "high": 1.5,
            "severe": 2.0,
        }

        self.cooldown_days = cooldown_days
        self.last_adaptation_date = None

    def feature_drift_score(self, reference: Sequence[float], window: Sequence[float]) -> float:
        if not self.feature_detectors:
            return 0.0
        scores = []
        for det in self.feature_detectors:
            if hasattr(det, "compute_psi"):
                scores.append(min(det.compute_psi(reference, window) / 0.5, 1.0))  # PSI [0,∞) → [0,1]
            elif hasattr(det, "score"):
                scores.append(min(det.score(reference, window) / 0.693, 1.0))  # JS [0,ln2] → [0,1]
            elif hasattr(det, "ks_statistic"):
                scores.append(float(det.ks_statistic(reference, window)))  # KS already [0,1]
        if not scores:
            return 0.0
        return float(max(scores))

    def prediction_drift_score(self, ref_preds, cur_preds) -> float:
        if self.prediction_detector is None:
            return 0.0
        return float(min(self.prediction_detector.score(ref_preds, cur_preds) / 0.693, 1.0))  # JS → [0,1]

    def performance_drift_score(self, loss_value: float) -> float:
        if self.performance_detector is None:
            return 0.0
        triggered = self.performance_detector.update(loss_value)
        stat = self.performance_detector.statistic()
        if triggered:
            return float(stat)
        return float(stat)

    def compute_drift_index(
        self,
        feature_score: float,
        prediction_score: float,
        performance_score: float,
    ) -> float:
        return (
            self.weights["feature"] * feature_score
            + self.weights["prediction"] * prediction_score
            + self.weights["performance"] * performance_score
        )

    def decide_action(self, drift_index: float) -> str:
        if drift_index < self.control_limits["low"]:
            return "none"
        elif drift_index < self.control_limits["moderate"]:
            return "moderate"
        elif drift_index < self.control_limits["high"]:
            return "high"
        elif drift_index < self.control_limits["severe"]:
            return "severe"
        else:
            return "severe"

    def in_cooldown(self, current_date) -> bool:
        if self.last_adaptation_date is None:
            return False
        return (current_date - self.last_adaptation_date).days < self.cooldown_days

    def register_adaptation(self, current_date):
        self.last_adaptation_date = current_date
