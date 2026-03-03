# src/drift_detectors/page_hinkley_detector.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkleyState:
    t: int = 0
    mean: float = 0.0
    m_t: float = 0.0
    m_min: float = 0.0
    m_max: float = 0.0


class PageHinkleyDetector:
    """
    Streaming Page-Hinkley drift detector.

    - update(x) -> bool : drift event trigger (keeps tests compatible)
    - statistic() -> float : continuous drift magnitude for logging/plots
    - direction: "increase", "decrease", or "both"
      "increase" detects upward mean shifts
      "decrease" detects downward mean shifts
      "both" detects either (using max of both-sided stats)
    """

    def __init__(self, threshold: float = 50.0, delta: float = 0.0, direction: str = "increase"):
        self.threshold = float(threshold)
        self.delta = float(delta)
        self.direction = direction.lower().strip()
        if self.direction not in {"increase", "decrease", "both"}:
            raise ValueError("direction must be one of: 'increase', 'decrease', 'both'")
        self.last_direction: str | None = None
        self.reset()

    def reset(self) -> None:
        self.state = PageHinkleyState()
        self.last_direction = None

    def update(self, x: float) -> bool:
        s = self.state
        s.t += 1

        # Online mean
        s.mean += (x - s.mean) / s.t

        # Cumulative deviation (centered by mean and delta)
        # For "increase": accumulates positive shifts
        # For "decrease": accumulates negative shifts
        s.m_t += (x - s.mean - self.delta)

        # Track extremas for both-sided detection
        if s.m_t < s.m_min:
            s.m_min = s.m_t
        if s.m_t > s.m_max:
            s.m_max = s.m_t

        stat_inc = s.m_t - s.m_min          # upward shift evidence
        stat_dec = s.m_max - s.m_t          # downward shift evidence (mirror)

        if self.direction == "increase":
            drift = stat_inc > self.threshold
            self.last_direction = "increase" if drift else None
            return drift

        if self.direction == "decrease":
            drift = stat_dec > self.threshold
            self.last_direction = "decrease" if drift else None
            return drift

        # both
        if stat_inc >= stat_dec:
            drift = stat_inc > self.threshold
            self.last_direction = "increase" if drift else None
        else:
            drift = stat_dec > self.threshold
            self.last_direction = "decrease" if drift else None
        return drift

    def statistic(self) -> float:
        """Continuous PH drift magnitude (>=0)."""
        s = self.state
        stat_inc = s.m_t - s.m_min
        stat_dec = s.m_max - s.m_t
        if self.direction == "increase":
            return float(stat_inc)
        if self.direction == "decrease":
            return float(stat_dec)
        return float(max(stat_inc, stat_dec))
