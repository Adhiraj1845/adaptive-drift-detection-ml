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
    """Page-Hinkley sequential change-point detector (Page, 1954).

    Accumulates deviations of an observed signal from its running mean.
    Triggers when the cumulative sum exceeds a threshold:
        M_t = sum_{i=1}^{t} (x_i - mean_t - delta)
        PH_t = M_t - min_{i<=t}(M_i)   (upward shift)
        PH_t = max_{i<=t}(M_i) - M_t   (downward shift)

    Operates in O(1) time and O(1) space per update — the most
    computationally efficient detector in the ensemble.
    Applied to prediction error rate to detect performance drift.

    Parameters
    ----------
    threshold : float
        Detection threshold on the PH statistic. Default 50.0.
    delta : float
        Allowable slack (magnitude of acceptable change). Default 0.0.
    direction : str
        One of 'increase', 'decrease', or 'both'. Default 'increase'.
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
        s.mean += (x - s.mean) / s.t  # online mean
        s.m_t += (x - s.mean - self.delta)
        if s.m_t < s.m_min:
            s.m_min = s.m_t
        if s.m_t > s.m_max:
            s.m_max = s.m_t

        stat_inc = s.m_t - s.m_min  # upward evidence
        stat_dec = s.m_max - s.m_t  # downward evidence

        if self.direction == "increase":
            drift = stat_inc > self.threshold
            self.last_direction = "increase" if drift else None
            return drift
        if self.direction == "decrease":
            drift = stat_dec > self.threshold
            self.last_direction = "decrease" if drift else None
            return drift
        if stat_inc >= stat_dec:
            drift = stat_inc > self.threshold
            self.last_direction = "increase" if drift else None
        else:
            drift = stat_dec > self.threshold
            self.last_direction = "decrease" if drift else None
        return drift

    def statistic(self) -> float:
        s = self.state
        stat_inc = s.m_t - s.m_min
        stat_dec = s.m_max - s.m_t
        if self.direction == "increase":
            return float(stat_inc)
        if self.direction == "decrease":
            return float(stat_dec)
        return float(max(stat_inc, stat_dec))
