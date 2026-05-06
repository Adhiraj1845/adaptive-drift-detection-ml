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
    """Page-Hinkley sequential change-point detector (Page, 1954); O(1) time and space per update."""

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
        s.mean += (x - s.mean) / s.t
        s.m_t += (x - s.mean - self.delta)
        if s.m_t < s.m_min:
            s.m_min = s.m_t
        if s.m_t > s.m_max:
            s.m_max = s.m_t

        stat_inc = s.m_t - s.m_min
        stat_dec = s.m_max - s.m_t

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
