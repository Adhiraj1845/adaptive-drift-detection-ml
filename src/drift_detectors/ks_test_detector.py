from math import sqrt
from typing import Sequence


class KSTestDetector:
    """Two-sample Kolmogorov-Smirnov drift detector.

    Measures the maximum absolute difference between two empirical CDFs:
        D = sup_x |F_ref(x) - F_cur(x)|

    D is bounded in [0, 1] and requires no distributional assumptions.
    Sensitive to shifts anywhere in the distribution, including tails.
    Critical value at alpha=0.05: D_alpha ≈ 1.36 * sqrt((n+m)/(n*m)).

    Parameters
    ----------
    p_threshold : float
        Detection threshold on D (not a p-value). Default 0.05.
    """

    def __init__(self, p_threshold: float = 0.05):
        self.threshold = float(p_threshold)

    def detect(self, reference_data: Sequence[float], new_data: Sequence[float]) -> bool:
        return self.ks_statistic(reference_data, new_data) > self.threshold

    @staticmethod
    def critical_value(alpha: float, n: int, m: int) -> float:
        # D_alpha ≈ c(alpha) * sqrt((n+m)/(n*m)); c: 0.10→1.22, 0.05→1.36, 0.01→1.63
        if n <= 0 or m <= 0:
            return 0.0
        c = 1.63 if alpha <= 0.01 else (1.36 if alpha <= 0.05 else 1.22)
        return float(c * sqrt((n + m) / (n * m)))

    def ks_statistic(self, a: Sequence[float], b: Sequence[float]) -> float:
        a = sorted(a)
        b = sorted(b)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0

        i = j = 0
        d_max = 0.0

        while i < n and j < m:
            x = a[i] if a[i] <= b[j] else b[j]
            while i < n and a[i] == x:
                i += 1
            while j < m and b[j] == x:
                j += 1
            d = abs(i / n - j / m)
            if d > d_max:
                d_max = d

        while i < n: # drain tail of a
            x = a[i]
            while i < n and a[i] == x:
                i += 1
            d = abs(i / n - j / m)
            if d > d_max:
                d_max = d

        while j < m: # drain tail of b
            x = b[j]
            while j < m and b[j] == x:
                j += 1
            d = abs(i / n - j / m)
            if d > d_max:
                d_max = d

        return float(d_max)
