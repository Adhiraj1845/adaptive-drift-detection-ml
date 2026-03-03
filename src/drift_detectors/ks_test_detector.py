# src/drift_detectors/ks_test_detector.py
from math import sqrt
from typing import Sequence


class KSTestDetector:
    """
    Batch KS detector (two-sample), computing KS statistic D.

    IMPORTANT (compatibility): p_threshold is treated as a threshold on D,
    not a p-value, because we are not computing exact p-values (no scipy).

    Matches tests:
      detect(reference_data, new_data) -> bool
    """

    def __init__(self, p_threshold: float = 0.05):
        self.threshold = float(p_threshold)

    def detect(self, reference_data: Sequence[float], new_data: Sequence[float]) -> bool:
        d = self.ks_statistic(reference_data, new_data)
        return d > self.threshold

    @staticmethod
    def critical_value(alpha: float, n: int, m: int) -> float:
        """
        Approximate two-sample KS critical value:
          D_alpha ≈ c(alpha) * sqrt((n+m)/(n*m))
        with common constants c(alpha):
          alpha=0.10 -> 1.22
          alpha=0.05 -> 1.36
          alpha=0.01 -> 1.63

        Useful for dissertation discussion even if you keep threshold as D.
        """
        if n <= 0 or m <= 0:
            return 0.0
        a = float(alpha)
        if a <= 0.01:
            c = 1.63
        elif a <= 0.05:
            c = 1.36
        else:
            c = 1.22
        return float(c * sqrt((n + m) / (n * m)))

    def ks_statistic(self, a: Sequence[float], b: Sequence[float]) -> float:
        a = sorted(a)
        b = sorted(b)
        n = len(a)
        m = len(b)

        if n == 0 or m == 0:
            return 0.0

        i = 0
        j = 0
        d_max = 0.0

        # Walk through sorted values
        while i < n and j < m:
            if a[i] <= b[j]:
                x = a[i]
            else:
                x = b[j]

            while i < n and a[i] == x:
                i += 1
            while j < m and b[j] == x:
                j += 1

            cdf_a = i / n
            cdf_b = j / m
            d = abs(cdf_a - cdf_b)
            if d > d_max:
                d_max = d

        # Handle tails: if one list ended earlier, its cdf is 1.0 while other may not be
        while i < n:
            x = a[i]
            while i < n and a[i] == x:
                i += 1
            cdf_a = i / n
            cdf_b = j / m
            d = abs(cdf_a - cdf_b)
            if d > d_max:
                d_max = d

        while j < m:
            x = b[j]
            while j < m and b[j] == x:
                j += 1
            cdf_a = i / n
            cdf_b = j / m
            d = abs(cdf_a - cdf_b)
            if d > d_max:
                d_max = d

        return float(d_max)
