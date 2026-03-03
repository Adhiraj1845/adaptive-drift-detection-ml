import numpy as np


class JSDivergenceDetector:
    """
    Jensen–Shannon divergence using histogram approximation.
    """

    def __init__(self, bins: int = 20):
        self.bins = bins

    def _hist(self, x):
        hist, _ = np.histogram(x, bins=self.bins, density=True)
        hist = hist / (hist.sum() + 1e-12)
        return hist

    def score(self, ref, cur):
        p = self._hist(ref)
        q = self._hist(cur)

        m = 0.5 * (p + q)

        def kl(a, b):
            return np.sum(a * np.log((a + 1e-12) / (b + 1e-12)))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
