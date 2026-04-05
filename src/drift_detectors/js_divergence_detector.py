import numpy as np


class JSDivergenceDetector:
    # bin edges span combined ref+cur so non-overlapping ranges are scored near ln(2)

    def __init__(self, bins: int = 20):
        self.bins = bins

    def _pmf_pair(self, ref, cur):
        ref = np.asarray(ref, dtype=float)
        cur = np.asarray(cur, dtype=float)
        combined = np.concatenate([ref, cur])
        lo, hi = combined.min(), combined.max()
        if lo == hi:
            return np.array([1.0]), np.array([1.0])
        edges = np.linspace(lo, hi, self.bins + 1)
        p = np.histogram(ref, bins=edges)[0].astype(float)
        q = np.histogram(cur, bins=edges)[0].astype(float)
        p /= p.sum() + 1e-12
        q /= q.sum() + 1e-12
        return p, q

    def score(self, ref, cur) -> float:
        p, q = self._pmf_pair(ref, cur)
        m = 0.5 * (p + q)

        def kl(a, b):
            return float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
