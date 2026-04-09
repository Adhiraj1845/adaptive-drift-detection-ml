import numpy as np


class JSDivergenceDetector:
    """Jensen-Shannon divergence drift detector.

    Symmetrised, smoothed version of KL divergence bounded in [0, ln(2)]:
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M),  M = 0.5*(P+Q)

    Bin edges span the combined reference+current range so that
    non-overlapping distributions score near ln(2) ≈ 0.693.
    Normalise to [0, 1] by dividing by ln(2) for use as a drift score.

    Unlike KL divergence, JSD is symmetric and always finite — useful
    for detecting correlation-regime shifts where marginals are stable
    but joint distributions change (e.g., equity-bond correlation inversion).

    Parameters
    ----------
    bins : int
        Number of histogram bins. Default 20.
    """

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
