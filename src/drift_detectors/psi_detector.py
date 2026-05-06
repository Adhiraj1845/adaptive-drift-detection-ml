import math
from bisect import bisect_right
from typing import Sequence, List, Optional


class PSIDetector:
    """Population Stability Index drift detector; epsilon smoothing prevents log(0)."""

    def __init__(self, threshold: float = 0.2, num_bins: int = 10, epsilon: float = 1e-6):
        self.threshold = float(threshold)
        self.num_bins = int(num_bins)
        self.epsilon = float(epsilon)
        self._edges: Optional[List[float]] = None  # cached after fit()

    def fit(self, reference_data: Sequence[float]) -> None:
        ref = list(reference_data)
        if len(ref) == 0:
            self._edges = None
            return
        self._edges = self._quantile_edges(sorted(ref), self.num_bins)

    def detect(self, reference_data: Sequence[float], new_data: Sequence[float]) -> bool:
        return self.compute_psi(reference_data, new_data) > self.threshold

    def compute_psi(self, reference_data: Sequence[float], new_data: Sequence[float]) -> float:
        ref = list(reference_data)
        new = list(new_data)
        if len(ref) == 0 or len(new) == 0:
            return 0.0
        if min(ref) == max(ref) and min(new) == max(new) and ref[0] == new[0]:
            return 0.0
        edges = self._edges if self._edges is not None else self._quantile_edges(sorted(ref), self.num_bins)
        ref_probs = self._hist_probs(ref, edges)
        new_probs = self._hist_probs(new, edges)
        psi = 0.0
        for p_ref, p_new in zip(ref_probs, new_probs):
            p_ref = max(float(p_ref), self.epsilon)
            p_new = max(float(p_new), self.epsilon)
            psi += (p_new - p_ref) * math.log(p_new / p_ref)
        return float(psi)

    def _quantile_edges(self, sorted_data: List[float], num_bins: int) -> List[float]:
        n = len(sorted_data)
        if n == 0:
            return [0.0, 1.0]
        raw = [sorted_data[0]]
        for i in range(1, num_bins):
            raw.append(sorted_data[int(i / num_bins * (n - 1))])
        raw.append(sorted_data[-1])
        if raw[0] == raw[-1]:
            return [raw[0], raw[-1]]
        edges = [raw[0]]
        for v in raw[1:]:
            if v > edges[-1]:
                edges.append(v)
        if len(edges) < 2:
            edges = [raw[0], raw[-1]]
        return edges

    def _hist_probs(self, data: List[float], edges: List[float]) -> List[float]:
        k = max(len(edges) - 1, 1)
        if edges[0] == edges[-1]:
            return [1.0]
        counts = [0] * k
        for x in data:
            counts[self._find_bin(x, edges)] += 1
        total = sum(counts)
        if total == 0:
            return [0.0] * k
        return [c / total for c in counts]

    def _find_bin(self, x: float, edges: List[float]) -> int:
        k = len(edges) - 1
        if k <= 1:
            return 0
        if x <= edges[0]:
            return 0
        if x >= edges[-1]:
            return k - 1
        j = bisect_right(edges, x) - 1
        return max(0, min(j, k - 1))
