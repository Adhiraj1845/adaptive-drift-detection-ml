# src/drift_detectors/psi_detector.py
import math
from bisect import bisect_right
from typing import Sequence, List, Optional


class PSIDetector:
    """
    Batch PSI detector using reference quantile bins.

    - Bins are derived from the reference distribution (quantiles).
    - Handles duplicate quantile edges by collapsing bins safely.
    - Optional fit() allows caching edges for repeated streaming calls.

    Matches tests:
      detect(reference_data, new_data) -> bool
    """

    def __init__(self, threshold: float = 0.2, num_bins: int = 10, epsilon: float = 1e-6):
        self.threshold = float(threshold)
        self.num_bins = int(num_bins)
        self.epsilon = float(epsilon)
        self._edges: Optional[List[float]] = None  # cached edges after fit()

    def fit(self, reference_data: Sequence[float]) -> None:
        """Cache bin edges from reference data for faster repeated calls."""
        ref = list(reference_data)
        if len(ref) == 0:
            self._edges = None
            return
        ref_sorted = sorted(ref)
        self._edges = self._quantile_edges(ref_sorted, self.num_bins)

    def detect(self, reference_data: Sequence[float], new_data: Sequence[float]) -> bool:
        return self.compute_psi(reference_data, new_data) > self.threshold

    def compute_psi(self, reference_data: Sequence[float], new_data: Sequence[float]) -> float:
        ref = list(reference_data)
        new = list(new_data)

        if len(ref) == 0 or len(new) == 0:
            return 0.0

        # Fast path: both constant and equal -> PSI = 0
        if min(ref) == max(ref) and min(new) == max(new) and ref[0] == new[0]:
            return 0.0

        # Decide edges: use cached if available and matches current reference regime,
        # otherwise compute from ref.
        if self._edges is None:
            ref_sorted = sorted(ref)
            edges = self._quantile_edges(ref_sorted, self.num_bins)
        else:
            edges = self._edges

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

        # Raw quantile edges (may have duplicates)
        raw = [sorted_data[0]]
        for i in range(1, num_bins):
            q = i / num_bins
            idx = int(q * (n - 1))
            raw.append(sorted_data[idx])
        raw.append(sorted_data[-1])

        # If flat reference, return 2-edge "single bin"
        if raw[0] == raw[-1]:
            return [raw[0], raw[-1]]

        # Collapse duplicates to ensure edges are strictly increasing.
        # This reduces the effective number of bins (fine and correct).
        edges = [raw[0]]
        for v in raw[1:]:
            if v > edges[-1]:
                edges.append(v)

        # Safety: if everything collapsed (shouldn't happen unless numerical weirdness)
        if len(edges) < 2:
            edges = [raw[0], raw[-1]]

        return edges

    def _hist_probs(self, data: List[float], edges: List[float]) -> List[float]:
        k = max(len(edges) - 1, 1)
        counts = [0] * k

        lo = edges[0]
        hi = edges[-1]
        if lo == hi:
            # Single bin: everything goes here
            return [1.0]

        for x in data:
            idx = self._find_bin(x, edges)
            counts[idx] += 1

        total = sum(counts)
        if total == 0:
            return [0.0] * k

        return [c / total for c in counts]

    def _find_bin(self, x: float, edges: List[float]) -> int:
        # bins are [edges[i], edges[i+1]) except last which includes right edge
        k = len(edges) - 1
        if k <= 1:
            return 0

        if x <= edges[0]:
            return 0
        if x >= edges[-1]:
            return k - 1

        # bisect_right gives insertion point; convert to bin index
        j = bisect_right(edges, x) - 1
        if j < 0:
            return 0
        if j >= k:
            return k - 1
        return j
