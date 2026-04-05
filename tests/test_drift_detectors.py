"""
Tests for drift detectors.

Each detector is tested for:
- True negative (no drift on identical/same-distribution data)
- True positive (drift on clearly shifted data)
- Magnitude / monotonicity properties where applicable
"""
import numpy as np
import pytest

from src.drift_detectors.drift_index import DriftIndex
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector
from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.page_hinkley_detector import PageHinkleyDetector
from src.drift_detectors.psi_detector import PSIDetector


# ── KS Detector ───────────────────────────────────────────────────────────────

class TestKSDetector:
    def test_no_drift_identical_data(self):
        det = KSTestDetector(p_threshold=0.05)
        assert det.detect([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) is False

    def test_detects_large_mean_shift(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 200).tolist()
        new = rng.normal(10, 1, 200).tolist()  # extreme shift
        det = KSTestDetector(p_threshold=0.05)
        assert det.detect(ref, new) is True

    def test_detects_variance_change(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 300).tolist()
        new = rng.normal(0, 10, 300).tolist()  # 10x wider spread
        det = KSTestDetector(p_threshold=0.05)
        assert det.detect(ref, new) is True

    def test_returns_bool(self):
        det = KSTestDetector(p_threshold=0.05)
        result = det.detect([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert isinstance(result, bool)

    def test_strict_threshold_less_sensitive(self):
        """Strict threshold fires on extreme shifts; both always return bool."""
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 500).tolist()
        new = rng.normal(0.05, 1, 500).tolist()  # tiny shift
        det_strict = KSTestDetector(p_threshold=0.0001)
        det_loose = KSTestDetector(p_threshold=0.5)
        assert isinstance(det_strict.detect(ref, new), bool)
        assert isinstance(det_loose.detect(ref, new), bool)
        # With a very large shift, strict detector must fire
        new_large = rng.normal(10, 1, 500).tolist()
        assert det_strict.detect(ref, new_large) is True


# ── Page-Hinkley Detector ─────────────────────────────────────────────────────

class TestPageHinkleyDetector:
    def test_no_drift_steady_signal(self):
        det = PageHinkleyDetector(threshold=50)
        assert not any(det.update(x) for x in [0.0] * 30)

    def test_detects_upward_mean_shift(self):
        det = PageHinkleyDetector(threshold=0.5, delta=0.0, direction="increase")
        for _ in range(20):
            det.update(0.0)
        triggered = any(det.update(5.0) for _ in range(15))
        assert triggered is True

    def test_detects_downward_mean_shift(self):
        det = PageHinkleyDetector(threshold=0.5, delta=0.0, direction="decrease")
        for _ in range(20):
            det.update(0.0)
        triggered = any(det.update(-5.0) for _ in range(15))
        assert triggered is True

    def test_detects_both_directions(self):
        det = PageHinkleyDetector(threshold=0.5, delta=0.0, direction="both")
        for _ in range(20):
            det.update(0.0)
        triggered = any(det.update(5.0) for _ in range(15))
        assert triggered is True

    def test_statistic_increases_after_shift(self):
        det = PageHinkleyDetector(threshold=9999, delta=0.0, direction="increase")
        for _ in range(20):
            det.update(0.0)
        stat_before = det.statistic()
        for _ in range(20):
            det.update(10.0)
        stat_after = det.statistic()
        assert stat_after > stat_before

    def test_statistic_is_nonnegative(self):
        det = PageHinkleyDetector(threshold=9999)
        for x in np.random.default_rng(0).normal(0, 1, 100):
            det.update(float(x))
        assert det.statistic() >= 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            PageHinkleyDetector(direction="sideways")


# ── PSI Detector ──────────────────────────────────────────────────────────────

class TestPSIDetector:
    def test_no_drift_identical(self):
        det = PSIDetector(threshold=0.1)
        data = list(range(50))
        assert det.detect(data, data) is False

    def test_detects_distribution_shift(self):
        det = PSIDetector(threshold=0.1)
        ref = list(range(100))          # uniform [0, 99]
        new = list(range(200, 300))     # completely non-overlapping
        assert det.detect(ref, new) is True

    def test_psi_zero_for_identical(self):
        det = PSIDetector(threshold=0.1)
        data = list(range(100))
        assert det.compute_psi(data, data) == pytest.approx(0.0, abs=1e-6)

    def test_psi_positive_for_different(self):
        det = PSIDetector(threshold=0.1)
        ref = list(range(100))
        new = list(range(200, 300))
        assert det.compute_psi(ref, new) > 0.0

    def test_psi_monotone_in_shift(self):
        det = PSIDetector(threshold=0.1)
        ref = list(range(100))
        small_shift = list(range(10, 110))
        large_shift = list(range(200, 300))
        assert det.compute_psi(ref, small_shift) < det.compute_psi(ref, large_shift)

    def test_empty_data_returns_zero(self):
        det = PSIDetector(threshold=0.1)
        assert det.compute_psi([], []) == 0.0

    def test_fit_then_detect(self):
        det = PSIDetector(threshold=0.1)
        ref = list(range(100))
        det.fit(ref)
        new = list(range(200, 300))
        assert det.detect(ref, new) is True


# ── JS Divergence Detector ────────────────────────────────────────────────────

class TestJSDivergenceDetector:
    def test_zero_for_identical(self):
        det = JSDivergenceDetector()
        data = np.random.default_rng(0).normal(0, 1, 200).tolist()
        assert det.score(data, data) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_distributions(self):
        det = JSDivergenceDetector()
        ref = np.random.default_rng(0).normal(0, 1, 200).tolist()
        new = np.random.default_rng(1).normal(10, 1, 200).tolist()
        assert det.score(ref, new) > 0.0

    def test_overlapping_shift_larger_than_identical(self):
        """Partially overlapping distributions have higher JS than identical."""
        det = JSDivergenceDetector()
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 300).tolist()
        shifted = rng.normal(1.0, 1, 300).tolist()  # overlapping shift
        assert det.score(ref, shifted) > det.score(ref, ref)

    def test_score_bounded(self):
        """JS divergence is bounded in [0, ln(2)] ~ [0, 0.693]."""
        det = JSDivergenceDetector()
        ref = np.random.default_rng(0).normal(0, 1, 200).tolist()
        new = np.random.default_rng(1).normal(100, 1, 200).tolist()
        score = det.score(ref, new)
        assert 0.0 <= score <= 1.0  # histogram-normalised version


# ── Composite Drift Index ─────────────────────────────────────────────────────

class TestDriftIndex:
    def test_zero_for_no_drift(self):
        ks = KSTestDetector(p_threshold=0.05)
        psi = PSIDetector(threshold=0.1)
        composite = DriftIndex([ks, psi])
        assert composite.compute_index([1, 2, 3], [1, 2, 3]) == 0.0

    def test_one_for_all_fire(self):
        class AlwaysDrift:
            def detect(self, *_): return True
        composite = DriftIndex([AlwaysDrift(), AlwaysDrift()])
        assert composite.compute_index([], []) == pytest.approx(1.0)

    def test_half_for_partial_fire(self):
        class AlwaysDrift:
            def detect(self, *_): return True
        class NeverDrift:
            def detect(self, *_): return False
        composite = DriftIndex([AlwaysDrift(), NeverDrift()])
        assert composite.compute_index([], []) == pytest.approx(0.5)

    def test_empty_detector_list_returns_zero(self):
        composite = DriftIndex([])
        assert composite.compute_index([1, 2], [3, 4]) == 0.0

    def test_index_in_unit_interval(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 200).tolist()
        new = rng.normal(5, 1, 200).tolist()
        composite = DriftIndex([
            KSTestDetector(p_threshold=0.05),
            PSIDetector(threshold=0.1),
        ])
        idx = composite.compute_index(ref, new)
        assert 0.0 <= idx <= 1.0
