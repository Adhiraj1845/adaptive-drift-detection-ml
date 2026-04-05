from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector

N_PRE           = 500
N_POST          = 500
REFERENCE_SIZE  = 250
WINDOW_SIZE     = 100
DRIFT_MAGNITUDE = 1.5  # mean shift in σ
VOL_SCALE       = 1.5  # post-drift σ multiplier
FPR_TARGET      = 0.05
SEED            = 42


def generate_stream(
    n_pre: int = N_PRE,
    n_post: int = N_POST,
    drift_magnitude: float = DRIFT_MAGNITUDE,
    vol_scale: float = VOL_SCALE,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    rng  = np.random.RandomState(seed)
    pre  = rng.normal(0.0, 1.0, n_pre)
    post = rng.normal(drift_magnitude, vol_scale, n_post)
    return np.concatenate([pre, post]), n_pre


def generate_gradual_drift_stream(
    n_pre: int = N_PRE,
    n_post: int = N_POST,
    drift_magnitude: float = DRIFT_MAGNITUDE,
    vol_scale: float = VOL_SCALE,
    ramp_length: int = 200,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    rng      = np.random.RandomState(seed)
    pre      = rng.normal(0.0, 1.0, n_pre)
    ramp_n   = min(ramp_length, n_post)
    ramp_means = np.linspace(0.0, drift_magnitude, ramp_n + 1)[1:]  # exclude 0
    post_ramp  = rng.normal(ramp_means, vol_scale)
    flat_n     = n_post - ramp_n
    post_flat  = rng.normal(drift_magnitude, vol_scale, flat_n) if flat_n > 0 else np.array([])
    return np.concatenate([pre, post_ramp, post_flat]), n_pre


def generate_concept_drift_stream(
    n_pre: int = N_PRE,
    n_post: int = N_POST,
    seed: int = SEED,
) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    # X ~ N(0,1) throughout; only the label sign flips — feature detectors should be blind
    rng = np.random.RandomState(seed)
    n   = n_pre + n_post
    X   = rng.normal(0.0, 1.0, n)

    def _logistic(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    p_pre  = _logistic( X[:n_pre])
    p_post = _logistic(-X[n_pre:])
    p_all  = np.concatenate([p_pre, p_post])
    y      = (rng.uniform(size=n) < p_all).astype(int)
    return (X, y), n_pre


def sliding_scores(
    stream: np.ndarray,
    reference: list,
    window_size: int,
    score_fn,
) -> np.ndarray:
    out = np.full(len(stream), np.nan)
    for t in range(window_size, len(stream)):
        window   = stream[t - window_size : t].tolist()
        out[t]   = score_fn(reference, window)
    return out


def calibrate_threshold(pre_scores: np.ndarray, fpr: float) -> float:
    valid = pre_scores[~np.isnan(pre_scores)]
    return float(np.quantile(valid, 1.0 - fpr)) if len(valid) > 0 else 0.0


def evaluate_detector(
    scores: np.ndarray,
    drift_day: int,
    threshold: float,
) -> dict:
    pre_valid  = scores[:drift_day][~np.isnan(scores[:drift_day])]
    post_valid = scores[drift_day:][~np.isnan(scores[drift_day:])]

    fpr = float(np.mean(pre_valid  > threshold)) if len(pre_valid)  > 0 else float("nan")
    tpr = float(np.mean(post_valid > threshold)) if len(post_valid) > 0 else float("nan")

    post_alarms = np.where(scores[drift_day:] > threshold)[0]
    latency     = int(post_alarms[0]) if len(post_alarms) > 0 else None

    return {"threshold": threshold, "fpr": fpr, "tpr": tpr, "latency": latency}


def _detector_configs():
    ks  = KSTestDetector()
    psi = PSIDetector()
    js  = JSDivergenceDetector()
    return {
        "KS":  lambda ref, win: ks.ks_statistic(ref, win),
        "PSI": lambda ref, win: psi.compute_psi(ref, win),
        "JS":  lambda ref, win: js.score(ref, win),
        "Composite (KS+PSI+JS)": lambda ref, win: max(
            ks.ks_statistic(ref, win),
            psi.compute_psi(ref, win),
            js.score(ref, win),
        ),
    }


def _evaluate_streaming_detector(stream: np.ndarray, drift_day: int, detector) -> dict:
    """
    Evaluate a river-style streaming detector that exposes .update(x)
    and .drift_detected.  Returns the same metrics dict as evaluate_detector().
    """
    alarm_days = []
    for i, x in enumerate(stream):
        detector.update(float(x))
        if detector.drift_detected:
            alarm_days.append(i)

    pre_alarms  = [d for d in alarm_days if d < drift_day]
    post_alarms = [d for d in alarm_days if d >= drift_day]

    fpr     = len(pre_alarms) / max(drift_day, 1)
    tpr     = len(post_alarms) / max(len(stream) - drift_day, 1)
    latency = (post_alarms[0] - drift_day) if post_alarms else None

    return {"threshold": float("nan"), "fpr": fpr, "tpr": tpr, "latency": latency}


def _streaming_detector_configs(stream: np.ndarray, drift_day: int) -> dict[str, dict]:
    """Run ADWIN and KSWIN (from river) on the stream and return metrics dicts."""
    results = {}
    try:
        from river.drift import ADWIN, KSWIN

        results["ADWIN"] = _evaluate_streaming_detector(
            stream, drift_day, ADWIN(delta=0.002)
        )
        results["KSWIN"] = _evaluate_streaming_detector(
            stream, drift_day, KSWIN(window_size=WINDOW_SIZE, seed=SEED)
        )
    except ImportError:
        print("  [river not installed — skipping ADWIN/KSWIN baselines]")

    return results


def _run_scenario_table(stream: np.ndarray, drift_day: int, title: str) -> None:
    reference = stream[:REFERENCE_SIZE].tolist()
    configs   = _detector_configs()
    print(f"\n--- {title} ---")
    print(f"  {'Detector':<26} {'Type':>6} {'Threshold':>10} {'FPR':>8} {'TPR':>8} {'Latency':>10}")
    print("  " + "-" * 72)
    for name, fn in configs.items():
        scores    = sliding_scores(stream, reference, WINDOW_SIZE, fn)
        threshold = calibrate_threshold(scores[:drift_day], FPR_TARGET)
        ev        = evaluate_detector(scores, drift_day, threshold)
        lat = f"{ev['latency']} days" if ev["latency"] is not None else "  never"
        print(f"  {name:<26} {'Batch':>6} {ev['threshold']:>10.4f} "
              f"{ev['fpr']:>8.3f} {ev['tpr']:>8.3f} {lat:>10}")
    streaming = _streaming_detector_configs(stream, drift_day)
    for name, ev in streaming.items():
        lat = f"{ev['latency']} days" if ev["latency"] is not None else "  never"
        print(f"  {name:<26} {'Online':>6} {'adaptive':>10} "
              f"{ev['fpr']:>8.3f} {ev['tpr']:>8.3f} {lat:>10}")
    print()


def run_benchmark(
    *,
    chart_path: str = "results/chart_synthetic_benchmark.png",
    include_new_scenarios: bool = True,
) -> dict[str, dict]:
    stream, drift_day = generate_stream()
    reference         = stream[:REFERENCE_SIZE].tolist()
    configs           = _detector_configs()

    results: dict[str, dict] = {}
    for name, fn in configs.items():
        scores    = sliding_scores(stream, reference, WINDOW_SIZE, fn)
        threshold = calibrate_threshold(scores[:drift_day], FPR_TARGET)
        ev        = evaluate_detector(scores, drift_day, threshold)
        ev["scores"] = scores
        ev["detector_type"] = "Batch"
        results[name] = ev

    # Add ADWIN and KSWIN streaming baselines
    streaming = _streaming_detector_configs(stream, drift_day)
    for name, ev in streaming.items():
        ev["scores"] = np.full(len(stream), np.nan)  # no continuous score for online detectors
        ev["detector_type"] = "Online"
        results[name] = ev

    _print_table(results, drift_day)
    _save_chart(results, drift_day, chart_path)

    if include_new_scenarios:
        grad_stream, grad_day = generate_gradual_drift_stream()
        _run_scenario_table(
            grad_stream, grad_day,
            f"Gradual Drift  (ramp=200 days, Δμ={DRIFT_MAGNITUDE}σ, vol×{VOL_SCALE})",
        )

        (concept_X, _), concept_day = generate_concept_drift_stream()
        _run_scenario_table(
            concept_X, concept_day,
            "Concept Drift  (feature marginal unchanged — TPR should ≈ FPR)",
        )
        print("  ^ Confirms: feature-distribution detectors are blind to concept drift.\n")

    return results


def _print_table(results: dict, drift_day: int) -> None:
    print("\n" + "=" * 82)
    print("  Synthetic Drift Detection Benchmark  —  Abrupt Drift")
    print(f"  Stream: {N_PRE + N_POST} days   |   True drift at day {drift_day}")
    print(f"  Pre-drift:  N(0, 1)   Post-drift: N({DRIFT_MAGNITUDE:.1f}, {VOL_SCALE:.1f})"
          f"   [Δμ = {DRIFT_MAGNITUDE}σ,  vol × {VOL_SCALE}]")
    print(f"  Reference/detection window: {REFERENCE_SIZE}/{WINDOW_SIZE} days"
          f"   |   Threshold @ {FPR_TARGET*100:.0f}% FPR (batch detectors)")
    print("=" * 82)
    print(f"  {'Detector':<26} {'Type':>6} {'Threshold':>10} {'FPR':>8} {'TPR':>8} {'Latency':>10}")
    print("  " + "-" * 70)
    for name, r in results.items():
        lat   = f"{r['latency']} days" if r["latency"] is not None else "  never"
        thr   = f"{r['threshold']:.4f}" if not np.isnan(r["threshold"]) else "   adaptive"
        dtype = r.get("detector_type", "Batch")
        print(
            f"  {name:<26} {dtype:>6} {thr:>10} "
            f"{r['fpr']:>8.3f} {r['tpr']:>8.3f} {lat:>10}"
        )
    print("=" * 82)
    print("  FPR = false-alarm rate (pre-drift)   TPR = detection rate (post-drift)")
    print("  Online detectors: ADWIN/KSWIN adapt threshold internally (no fixed threshold)\n")


def _save_chart(results: dict, drift_day: int, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colours = {
            "KS":                    "#2c7bb6",
            "PSI":                   "#1a9641",
            "JS":                    "#d7191c",
            "Composite (KS+PSI+JS)": "#756bb1",
        }

        names = list(results.keys())
        n     = len(names)
        t     = np.arange(N_PRE + N_POST)

        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, name in zip(axes, names):
            r      = results[name]
            scores = r["scores"]
            thresh = r["threshold"]
            c      = colours.get(name, "#555555")

            ax.plot(t, scores, color=c, lw=1.2, label=name)
            ax.axhline(
                thresh, color="grey", lw=1, ls="--", alpha=0.8,
                label=f"Threshold = {thresh:.4f}  (FPR = {r['fpr']:.3f})"
            )
            ax.axvline(drift_day, color="#fdae61", lw=2,
                       label=f"True drift (day {drift_day})")

            post_mask = (t >= drift_day) & (scores > thresh)
            ax.fill_between(t, thresh, scores,
                            where=post_mask, alpha=0.25, color=c,
                            label=f"True positives  (TPR = {r['tpr']:.3f})")

            pre_mask = (t < drift_day) & (scores > thresh)
            ax.fill_between(t, thresh, scores,
                            where=pre_mask, alpha=0.35, color="red",
                            label="False positives")

            if r["latency"] is not None:
                ax.axvline(drift_day + r["latency"], color=c, lw=1.2, ls=":",
                           label=f"First alarm (day {drift_day + r['latency']},"
                                 f" latency = {r['latency']}d)")

            ax.set_ylabel("Detector score")
            ax.legend(loc="upper left", fontsize=8, ncol=2)
            ax.set_title(name)

        axes[-1].set_xlabel("Day")
        fig.suptitle(
            f"Synthetic Drift Benchmark  —  Δμ = {DRIFT_MAGNITUDE}σ, vol×{VOL_SCALE}",
            fontsize=12,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved: {path}")
    except Exception as e:
        print(f"  [Chart skipped] {e}")


if __name__ == "__main__":
    chart = sys.argv[1] if len(sys.argv) > 1 else "results/chart_synthetic_benchmark.png"
    run_benchmark(chart_path=chart)
