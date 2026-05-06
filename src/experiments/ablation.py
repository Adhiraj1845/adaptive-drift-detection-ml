from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.experiments.synthetic_benchmark import (
    generate_stream,
    generate_gradual_drift_stream,
    generate_concept_drift_stream,
    sliding_scores,
    calibrate_threshold,
    evaluate_detector,
    REFERENCE_SIZE,
    WINDOW_SIZE,
    FPR_TARGET,
    N_POST,
)
from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector

DRIFT_MAGNITUDES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
VOL_SCALE        = 1.5
N_SEEDS          = 20  # seeds per magnitude

CONFIG_NAMES = ["KS", "PSI", "JS", "Composite"]
COLOURS      = {
    "KS":        "#2c7bb6",
    "PSI":       "#1a9641",
    "JS":        "#d7191c",
    "Composite": "#756bb1",
}
STYLES = {"KS": "--", "PSI": ":", "JS": "-.", "Composite": "-"}
WIDTHS = {"KS": 1.5, "PSI": 1.5, "JS": 1.5, "Composite": 2.4}


def _make_score_fns() -> dict:
    ks  = KSTestDetector()
    psi = PSIDetector()
    js  = JSDivergenceDetector()
    return {
        "KS":        lambda ref, win: ks.ks_statistic(ref, win),
        "PSI":       lambda ref, win: psi.compute_psi(ref, win),
        "JS":        lambda ref, win: js.score(ref, win),
        "Composite": lambda ref, win: max(
            ks.ks_statistic(ref, win),
            psi.compute_psi(ref, win),
            js.score(ref, win),
        ),
    }


def run_ablation(
    *,
    chart_path: str = "results/chart_ablation.png",
) -> dict:
    results: dict[str, dict[float, dict]] = {c: {} for c in CONFIG_NAMES}

    for mag in DRIFT_MAGNITUDES:
        seed_tpr     = {c: [] for c in CONFIG_NAMES}
        seed_fpr     = {c: [] for c in CONFIG_NAMES}
        seed_latency = {c: [] for c in CONFIG_NAMES}
        seed_never   = {c: [] for c in CONFIG_NAMES}

        for seed in range(N_SEEDS):
            stream, drift_day = generate_stream(
                drift_magnitude=mag, vol_scale=VOL_SCALE, seed=seed
            )
            reference = stream[:REFERENCE_SIZE].tolist()
            fns       = _make_score_fns()

            for name in CONFIG_NAMES:
                scores    = sliding_scores(stream, reference, WINDOW_SIZE, fns[name])
                threshold = calibrate_threshold(scores[:drift_day], FPR_TARGET)
                ev        = evaluate_detector(scores, drift_day, threshold)

                seed_tpr[name].append(ev["tpr"])
                seed_fpr[name].append(ev["fpr"])
                seed_never[name].append(0 if ev["latency"] is not None else 1)
                # Use N_POST as a ceiling latency for "never detected" seeds
                seed_latency[name].append(
                    ev["latency"] if ev["latency"] is not None else N_POST
                )

        for name in CONFIG_NAMES:
            results[name][mag] = {
                "tpr":         float(np.mean(seed_tpr[name])),
                "tpr_std":     float(np.std(seed_tpr[name])),
                "fpr":         float(np.mean(seed_fpr[name])),
                "latency_med": float(np.median(seed_latency[name])),
                "never_pct":   float(np.mean(seed_never[name])),
            }

    _print_ablation_table(results)
    _save_ablation_chart(results, chart_path)
    return results


def _print_ablation_table(results: dict) -> None:
    mags  = DRIFT_MAGNITUDES
    names = CONFIG_NAMES

    print("\n" + "=" * 85)
    print("  Detector Ablation Study")
    print(f"  {N_SEEDS} seeds per magnitude  |  FPR calibrated at {FPR_TARGET * 100:.0f}%  |  vol × {VOL_SCALE:.1f}")
    print("=" * 85)

    print(f"\n  True Positive Rate (TPR) at {FPR_TARGET * 100:.0f}% FPR  "
          f"[mean ± std over {N_SEEDS} seeds]:\n")
    col  = 15
    print("  " + f"{'Δμ (σ)':<10}" + "".join(f"{n:>{col}}" for n in names))
    print("  " + "-" * (10 + col * len(names)))
    for mag in mags:
        row = f"  {mag:<10.2f}"
        for n in names:
            r   = results[n][mag]
            row += f"  {r['tpr']:.3f}±{r['tpr_std']:.3f}"
        print(row)

    print(f"\n  Median Detection Latency (days after drift)  "
          f"[{N_POST} = never detected]:\n")
    print("  " + f"{'Δμ (σ)':<10}" + "".join(f"{n:>{col}}" for n in names))
    print("  " + "-" * (10 + col * len(names)))
    for mag in mags:
        row = f"  {mag:<10.2f}"
        for n in names:
            lat = results[n][mag]["latency_med"]
            row += f"  {lat:>13.1f}"
        print(row)

    print(f"\n  Fraction of seeds where drift was NEVER detected:\n")
    print("  " + f"{'Δμ (σ)':<10}" + "".join(f"{n:>{col}}" for n in names))
    print("  " + "-" * (10 + col * len(names)))
    for mag in mags:
        row = f"  {mag:<10.2f}"
        for n in names:
            pct = results[n][mag]["never_pct"]
            row += f"  {pct:>13.2f}"
        print(row)

    print("\n" + "=" * 85)
    print("  Composite = max(KS, PSI, JS) — the full system detector")
    print("  Lower latency and higher TPR at a given FPR = better\n")



def _save_ablation_chart(results: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mags  = DRIFT_MAGNITUDES
        names = CONFIG_NAMES

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for name in names:
            tprs = [results[name][m]["tpr"]         for m in mags]
            stds = [results[name][m]["tpr_std"]      for m in mags]
            lats = [results[name][m]["latency_med"]  for m in mags]

            lo = [max(0.0, t - s) for t, s in zip(tprs, stds)]
            hi = [min(1.0, t + s) for t, s in zip(tprs, stds)]

            c  = COLOURS[name]
            ls = STYLES[name]
            lw = WIDTHS[name]

            ax1.plot(mags, tprs, color=c, ls=ls, lw=lw, marker="o", ms=5, label=name)
            ax1.fill_between(mags, lo, hi, alpha=0.12, color=c)

            ax2.plot(mags, lats, color=c, ls=ls, lw=lw, marker="o", ms=5, label=name)

        ax1.axhline(0.8, color="grey", lw=0.8, ls="--", alpha=0.5, label="80% TPR")
        ax1.axhline(FPR_TARGET, color="red", lw=0.8, ls=":", alpha=0.5,
                    label=f"FPR target ({FPR_TARGET * 100:.0f}%)")
        ax2.axhline(N_POST, color="grey", lw=0.8, ls="--", alpha=0.5,
                    label=f"Never detected ({N_POST}d ceiling)")

        ax1.set_xlabel("Drift magnitude (Δμ / σ)")
        ax1.set_ylabel(f"True Positive Rate @ {FPR_TARGET * 100:.0f}% FPR")
        ax1.set_title("Detection Sensitivity")
        ax1.legend(fontsize=9)
        ax1.set_ylim(-0.05, 1.10)

        ax2.set_xlabel("Drift magnitude (Δμ / σ)")
        ax2.set_ylabel("Median detection latency (days)   ↓ better")
        ax2.set_title("Detection Latency")
        ax2.legend(fontsize=9)

        fig.suptitle(
            f"Detector Ablation Study  ({N_SEEDS} seeds per magnitude, vol×{VOL_SCALE})",
            fontsize=12,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Ablation chart saved: {path}")
    except Exception as e:
        print(f"  [Chart skipped] {e}")


def run_ablation_all_scenarios(
    *,
    chart_path: str = "results/chart_ablation_scenarios.png",
) -> dict:
    """Run detector ablation over abrupt, gradual, and concept drift scenarios."""
    all_results: dict = {}

    print("\n=== Ablation Scenario 1/3: Abrupt Drift ===")
    all_results["abrupt"] = run_ablation(
        chart_path=chart_path.replace(".png", "_abrupt.png")
    )

    print("\n=== Ablation Scenario 2/3: Gradual Drift (ramp=200 days) ===")
    gradual_results: dict[str, dict[float, dict]] = {c: {} for c in CONFIG_NAMES}

    for mag in DRIFT_MAGNITUDES:
        seed_tpr     = {c: [] for c in CONFIG_NAMES}
        seed_fpr     = {c: [] for c in CONFIG_NAMES}
        seed_latency = {c: [] for c in CONFIG_NAMES}
        seed_never   = {c: [] for c in CONFIG_NAMES}

        for seed in range(N_SEEDS):
            stream, drift_day = generate_gradual_drift_stream(
                drift_magnitude=mag, vol_scale=VOL_SCALE, seed=seed
            )
            reference = stream[:REFERENCE_SIZE].tolist()
            fns       = _make_score_fns()

            for name in CONFIG_NAMES:
                scores    = sliding_scores(stream, reference, WINDOW_SIZE, fns[name])
                threshold = calibrate_threshold(scores[:drift_day], FPR_TARGET)
                ev        = evaluate_detector(scores, drift_day, threshold)
                seed_tpr[name].append(ev["tpr"])
                seed_fpr[name].append(ev["fpr"])
                seed_never[name].append(0 if ev["latency"] is not None else 1)
                seed_latency[name].append(
                    ev["latency"] if ev["latency"] is not None else N_POST
                )

        for name in CONFIG_NAMES:
            gradual_results[name][mag] = {
                "tpr":         float(np.mean(seed_tpr[name])),
                "tpr_std":     float(np.std(seed_tpr[name])),
                "fpr":         float(np.mean(seed_fpr[name])),
                "latency_med": float(np.median(seed_latency[name])),
                "never_pct":   float(np.mean(seed_never[name])),
            }

    _print_ablation_table(gradual_results)
    all_results["gradual"] = gradual_results

    print("\n=== Ablation Scenario 3/3: Concept Drift (feature-label sign flip) ===")
    print(f"  X ~ N(0,1) throughout — feature marginal does NOT change.")
    print(f"  Expected: all feature detectors TPR ≈ FPR ≈ {FPR_TARGET}  "
          f"({N_SEEDS} seeds)\n")

    concept_results: dict[str, dict] = {c: {} for c in CONFIG_NAMES}
    seed_tpr = {c: [] for c in CONFIG_NAMES}
    seed_fpr = {c: [] for c in CONFIG_NAMES}

    for seed in range(N_SEEDS):
        (X, _), drift_day = generate_concept_drift_stream(seed=seed)
        reference = X[:REFERENCE_SIZE].tolist()
        fns       = _make_score_fns()

        for name in CONFIG_NAMES:
            scores    = sliding_scores(X, reference, WINDOW_SIZE, fns[name])
            threshold = calibrate_threshold(scores[:drift_day], FPR_TARGET)
            ev        = evaluate_detector(scores, drift_day, threshold)
            seed_tpr[name].append(ev["tpr"])
            seed_fpr[name].append(ev["fpr"])

    print(f"  {'Detector':<14} {'mean TPR':>9} {'mean FPR':>9}")
    print("  " + "-" * 34)
    for name in CONFIG_NAMES:
        tpr_m = float(np.mean(seed_tpr[name]))
        fpr_m = float(np.mean(seed_fpr[name]))
        concept_results[name] = {"tpr": tpr_m, "fpr": fpr_m}
        print(f"  {name:<14} {tpr_m:>9.3f} {fpr_m:>9.3f}")
    print("  (TPR ≈ FPR confirms feature detectors cannot detect concept drift)\n")
    all_results["concept"] = concept_results
    return all_results


if __name__ == "__main__":
    chart = sys.argv[1] if len(sys.argv) > 1 else "results/chart_ablation.png"
    run_ablation(chart_path=chart)
    run_ablation_all_scenarios(
        chart_path=chart.replace(".png", "_scenarios.png")
    )
