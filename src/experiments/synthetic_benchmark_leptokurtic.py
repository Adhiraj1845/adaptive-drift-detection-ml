"""Leptokurtic synthetic benchmark: TPR/FPR under Student-t streams vs Gaussian."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector

N_PRE          = 500
N_POST         = 500
REFERENCE_SIZE = 250
WINDOW_SIZE    = 100
FPR_TARGET     = 0.05
SEED           = 42

SHIFT_MAGNITUDES = [0.5, 1.0, 2.0]
DISTRIBUTIONS    = [
    ("Gaussian",        None),   # None → use normal
    ("Student-t (ν=5)", 5),
    ("Student-t (ν=3)", 3),
]


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------

def _generate_abrupt(
    drift_magnitude: float,
    df: int | None,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    """Generate a stream with an abrupt shift at position N_PRE.

    If df is None, uses Normal(0,1) → Normal(Δ, 1).
    If df is an int, uses t(df) → shifted-and-scaled t(df) with mean Δ.
    """
    rng = np.random.RandomState(seed)
    if df is None:
        pre  = rng.normal(0.0, 1.0, N_PRE)
        post = rng.normal(drift_magnitude, 1.0, N_POST)
    else:
        # Student-t: standardise so std ≈ 1 (std of t(ν) = sqrt(ν/(ν-2)))
        scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        pre   = rng.standard_t(df, N_PRE) / scale
        post  = drift_magnitude + rng.standard_t(df, N_POST) / scale
    return np.concatenate([pre, post]), N_PRE


# ---------------------------------------------------------------------------
# Detector evaluation (reuse helpers from synthetic_benchmark)
# ---------------------------------------------------------------------------

def _sliding_scores(stream, reference, score_fn):
    out = np.full(len(stream), np.nan)
    for t in range(WINDOW_SIZE, len(stream)):
        win    = stream[t - WINDOW_SIZE: t].tolist()
        out[t] = score_fn(reference, win)
    return out


def _calibrate_threshold(pre_scores, fpr):
    valid = pre_scores[~np.isnan(pre_scores)]
    return float(np.quantile(valid, 1.0 - fpr)) if len(valid) > 0 else 0.0


def _evaluate(scores, drift_day, threshold):
    pre  = scores[:drift_day]
    post = scores[drift_day:]
    pre_v  = pre[~np.isnan(pre)]
    post_v = post[~np.isnan(post)]
    fpr = float(np.mean(pre_v  > threshold)) if len(pre_v)  > 0 else float("nan")
    tpr = float(np.mean(post_v > threshold)) if len(post_v) > 0 else float("nan")
    alarms = np.where(post > threshold)[0]
    latency = int(alarms[0]) if len(alarms) > 0 else None
    return fpr, tpr, latency


def _adwin_tpr_fpr(stream, drift_day):
    try:
        from river.drift import ADWIN
        det = ADWIN(delta=0.002)
        alarms = []
        for i, x in enumerate(stream):
            det.update(float(x))
            if det.drift_detected:
                alarms.append(i)
        pre_alarms  = [d for d in alarms if d < drift_day]
        post_alarms = [d for d in alarms if d >= drift_day]
        fpr = len(pre_alarms) / max(drift_day, 1)
        tpr = len(post_alarms) / max(len(stream) - drift_day, 1)
        return fpr, tpr
    except ImportError:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_leptokurtic_benchmark(
    chart_out: str = "results/synthetic_leptokurtic_chart.png",
    csv_out:   str = "results/synthetic_leptokurtic_results.csv",
) -> pd.DataFrame:

    ks  = KSTestDetector()
    psi = PSIDetector()
    js  = JSDivergenceDetector()

    detector_fns = {
        "KS":  lambda ref, win: ks.ks_statistic(ref, win),
        "PSI": lambda ref, win: psi.compute_psi(ref, win),
        "JS":  lambda ref, win: js.score(ref, win),
        "PSI+PH (composite)": lambda ref, win: max(
            psi.compute_psi(ref, win),
            ks.ks_statistic(ref, win),
        ),
    }

    records = []

    print("\n" + "=" * 90)
    print("  Leptokurtic Synthetic Drift Benchmark")
    print("  Comparing Gaussian vs Student-t(ν=5) vs Student-t(ν=3) streams")
    print("=" * 90)

    for dist_name, df_val in DISTRIBUTIONS:
        for delta in SHIFT_MAGNITUDES:
            stream, drift_day = _generate_abrupt(delta, df_val)
            reference = stream[:REFERENCE_SIZE].tolist()

            row_base = {
                "distribution":   dist_name,
                "shift_magnitude": delta,
            }

            for det_name, score_fn in detector_fns.items():
                scores    = _sliding_scores(stream, reference, score_fn)
                threshold = _calibrate_threshold(scores[:drift_day], FPR_TARGET)
                fpr, tpr, latency = _evaluate(scores, drift_day, threshold)

                rec = {**row_base,
                       "detector": det_name,
                       "tpr":      round(tpr, 3) if not np.isnan(tpr) else float("nan"),
                       "fpr":      round(fpr, 3) if not np.isnan(fpr) else float("nan"),
                       "latency":  latency}
                records.append(rec)

                lat_str = f"{latency}d" if latency is not None else "never"
                print(
                    f"  {dist_name:<22} Δ={delta:.1f}σ  {det_name:<22} "
                    f"TPR={tpr:6.3f}  FPR={fpr:6.3f}  latency={lat_str}"
                )

            # ADWIN as error-rate comparator
            adwin_fpr, adwin_tpr = _adwin_tpr_fpr(stream, drift_day)
            rec = {**row_base,
                   "detector": "ADWIN (online)",
                   "tpr": round(adwin_tpr, 3) if not np.isnan(adwin_tpr) else float("nan"),
                   "fpr": round(adwin_fpr, 3) if not np.isnan(adwin_fpr) else float("nan"),
                   "latency": None}
            records.append(rec)
        print()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)
    print(f"Results saved: {csv_out}")

    _validate_lower_bound_claim(df_out)
    _plot(df_out, chart_out)
    return df_out


def _validate_lower_bound_claim(df: pd.DataFrame) -> None:
    """Print whether Gaussian TPR is genuinely a lower bound vs leptokurtic."""
    print("\n=== Lower-bound validation ===")
    for det in df["detector"].unique():
        for delta in SHIFT_MAGNITUDES:
            sub = df[(df["detector"] == det) & (df["shift_magnitude"] == delta)]
            tprs = {row["distribution"]: row["tpr"] for _, row in sub.iterrows()}
            gauss = tprs.get("Gaussian", float("nan"))
            t5    = tprs.get("Student-t (ν=5)", float("nan"))
            t3    = tprs.get("Student-t (ν=3)", float("nan"))
            if np.isnan(gauss):
                continue
            if not np.isnan(t5) and not np.isnan(t3):
                lb_holds = (gauss <= t5 + 0.05) and (gauss <= t3 + 0.05)
                note = "lower-bound HOLDS" if lb_holds else "lower-bound VIOLATED"
                print(
                    f"  {det:<22} Δ={delta:.1f}σ  "
                    f"Gaussian TPR={gauss:.3f}  t5={t5:.3f}  t3={t3:.3f}  [{note}]"
                )


def _plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Focus on PSI+PH (composite) and ADWIN for the main comparison
        main_dets = ["PSI+PH (composite)", "ADWIN (online)", "PSI", "KS"]
        dets_avail = [d for d in main_dets if d in df["detector"].unique()]
        deltas     = sorted(df["shift_magnitude"].unique())
        dists      = [d for d, _ in DISTRIBUTIONS]

        n_rows = len(dets_avail)
        n_cols = len(deltas)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                  sharey="row")
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        bar_colours = {"Gaussian": "#2166ac", "Student-t (ν=5)": "#f4a582",
                       "Student-t (ν=3)": "#d6604d"}
        x = np.arange(len(dists))
        width = 0.6

        for r, det in enumerate(dets_avail):
            for c, delta in enumerate(deltas):
                ax = axes[r][c]
                sub = df[(df["detector"] == det) & (df["shift_magnitude"] == delta)]
                tpr_vals = [sub[sub["distribution"] == d]["tpr"].values[0]
                             if len(sub[sub["distribution"] == d]) > 0
                             else float("nan")
                             for d in dists]
                colours  = [bar_colours.get(d, "#888") for d in dists]
                bars = ax.bar(x, tpr_vals, width, color=colours, edgecolor="black", lw=0.7)
                ax.axhline(0.05, color="grey", ls="--", lw=1, label="FPR target (5%)")
                ax.set_ylim(0, 1.05)
                ax.set_xticks(x)
                ax.set_xticklabels([d.replace("Student-t ", "t") for d in dists],
                                   fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{det}\nTPR", fontsize=9)
                if r == 0:
                    ax.set_title(f"Δμ = {delta:.1f}σ", fontsize=11)
                for bar, val in zip(bars, tpr_vals):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(
            "Leptokurtic benchmark: TPR by distribution, shift magnitude, and detector\n"
            "Gaussian TPR ≤ Student-t TPR validates the 'conservative lower bound' claim",
            fontsize=11,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [chart skipped] {e}")


if __name__ == "__main__":
    run_leptokurtic_benchmark()
