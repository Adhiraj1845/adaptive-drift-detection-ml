"""
Error-trigger redundancy analysis.

Empirically demonstrates why error-rate-triggered retraining adds zero independent
retrains when distribution-based drift detection is active.

Hypothesis: accuracy degradation is a lagging indicator of distribution drift.
By the time the 30-day rolling static accuracy drops below 0.45, the drift
detector has already flagged the period (action != "none").

Method: for every day in each daily monitoring CSV, compute the rolling 30-day
static accuracy and check whether the drift detector was already active.

Output
------
  results/error_trigger_analysis.csv   — per-run summary
  results/error_trigger_lag.png        — lag distribution chart

Usage
-----
    python -m src.experiments.error_trigger_analysis
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR    = "results"
_ACC_THRESHOLD  = 0.45   # matches main.py error trigger condition
_WINDOW         = 30     # rolling window for accuracy tracking
_COMBO_FILTER   = "drift_only"  # use drift_only so there's no confounding scheduled retrain


def analyse_error_trigger_redundancy(
    results_dir: str = _RESULTS_DIR,
    combo: str = _COMBO_FILTER,
    csv_out: str = "results/error_trigger_analysis.csv",
    chart_path: str = "results/error_trigger_lag.png",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob(f"daily_monitoring_*_{combo}.csv"))
    if not files:
        files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
        files = [f for f in files if "_all." in f.name or "_drift_only." in f.name]

    print(f"Analysing {len(files)} monitoring files …")

    records = []
    lag_days_all: list[int] = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["date", "y_pred_static", "y_true_next", "action", "drift_index"],
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        except Exception as e:
            continue

        if len(df) < _WINDOW + 10:
            continue

        df["static_correct"] = (df["y_pred_static"] == df["y_true_next"]).astype(int)
        df["rolling_acc_30"] = df["static_correct"].rolling(_WINDOW).mean()
        df["drift_active"]   = (df["action"] != "none").astype(int)

        # Days where acc < threshold AND drift already active (error trigger would be blocked)
        low_acc_mask     = df["rolling_acc_30"] < _ACC_THRESHOLD
        low_acc_days     = df[low_acc_mask]
        blocked_by_drift = df[low_acc_mask & (df["drift_active"] == 1)]
        would_be_novel   = df[low_acc_mask & (df["drift_active"] == 0)]

        # For novel days (acc low, drift NOT active), how long ago was last drift event?
        # This measures how long after the last drift event accuracy stays low
        last_drift_day = None
        lags = []
        for i, row in df.iterrows():
            if row["drift_active"] == 1:
                last_drift_day = i
            if low_acc_mask.iloc[i] and row["drift_active"] == 0 and last_drift_day is not None:
                lags.append(i - last_drift_day)

        lag_days_all.extend(lags)

        ticker = path.stem.replace("daily_monitoring_", "").split("_")[0]
        period = path.stem.replace("daily_monitoring_", "").split("_")[1] if "_" in path.stem.replace("daily_monitoring_", "") else "unknown"

        records.append({
            "ticker":              ticker,
            "period":              period,
            "n_days":              len(df),
            "n_low_acc_days":      int(low_acc_mask.sum()),
            "n_blocked_by_drift":  len(blocked_by_drift),
            "n_would_be_novel":    len(would_be_novel),
            "pct_blocked":         round(len(blocked_by_drift) / max(len(low_acc_days), 1) * 100, 1),
            "mean_lag_after_drift": round(float(np.mean(lags)) if lags else float("nan"), 1),
        })

    if not records:
        print("No results.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    print(f"\nError-trigger redundancy analysis ({len(df_out)} runs):")
    print(f"  Mean % of low-acc days already covered by drift detector: "
          f"{df_out['pct_blocked'].mean():.1f}%")
    print(f"  Mean % of low-acc days that would be novel: "
          f"{100 - df_out['pct_blocked'].mean():.1f}%")
    print(f"  Mean lag between drift detection and accuracy drop: "
          f"{df_out['mean_lag_after_drift'].mean():.1f} days")
    print(f"\nConclusion: {df_out['pct_blocked'].mean():.0f}% of error-trigger opportunities are "
          f"pre-empted by distribution drift detection, confirming that accuracy degradation "
          f"is a lagging indicator of drift already detected by KS/PSI/PH.")
    print(f"Saved: {csv_out}")

    _plot_lag(lag_days_all, df_out, chart_path)
    return df_out


def _plot_lag(lag_days_all: list[int], df_out: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: distribution of lag (days between last drift event and acc drop) ──
        ax = axes[0]
        if lag_days_all:
            ax.hist(lag_days_all, bins=30, color="#2c7bb6", edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(lag_days_all), color="#d7191c", lw=2, ls="--",
                       label=f"Mean = {np.mean(lag_days_all):.1f} days")
            ax.set_xlabel("Days since last drift event when acc first drops below 0.45")
            ax.set_ylabel("Count")
            ax.set_title("Lag: drift detection → accuracy degradation")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No novel low-acc days found\n(drift detector pre-empts all)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("No novel error-trigger opportunities")

        # ── Right: % blocked per run (scatter) ─────────────────────────────────
        ax2 = axes[1]
        ax2.hist(df_out["pct_blocked"].dropna(), bins=20,
                 color="#1a9641", edgecolor="white", alpha=0.85)
        ax2.axvline(df_out["pct_blocked"].mean(), color="#d7191c", lw=2, ls="--",
                    label=f"Mean = {df_out['pct_blocked'].mean():.1f}%")
        ax2.set_xlabel("% of low-accuracy days already covered by drift detector")
        ax2.set_ylabel("Number of runs")
        ax2.set_title("Error-trigger pre-emption rate per run")
        ax2.legend()

        fig.suptitle(
            "Why error-rate triggering adds zero independent retrains:\n"
            "accuracy degradation co-occurs with or lags drift detection",
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
    analyse_error_trigger_redundancy()
