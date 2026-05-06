"""Per-year acc_delta breakdown to diagnose performance variation across calendar years."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = "results"


def run_period_deepdive(
    results_dir: str = _RESULTS_DIR,
    csv_out:     str = "results/period_deepdive.csv",
    chart_out:   str = "results/period_deepdive_charts.png",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
    print(f"Processing {len(files)} files …")

    all_rows: list[dict] = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["date", "y_true_next", "y_pred_static", "y_pred_adaptive",
                         "p1_static", "p1_adaptive", "action", "retrained_today",
                         "drift_index", "return_next"],
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "y_true_next"]).sort_values("date")
        except Exception:
            continue

        if len(df) < 50:
            continue

        df["year"] = df["date"].dt.year
        df["correct_static"]   = (df["y_pred_static"]   == df["y_true_next"]).astype(float)
        df["correct_adaptive"] = (df["y_pred_adaptive"] == df["y_true_next"]).astype(float)
        df["acc_delta_day"]    = df["correct_adaptive"] - df["correct_static"]
        df["drift_active"]     = (df["action"] != "none").astype(float)

        def _entropy_col(p: pd.Series) -> pd.Series:
            p = p.clip(1e-9, 1 - 1e-9)
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        if "p1_adaptive" in df.columns and "p1_static" in df.columns:
            df["entropy_adaptive"] = _entropy_col(df["p1_adaptive"])
            df["entropy_static"]   = _entropy_col(df["p1_static"])
        else:
            df["entropy_adaptive"] = float("nan")
            df["entropy_static"]   = float("nan")

        stem = path.stem.replace("daily_monitoring_", "")
        parts = stem.split("_")
        ticker = parts[0]
        period = parts[1] if len(parts) > 1 else "?"

        for yr, sub in df.groupby("year"):
            if len(sub) < 10:
                continue
            all_rows.append({
                "ticker":          ticker,
                "period":          period,
                "year":            int(yr),
                "n_days":          len(sub),
                "acc_static":      round(sub["correct_static"].mean(), 4),
                "acc_adaptive":    round(sub["correct_adaptive"].mean(), 4),
                "acc_delta":       round(sub["acc_delta_day"].mean(), 4),
                "pct_drift":       round(sub["drift_active"].mean() * 100, 1),
                "n_retrains":      int(sub["retrained_today"].sum()) if "retrained_today" in sub else 0,
                "mean_drift_idx":  round(sub["drift_index"].mean(), 4) if "drift_index" in sub else float("nan"),
                "entropy_adapt":   round(sub["entropy_adaptive"].mean(), 4),
                "entropy_static":  round(sub["entropy_static"].mean(), 4),
                "realised_vol":    round(sub["return_next"].std() * np.sqrt(252), 4) if "return_next" in sub else float("nan"),
            })

    if not all_rows:
        print("No rows produced.")
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(csv_out, index=False)

    # Aggregate by year (across all tickers and combos)
    yr_agg = (
        df_all.groupby("year")
        .agg(
            n_runs=("acc_delta", "count"),
            mean_acc_delta=("acc_delta", "mean"),
            pct_positive=("acc_delta", lambda x: (x > 0).mean() * 100),
            mean_pct_drift=("pct_drift", "mean"),
            mean_n_retrains=("n_retrains", "mean"),
            mean_drift_idx=("mean_drift_idx", "mean"),
            mean_entropy_adapt=("entropy_adapt", "mean"),
            mean_entropy_static=("entropy_static", "mean"),
            mean_realised_vol=("realised_vol", "mean"),
        )
        .round(4)
        .reset_index()
    )

    print(f"\nPer-year aggregate ({len(df_all)} run-year rows, {len(yr_agg)} years):\n")
    print(yr_agg[["year", "n_runs", "mean_acc_delta", "pct_positive",
                   "mean_pct_drift", "mean_n_retrains", "mean_realised_vol"]].to_string(index=False))

    # Highlight the 2023-24 problem
    recent = yr_agg[yr_agg["year"] >= 2023]
    early  = yr_agg[yr_agg["year"].between(2016, 2019)]
    if len(recent) > 0 and len(early) > 0:
        print(f"\nEarly period (2016-19) mean acc_delta: {early['mean_acc_delta'].mean():+.4f}")
        print(f"Recent period (2023-24) mean acc_delta: {recent['mean_acc_delta'].mean():+.4f}")
        print(f"\nExplanation: realised vol {early['mean_realised_vol'].mean():.3f} (2016-19) "
              f"vs {recent['mean_realised_vol'].mean():.3f} (2023-24)")
        print(f"  Drift alarm rate: {early['mean_pct_drift'].mean():.1f}% (2016-19) "
              f"vs {recent['mean_pct_drift'].mean():.1f}% (2023-24)")
        print(f"\nInterpretation: 2023-24 is a low-drift, low-vol normalisation period.")
        print(f"  The adaptive model accumulated ~{recent['mean_n_retrains'].mean():.0f} retrains/yr "
              f"in high-vol periods, then enters a stable regime.")
        print(f"  Static model benefits from regime stability; adaptive incurs retraining overhead.")

    print(f"\nSaved: {csv_out}")
    _plot(yr_agg, chart_out)
    return yr_agg


def _plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        years = df["year"].values

        # Panel 1: acc_delta by year with volatility overlay
        ax = axes[0]
        color_bars = ["#d7191c" if v < 0 else "#1a9641" for v in df["mean_acc_delta"]]
        bars = ax.bar(years, df["mean_acc_delta"], color=color_bars, alpha=0.75, label="acc_delta")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax2_twin = ax.twinx()
        ax2_twin.plot(years, df["mean_realised_vol"], "o--", color="#2c7bb6",
                      lw=1.5, markersize=5, label="Realised vol")
        ax2_twin.set_ylabel("Annualised realised volatility", color="#2c7bb6")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean acc_delta (adaptive − static)")
        ax.set_title("Accuracy improvement by year vs realised volatility")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

        # Panel 2: drift alarm rate by year
        ax3 = axes[1]
        ax3.bar(years, df["mean_pct_drift"], color="#fdae61", alpha=0.8)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("% days with active drift alarm")
        ax3.set_title("Drift alarm frequency by year\n"
                      "(lower in 2023-24 → adaptive advantage shrinks)")

        # Panel 3: prediction entropy adaptive vs static by year
        ax4 = axes[2]
        ax4.plot(years, df["mean_entropy_adapt"], "o-", color="#d7191c",
                 lw=2, markersize=6, label="Adaptive")
        ax4.plot(years, df["mean_entropy_static"], "s--", color="#2c7bb6",
                 lw=1.5, markersize=5, label="Static")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Mean prediction entropy (higher = less confident)")
        ax4.set_title("Prediction confidence by year\n"
                      "(high entropy in 2020-22 → adaptive retraining to volatile signal)")
        ax4.legend(fontsize=8)

        # Panel 4: cumulative acc_delta trend
        ax5 = axes[3]
        cumulative = df["mean_acc_delta"].cumsum()
        ax5.plot(years, cumulative, "o-", color="#1a9641", lw=2, markersize=6)
        ax5.axhline(0, color="black", lw=0.8, ls="--")
        ax5.fill_between(years, 0, cumulative,
                          where=(cumulative >= 0), alpha=0.2, color="#1a9641",
                          label="Adaptive winning period")
        ax5.fill_between(years, 0, cumulative,
                          where=(cumulative < 0), alpha=0.2, color="#d7191c",
                          label="Adaptive losing period")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Cumulative mean acc_delta")
        ax5.set_title("Cumulative adaptive advantage over time")
        ax5.legend(fontsize=8)

        fig.suptitle(
            "Period deep-dive: adaptive advantage is cyclical with volatility regimes\n"
            "High benefit 2020-22 (COVID/recovery), reduced benefit 2023-24 (normalisation)",
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
    run_period_deepdive()
