"""
Computational cost analysis.

Analyses the elapsed_s column from ablation_summary.csv to answer:
  1. How many detectors ON vs runtime?
  2. Do more detectors = more retrains = more runtime?
  3. What is the cost-efficiency frontier (acc_delta / elapsed_s)?
  4. PSI+PH vs full ensemble: same accuracy at what fraction of cost?

Output
------
  results/cost_analysis.csv          — per-combo cost-efficiency stats
  results/cost_analysis_charts.png   — 4-panel chart

Usage
-----
    python -m src.experiments.computational_cost_analysis
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

_ABLATION_CSV = "results/ablation_summary.csv"
_RETRAIN_CSV  = "results/retrain_summary.csv"


def _detector_count(row: pd.Series) -> int:
    cols = ["use_ks", "use_psi", "use_js", "use_page_hinkley"]
    return int(sum(bool(row[c]) for c in cols if c in row.index))


def _combo_label(row: pd.Series) -> str:
    parts = []
    if row.get("use_ks"):       parts.append("KS")
    if row.get("use_psi"):      parts.append("PSI")
    if row.get("use_js"):       parts.append("JS")
    if row.get("use_page_hinkley"): parts.append("PH")
    return "+".join(parts) if parts else "none"


def run_cost_analysis(
    ablation_csv: str = _ABLATION_CSV,
    retrain_csv:  str = _RETRAIN_CSV,
    chart_path:   str = "results/cost_analysis_charts.png",
    csv_out:      str = "results/cost_analysis.csv",
) -> pd.DataFrame:

    if not os.path.exists(ablation_csv):
        print(f"ERROR: {ablation_csv} not found.")
        return pd.DataFrame()

    df = pd.read_csv(ablation_csv)
    df = df[df["status"] == "ok"].copy()
    print(f"Loaded {len(df)} successful runs from {ablation_csv}")

    df["n_detectors"]  = df.apply(_detector_count, axis=1)
    df["combo_label"]  = df.apply(_combo_label, axis=1)
    df["cost_eff"]     = (df["acc_delta"] / df["elapsed_s"].clip(lower=0.1)).round(6)

    # ── Per-combo aggregate ────────────────────────────────────────────────────
    combo_stats = (
        df.groupby(["combo_label", "n_detectors"])
        .agg(
            n_runs=("elapsed_s", "count"),
            mean_elapsed_s=("elapsed_s", "mean"),
            median_elapsed_s=("elapsed_s", "median"),
            mean_acc_delta=("acc_delta", "mean"),
            mean_n_retrains=("n_retrains", "mean"),
            mean_n_drift_events=("n_drift_events", "mean"),
            mean_sharpe_delta=("sharpe_delta", "mean"),
            mean_cost_eff=("cost_eff", "mean"),
        )
        .round(4)
        .reset_index()
        .sort_values("n_detectors")
    )
    combo_stats.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    print("\nCost-efficiency by detector combo:")
    print(combo_stats.to_string(index=False))

    # ── PSI+PH vs full-ensemble spotlight ─────────────────────────────────────
    full_ensemble = df[df["combo_label"] == "KS+PSI+JS+PH"]
    psi_ph        = df[df["combo_label"] == "PSI+PH"]
    if len(full_ensemble) > 0 and len(psi_ph) > 0:
        fe_t  = full_ensemble["elapsed_s"].mean()
        pp_t  = psi_ph["elapsed_s"].mean()
        fe_da = full_ensemble["acc_delta"].mean()
        pp_da = psi_ph["acc_delta"].mean()
        print(f"\nSpotlight — PSI+PH vs full ensemble (KS+PSI+JS+PH):")
        print(f"  PSI+PH       : {pp_t:7.1f}s/run   acc_delta={pp_da:+.4f}")
        print(f"  Full ensemble: {fe_t:7.1f}s/run   acc_delta={fe_da:+.4f}")
        print(f"  Cost saving  : {100*(1-pp_t/fe_t):.1f}%   acc difference: {pp_da-fe_da:+.4f}")

    _plot_cost(df, combo_stats, chart_path)
    return combo_stats


def _plot_cost(df: pd.DataFrame, combo_stats: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # ── Panel 1: n_detectors vs mean elapsed_s (scatter + mean line) ──────
        ax = axes[0]
        jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(df))
        ax.scatter(df["n_detectors"] + jitter, df["elapsed_s"],
                   alpha=0.15, s=6, color="#2c7bb6")
        mean_line = df.groupby("n_detectors")["elapsed_s"].mean()
        ax.plot(mean_line.index, mean_line.values, "o-", color="#d7191c",
                lw=2, markersize=6, label="Mean")
        ax.set_xlabel("Number of active detectors (excl. PredictionDrift)")
        ax.set_ylabel("Elapsed time (s)")
        ax.set_title("Runtime vs detector count")
        ax.legend()

        # ── Panel 2: n_retrains vs elapsed_s ─────────────────────────────────
        ax = axes[1]
        ax.scatter(df["n_retrains"], df["elapsed_s"],
                   alpha=0.2, s=8, color="#1a9641")
        # Bin by retrains
        bins = pd.cut(df["n_retrains"], bins=10)
        mean_by_bin = df.groupby(bins, observed=False)["elapsed_s"].mean()
        midpoints = [iv.mid for iv in mean_by_bin.index]
        ax.plot(midpoints, mean_by_bin.values, "o-", color="#d7191c",
                lw=2, markersize=6, label="Bin mean")
        ax.set_xlabel("Number of retrains (evaluation period)")
        ax.set_ylabel("Elapsed time (s)")
        ax.set_title("Runtime vs retrain count")
        ax.legend()

        # ── Panel 3: combo_label vs mean elapsed_s (horizontal bar) ──────────
        ax = axes[2]
        cs = combo_stats.sort_values("mean_elapsed_s")
        colours_bar = plt.cm.viridis(
            np.linspace(0.2, 0.8, len(cs))
        )
        ax.barh(cs["combo_label"], cs["mean_elapsed_s"], color=colours_bar)
        ax.set_xlabel("Mean elapsed time (s)")
        ax.set_title("Mean runtime by detector combo")
        ax.tick_params(axis="y", labelsize=8)

        # ── Panel 4: cost-efficiency frontier (acc_delta vs elapsed_s) ────────
        ax = axes[3]
        combo_means = df.groupby("combo_label").agg(
            x=("elapsed_s", "mean"),
            y=("acc_delta", "mean"),
            n=("elapsed_s", "count"),
        ).reset_index()
        sc = ax.scatter(combo_means["x"], combo_means["y"],
                        s=combo_means["n"] * 0.3, alpha=0.75,
                        c=combo_means["n"], cmap="plasma")
        for _, row in combo_means.iterrows():
            ax.annotate(row["combo_label"], (row["x"], row["y"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=6, alpha=0.8)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Mean elapsed time (s)")
        ax.set_ylabel("Mean accuracy delta (adaptive − static)")
        ax.set_title("Cost-efficiency frontier")
        plt.colorbar(sc, ax=ax, label="n runs")

        fig.suptitle("Computational cost analysis — ablation experiments", fontsize=13)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [chart skipped] {e}")


if __name__ == "__main__":
    run_cost_analysis()
