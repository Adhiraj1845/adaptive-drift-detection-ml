"""
Sharpe decomposition: drift detection vs position sizing.

Separates the Sharpe improvement into two components:
  1. Prediction quality gain  — from better adaptive predictions (flat 0.5 position)
  2. Position sizing gain     — from conviction-weighted position scaling

Uses existing daily_monitoring_*.csv files — no re-runs needed.

Method: for each run, recompute equity curves using three position strategies:
  A. Flat 0.5 (static)   — static predictions, fixed half-position
  B. Flat 0.5 (adaptive) — adaptive predictions, fixed half-position
  C. Conviction (static) — original static equity (from main.py)
  D. Conviction (adaptive)— original adaptive equity (from main.py)

Sharpe(B) - Sharpe(A) = prediction quality gain (pure information)
Sharpe(D) - Sharpe(B) = position sizing gain (from conviction scaling)
Sharpe(D) - Sharpe(A) = total gain (matches existing sharpe_delta approximately)

Output
------
  results/sharpe_decomposition.csv    — per-run decomposition
  results/sharpe_decomposition.png    — attribution bar chart

Usage
-----
    python -m src.experiments.sharpe_decomposition
    python -m src.experiments.sharpe_decomposition --combo all
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    if len(returns) < 2:
        return float("nan")
    mu  = float(np.mean(returns))
    sig = float(np.std(returns))
    return float((mu / (sig + eps)) * math.sqrt(252.0))


def _equity_flat(p1: np.ndarray, returns: np.ndarray, pos: float = 0.5) -> np.ndarray:
    """Equity curve with fixed position = pos whenever p1 >= 0.5, else 0."""
    positions = np.where(p1 >= 0.5, pos, 0.0)
    eq = np.ones(len(returns) + 1)
    for i, r in enumerate(returns):
        eq[i + 1] = eq[i] * (1.0 + positions[i] * r)
    return eq


def _parse_ticker_period(stem: str) -> tuple[str, str, str]:
    body  = stem.replace("daily_monitoring_", "")
    parts = body.split("_")
    return parts[0], parts[1] if len(parts) > 1 else "?", "_".join(parts[2:]) if len(parts) > 2 else "?"


def run_sharpe_decomposition(
    results_dir: str = "results",
    combo_filter: str | None = None,
    csv_out:   str = "results/sharpe_decomposition.csv",
    chart_out: str = "results/sharpe_decomposition.png",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
    if combo_filter:
        files = [f for f in files if f.name.endswith(f"_{combo_filter}.csv")]

    print(f"Processing {len(files)} files (combo_filter={combo_filter!r}) …")

    records = []
    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["p1_static", "p1_adaptive", "return_next",
                         "pos_long_static", "pos_long_adaptive"],
            )
            df = df.dropna()
        except Exception:
            continue

        if len(df) < 50:
            continue

        p1_s  = df["p1_static"].values
        p1_a  = df["p1_adaptive"].values
        rets  = df["return_next"].values

        # ── Flat-position equity curves ────────────────────────────────────────
        eq_flat_s = _equity_flat(p1_s, rets, pos=0.5)
        eq_flat_a = _equity_flat(p1_a, rets, pos=0.5)

        ret_flat_s = eq_flat_s[1:] / eq_flat_s[:-1] - 1.0
        ret_flat_a = eq_flat_a[1:] / eq_flat_a[:-1] - 1.0

        # ── Original conviction-weighted equity (recompute from stored positions) ─
        eq_conv_s = np.ones(len(rets) + 1)
        eq_conv_a = np.ones(len(rets) + 1)
        pos_s = df["pos_long_static"].values
        pos_a = df["pos_long_adaptive"].values
        for i, r in enumerate(rets):
            eq_conv_s[i + 1] = eq_conv_s[i] * (1.0 + pos_s[i] * r)
            eq_conv_a[i + 1] = eq_conv_a[i] * (1.0 + pos_a[i] * r)

        ret_conv_s = eq_conv_s[1:] / eq_conv_s[:-1] - 1.0
        ret_conv_a = eq_conv_a[1:] / eq_conv_a[:-1] - 1.0

        sharpe_flat_s = _sharpe(ret_flat_s)
        sharpe_flat_a = _sharpe(ret_flat_a)
        sharpe_conv_s = _sharpe(ret_conv_s)
        sharpe_conv_a = _sharpe(ret_conv_a)

        prediction_gain  = sharpe_flat_a  - sharpe_flat_s   # pure info gain
        sizing_gain      = sharpe_conv_a  - sharpe_flat_a   # position scaling gain
        total_gain       = sharpe_conv_a  - sharpe_conv_s   # matches existing sharpe_delta

        ticker, period, combo = _parse_ticker_period(path.stem)
        records.append({
            "ticker":            ticker,
            "period":            period,
            "combo":             combo,
            "sharpe_flat_static":  round(sharpe_flat_s,  4),
            "sharpe_flat_adaptive":round(sharpe_flat_a,  4),
            "sharpe_conv_static":  round(sharpe_conv_s,  4),
            "sharpe_conv_adaptive":round(sharpe_conv_a,  4),
            "prediction_gain":   round(prediction_gain,  4),
            "sizing_gain":       round(sizing_gain,      4),
            "total_gain":        round(total_gain,       4),
            "pct_prediction":    round(prediction_gain / (abs(total_gain) + 1e-9) * 100, 1),
        })

    if not records:
        print("No records produced.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    ok = df_out[np.isfinite(df_out["total_gain"])]
    print(f"\nSharpe decomposition  ({len(ok)} runs):")
    print(f"  Mean sharpe_flat_static:   {ok['sharpe_flat_static'].mean():+.4f}")
    print(f"  Mean sharpe_flat_adaptive: {ok['sharpe_flat_adaptive'].mean():+.4f}")
    print(f"  Mean prediction gain:      {ok['prediction_gain'].mean():+.4f}  "
          f"({100*ok['prediction_gain'].mean()/max(abs(ok['total_gain'].mean()),1e-6):.0f}% of total)")
    print(f"  Mean sizing gain:          {ok['sizing_gain'].mean():+.4f}  "
          f"({100*ok['sizing_gain'].mean()/max(abs(ok['total_gain'].mean()),1e-6):.0f}% of total)")
    print(f"  Mean total gain:           {ok['total_gain'].mean():+.4f}")
    print(f"  % runs: prediction_gain > 0: {(ok['prediction_gain']>0).mean()*100:.1f}%")
    print(f"  % runs: sizing_gain > 0:     {(ok['sizing_gain']>0).mean()*100:.1f}%")
    print(f"Saved: {csv_out}")

    _plot_decomposition(ok, chart_out)
    return df_out


def _plot_decomposition(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: stacked attribution bar chart ────────────────────────────────
        ax = axes[0]
        labels   = ["Prediction gain\n(flat position)", "Sizing gain\n(conviction scale)", "Total gain"]
        means    = [df["prediction_gain"].mean(), df["sizing_gain"].mean(), df["total_gain"].mean()]
        sems     = [df["prediction_gain"].sem(),  df["sizing_gain"].sem(),  df["total_gain"].sem()]
        colors   = ["#2c7bb6", "#1a9641", "#756bb1"]
        bars = ax.bar(labels, means, color=colors, alpha=0.85, capsize=5)
        ax.errorbar(labels, means, yerr=[1.96 * s for s in sems],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean Sharpe improvement")
        ax.set_title("Sharpe attribution: prediction quality vs position sizing")

        # ── Right: scatter prediction_gain vs sizing_gain ──────────────────────
        ax2 = axes[1]
        ax2.scatter(df["prediction_gain"], df["sizing_gain"],
                    alpha=0.15, s=8, color="#2c7bb6")
        ax2.axhline(0, color="grey", lw=0.7, ls="--")
        ax2.axvline(0, color="grey", lw=0.7, ls="--")
        ax2.set_xlabel("Prediction gain (flat-position Sharpe improvement)")
        ax2.set_ylabel("Sizing gain (conviction-scale Sharpe improvement)")
        ax2.set_title(f"Prediction vs sizing attribution  (n={len(df)})")
        # Annotate quadrants
        xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
        for q_x, q_y, label in [
            (0.75, 0.85, "Both win"),
            (0.25, 0.85, "Sizing wins,\npred loses"),
            (0.75, 0.15, "Pred wins,\nsizing loses"),
            (0.25, 0.15, "Both lose"),
        ]:
            ax2.text(q_x, q_y, label, transform=ax2.transAxes,
                     ha="center", fontsize=7, color="grey", alpha=0.7)

        fig.suptitle(
            "Sharpe decomposition: how much comes from better predictions\n"
            "vs how much from conviction-weighted position sizing?",
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
    parser = argparse.ArgumentParser(description="Sharpe decomposition: prediction vs position sizing")
    parser.add_argument("--combo", default=None, help="Filter by combo tag (e.g. 'all')")
    args = parser.parse_args()
    run_sharpe_decomposition(combo_filter=args.combo)
