"""Transaction cost break-even: Sharpe(adaptive, c*) = Sharpe(static, c*) per run."""

Usage
-----
    python -m src.experiments.transaction_cost_analysis
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = "results"
_COST_GRID   = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.010, 0.020]
_TRADING_DAYS = 252


def _sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    if len(returns) < 2:
        return float("nan")
    return float((np.mean(returns) / (np.std(returns) + eps)) * math.sqrt(_TRADING_DAYS))


def _net_returns(pos: np.ndarray, market_ret: np.ndarray, cost_per_trade: float) -> np.ndarray:
    """Returns after deducting round-trip costs on position changes."""
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * market_ret - turnover * cost_per_trade


def _break_even_cost(pos_s: np.ndarray, pos_a: np.ndarray,
                      market_ret: np.ndarray) -> float:
    """
    Binary-search for cost c* where Sharpe(adaptive, c*) = Sharpe(static, c*).
    Returns NaN if no crossing found in [0, 10%].
    """
    lo, hi = 0.0, 0.10
    for _ in range(60):
        mid = (lo + hi) / 2.0
        r_s = _net_returns(pos_s, market_ret, mid)
        r_a = _net_returns(pos_a, market_ret, mid)
        sh_s = _sharpe(r_s)
        sh_a = _sharpe(r_a)
        if np.isnan(sh_s) or np.isnan(sh_a):
            return float("nan")
        if sh_a > sh_s:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    # Check that a crossing actually occurred
    r_s0 = _net_returns(pos_s, market_ret, 0.0)
    r_a0 = _net_returns(pos_a, market_ret, 0.0)
    if _sharpe(r_a0) <= _sharpe(r_s0):
        return 0.0   # adaptive already losing at zero cost
    return float((lo + hi) / 2.0)


def run_transaction_cost_analysis(
    results_dir: str = _RESULTS_DIR,
    csv_out:     str = "results/transaction_cost.csv",
    chart_out:   str = "results/transaction_cost_charts.png",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
    print(f"Processing {len(files)} files …")

    records = []
    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["return_next", "pos_long_static", "pos_long_adaptive"],
            )
            df = df.dropna()
        except Exception:
            continue

        if len(df) < 50:
            continue

        mkt  = df["return_next"].values
        pos_s = df["pos_long_static"].values
        pos_a = df["pos_long_adaptive"].values

        # Turnover (annualised)
        turnover_s = float(np.abs(np.diff(np.concatenate([[0.0], pos_s]))).sum() * _TRADING_DAYS / len(pos_s))
        turnover_a = float(np.abs(np.diff(np.concatenate([[0.0], pos_a]))).sum() * _TRADING_DAYS / len(pos_a))

        # Break-even cost
        be_cost = _break_even_cost(pos_s, pos_a, mkt)

        # Sharpe at different cost levels
        sharpe_at_cost = {}
        for c in _COST_GRID:
            r_s = _net_returns(pos_s, mkt, c)
            r_a = _net_returns(pos_a, mkt, c)
            sharpe_at_cost[f"sharpe_s_{int(c*10000)}bps"] = round(_sharpe(r_s), 4)
            sharpe_at_cost[f"sharpe_a_{int(c*10000)}bps"] = round(_sharpe(r_a), 4)

        stem = path.stem.replace("daily_monitoring_", "")
        parts = stem.split("_")
        rec = {
            "ticker":          parts[0],
            "period":          parts[1] if len(parts) > 1 else "?",
            "combo":           "_".join(parts[2:]) if len(parts) > 2 else "?",
            "turnover_static_ann":  round(turnover_s, 2),
            "turnover_adapt_ann":   round(turnover_a, 2),
            "break_even_bps":       round(be_cost * 10000, 2) if not np.isnan(be_cost) else float("nan"),
            "n_days":               len(df),
        }
        rec.update(sharpe_at_cost)
        records.append(rec)

    if not records:
        print("No records.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    ok = df_out.dropna(subset=["break_even_bps"])

    print(f"\nTransaction cost analysis ({len(ok)} runs):\n")
    print(f"  Annualised turnover — static:   {ok['turnover_static_ann'].mean():.2f}x")
    print(f"  Annualised turnover — adaptive: {ok['turnover_adapt_ann'].mean():.2f}x")

    be = ok["break_even_bps"]
    be_pos = be[be > 0]
    print(f"\n  Break-even cost (bps):")
    print(f"    Mean: {be_pos.mean():.1f} bps   Median: {be_pos.median():.1f} bps")
    print(f"    % runs with BE > 5 bps (retail threshold):   {(be_pos > 5).mean()*100:.1f}%")
    print(f"    % runs with BE > 20 bps (expensive retail):  {(be_pos > 20).mean()*100:.1f}%")
    print(f"    % runs where adaptive already losing (BE=0): {(be <= 0).mean()*100:.1f}%")

    print(f"\n  Sharpe delta at various cost levels:")
    for c in [0.0, 0.0005, 0.001, 0.002, 0.005]:
        bps = int(c * 10000)
        cs  = f"sharpe_s_{bps}bps"
        ca  = f"sharpe_a_{bps}bps"
        if cs in ok.columns and ca in ok.columns:
            delta = (ok[ca] - ok[cs]).mean()
            pct_pos = ((ok[ca] - ok[cs]) > 0).mean() * 100
            print(f"    {bps:4d} bps: sharpe_delta={delta:+.4f}  ({pct_pos:.1f}% positive)")

    print(f"\n  Industry context:")
    print(f"    Retail broker (5-20 bps): {'viable' if be_pos.median() > 10 else 'marginal'}")
    print(f"    Institutional (<1 bps):   {'viable' if be_pos.median() > 1 else 'marginal'}")
    print(f"\nSaved: {csv_out}")

    _plot(ok, chart_out)
    return df_out


def _plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Panel 1: break-even cost distribution
        ax = axes[0]
        be = df["break_even_bps"].dropna()
        be = be[be.between(-10, 200)]   # clip extreme outliers for display
        ax.hist(be, bins=60, color="#2c7bb6", edgecolor="white", alpha=0.85)
        ax.axvline(5,  color="#fdae61", lw=1.5, ls="--", label="5 bps  (online retail)")
        ax.axvline(20, color="#d7191c", lw=1.5, ls="--", label="20 bps (expensive retail)")
        ax.axvline(be.median(), color="#1a9641", lw=2, ls="--",
                   label=f"Median={be.median():.1f} bps")
        ax.set_xlabel("Break-even transaction cost (bps per trade)")
        ax.set_ylabel("Count")
        ax.set_title("Break-even cost distribution\n"
                     "(right of dashed lines → adaptive profitable after costs)")
        ax.legend(fontsize=8)

        # Panel 2: Sharpe vs cost level (mean across all runs)
        ax2 = axes[1]
        cost_bps   = [int(c * 10000) for c in _COST_GRID]
        sharpe_s_means = []
        sharpe_a_means = []
        for c in _COST_GRID:
            bps = int(c * 10000)
            cs, ca = f"sharpe_s_{bps}bps", f"sharpe_a_{bps}bps"
            if cs in df.columns:
                sharpe_s_means.append(df[cs].mean())
                sharpe_a_means.append(df[ca].mean())
        if sharpe_s_means:
            ax2.plot(cost_bps[:len(sharpe_s_means)], sharpe_s_means, "o-",
                     color="#2c7bb6", lw=2, markersize=6, label="Static")
            ax2.plot(cost_bps[:len(sharpe_a_means)], sharpe_a_means, "s-",
                     color="#d7191c", lw=2, markersize=6, label="Adaptive")
            ax2.axvline(5,  color="#fdae61", lw=1.0, ls="--", alpha=0.7)
            ax2.axvline(20, color="#d7191c", lw=1.0, ls="--", alpha=0.5)
        ax2.set_xlabel("Transaction cost (bps per trade)")
        ax2.set_ylabel("Mean Sharpe ratio")
        ax2.set_title("Mean Sharpe vs transaction cost\n"
                      "(crossing point = break-even cost)")
        ax2.legend()

        # Panel 3: turnover comparison
        ax3 = axes[2]
        parts3 = ax3.violinplot(
            [df["turnover_static_ann"].dropna().clip(0, 20).values,
             df["turnover_adapt_ann"].dropna().clip(0, 20).values],
            positions=[1, 2], showmedians=True,
        )
        for pc, col in zip(parts3["bodies"], ["#2c7bb6", "#d7191c"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(["Static", "Adaptive"])
        ax3.set_ylabel("Annual position turnover (× per year)")
        t_s = df["turnover_static_ann"].mean()
        t_a = df["turnover_adapt_ann"].mean()
        ax3.set_title(f"Annual turnover\nStatic={t_s:.1f}×  Adaptive={t_a:.1f}×\n"
                      f"(Adaptive {t_a/t_s:.1f}× more trading activity)")

        # Panel 4: % runs profitable after cost at different thresholds
        ax4 = axes[3]
        thresholds = [0, 1, 2, 5, 10, 20, 50]
        pct_viable = []
        for thresh in thresholds:
            pct = (df["break_even_bps"].dropna() >= thresh).mean() * 100
            pct_viable.append(pct)
        ax4.bar([str(t) for t in thresholds], pct_viable, color="#1a9641", alpha=0.85)
        ax4.axhline(50, color="grey", lw=0.8, ls="--", label="50% threshold")
        ax4.set_xlabel("Cost threshold (bps per trade)")
        ax4.set_ylabel("% runs where adaptive beats static after costs")
        ax4.set_title("Fraction of runs with adaptive advantage\nat each cost level")
        ax4.legend(fontsize=8)
        for i, v in enumerate(pct_viable):
            ax4.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)

        fig.suptitle(
            "Transaction cost break-even analysis\n"
            "Quantifies practical exploitability of the adaptive framework",
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
    run_transaction_cost_analysis()
