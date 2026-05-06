"""Financial strategy baselines: buy-and-hold, SMA crossover, momentum, static ML, adaptive ML."""
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR  = "results"
_RF_DAILY     = 0.0   # use 0 for cross-strategy comparison (avoids position-size bias)
_TRADING_DAYS = 252
_SMA_FAST     = 20
_SMA_SLOW     = 50
_MOM_WINDOW   = 20


def _sharpe(returns: np.ndarray, rf: float = _RF_DAILY) -> float:
    if len(returns) < 2:
        return float("nan")
    std = float(np.std(returns))
    if std < 1e-8:
        return float("nan")
    excess = returns - rf
    return float((np.mean(excess) / std) * math.sqrt(_TRADING_DAYS))


def _max_dd(returns: np.ndarray) -> float:
    eq = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / (peak + 1e-12)))


def _cagr(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    total = float(np.prod(1.0 + returns))
    years = len(returns) / _TRADING_DAYS
    return float(total ** (1.0 / years) - 1.0)


def _beta(port_ret: np.ndarray, market_ret: np.ndarray) -> float:
    if len(port_ret) < 5:
        return float("nan")
    cov = np.cov(port_ret, market_ret)
    if cov[1, 1] < 1e-12:
        return float("nan")
    return float(cov[0, 1] / cov[1, 1])


def _sma_signal(prices: np.ndarray) -> np.ndarray:
    """Returns 1 (long) when fast SMA > slow SMA, else 0."""
    if len(prices) < _SMA_SLOW:
        return np.zeros(len(prices))
    fast = pd.Series(prices).rolling(_SMA_FAST).mean().values
    slow = pd.Series(prices).rolling(_SMA_SLOW).mean().values
    signal = np.where(fast > slow, 1.0, 0.0)
    signal[:_SMA_SLOW] = 0.0   # no signal before enough data
    return signal


def _momentum_signal(returns: np.ndarray) -> np.ndarray:
    """Long when trailing 20-day return > 0, else flat."""
    cum = pd.Series(returns).rolling(_MOM_WINDOW).sum().values
    signal = np.where(cum > 0, 1.0, 0.0)
    signal[:_MOM_WINDOW] = 0.0
    return signal


def run_financial_baselines(
    results_dir: str = _RESULTS_DIR,
    csv_out:     str = "results/financial_baselines.csv",
    chart_out:   str = "results/financial_baselines_charts.png",
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

        if len(df) < _SMA_SLOW + 20:
            continue

        mkt   = df["return_next"].values
        pos_s = df["pos_long_static"].values
        pos_a = df["pos_long_adaptive"].values

        # Reconstruct a rough price series for SMA (use cumulative product)
        prices = np.cumprod(1.0 + mkt)

        # Baseline signals
        bah_pos  = np.ones(len(mkt))                   # buy and hold
        sma_pos  = _sma_signal(prices)                  # SMA crossover
        mom_pos  = _momentum_signal(mkt)                # momentum

        # Returns for each strategy
        ret_bah  = bah_pos  * mkt
        ret_sma  = sma_pos  * mkt
        ret_mom  = mom_pos  * mkt
        ret_s    = pos_s    * mkt
        ret_a    = pos_a    * mkt

        def _metrics(ret: np.ndarray, name: str) -> dict:
            return {
                f"sharpe_{name}":  round(_sharpe(ret),  4),
                f"cagr_{name}":    round(_cagr(ret),    4),
                f"maxdd_{name}":   round(_max_dd(ret),  4),
                f"beta_{name}":    round(_beta(ret, mkt), 4),
            }

        stem = path.stem.replace("daily_monitoring_", "")
        parts = stem.split("_")
        rec = {
            "ticker": parts[0],
            "period": parts[1] if len(parts) > 1 else "?",
            "combo":  "_".join(parts[2:]) if len(parts) > 2 else "?",
            "n_days": len(df),
        }
        for name, ret in [("bah", ret_bah), ("sma", ret_sma),
                           ("mom", ret_mom), ("static", ret_s), ("adaptive", ret_a)]:
            rec.update(_metrics(ret, name))

        # Alpha vs buy-and-hold
        rec["alpha_static_vs_bah"]   = round(rec["cagr_static"]   - rec["cagr_bah"],   4)
        rec["alpha_adaptive_vs_bah"] = round(rec["cagr_adaptive"] - rec["cagr_bah"],   4)
        rec["sharpe_delta_vs_bah_s"] = round(rec["sharpe_static"]   - rec["sharpe_bah"], 4)
        rec["sharpe_delta_vs_bah_a"] = round(rec["sharpe_adaptive"] - rec["sharpe_bah"], 4)

        records.append(rec)

    if not records:
        print("No records.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    ok = df_out.dropna(subset=["sharpe_bah", "sharpe_static", "sharpe_adaptive"])

    print(f"\nFinancial baselines ({len(ok)} runs):\n")
    strategies = [
        ("Buy & hold",  "bah"),
        ("SMA crossover","sma"),
        ("Momentum",    "mom"),
        ("Static model","static"),
        ("Adaptive model","adaptive"),
    ]
    for name, tag in strategies:
        sh = ok[f"sharpe_{tag}"].mean()
        cg = ok[f"cagr_{tag}"].mean()
        md = ok[f"maxdd_{tag}"].mean()
        bt = ok[f"beta_{tag}"].mean()
        print(f"  {name:20s}: Sharpe={sh:+.3f}  CAGR={cg:+.3f}  MaxDD={md:.3f}  Beta={bt:.3f}")

    print(f"\n  Adaptive vs buy-and-hold:")
    print(f"    CAGR alpha:     {ok['alpha_adaptive_vs_bah'].mean():+.4f}  "
          f"({(ok['alpha_adaptive_vs_bah']>0).mean()*100:.1f}% positive)")
    print(f"    Sharpe alpha:   {ok['sharpe_delta_vs_bah_a'].mean():+.4f}  "
          f"({(ok['sharpe_delta_vs_bah_a']>0).mean()*100:.1f}% positive)")
    print(f"\n  Adaptive vs SMA crossover:")
    diff_sma = ok["sharpe_adaptive"] - ok["sharpe_sma"]
    print(f"    Sharpe delta:   {diff_sma.mean():+.4f}  ({(diff_sma>0).mean()*100:.1f}% positive)")
    print(f"  Adaptive vs momentum:")
    diff_mom = ok["sharpe_adaptive"] - ok["sharpe_mom"]
    print(f"    Sharpe delta:   {diff_mom.mean():+.4f}  ({(diff_mom>0).mean()*100:.1f}% positive)")

    # t-tests for significance
    for desc, data in [
        ("Adaptive Sharpe > static",  ok["sharpe_adaptive"] - ok["sharpe_static"]),
        ("Adaptive Sharpe > SMA",     ok["sharpe_adaptive"] - ok["sharpe_sma"]),
        ("Adaptive Sharpe > momentum",ok["sharpe_adaptive"] - ok["sharpe_mom"]),
        ("Adaptive Sharpe > buy-hold",ok["sharpe_delta_vs_bah_a"]),
    ]:
        t, p = sp_stats.ttest_1samp(data.dropna(), 0)
        print(f"  t-test {desc}: t={t:.2f}, p={p:.4f}")

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

        strategies = ["bah", "sma", "mom", "static", "adaptive"]
        labels     = ["Buy &\nHold", "SMA\nCrossover", "Momentum", "Static\nModel", "Adaptive\nModel"]
        colors     = ["#636363", "#fdae61", "#2c7bb6", "#756bb1", "#1a9641"]

        # Panel 1: mean Sharpe comparison
        ax = axes[0]
        means  = [df[f"sharpe_{s}"].mean() for s in strategies]
        sems   = [df[f"sharpe_{s}"].sem()  for s in strategies]
        bars = ax.bar(labels, means, color=colors, alpha=0.85)
        ax.errorbar(labels, means, yerr=[1.96 * s for s in sems],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean Sharpe ratio")
        ax.set_title("Sharpe ratio: all strategies vs baselines\n"
                     "(error bars = 1.96 × SEM)")

        # Panel 2: mean CAGR comparison
        ax2 = axes[1]
        means2 = [df[f"cagr_{s}"].mean() for s in strategies]
        sems2  = [df[f"cagr_{s}"].sem()  for s in strategies]
        bars2 = ax2.bar(labels, means2, color=colors, alpha=0.85)
        ax2.errorbar(labels, means2, yerr=[1.96 * s for s in sems2],
                     fmt="none", color="black", capsize=5, lw=1.5)
        ax2.axhline(0, color="black", lw=0.8, ls="--")
        for bar, val in zip(bars2, means2):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                     f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Mean annualised CAGR")
        ax2.set_title("CAGR comparison across strategies")

        # Panel 3: max drawdown comparison
        ax3 = axes[2]
        means3 = [df[f"maxdd_{s}"].mean() for s in strategies]
        bars3 = ax3.bar(labels, means3, color=colors, alpha=0.85)
        ax3.axhline(0, color="black", lw=0.8, ls="--")
        for bar, val in zip(bars3, means3):
            ax3.text(bar.get_x() + bar.get_width() / 2, val - 0.003,
                     f"{val:.3f}", ha="center", va="top", fontsize=9)
        ax3.set_ylabel("Mean maximum drawdown")
        ax3.set_title("Maximum drawdown (less negative = better)")

        # Panel 4: adaptive Sharpe vs each baseline (violin of difference)
        ax4 = axes[3]
        diffs = [
            df["sharpe_adaptive"] - df["sharpe_bah"],
            df["sharpe_adaptive"] - df["sharpe_sma"],
            df["sharpe_adaptive"] - df["sharpe_mom"],
            df["sharpe_adaptive"] - df["sharpe_static"],
        ]
        diff_labels = ["vs Buy\n&Hold", "vs SMA", "vs\nMomentum", "vs Static\nmodel"]
        parts = ax4.violinplot([d.dropna().values for d in diffs],
                                positions=range(1, 5), showmedians=True)
        diff_colors = ["#636363", "#fdae61", "#2c7bb6", "#756bb1"]
        for pc, col in zip(parts["bodies"], diff_colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax4.axhline(0, color="black", lw=0.8, ls="--")
        ax4.set_xticks(range(1, 5))
        ax4.set_xticklabels(diff_labels)
        ax4.set_ylabel("Adaptive Sharpe minus baseline Sharpe")
        ax4.set_title("Adaptive advantage over each baseline\n"
                      "(above 0 = adaptive wins)")
        for i, d in enumerate(diffs, 1):
            ax4.text(i, d.mean() + 0.005, f"μ={d.mean():+.3f}",
                     ha="center", fontsize=8)

        fig.suptitle(
            "Financial strategy baseline comparison\n"
            "Buy-and-hold, SMA crossover, and momentum are standard quant benchmarks",
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
    run_financial_baselines()
