"""Hyperparameter sensitivity: window_size × cooldown_days grid across representative tickers."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Default sensitivity grid
_WINDOW_SIZES   = [50, 75, 100, 125, 150]
_COOLDOWN_DAYS  = [2, 3, 5, 7, 10]

_DEFAULT_TICKERS = [
    "AAPL", "MSFT", "BTC-USD", "ETH-USD",
    "^GSPC", "^DJI", "AGG", "GLD",
]

_PERIODS = {
    "std":   ("2010-01-01", "2017-12-31", "2018-01-01", "2024-12-31"),
    "covid": ("2010-01-01", "2019-12-31", "2020-01-01", "2024-12-31"),
}


def _make_cfg(ticker: str, period: str, window_size: int, cooldown: int):
    from src.utils.cli import RunConfig
    ds, te, es, ee = _PERIODS[period]
    return RunConfig(
        source="yahoo",
        ticker_or_series=ticker,
        csv_path=None,
        date_col=None,
        schema_mode="auto",
        close_col=None, open_col=None, high_col=None, low_col=None, volume_col=None,
        data_start="2008-01-01",
        data_end=ee,
        train_start=ds,
        train_end=te,
        eval_start=es,
        eval_end=ee,
        retrain_lookback_years=3,
        min_retrain_rows=400,
        model_name="gradient_boosting",
        run_tag=f"sens_{ticker}_{period}_w{window_size}_c{cooldown}",
        use_ks=False, use_psi=True, use_js=False,
        use_prediction_drift=True, use_page_hinkley=True,
        use_sgd_online=True, use_scheduled_retrain=False,
        use_error_rate_trigger=False,
    )


def _run_one(args: tuple) -> dict:
    """Worker function for parallel execution."""
    ticker, period, ws, cd = args
    try:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from main import run_pipeline
        cfg = _make_cfg(ticker, period, ws, cd)
        t0 = time.time()
        res = run_pipeline(cfg, _quiet=True, _window_size=ws, _cooldown_days=cd)
        elapsed = time.time() - t0
        return {
            "ticker":         ticker,
            "period":         period,
            "window_size":    ws,
            "cooldown_days":  cd,
            "acc_delta":      round(res.get("acc_delta",      float("nan")), 4),
            "sharpe_delta":   round(res.get("sharpe_delta",   float("nan")), 4),
            "n_retrains":     res.get("n_retrains", 0),
            "n_drift_events": res.get("n_drift_events", 0),
            "elapsed_s":      round(elapsed, 1),
            "status":         "ok",
        }
    except Exception as e:
        return {
            "ticker": ticker, "period": period,
            "window_size": ws, "cooldown_days": cd,
            "status": "error", "error": str(e)[:120],
        }


def run_sensitivity(
    tickers: list[str] | None = None,
    periods: list[str] | None = None,
    window_sizes: list[int] = _WINDOW_SIZES,
    cooldown_days: list[int] = _COOLDOWN_DAYS,
    csv_out:   str = "results/sensitivity_analysis.csv",
    chart_out: str = "results/sensitivity_charts.png",
    n_workers: int = 8,
) -> pd.DataFrame:

    tickers = tickers or _DEFAULT_TICKERS
    periods = periods or list(_PERIODS.keys())

    combos = list(product(tickers, periods, window_sizes, cooldown_days))
    print(f"Sensitivity grid: {len(window_sizes)}×{len(cooldown_days)} params × "
          f"{len(tickers)} tickers × {len(periods)} periods = {len(combos)} runs "
          f"(workers={n_workers})")

    records = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, combo): combo for combo in combos}
        for fut in as_completed(futures):
            completed += 1
            rec = fut.result()
            records.append(rec)
            ticker, period, ws, cd = futures[fut]
            if rec["status"] == "ok":
                print(f"  [{completed:3d}/{len(combos)}] {ticker} {period} w={ws:3d} c={cd:2d} "
                      f"→ acc_delta={rec['acc_delta']:+.4f}  "
                      f"sharpe_delta={rec['sharpe_delta']:+.4f}")
            else:
                print(f"  [{completed:3d}/{len(combos)}] {ticker} {period} w={ws} c={cd} "
                      f"→ ERROR: {rec.get('error', '?')}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    ok = df_out[df_out["status"] == "ok"].copy()
    if len(ok) == 0:
        print("No successful runs.")
        return df_out

    print(f"\n=== Sensitivity summary ({len(ok)} successful runs) ===")
    print(f"  Overall acc_delta:   {ok['acc_delta'].mean():+.4f} ± {ok['acc_delta'].std():.4f}")
    print(f"  Overall sharpe_delta:{ok['sharpe_delta'].mean():+.4f} ± {ok['sharpe_delta'].std():.4f}")

    print("\n  acc_delta by window_size:")
    for ws, g in ok.groupby("window_size"):
        print(f"    w={ws:3d}: {g['acc_delta'].mean():+.4f} ± {g['acc_delta'].std():.4f}  "
              f"({(g['acc_delta']>0).mean()*100:.0f}% positive)")

    print("\n  acc_delta by cooldown_days:")
    for cd, g in ok.groupby("cooldown_days"):
        print(f"    c={cd:2d}: {g['acc_delta'].mean():+.4f} ± {g['acc_delta'].std():.4f}  "
              f"({(g['acc_delta']>0).mean()*100:.0f}% positive)")

    # Stability metric: coefficient of variation
    cv_acc    = ok["acc_delta"].std() / (abs(ok["acc_delta"].mean()) + 1e-9)
    cv_sharpe = ok["sharpe_delta"].std() / (abs(ok["sharpe_delta"].mean()) + 1e-9)
    print(f"\n  Coefficient of variation (lower = more stable):")
    print(f"    acc_delta CV:    {cv_acc:.2f}")
    print(f"    sharpe_delta CV: {cv_sharpe:.2f}")
    if cv_acc < 3.0:
        print("  → Results are robust to hyperparameter choice (CV < 3)")

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

        # Panel 1: heatmap acc_delta vs window_size × cooldown_days
        ax = axes[0]
        pivot = df.groupby(["window_size", "cooldown_days"])["acc_delta"].mean().unstack()
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-0.02, vmax=0.02)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"c={c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"w={w}" for w in pivot.index])
        plt.colorbar(im, ax=ax, label="Mean acc_delta")
        ax.set_title("Accuracy delta heatmap\n(window_size × cooldown_days)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            fontsize=7, color="black")

        # Panel 2: heatmap sharpe_delta
        ax2 = axes[1]
        pivot2 = df.groupby(["window_size", "cooldown_days"])["sharpe_delta"].mean().unstack()
        im2 = ax2.imshow(pivot2.values, aspect="auto", cmap="RdYlGn",
                          vmin=-0.2, vmax=0.2)
        ax2.set_xticks(range(len(pivot2.columns)))
        ax2.set_xticklabels([f"c={c}" for c in pivot2.columns])
        ax2.set_yticks(range(len(pivot2.index)))
        ax2.set_yticklabels([f"w={w}" for w in pivot2.index])
        plt.colorbar(im2, ax=ax2, label="Mean sharpe_delta")
        ax2.set_title("Sharpe delta heatmap\n(window_size × cooldown_days)")

        # Panel 3: acc_delta distribution by window_size
        ax3 = axes[2]
        ws_vals = sorted(df["window_size"].unique())
        data3 = [df[df["window_size"] == w]["acc_delta"].dropna().values for w in ws_vals]
        bp3 = ax3.boxplot(data3, labels=[f"w={w}" for w in ws_vals], patch_artist=True)
        colors3 = plt.cm.Blues(np.linspace(0.3, 0.8, len(ws_vals)))
        for patch, col in zip(bp3["boxes"], colors3):
            patch.set_facecolor(col)
        ax3.axhline(0, color="black", lw=0.8, ls="--")
        ax3.set_ylabel("acc_delta")
        ax3.set_title("acc_delta stability across window sizes\n"
                      "(narrow spread → robust to window_size choice)")

        # Panel 4: acc_delta distribution by cooldown_days
        ax4 = axes[3]
        cd_vals = sorted(df["cooldown_days"].unique())
        data4 = [df[df["cooldown_days"] == c]["acc_delta"].dropna().values for c in cd_vals]
        bp4 = ax4.boxplot(data4, labels=[f"c={c}" for c in cd_vals], patch_artist=True)
        colors4 = plt.cm.Oranges(np.linspace(0.3, 0.8, len(cd_vals)))
        for patch, col in zip(bp4["boxes"], colors4):
            patch.set_facecolor(col)
        ax4.axhline(0, color="black", lw=0.8, ls="--")
        ax4.set_ylabel("acc_delta")
        ax4.set_title("acc_delta stability across cooldown values\n"
                      "(narrow spread → robust to cooldown_days choice)")

        fig.suptitle(
            "Hyperparameter sensitivity analysis: window_size × cooldown_days grid\n"
            "Stable heatmaps and narrow boxplots indicate robust, hyperparameter-insensitive results",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--periods", nargs="+", default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=_WINDOW_SIZES)
    parser.add_argument("--cooldowns", nargs="+", type=int, default=_COOLDOWN_DAYS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_sensitivity(
        tickers=args.tickers,
        periods=args.periods,
        window_sizes=args.windows,
        cooldown_days=args.cooldowns,
        n_workers=args.workers,
    )
