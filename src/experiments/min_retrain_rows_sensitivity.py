"""
min_retrain_rows sensitivity analysis.

Tests whether the 2022 Federal Reserve tightening cycle degradation is an
architectural constraint or a tunable parameter issue.

The hypothesis: no value of min_retrain_rows within the feasible range
{200, 300, 400, 500} recovers performance in the 2021-2022 sub-period,
confirming that the 400-row minimum is a batch-learning floor, not a
mis-set parameter.

Grid: 4 values × 12 tickers × 2 periods (2022-specific and full eval) = 96 runs.

Output
------
  results/min_retrain_rows_sensitivity.csv
  results/min_retrain_rows_chart.png
"""
from __future__ import annotations

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

_MIN_RETRAIN_ROWS_GRID = [200, 300, 400, 500]

# 12 tickers spanning US equity, index, crypto, bonds, international
_TICKERS = [
    "AAPL", "MSFT", "^GSPC", "^DJI",
    "GLD", "TLT", "BTC-USD", "ETH-USD",
    "VEA", "EEM", "AGG", "XOM",
]

# Two periods: 2022 specifically (architectural failure period) + baseline period
_PERIODS = {
    "2022_period": ("2019-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    "baseline":    ("2016-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
}


def _make_cfg(ticker: str, period: str, min_retrain_rows: int):
    from src.utils.cli import RunConfig
    ds, te, es, ee = _PERIODS[period]
    return RunConfig(
        source="yahoo",
        ticker_or_series=ticker,
        csv_path=None,
        date_col=None,
        schema_mode="auto",
        close_col=None, open_col=None, high_col=None, low_col=None, volume_col=None,
        data_start="2014-01-01",
        data_end=ee,
        train_start=ds,
        train_end=te,
        eval_start=es,
        eval_end=ee,
        retrain_lookback_years=2,
        min_retrain_rows=min_retrain_rows,
        model_name="gradient_boosting",
        run_tag=f"minrows_{ticker}_{period}_r{min_retrain_rows}",
        use_ks=False, use_psi=True, use_js=False,
        use_prediction_drift=True, use_page_hinkley=True,
        use_sgd_online=True, use_scheduled_retrain=False,
        use_error_rate_trigger=False,
    )


def _run_one(args: tuple) -> dict:
    ticker, period, mrr = args
    try:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from main import run_pipeline
        cfg = _make_cfg(ticker, period, mrr)
        t0 = time.time()
        res = run_pipeline(cfg, _quiet=True)
        elapsed = time.time() - t0
        return {
            "ticker":           ticker,
            "period":           period,
            "min_retrain_rows": mrr,
            "acc_delta":        round(res.get("acc_delta",    float("nan")), 4),
            "sharpe_delta":     round(res.get("sharpe_delta", float("nan")), 4),
            "n_retrains":       res.get("n_retrains", 0),
            "elapsed_s":        round(elapsed, 1),
            "status":           "ok",
        }
    except Exception as e:
        return {
            "ticker": ticker, "period": period,
            "min_retrain_rows": mrr,
            "status": "error", "error": str(e)[:200],
        }


def run_min_retrain_sensitivity(
    tickers: list[str] | None = None,
    min_rows_grid: list[int] = _MIN_RETRAIN_ROWS_GRID,
    csv_out:   str = "results/min_retrain_rows_sensitivity.csv",
    chart_out: str = "results/min_retrain_rows_chart.png",
    n_workers: int = 6,
) -> pd.DataFrame:

    tickers = tickers or _TICKERS
    combos  = list(product(tickers, list(_PERIODS.keys()), min_rows_grid))

    print(
        f"min_retrain_rows sensitivity: {len(min_rows_grid)} values × "
        f"{len(tickers)} tickers × {len(_PERIODS)} periods = {len(combos)} runs "
        f"(workers={n_workers})"
    )

    records = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, combo): combo for combo in combos}
        for fut in as_completed(futures):
            completed += 1
            rec = fut.result()
            records.append(rec)
            ticker, period, mrr = futures[fut]
            if rec["status"] == "ok":
                print(
                    f"  [{completed:3d}/{len(combos)}] {ticker:<10} {period:<15} "
                    f"min_rows={mrr}  "
                    f"acc_delta={rec['acc_delta']:+.4f}  "
                    f"sharpe_delta={rec['sharpe_delta']:+.4f}"
                )
            else:
                print(
                    f"  [{completed:3d}/{len(combos)}] {ticker} {period} "
                    f"min_rows={mrr}  ERROR: {rec.get('error', '?')}"
                )

    df = pd.DataFrame(records)
    df.to_csv(csv_out, index=False)

    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        print("No successful runs.")
        return df

    _print_summary(ok)
    _plot(ok, chart_out)
    return df


def _print_summary(ok: pd.DataFrame) -> None:
    print("\n=== min_retrain_rows sensitivity summary ===")
    for period in ok["period"].unique():
        sub = ok[ok["period"] == period]
        print(f"\n  Period: {period}")
        for mrr, g in sub.groupby("min_retrain_rows"):
            pos_pct = (g["acc_delta"] > 0).mean() * 100
            print(
                f"    min_rows={mrr:3d}: "
                f"acc_delta={g['acc_delta'].mean():+.4f} ± {g['acc_delta'].std():.4f}  "
                f"({pos_pct:.0f}% positive)  "
                f"sharpe={g['sharpe_delta'].mean():+.4f}"
            )

    # Key test: does any min_retrain_rows value produce positive acc_delta in 2022?
    p22 = ok[ok["period"] == "2022_period"]
    if len(p22) > 0:
        print("\n  KEY FINDING — 2022 period acc_delta by min_retrain_rows:")
        for mrr, g in p22.groupby("min_retrain_rows"):
            mean_d = g["acc_delta"].mean()
            flag   = "POSITIVE" if mean_d > 0 else "negative"
            print(f"    min_rows={mrr}: {mean_d:+.4f}  [{flag}]")
        best = p22.groupby("min_retrain_rows")["acc_delta"].mean().idxmax()
        best_val = p22.groupby("min_retrain_rows")["acc_delta"].mean().max()
        print(
            f"\n  Best 2022 result: min_rows={best} → acc_delta={best_val:+.4f}"
        )
        if best_val <= 0:
            print(
                "  → CONFIRMS architectural constraint: "
                "no min_retrain_rows value recovers 2022 performance."
            )
        else:
            print(
                f"  → NOTE: min_rows={best} shows marginal positive delta "
                "in 2022; interpret with caution (small n per cell)."
            )


def _plot(ok: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        periods   = sorted(ok["period"].unique())
        mrr_vals  = sorted(ok["min_retrain_rows"].unique())
        n_periods = len(periods)

        fig, axes = plt.subplots(1, n_periods, figsize=(7 * n_periods, 6), sharey=False)
        if n_periods == 1:
            axes = [axes]

        period_labels = {
            "2022_period": "2021–2022 (Fed tightening cycle)",
            "baseline":    "2019–2020 (baseline period)",
        }
        colours = ["#d73027", "#fc8d59", "#4575b4", "#313695"]

        for ax, period in zip(axes, periods):
            sub = ok[ok["period"] == period]
            means = sub.groupby("min_retrain_rows")["acc_delta"].mean()
            stds  = sub.groupby("min_retrain_rows")["acc_delta"].std()

            bars = ax.bar(
                [str(m) for m in mrr_vals],
                [means.get(m, 0) for m in mrr_vals],
                yerr=[stds.get(m, 0) for m in mrr_vals],
                color=colours[:len(mrr_vals)],
                capsize=5,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
            ax.set_xlabel("min_retrain_rows", fontsize=11)
            ax.set_ylabel("Mean acc_delta (adaptive − static)", fontsize=11)
            ax.set_title(period_labels.get(period, period), fontsize=12)

            # Annotate bars
            for bar, mrr in zip(bars, mrr_vals):
                val = means.get(mrr, 0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.0003 if val >= 0 else -0.0006),
                    f"{val:+.4f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9,
                )

        fig.suptitle(
            "min_retrain_rows sensitivity: acc_delta by period\n"
            "If 2022 bars remain negative across all values, "
            "degradation is architectural not parametric.",
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--min-rows", nargs="+", type=int, default=_MIN_RETRAIN_ROWS_GRID)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_min_retrain_sensitivity(
        tickers=args.tickers,
        min_rows_grid=args.min_rows,
        n_workers=args.workers,
    )
