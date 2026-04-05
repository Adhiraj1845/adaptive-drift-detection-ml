"""
Systematic detector ablation across 50 assets × 4 time windows × 16 detector combos.

Design
------
Assets  : 50 diverse instruments (mega-cap equity, sector/index ETFs, international,
          fixed income, commodities, crypto).
Periods : 4 non-overlapping / partially-overlapping windows covering different
          market regimes:
            early  – 2010-2014 train / 2015-2019 eval  (pre-COVID bull market)
            std    – 2010-2017 train / 2018-2024 eval  (standard benchmark window)
            covid  – 2014-2019 train / 2020-2024 eval  (COVID crash + recovery)
            recent – 2015-2021 train / 2022-2024 eval  (post-COVID, rate-hike era)
Combos  : 2^4 = 16 combinations of {KS, PSI, JS, PageHinkley} toggles.
          PredictionDrift is always on (it is model-output-specific and without it
          the adaptation signal is blind to model degradation).

Valid configs: assets whose data_start precedes a period's train_end are skipped
automatically, giving ~180 valid (asset, period) pairs × 16 combos ≈ 2,880 runs.

Usage
-----
    python -m src.experiments.detector_ablation               # all CPU cores
    python -m src.experiments.detector_ablation --workers -1  # same
    python -m src.experiments.detector_ablation --workers 8   # explicit count
    python -m src.experiments.detector_ablation --dry-run     # list jobs only
    python -m src.experiments.detector_ablation --manifest    # write manifest CSV and exit
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
import traceback
from typing import Any

import pandas as pd

# ── 50 assets ──────────────────────────────────────────────────────────────
# data_start = earliest date we try to fetch.  If a period's train_end is
# before data_start the config is skipped automatically.
_BASE_ASSETS: list[dict[str, str]] = [
    # US mega-cap equities (20)
    {"ticker": "AAPL",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "MSFT",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "GOOGL",   "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "AMZN",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "NVDA",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "META",    "data_start": "2013-01-01", "group": "us_equity"},   # IPO May 2012
    {"ticker": "TSLA",    "data_start": "2011-01-01", "group": "us_equity"},   # IPO Jun 2010
    {"ticker": "JPM",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "BAC",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "JNJ",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "V",       "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "WMT",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "XOM",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "CVX",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "HD",      "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "PG",      "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "UNH",     "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "MA",      "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "NFLX",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "AMD",     "data_start": "2010-01-01", "group": "us_equity"},
    # US index ETFs (5)
    {"ticker": "SPY",     "data_start": "2010-01-01", "group": "us_index"},
    {"ticker": "QQQ",     "data_start": "2010-01-01", "group": "us_index"},
    {"ticker": "DIA",     "data_start": "2010-01-01", "group": "us_index"},
    {"ticker": "IWM",     "data_start": "2010-01-01", "group": "us_index"},
    {"ticker": "VTI",     "data_start": "2010-01-01", "group": "us_index"},
    # Sector ETFs (5)
    {"ticker": "XLF",     "data_start": "2010-01-01", "group": "sector"},
    {"ticker": "XLK",     "data_start": "2010-01-01", "group": "sector"},
    {"ticker": "XLE",     "data_start": "2010-01-01", "group": "sector"},
    {"ticker": "XLV",     "data_start": "2010-01-01", "group": "sector"},
    {"ticker": "XLI",     "data_start": "2010-01-01", "group": "sector"},
    # International ETFs (5)
    {"ticker": "EWJ",     "data_start": "2010-01-01", "group": "international"},
    {"ticker": "EWZ",     "data_start": "2010-01-01", "group": "international"},
    {"ticker": "EEM",     "data_start": "2010-01-01", "group": "international"},
    {"ticker": "VEU",     "data_start": "2010-01-01", "group": "international"},
    {"ticker": "FXI",     "data_start": "2010-01-01", "group": "international"},
    # Fixed income ETFs (5)
    {"ticker": "TLT",     "data_start": "2010-01-01", "group": "fixed_income"},
    {"ticker": "AGG",     "data_start": "2010-01-01", "group": "fixed_income"},
    {"ticker": "HYG",     "data_start": "2010-01-01", "group": "fixed_income"},
    {"ticker": "LQD",     "data_start": "2010-01-01", "group": "fixed_income"},
    {"ticker": "SHY",     "data_start": "2010-01-01", "group": "fixed_income"},
    # Commodities / alternatives (5)
    {"ticker": "GLD",     "data_start": "2010-01-01", "group": "commodity"},
    {"ticker": "SLV",     "data_start": "2010-01-01", "group": "commodity"},
    {"ticker": "USO",     "data_start": "2010-01-01", "group": "commodity"},
    {"ticker": "GDX",     "data_start": "2010-01-01", "group": "commodity"},
    {"ticker": "DBC",     "data_start": "2010-01-01", "group": "commodity"},
    # Crypto (2)
    {"ticker": "BTC-USD", "data_start": "2015-01-01", "group": "crypto"},
    {"ticker": "ETH-USD", "data_start": "2018-01-01", "group": "crypto"},
    # Other (3)
    {"ticker": "INTC",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "ARKK",    "data_start": "2015-01-01", "group": "us_index"},   # inception Oct 2014
    {"ticker": "SMH",     "data_start": "2010-01-01", "group": "sector"},
]

assert len(_BASE_ASSETS) == 50, f"Expected 50 base assets, got {len(_BASE_ASSETS)}"

# ── 4 time-window variants ─────────────────────────────────────────────────
# Each covers a distinct market regime; using multiple windows guards against
# results being artefacts of any one period.
#
#  early  : pre-COVID bull market, low volatility regime
#  std    : benchmark window — long train / long eval (used in prior experiments)
#  covid  : COVID crash (Feb-Mar 2020), recovery, rate hikes begin 2022
#  recent : short, recent eval — stress-tests whether gains hold post-2022
_PERIOD_VARIANTS: list[dict[str, str]] = [
    {
        "label":      "early",
        "train_end":  "2014-12-31",
        "eval_start": "2015-01-01",
        "eval_end":   "2019-12-31",
        "note":       "pre-COVID bull market (2010-2019)",
    },
    {
        "label":      "std",
        "train_end":  "2017-12-31",
        "eval_start": "2018-01-01",
        "eval_end":   "2024-12-31",
        "note":       "standard benchmark window (2010-2024)",
    },
    {
        "label":      "covid",
        "train_end":  "2019-12-31",
        "eval_start": "2020-01-01",
        "eval_end":   "2024-12-31",
        "note":       "COVID crash + rate-hike recovery (2014-2024)",
    },
    {
        "label":      "recent",
        "train_end":  "2021-12-31",
        "eval_start": "2022-01-01",
        "eval_end":   "2024-12-31",
        "note":       "post-COVID, rate-hike era (2015-2024)",
    },
]

# ── Detector toggle combinations ──────────────────────────────────────────
# 2^4 = 16 combos of {KS, PSI, JS, PageHinkley} with PredictionDrift always on,
# plus one explicit pure_static baseline (all adaptation disabled).
#
# Critically, SGD online learning and scheduled retraining are DISABLED for all
# combos so the only source of adaptation is the drift-triggered batch retrain.
# This makes the "all detectors off" combo a true no-adaptation baseline and
# isolates the causal effect of each detector.  The pure_static combo is an
# additional sanity-check baseline where even prediction drift is disabled.
_DETECTOR_KEYS = ("use_ks", "use_psi", "use_js", "use_page_hinkley")
_COMBOS: list[dict] = [
    dict(zip(_DETECTOR_KEYS, combo))
    for combo in itertools.product([False, True], repeat=4)
]
# 17th entry: pure static — no detection at all, no prediction drift either
_COMBOS.append({
    "use_ks": False, "use_psi": False, "use_js": False, "use_page_hinkley": False,
    "_pure_static": True,   # sentinel — picked up in _build_cfg
})
assert len(_COMBOS) == 17


def _combo_label(combo: dict) -> str:
    if combo.get("_pure_static") or not combo.get("use_prediction_drift", True):
        return "pure_static"
    parts = [
        "ks"   if combo["use_ks"]           else "noKS",
        "psi"  if combo["use_psi"]          else "noPSI",
        "js"   if combo["use_js"]           else "noJS",
        "ph"   if combo["use_page_hinkley"] else "noPH",
    ]
    return "_".join(parts)


def _build_cfg(asset: dict[str, str], period: dict[str, str], combo: dict[str, bool]):
    """
    Return a RunConfig for (asset, period, combo), or None if the period is
    invalid for this asset (data_start after train_end).
    """
    from src.utils.cli import RunConfig

    data_start = asset["data_start"]
    ticker     = asset["ticker"]
    train_end  = period["train_end"]

    # Skip if asset has no data before the training window ends
    if pd.Timestamp(data_start) > pd.Timestamp(train_end):
        return None

    # data_end = max of eval_end or today (capped at 2024-12-31 for reproducibility)
    data_end = "2024-12-31"

    tag = f"{ticker.replace('-', '')}_{period['label']}_{_combo_label(combo)}"

    is_pure_static = bool(combo.get("_pure_static", False))

    return RunConfig(
        source="yahoo",
        ticker_or_series=ticker,
        csv_path=None,
        date_col=None,
        schema_mode="auto",
        close_col=None,
        open_col=None,
        high_col=None,
        low_col=None,
        volume_col=None,
        data_start=data_start,
        data_end=data_end,
        train_start=data_start,
        train_end=train_end,
        eval_start=period["eval_start"],
        eval_end=period["eval_end"],
        retrain_lookback_years=5,
        min_retrain_rows=400,
        model_name="gradient_boosting",
        run_tag=tag,
        use_ks=combo["use_ks"],
        use_psi=combo["use_psi"],
        use_js=combo["use_js"],
        use_prediction_drift=False if is_pure_static else True,
        use_page_hinkley=combo["use_page_hinkley"],
        # Disable all three confounding adaptation sources for the detector ablation.
        # Only drift-triggered batch retrain varies across combos — this is the
        # clean experimental design required to isolate detector contribution.
        use_sgd_online=False,
        use_scheduled_retrain=False,
        use_error_rate_trigger=False,
    )


def build_jobs(log_dir: str = "results/ablation_logs") -> list[tuple]:
    jobs = []
    for asset in _BASE_ASSETS:
        for period in _PERIOD_VARIANTS:
            for combo in _COMBOS:
                cfg = _build_cfg(asset, period, combo)
                if cfg is not None:
                    jobs.append((cfg, log_dir))
    return jobs


def write_manifest(jobs: list[tuple], path: str = "results/ablation_manifest.csv") -> None:
    """Save a human-readable manifest of every planned run."""
    rows = []
    for cfg, _ in jobs:
        rows.append({
            "run_tag":             cfg.run_tag,
            "ticker":              cfg.ticker_or_series,
            "group":               next(
                a["group"] for a in _BASE_ASSETS if a["ticker"] == cfg.ticker_or_series
            ),
            "data_start":          cfg.data_start,
            "data_end":            cfg.data_end,
            "train_start":         cfg.train_start,
            "train_end":           cfg.train_end,
            "eval_start":          cfg.eval_start,
            "eval_end":            cfg.eval_end,
            "period_label":        cfg.run_tag.split("_")[1] if "_" in cfg.run_tag else "",
            "use_ks":               cfg.use_ks,
            "use_psi":              cfg.use_psi,
            "use_js":               cfg.use_js,
            "use_prediction_drift": cfg.use_prediction_drift,
            "use_page_hinkley":     cfg.use_page_hinkley,
            "use_sgd_online":        cfg.use_sgd_online,
            "use_scheduled_retrain": cfg.use_scheduled_retrain,
            "use_error_rate_trigger":cfg.use_error_rate_trigger,
            "model":                 cfg.model_name,
            "combo_label":         _combo_label({
                "use_ks":              cfg.use_ks,
                "use_psi":             cfg.use_psi,
                "use_js":              cfg.use_js,
                "use_page_hinkley":    cfg.use_page_hinkley,
                "use_prediction_drift":cfg.use_prediction_drift,
                "_pure_static":        not cfg.use_prediction_drift,
            }),
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Manifest written to {path}  ({len(df)} rows)", flush=True)


# ── Data pre-fetch ─────────────────────────────────────────────────────────

def _prefetch_all_data(jobs: list[tuple]) -> None:
    """
    Download and cache raw price data for every unique (ticker, data_start, data_end)
    combination before the parallel pool starts.

    Uses a single yfinance batch call per unique (data_start, data_end) group so
    all tickers are fetched in one request — far less likely to trigger Yahoo Finance
    rate limits than 35+ sequential or parallel calls.  Once the CSV is cached in
    data/raw/, every worker uses the local file and never touches the network.
    """
    import yfinance as yf

    # Collect unique (ticker, data_start, data_end, cache_path) that aren't cached yet
    seen: set[str] = set()
    to_fetch: list[tuple[str, str, str, str]] = []

    for cfg, _ in jobs:
        if cfg.source != "yahoo":
            continue
        cache_name = (
            f"data/raw/{cfg.ticker_or_series}_{cfg.data_start}_{cfg.data_end}.csv"
        ).replace(":", "")
        if cache_name not in seen:
            seen.add(cache_name)
            if not os.path.exists(cache_name):
                to_fetch.append((cfg.ticker_or_series, cfg.data_start, cfg.data_end, cache_name))

    if not to_fetch:
        print("Pre-fetch: all data already cached.", flush=True)
        return

    os.makedirs("data/raw", exist_ok=True)
    print(f"Pre-fetch: batch-downloading {len(to_fetch)} ticker(s) in one request...", flush=True)

    # Group by (data_start, data_end) so each group is one batch call
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for ticker, start, end, cache_name in to_fetch:
        groups[(start, end)].append((ticker, cache_name))

    total_ok = 0
    total_fail = 0
    for (start, end), items in groups.items():
        tickers = [t for t, _ in items]
        cache_map = {t: c for t, c in items}
        try:
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, group_by="ticker",
            )
        except Exception as exc:
            print(f"  Batch download failed ({start}->{end}): {exc}", flush=True)
            total_fail += len(tickers)
            continue

        for ticker, cache_name in items:
            try:
                if len(tickers) == 1:
                    # Single-ticker download returns flat columns
                    df = raw.copy()
                else:
                    df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(how="all")
                if df.empty:
                    print(f"  {ticker}: WARN empty", flush=True)
                    total_fail += 1
                else:
                    df.to_csv(cache_name)
                    print(f"  {ticker}: cached ({len(df)} rows)", flush=True)
                    total_ok += 1
            except Exception as exc:
                print(f"  {ticker}: ERROR {exc}", flush=True)
                total_fail += 1

    print(f"Pre-fetch complete: {total_ok} cached, {total_fail} failed.\n", flush=True)


# ── Worker ─────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict[str, Any]:
    """Run one pipeline job; returns a summary dict (never raises)."""
    cfg, log_dir = args

    log_path = os.path.join(log_dir, f"{cfg.run_tag}.log")
    os.makedirs(log_dir, exist_ok=True)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    try:
        with open(log_path, "w", buffering=1) as log_f:
            sys.stdout = log_f
            sys.stderr = log_f
            try:
                from main import run_pipeline
                result = run_pipeline(cfg, _quiet=True)
                result["status"] = "ok"
                return result
            except Exception:
                tb = traceback.format_exc()
                log_f.write(f"\n\nERROR:\n{tb}\n")
                return {
                    "run_tag":              cfg.run_tag,
                    "ticker":               cfg.ticker_or_series,
                    "period_label":         cfg.run_tag.split("_")[1] if "_" in cfg.run_tag else "",
                    "use_ks":               cfg.use_ks,
                    "use_psi":              cfg.use_psi,
                    "use_js":               cfg.use_js,
                    "use_prediction_drift": cfg.use_prediction_drift,
                    "use_page_hinkley":     cfg.use_page_hinkley,
                    "status":               "error",
                    "error":                traceback.format_exc().splitlines()[-1],
                }
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


# ── Main entry ─────────────────────────────────────────────────────────────

def run_ablation(
    n_workers: int | None = None,
    log_dir: str = "results/ablation_logs",
    dry_run: bool = False,
    manifest_only: bool = False,
) -> pd.DataFrame:

    jobs  = build_jobs(log_dir)
    n_jobs = len(jobs)

    if n_workers is None or n_workers == -1:
        n_workers = multiprocessing.cpu_count()

    # Count valid (asset, period) pairs for reporting
    asset_period_pairs = len({(cfg.ticker_or_series, cfg.train_end) for cfg, _ in jobs})

    print(
        f"Detector ablation:\n"
        f"  Assets          : {len(_BASE_ASSETS)}\n"
        f"  Period variants : {len(_PERIOD_VARIANTS)}\n"
        f"  Valid (asset,period) pairs: {asset_period_pairs}\n"
        f"  Detector combos : {len(_COMBOS)}\n"
        f"  Total runs      : {n_jobs}\n"
        f"  Workers         : {n_workers}",
        flush=True,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    manifest_path = "results/ablation_manifest.csv"
    write_manifest(jobs, manifest_path)

    if dry_run:
        for cfg, _ in jobs:
            print(f"  {cfg.run_tag}")
        return pd.DataFrame()

    if manifest_only:
        return pd.DataFrame()

    # Pre-fetch all ticker data sequentially to avoid Yahoo Finance rate-limits
    # when N workers simultaneously request the same (or different) tickers.
    _prefetch_all_data(jobs)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
            status = res.get("status", "?")
            tag    = res.get("run_tag", "?")
            acc_a  = res.get("acc_adaptive", float("nan"))
            acc_s  = res.get("acc_static",  float("nan"))
            print(
                f"[{i:>5}/{n_jobs}] {tag:<55}  "
                f"status={status}  acc_s={acc_s:.3f}  acc_a={acc_a:.3f}",
                flush=True,
            )
            results.append(res)

    df = pd.DataFrame(results)

    if not df.empty:
        # Attach group and period_label from run_tag where missing
        ticker_to_group = {a["ticker"]: a["group"] for a in _BASE_ASSETS}
        if "ticker" in df.columns and "group" not in df.columns:
            df["group"] = df["ticker"].map(ticker_to_group)
        if "period_label" not in df.columns and "run_tag" in df.columns:
            df["period_label"] = df["run_tag"].apply(
                lambda t: t.split("_")[1] if isinstance(t, str) and "_" in t else ""
            )
        df = df.sort_values(["ticker", "period_label", "run_tag"]).reset_index(drop=True)

    summary_path = "results/ablation_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to {summary_path}  ({len(df)} rows)", flush=True)

    if not df.empty and "status" in df.columns:
        ok  = (df["status"] == "ok").sum()
        err = (df["status"] == "error").sum()
        print(f"Completed: {ok} ok  |  {err} errors", flush=True)

    # ── Per-combo aggregate (averaged over all assets + periods) ───────────
    combo_cols = ["use_ks", "use_psi", "use_js", "use_page_hinkley", "use_prediction_drift"]
    ok_df = df[df.get("status", pd.Series(dtype=str)) == "ok"] if "status" in df.columns else df
    if not ok_df.empty and all(c in ok_df.columns for c in combo_cols + ["acc_static", "acc_adaptive"]):
        agg_cols = [c for c in [
            "acc_static", "acc_adaptive", "acc_delta",
            "sharpe_static", "sharpe_adaptive", "sharpe_delta",
            "sharpe_ls_static", "sharpe_ls_adaptive", "sharpe_ls_delta",
            "cagr_static", "cagr_adaptive", "cagr_market",
            "maxdd_static", "maxdd_adaptive",
            "mean_logloss_static", "mean_logloss_adaptive", "logloss_delta",
            "n_drift_events", "n_retrains", "drift_vol_corr",
        ] if c in ok_df.columns]
        combo_summary = (
            ok_df
            .groupby(combo_cols)[agg_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        combo_summary.to_csv("results/ablation_combo_summary.csv", index=False)
        print("\nMean metrics by detector combo (across all assets & periods):")
        print(combo_summary.to_string(index=False))

    # ── Per-asset aggregate (averaged over combos) ─────────────────────────
    if not ok_df.empty and "ticker" in ok_df.columns:
        agg_cols2 = [c for c in [
            "acc_static", "acc_adaptive", "acc_delta",
            "sharpe_static", "sharpe_adaptive", "sharpe_delta",
            "cagr_static", "cagr_adaptive", "cagr_market",
            "maxdd_static", "maxdd_adaptive",
            "mean_logloss_static", "mean_logloss_adaptive", "logloss_delta",
            "n_drift_events", "n_retrains", "drift_vol_corr",
        ] if c in ok_df.columns]
        asset_summary = (
            ok_df
            .groupby(["ticker", "period_label"] if "period_label" in ok_df.columns else ["ticker"])[agg_cols2]
            .mean()
            .round(4)
            .reset_index()
        )
        asset_summary.to_csv("results/ablation_asset_summary.csv", index=False)
        print(f"\nPer-asset summary saved to results/ablation_asset_summary.csv")

    return df


if __name__ == "__main__":
    # Required on Windows for multiprocessing spawn
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Run detector ablation: 50 assets × 4 periods × 16 detector combos"
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Parallel workers (-1 = all CPU cores, default: -1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all job tags without running",
    )
    parser.add_argument(
        "--manifest", action="store_true",
        help="Write manifest CSV and exit (no runs)",
    )
    parser.add_argument(
        "--log-dir", default="results/ablation_logs",
        help="Directory for per-run log files (default: results/ablation_logs)",
    )
    parser.add_argument(
        "--rerun-errors", action="store_true",
        help="Re-run only the jobs that errored in results/ablation_summary.csv, "
             "then merge their results back into that file.",
    )
    args = parser.parse_args()

    if args.rerun_errors:
        summary_path = "results/ablation_summary.csv"
        if not os.path.exists(summary_path):
            print(f"ERROR: {summary_path} not found. Run the full ablation first.")
            sys.exit(1)

        existing = pd.read_csv(summary_path)
        errored_tags = set(existing.loc[existing["status"] == "error", "run_tag"])
        print(f"Re-running {len(errored_tags)} errored jobs from {summary_path}", flush=True)

        all_jobs = build_jobs(args.log_dir)
        jobs = [(cfg, ld) for cfg, ld in all_jobs if cfg.run_tag in errored_tags]
        if not jobs:
            print("No matching jobs found in build_jobs(). Nothing to do.")
            sys.exit(0)

        n_workers = multiprocessing.cpu_count() if args.workers == -1 else args.workers

        _prefetch_all_data(jobs)

        results = []
        n_jobs = len(jobs)
        with multiprocessing.Pool(processes=n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
                status = res.get("status", "?")
                tag    = res.get("run_tag", "?")
                acc_a  = res.get("acc_adaptive", float("nan"))
                acc_s  = res.get("acc_static",  float("nan"))
                print(
                    f"[{i:>4}/{n_jobs}] {tag:<55}  "
                    f"status={status}  acc_s={acc_s:.3f}  acc_a={acc_a:.3f}",
                    flush=True,
                )
                results.append(res)

        new_df = pd.DataFrame(results)

        # Attach group / period_label
        ticker_to_group = {a["ticker"]: a["group"] for a in _BASE_ASSETS}
        if "ticker" in new_df.columns and "group" not in new_df.columns:
            new_df["group"] = new_df["ticker"].map(ticker_to_group)
        if "period_label" not in new_df.columns and "run_tag" in new_df.columns:
            new_df["period_label"] = new_df["run_tag"].apply(
                lambda t: t.split("_")[1] if isinstance(t, str) and "_" in t else ""
            )

        # Merge: drop old errored rows, append new results
        merged = pd.concat([
            existing[~existing["run_tag"].isin(errored_tags)],
            new_df,
        ], ignore_index=True)
        merged = merged.sort_values(["ticker", "period_label", "run_tag"]).reset_index(drop=True)
        merged.to_csv(summary_path, index=False)

        ok  = (merged["status"] == "ok").sum()
        err = (merged["status"] == "error").sum()
        print(f"\nUpdated {summary_path}: {ok} ok  |  {err} errors  ({len(merged)} total)", flush=True)
        sys.exit(0)

    run_ablation(
        n_workers=args.workers,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
        manifest_only=args.manifest,
    )
