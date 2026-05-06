"""
Quick sanity-check experiment.

8 assets (one per class) × 5 detector combos × 4 time periods ≈ 160 runs.
Results saved to quick_check/ at the repo root.

Usage
-----
    python quick_check_experiment.py            # all CPU cores
    python quick_check_experiment.py --workers 4
    python quick_check_experiment.py --dry-run
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import traceback
from typing import Any

import pandas as pd

# ── Assets: one representative per asset class ─────────────────────────────
ASSETS = [
    {"ticker": "AAPL",    "data_start": "2010-01-01", "group": "us_equity"},
    {"ticker": "NVDA",    "data_start": "2010-01-01", "group": "us_equity_volatile"},
    {"ticker": "SPY",     "data_start": "2010-01-01", "group": "us_index"},
    {"ticker": "EWJ",     "data_start": "2010-01-01", "group": "international"},
    {"ticker": "TLT",     "data_start": "2010-01-01", "group": "fixed_income"},
    {"ticker": "GLD",     "data_start": "2010-01-01", "group": "commodity"},
    {"ticker": "XLK",     "data_start": "2010-01-01", "group": "sector"},
    {"ticker": "BTC-USD", "data_start": "2015-01-01", "group": "crypto"},
]

# ── Time periods: all four evaluation windows ──────────────────────────────
PERIODS = [
    {"label": "early",  "train_end": "2014-12-31", "eval_start": "2015-01-01", "eval_end": "2019-12-31"},
    {"label": "std",    "train_end": "2017-12-31", "eval_start": "2018-01-01", "eval_end": "2024-12-31"},
    {"label": "covid",  "train_end": "2019-12-31", "eval_start": "2020-01-01", "eval_end": "2024-12-31"},
    {"label": "recent", "train_end": "2021-12-31", "eval_start": "2022-01-01", "eval_end": "2024-12-31"},
]

# ── Detector combos: 5 key configurations ─────────────────────────────────
#   ph_only       — Pareto-optimal on accuracy/runtime frontier
#   psi_ph        — next best in ablation
#   full          — full ensemble (KS+PSI+JS+PH)
#   no_detectors  — all feature detectors off, prediction drift still on (adaptation baseline)
#   pure_static   — zero adaptation of any kind (true static baseline)
COMBOS = [
    {"label": "ph_only",      "use_ks": False, "use_psi": False, "use_js": False, "use_page_hinkley": True,  "use_prediction_drift": True},
    {"label": "psi_ph",       "use_ks": False, "use_psi": True,  "use_js": False, "use_page_hinkley": True,  "use_prediction_drift": True},
    {"label": "full",         "use_ks": True,  "use_psi": True,  "use_js": True,  "use_page_hinkley": True,  "use_prediction_drift": True},
    {"label": "no_detectors", "use_ks": False, "use_psi": False, "use_js": False, "use_page_hinkley": False, "use_prediction_drift": True},
    {"label": "pure_static",  "use_ks": False, "use_psi": False, "use_js": False, "use_page_hinkley": False, "use_prediction_drift": False},
]

OUT_DIR  = "quick_check"
LOG_DIR  = os.path.join(OUT_DIR, "logs")
CSV_PATH = os.path.join(OUT_DIR, "results.csv")


# ── Job builder ────────────────────────────────────────────────────────────

def _build_jobs() -> list[tuple]:
    from src.utils.cli import RunConfig
    jobs = []
    for asset in ASSETS:
        for period in PERIODS:
            # Skip if asset has no data before the training window ends
            if pd.Timestamp(asset["data_start"]) > pd.Timestamp(period["train_end"]):
                continue
            for combo in COMBOS:
                tag = f"{asset['ticker'].replace('-','')}_{period['label']}_{combo['label']}"
                cfg = RunConfig(
                    source="yahoo",
                    ticker_or_series=asset["ticker"],
                    csv_path=None,
                    date_col=None,
                    schema_mode="auto",
                    close_col=None,
                    open_col=None,
                    high_col=None,
                    low_col=None,
                    volume_col=None,
                    data_start=asset["data_start"],
                    data_end="2024-12-31",
                    train_start=asset["data_start"],
                    train_end=period["train_end"],
                    eval_start=period["eval_start"],
                    eval_end=period["eval_end"],
                    retrain_lookback_years=5,
                    min_retrain_rows=400,
                    model_name="gradient_boosting",
                    run_tag=tag,
                    use_ks=combo["use_ks"],
                    use_psi=combo["use_psi"],
                    use_js=combo["use_js"],
                    use_prediction_drift=combo["use_prediction_drift"],
                    use_page_hinkley=combo["use_page_hinkley"],
                    use_sgd_online=False,
                    use_scheduled_retrain=False,
                    use_error_rate_trigger=False,
                )
                jobs.append((cfg, combo["label"], asset["group"]))
    return jobs


# ── Data pre-fetch ─────────────────────────────────────────────────────────

def _prefetch(jobs: list[tuple]) -> None:
    import yfinance as yf
    from collections import defaultdict

    seen: set[str] = set()
    to_fetch: list[tuple] = []
    for cfg, *_ in jobs:
        cache = f"data/raw/{cfg.ticker_or_series}_{cfg.data_start}_{cfg.data_end}.csv".replace(":", "")
        if cache not in seen:
            seen.add(cache)
            if not os.path.exists(cache):
                to_fetch.append((cfg.ticker_or_series, cfg.data_start, cfg.data_end, cache))

    if not to_fetch:
        print("Pre-fetch: all data already cached.", flush=True)
        return

    os.makedirs("data/raw", exist_ok=True)
    print(f"Pre-fetch: downloading {len(to_fetch)} ticker(s)...", flush=True)

    groups: dict[tuple, list] = defaultdict(list)
    for ticker, start, end, cache in to_fetch:
        groups[(start, end)].append((ticker, cache))

    for (start, end), items in groups.items():
        tickers = [t for t, _ in items]
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
        except Exception as exc:
            print(f"  Batch download failed: {exc}", flush=True)
            continue
        for ticker, cache in items:
            try:
                df = raw.copy() if len(tickers) == 1 else raw[ticker].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(how="all")
                if not df.empty:
                    df.to_csv(cache)
                    print(f"  {ticker}: {len(df)} rows cached", flush=True)
                else:
                    print(f"  {ticker}: empty — skipped", flush=True)
            except Exception as exc:
                print(f"  {ticker}: {exc}", flush=True)

    print("Pre-fetch done.\n", flush=True)


# ── Worker ─────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict[str, Any]:
    cfg, combo_label, group = args
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{cfg.run_tag}.log")

    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        with open(log_path, "w", buffering=1) as f:
            sys.stdout = f
            sys.stderr = f
            try:
                from main import run_pipeline
                result = run_pipeline(cfg, _quiet=True)
                result["status"] = "ok"
                result["combo_label"] = combo_label
                result["group"] = group
                return result
            except Exception:
                tb = traceback.format_exc()
                f.write(f"\n\nERROR:\n{tb}\n")
                return {
                    "run_tag":     cfg.run_tag,
                    "ticker":      cfg.ticker_or_series,
                    "combo_label": combo_label,
                    "group":       group,
                    "status":      "error",
                    "error":       tb.splitlines()[-1],
                }
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = _build_jobs()
    n_jobs = len(jobs)
    n_workers = multiprocessing.cpu_count() if args.workers == -1 else args.workers

    print(
        f"Quick-check experiment\n"
        f"  Assets  : {len(ASSETS)}\n"
        f"  Combos  : {len(COMBOS)}\n"
        f"  Periods : {len(PERIODS)}\n"
        f"  Jobs    : {n_jobs}\n"
        f"  Workers : {n_workers}\n"
        f"  Output  : {OUT_DIR}/",
        flush=True,
    )

    if args.dry_run:
        for cfg, combo_label, group in jobs:
            print(f"  {cfg.run_tag}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    _prefetch(jobs)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
            status = res.get("status", "?")
            tag    = res.get("run_tag", "?")
            acc_a  = res.get("acc_adaptive", float("nan"))
            acc_s  = res.get("acc_static",  float("nan"))
            delta  = acc_a - acc_s if (acc_a == acc_a and acc_s == acc_s) else float("nan")
            print(
                f"[{i:>4}/{n_jobs}] {tag:<55}  status={status}"
                f"  acc_s={acc_s:.3f}  acc_a={acc_a:.3f}  delta={delta:+.4f}",
                flush=True,
            )
            results.append(res)

    df = pd.DataFrame(results)

    # Compute acc_delta if not already present
    if "acc_delta" not in df.columns and {"acc_adaptive", "acc_static"}.issubset(df.columns):
        df["acc_delta"] = df["acc_adaptive"] - df["acc_static"]

    df.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}  ({len(df)} rows)", flush=True)

    # Quick summary table
    ok = df[df["status"] == "ok"]
    if not ok.empty and "acc_delta" in ok.columns:
        print("\nMean acc_delta by combo:", flush=True)
        print(
            ok.groupby("combo_label")["acc_delta"]
              .agg(["mean", "count"])
              .round(4)
              .to_string(),
            flush=True,
        )
        print("\nMean acc_delta by asset group:", flush=True)
        print(
            ok.groupby("group")["acc_delta"]
              .agg(["mean", "count"])
              .round(4)
              .to_string(),
            flush=True,
        )


if __name__ == "__main__":
    main()
