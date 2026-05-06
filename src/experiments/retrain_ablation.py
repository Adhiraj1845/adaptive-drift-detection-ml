"""Retrain-strategy ablation: isolates marginal contribution of each adaptation mechanism."""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
import traceback
from typing import Any

import pandas as pd

from src.experiments.detector_ablation import (
    _BASE_ASSETS,
    _PERIOD_VARIANTS,
    _prefetch_all_data,
)

_RETRAIN_COMBOS: list[dict] = [
    {
        "label":                   "drift_only",
        "use_sgd_online":          False,
        "use_scheduled_retrain":   False,
        "use_error_rate_trigger":  False,
    },
    {
        "label":                   "sgd",
        "use_sgd_online":          True,
        "use_scheduled_retrain":   False,
        "use_error_rate_trigger":  False,
    },
    {
        "label":                   "scheduled",
        "use_sgd_online":          False,
        "use_scheduled_retrain":   True,
        "use_error_rate_trigger":  False,
    },
    {
        "label":                   "error_trigger",
        "use_sgd_online":          False,
        "use_scheduled_retrain":   False,
        "use_error_rate_trigger":  True,
    },
    {
        "label":                   "all",
        "use_sgd_online":          True,
        "use_scheduled_retrain":   True,
        "use_error_rate_trigger":  True,
    },
]


def _build_cfg(asset: dict, period: dict, combo: dict):
    from src.utils.cli import RunConfig

    data_start = asset["data_start"]
    ticker     = asset["ticker"]
    train_end  = period["train_end"]

    if pd.Timestamp(data_start) > pd.Timestamp(train_end):
        return None

    data_end = "2024-12-31"
    tag = f"{ticker.replace('-', '')}_{period['label']}_{combo['label']}"

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
        # All detectors on
        use_ks=True,
        use_psi=True,
        use_js=True,
        use_prediction_drift=True,
        use_page_hinkley=True,
        # Retrain strategy varies
        use_sgd_online=combo["use_sgd_online"],
        use_scheduled_retrain=combo["use_scheduled_retrain"],
        use_error_rate_trigger=combo["use_error_rate_trigger"],
    )


def build_jobs(log_dir: str = "results/retrain_ablation_logs") -> list[tuple]:
    jobs = []
    for asset in _BASE_ASSETS:
        for period in _PERIOD_VARIANTS:
            for combo in _RETRAIN_COMBOS:
                cfg = _build_cfg(asset, period, combo)
                if cfg is not None:
                    jobs.append((cfg, log_dir))
    return jobs


def _worker(args: tuple) -> dict[str, Any]:
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
                    "run_tag":                cfg.run_tag,
                    "ticker":                 cfg.ticker_or_series,
                    "period_label":           cfg.run_tag.split("_")[1] if "_" in cfg.run_tag else "",
                    "use_sgd_online":         cfg.use_sgd_online,
                    "use_scheduled_retrain":  cfg.use_scheduled_retrain,
                    "use_error_rate_trigger": cfg.use_error_rate_trigger,
                    "status":                 "error",
                    "error":                  traceback.format_exc().splitlines()[-1],
                }
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def run_retrain_ablation(
    n_workers: int | None = None,
    log_dir: str = "results/retrain_ablation_logs",
    dry_run: bool = False,
) -> pd.DataFrame:

    jobs   = build_jobs(log_dir)
    n_jobs = len(jobs)

    if n_workers is None or n_workers == -1:
        n_workers = multiprocessing.cpu_count()

    asset_period_pairs = len({(cfg.ticker_or_series, cfg.train_end) for cfg, _ in jobs})

    print(
        f"Retrain strategy ablation:\n"
        f"  Assets          : {len(_BASE_ASSETS)}\n"
        f"  Period variants : {len(_PERIOD_VARIANTS)}\n"
        f"  Valid (asset,period) pairs: {asset_period_pairs}\n"
        f"  Retrain combos  : {len(_RETRAIN_COMBOS)}\n"
        f"  Total runs      : {n_jobs}\n"
        f"  Workers         : {n_workers}",
        flush=True,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if dry_run:
        for cfg, _ in jobs:
            print(f"  {cfg.run_tag}")
        return pd.DataFrame()

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
        ticker_to_group = {a["ticker"]: a["group"] for a in _BASE_ASSETS}
        if "ticker" in df.columns and "group" not in df.columns:
            df["group"] = df["ticker"].map(ticker_to_group)
        if "period_label" not in df.columns and "run_tag" in df.columns:
            df["period_label"] = df["run_tag"].apply(
                lambda t: t.split("_")[1] if isinstance(t, str) and "_" in t else ""
            )
        df = df.sort_values(["ticker", "period_label", "run_tag"]).reset_index(drop=True)

    summary_path = "results/retrain_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to {summary_path}  ({len(df)} rows)", flush=True)

    if not df.empty and "status" in df.columns:
        ok  = (df["status"] == "ok").sum()
        err = (df["status"] == "error").sum()
        print(f"Completed: {ok} ok  |  {err} errors", flush=True)

    # ── Per-combo aggregate ────────────────────────────────────────────────
    combo_cols = ["use_sgd_online", "use_scheduled_retrain", "use_error_rate_trigger"]
    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df
    agg_cols = [c for c in [
        "acc_static", "acc_adaptive", "acc_delta",
        "sharpe_static", "sharpe_adaptive", "sharpe_delta",
        "sharpe_ls_static", "sharpe_ls_adaptive", "sharpe_ls_delta",
        "cagr_static", "cagr_adaptive", "cagr_market",
        "maxdd_static", "maxdd_adaptive",
        "mean_logloss_static", "mean_logloss_adaptive", "logloss_delta",
        "n_drift_events", "n_retrains", "drift_vol_corr",
    ] if c in ok_df.columns]

    if not ok_df.empty and all(c in ok_df.columns for c in combo_cols):
        combo_summary = (
            ok_df
            .groupby(combo_cols)[agg_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        combo_summary.to_csv("results/retrain_combo_summary.csv", index=False)
        print("\nMean metrics by retrain strategy (across all assets & periods):")
        print(combo_summary.to_string(index=False))

    # ── Per-asset aggregate ───────────────────────────────────────────────
    if not ok_df.empty and "ticker" in ok_df.columns:
        group_by = ["ticker", "period_label"] if "period_label" in ok_df.columns else ["ticker"]
        asset_summary = (
            ok_df
            .groupby(group_by)[agg_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        asset_summary.to_csv("results/retrain_asset_summary.csv", index=False)
        print(f"\nPer-asset summary saved to results/retrain_asset_summary.csv")

    return df


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Retrain strategy ablation: 50 assets × 4 periods × 5 retrain combos"
    )
    parser.add_argument(
        "--workers", type=int, default=-1,
        help="Parallel workers (-1 = all CPU cores)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all job tags without running",
    )
    parser.add_argument(
        "--log-dir", default="results/retrain_ablation_logs",
        help="Directory for per-run log files",
    )
    parser.add_argument(
        "--rerun-errors", action="store_true",
        help="Re-run only errored jobs from results/retrain_summary.csv, then merge back.",
    )
    args = parser.parse_args()

    if args.rerun_errors:
        summary_path = "results/retrain_summary.csv"
        if not os.path.exists(summary_path):
            print(f"ERROR: {summary_path} not found. Run the full ablation first.")
            sys.exit(1)

        existing = pd.read_csv(summary_path)
        errored_tags = set(existing.loc[existing["status"] == "error", "run_tag"])
        print(f"Re-running {len(errored_tags)} errored jobs from {summary_path}", flush=True)

        all_jobs = build_jobs(args.log_dir)
        jobs = [(cfg, ld) for cfg, ld in all_jobs if cfg.run_tag in errored_tags]
        if not jobs:
            print("No matching jobs found. Nothing to do.")
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
        ticker_to_group = {a["ticker"]: a["group"] for a in _BASE_ASSETS}
        if "ticker" in new_df.columns and "group" not in new_df.columns:
            new_df["group"] = new_df["ticker"].map(ticker_to_group)
        if "period_label" not in new_df.columns and "run_tag" in new_df.columns:
            new_df["period_label"] = new_df["run_tag"].apply(
                lambda t: t.split("_")[1] if isinstance(t, str) and "_" in t else ""
            )

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

    run_retrain_ablation(
        n_workers=args.workers,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
    )
