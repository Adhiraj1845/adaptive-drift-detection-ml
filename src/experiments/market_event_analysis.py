"""Market event alignment: drift alarm counts within ±N days of known macro events."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_EVENTS: list[dict] = [
    {"label": "COVID crash start",      "date": "2020-02-20", "color": "#d7191c"},
    {"label": "COVID bottom",           "date": "2020-03-23", "color": "#d7191c"},
    {"label": "COVID recovery",         "date": "2020-04-09", "color": "#1a9641"},
    {"label": "GameStop squeeze",       "date": "2021-01-27", "color": "#fdae61"},
    {"label": "Fed tapering start",     "date": "2021-11-03", "color": "#756bb1"},
    {"label": "Russia invades Ukraine", "date": "2022-02-24", "color": "#e41a1c"},
    {"label": "First Fed rate hike",    "date": "2022-03-16", "color": "#756bb1"},
    {"label": "SVB collapse",           "date": "2023-03-10", "color": "#ff7f00"},
    {"label": "NVDA AI surge",          "date": "2023-05-25", "color": "#4daf4a"},
    {"label": "Rate-cut pivot signal",  "date": "2023-11-01", "color": "#984ea3"},
]

_RESULTS_DIR = "results"
_PROXIMITY_WINDOW = 15  # ±15 trading days


def _find_monitoring_files(results_dir: str, combo_filter: str | None) -> list[Path]:
    paths = list(Path(results_dir).glob("daily_monitoring_*.csv"))
    if combo_filter:
        paths = [p for p in paths if f"_{combo_filter}.csv" in p.name or p.name.endswith(f"_{combo_filter}.csv")]
    return sorted(paths)


def _parse_ticker_period(stem: str) -> tuple[str, str]:
    parts = stem.replace("daily_monitoring_", "").split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], "unknown"


def load_alarm_dates(path: Path) -> pd.DatetimeIndex:
    try:
        df = pd.read_csv(path, usecols=["date", "action"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return pd.DatetimeIndex(df.loc[df["action"] != "none", "date"])
    except Exception:
        return pd.DatetimeIndex([])


def proximity_score(alarm_dates: pd.DatetimeIndex, event_date: pd.Timestamp, window: int) -> dict:
    """Count alarms within ±window days of event_date."""
    if len(alarm_dates) == 0:
        return {"count": 0, "min_lag_days": None, "any_within_window": False}
    lags = (alarm_dates - event_date).days
    within = lags[(lags >= -window) & (lags <= window)]
    min_lag = int(within[np.abs(within).argmin()]) if len(within) > 0 else None
    return {
        "count": len(within),
        "min_lag_days": min_lag,
        "any_within_window": len(within) > 0,
    }


def run_market_event_analysis(
    results_dir: str = _RESULTS_DIR,
    combo_filter: str | None = "all",
    proximity_window: int = _PROXIMITY_WINDOW,
    chart_timeline: str = "results/market_event_timeline.png",
    chart_heatmap: str = "results/market_event_heatmap.png",
    csv_out: str = "results/market_event_proximity.csv",
) -> pd.DataFrame:

    files = _find_monitoring_files(results_dir, combo_filter)
    if not files:
        print(f"No daily_monitoring_*_{combo_filter or ''}.csv files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} monitoring files  (combo_filter={combo_filter!r})")

    records: list[dict] = []
    # Also collect all (date, drift_index) rows from the largest/most representative file
    # for the timeline chart — use AAPL covid or first available
    timeline_df: pd.DataFrame | None = None

    for path in files:
        ticker, period = _parse_ticker_period(path.stem)
        try:
            df = pd.read_csv(path, usecols=["date", "action", "drift_index"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        if timeline_df is None and ticker == "AAPL" and period == "covid":
            timeline_df = df.copy()

        alarm_dates = pd.DatetimeIndex(df.loc[df["action"] != "none", "date"])

        for ev in _EVENTS:
            ev_date = pd.Timestamp(ev["date"])
            # Only analyse if the monitoring period covers the event
            if df["date"].min() > ev_date or df["date"].max() < ev_date:
                continue
            prox = proximity_score(alarm_dates, ev_date, proximity_window)
            records.append({
                "event":             ev["label"],
                "event_date":        ev["date"],
                "ticker":            ticker,
                "period":            period,
                "alarms_in_window":  prox["count"],
                "min_lag_days":      prox["min_lag_days"],
                "any_within_window": prox["any_within_window"],
                "total_alarms":      len(alarm_dates),
            })

    if not records:
        print("No records produced — check that monitoring files cover the event dates.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}  ({len(df_out)} rows)")

    # ── Summary table ──────────────────────────────────────────────────────────
    summary = (
        df_out.groupby("event")
        .agg(
            n_assets_covered=("ticker", "count"),
            pct_with_alarm=("any_within_window", "mean"),
            mean_alarms=("alarms_in_window", "mean"),
            median_lag=("min_lag_days", "median"),
        )
        .round(3)
        .reset_index()
    )
    summary["pct_with_alarm"] = (summary["pct_with_alarm"] * 100).round(1)
    print("\nEvent alignment summary (±{} trading days):".format(proximity_window))
    print(summary.to_string(index=False))

    if timeline_df is not None:
        _plot_timeline(timeline_df, chart_timeline)
    _plot_heatmap(df_out, chart_heatmap, proximity_window)

    return df_out


def _plot_timeline(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(df["date"], df["drift_index"], color="#2c7bb6", lw=0.8, label="Drift index")

        alarm_mask = df["action"] != "none"
        ax.scatter(df.loc[alarm_mask, "date"], df.loc[alarm_mask, "drift_index"],
                   color="#d7191c", s=18, zorder=3, label="Alarm fired")

        for ev in _EVENTS:
            ev_date = pd.Timestamp(ev["date"])
            if df["date"].min() <= ev_date <= df["date"].max():
                ax.axvline(ev_date, color=ev["color"], lw=1.2, ls="--", alpha=0.75)
                ax.text(ev_date, ax.get_ylim()[1] * 0.92, ev["label"],
                        rotation=90, fontsize=6, ha="right", va="top", color=ev["color"])

        ax.set_xlabel("Date")
        ax.set_ylabel("Drift index")
        ax.set_title("AAPL (COVID period) — drift index vs known market events")
        ax.legend(fontsize=8)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [timeline chart skipped] {e}")


def _plot_heatmap(df_out: pd.DataFrame, path: str, window: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # pct_with_alarm per event × asset
        pivot = (
            df_out.groupby(["event", "ticker"])["any_within_window"]
            .mean()
            .unstack(fill_value=0.0)
        )

        # Order events chronologically
        event_order = [ev["label"] for ev in _EVENTS if ev["label"] in pivot.index]
        pivot = pivot.reindex(event_order)

        fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.45), max(4, len(pivot) * 0.55)))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        plt.colorbar(im, ax=ax, label="Fraction of periods with alarm")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color="black" if val < 0.7 else "white")

        ax.set_title(
            f"Market event alarm alignment  (±{window} trading days)\n"
            "Green = alarm fired near event for most periods; Red = missed",
            fontsize=10,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [heatmap skipped] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market event alignment analysis")
    parser.add_argument("--window", type=int, default=_PROXIMITY_WINDOW,
                        help="Proximity window in trading days (default 15)")
    parser.add_argument("--combo", default="all",
                        help="Combo tag to filter monitoring files (default: all)")
    args = parser.parse_args()
    run_market_event_analysis(combo_filter=args.combo, proximity_window=args.window)
