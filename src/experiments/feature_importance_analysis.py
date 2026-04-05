"""
Feature importance analysis across assets and market regimes.

Trains GBM on two non-overlapping windows per asset and compares which features
drive predictions before vs after structural breaks (2020 COVID, 2022 rate hikes).

Output
------
  results/feature_importance_bars.png   — per-window bar chart for each ticker
  results/feature_importance_heatmap.png — importance heatmap (features × assets)
  results/feature_importance.csv        — raw importance table

Usage
-----
    python -m src.experiments.feature_importance_analysis
    python -m src.experiments.feature_importance_analysis --tickers AAPL MSFT XOM SPY
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data_loader import download_yahoo, load_csv, save_csv, add_features, infer_schema

# Representative cross-asset panel: 2 equity, 1 energy, 1 fixed income, 1 crypto, 1 index
_DEFAULT_TICKERS = ["AAPL", "MSFT", "XOM", "TLT", "SPY", "QQQ"]

# Two training windows — early regime vs post-COVID/rate-hike regime
_WINDOWS = {
    "early_2013_2016":  ("2010-01-01", "2013-01-01", "2016-12-31"),  # (data_start, train_start, train_end)
    "covid_2018_2021":  ("2010-01-01", "2018-01-01", "2021-12-31"),
    "recent_2021_2024": ("2010-01-01", "2021-01-01", "2024-06-30"),
}

_CACHE_DIR = "data/raw"


def _load_or_fetch(ticker: str, start: str, end: str = "2024-12-31") -> pd.DataFrame:
    fname = f"{ticker.replace('-', '_')}_{start}_{end}.csv"
    path  = os.path.join(_CACHE_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return load_csv(path)
    # Fall back to a broader cached file if one exists
    for f in os.listdir(_CACHE_DIR):
        if f.startswith(ticker.replace("-", "_")) and f.endswith(".csv"):
            candidate = os.path.join(_CACHE_DIR, f)
            if os.path.getsize(candidate) > 500:
                return load_csv(candidate)
    print(f"  Downloading {ticker} …")
    df = download_yahoo(ticker, start, end)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    save_csv(df, path)
    return df


def _make_features(df: pd.DataFrame) -> list[str]:
    banned_sub = {"target", "future", "next", "label"}
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c != "Target"
        and not any(s in c.lower() for s in banned_sub)
    ]


def _train_window(df: pd.DataFrame, train_start: str, train_end: str) -> tuple[np.ndarray, list[str]] | None:
    """Train GBM on [train_start, train_end] and return (feature_importances_, feature_names)."""
    window = df.loc[train_start:train_end].copy()
    if len(window) < 200:
        return None
    features = _make_features(window)
    X = window[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = window["Target"]
    sc = RobustScaler()
    X_sc = sc.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_sc, y)
    return clf.feature_importances_, features


def run_feature_importance(
    tickers: list[str] | None = None,
    chart_bars: str = "results/feature_importance_bars.png",
    chart_heat: str = "results/feature_importance_heatmap.png",
    csv_path:   str = "results/feature_importance.csv",
) -> pd.DataFrame:
    if tickers is None:
        tickers = _DEFAULT_TICKERS

    os.makedirs("results", exist_ok=True)

    records: list[dict] = []

    for ticker in tickers:
        print(f"\n[{ticker}]")
        try:
            raw = _load_or_fetch(ticker, start="2010-01-01")
        except Exception as e:
            print(f"  SKIP — could not load: {e}")
            continue

        schema = infer_schema(raw)
        try:
            df = add_features(raw, schema=schema)
        except Exception as e:
            print(f"  SKIP — add_features failed: {e}")
            continue

        for window_label, (data_start, train_start, train_end) in _WINDOWS.items():
            result = _train_window(df, train_start, train_end)
            if result is None:
                print(f"  {window_label}: insufficient data (<200 rows), skipping")
                continue
            importances, features = result
            print(f"  {window_label}: {len(features)} features, top-3: "
                  + ", ".join(
                      f"{features[i]}={importances[i]:.3f}"
                      for i in np.argsort(importances)[::-1][:3]
                  ))
            for feat, imp in zip(features, importances):
                records.append({
                    "ticker": ticker,
                    "window": window_label,
                    "feature": feat,
                    "importance": round(float(imp), 6),
                })

    if not records:
        print("No results — aborting charts.")
        return pd.DataFrame()

    df_imp = pd.DataFrame(records)
    df_imp.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df_imp)} rows)")

    _plot_bars(df_imp, chart_bars)
    _plot_heatmap(df_imp, chart_heat)

    return df_imp


# ── Feature grouping for readability ──────────────────────────────────────────
def _group_feature(name: str) -> str:
    if name in {"Return", "LogReturn"}:
        return "Return"
    if name.startswith("Ret_Mean"):
        return "Rolling Mean"
    if name.startswith("Ret_Vol"):
        return "Rolling Vol"
    if name.startswith("Mom_Sum"):
        return "Momentum"
    if name.startswith("DD_"):
        return "Drawdown"
    if name.startswith("Price_Z"):
        return "Price Z-score"
    if name.startswith("Vol_Z"):
        return "Volume Z"
    if name in {"HL_Range", "CO_Return", "Vol_Change"}:
        return "Intraday"
    if "MA_cross" in name:
        return "MA Cross"
    return "Other"


def _plot_bars(df_imp: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tickers = df_imp["ticker"].unique()
        windows = list(_WINDOWS.keys())
        colours = ["#2c7bb6", "#1a9641", "#d7191c"]

        # Aggregate importance by feature group + window
        df_imp = df_imp.copy()
        df_imp["group"] = df_imp["feature"].apply(_group_feature)

        fig, axes = plt.subplots(
            len(tickers), 1,
            figsize=(14, 4 * len(tickers)),
            sharex=False,
        )
        if len(tickers) == 1:
            axes = [axes]

        for ax, ticker in zip(axes, tickers):
            sub = df_imp[df_imp["ticker"] == ticker]
            grp = (
                sub.groupby(["window", "group"])["importance"]
                .sum()
                .reset_index()
            )
            groups_present = grp["group"].unique()
            x = np.arange(len(groups_present))
            width = 0.25

            for i, (win, col) in enumerate(zip(windows, colours)):
                wdata = grp[grp["window"] == win].set_index("group")
                vals = [wdata.loc[g, "importance"] if g in wdata.index else 0.0
                        for g in groups_present]
                ax.bar(x + i * width, vals, width, label=win, color=col, alpha=0.85)

            ax.set_xticks(x + width)
            ax.set_xticklabels(groups_present, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Total importance")
            ax.set_title(ticker)
            ax.legend(fontsize=7, loc="upper right")

        fig.suptitle("GBM Feature Importance by Group and Regime", fontsize=13)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [bars chart skipped] {e}")


def _plot_heatmap(df_imp: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use the std window for cross-asset comparison
        std_window = "covid_2018_2021"
        sub = df_imp[df_imp["window"] == std_window].copy()
        sub["group"] = sub["feature"].apply(_group_feature)

        pivot = (
            sub.groupby(["ticker", "group"])["importance"]
            .sum()
            .unstack(fill_value=0.0)
        )
        # Normalise each row so values sum to 1
        pivot = pivot.div(pivot.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="Normalised importance")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.3 else "white")

        ax.set_title(
            f"Feature-group importance heatmap  (window: {std_window})\n"
            "Row-normalised — each row sums to 1.0",
            fontsize=11,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [heatmap skipped] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis across regimes")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to analyse (default: AAPL MSFT XOM TLT SPY QQQ)",
    )
    args = parser.parse_args()
    run_feature_importance(tickers=args.tickers)
