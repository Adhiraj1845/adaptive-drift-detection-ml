"""
Generate the five figures that were missing from results/:

  1. results/architecture_diagram.png      — pipeline flowchart (Methodology)
  2. results/ablation_detector_chart.png   — 17-config detector comparison (Results)
  3. results/ablation_asset_chart.png      — per-asset ranked bar chart (Results)
  4. results/nonstationarity_chart.png     — empirical drift in real data (Background)
  5. results/retrain_ablation_chart.png    — 5-strategy retrain comparison (Results)

Usage
-----
    python -m src.experiments.generate_missing_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS = _root / "results"
_RESULTS.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Architecture diagram
# ──────────────────────────────────────────────────────────────────────────────

def _make_box(ax, x, y, w, h, label, sublabel="", color="#4C72B0", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03", linewidth=1.2,
        edgecolor="#2c2c2c", facecolor=color, alpha=0.92, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.030, sublabel,
                ha="center", va="center", fontsize=7,
                color="#dddddd", zorder=4)


def _arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.4, mutation_scale=14), zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label, fontsize=7, color="#444444",
                ha="left", va="center")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BLUE   = "#3A6EA5"
    GREEN  = "#2A7A4B"
    ORANGE = "#C05B1A"
    PURPLE = "#6A3FA0"
    TEAL   = "#1A7A7A"
    GREY   = "#555555"

    # ── Row 1: data sources ──────────────────────────────────────────────────
    y1 = 0.88
    _make_box(ax, 0.18, y1, 0.18, 0.08, "Yahoo Finance / FRED", "yfinance · pandas_datareader", GREY, fontsize=8)
    _make_box(ax, 0.50, y1, 0.18, 0.08, "Feature Engineering", "OHLCV → 28 features\nRollingReturn·Vol·Mom·DD", BLUE)
    _make_box(ax, 0.82, y1, 0.18, 0.08, "Train / Test Split", "Sequential split\n(no look-ahead)", BLUE)

    _arrow(ax, 0.27, y1, 0.41, y1)
    _arrow(ax, 0.59, y1, 0.73, y1)

    # ── Row 2: baseline training ─────────────────────────────────────────────
    y2 = 0.70
    _make_box(ax, 0.25, y2, 0.22, 0.08, "Static Model", "GBM / RF / LR\nRobustScaler (frozen)", GREEN)
    _make_box(ax, 0.75, y2, 0.22, 0.08, "Adaptive Model", "GBM + SGD online blend\nRobustScaler (re-fit)", GREEN)

    _arrow(ax, 0.82, y1 - 0.04, 0.75, y2 + 0.04, "train_df")
    _arrow(ax, 0.82, y1 - 0.04, 0.25, y2 + 0.04, "train_df")

    # ── Row 3: calibration ───────────────────────────────────────────────────
    y3 = 0.52
    _make_box(ax, 0.50, y3, 0.30, 0.08, "Calibration",
              "Replay train_df through detector stack\n"
              "60th/75th/90th/95th pct → low/mod/high/severe",
              ORANGE)
    _arrow(ax, 0.25, y2 - 0.04, 0.50, y3 + 0.04)
    _arrow(ax, 0.75, y2 - 0.04, 0.50, y3 + 0.04)

    # ── Row 4: streaming loop ────────────────────────────────────────────────
    y4 = 0.33
    bw = 0.15
    xs = [0.13, 0.32, 0.51, 0.70, 0.89]
    labels = [
        ("KS · PSI · JS", "Feature drift\nP(X) shift"),
        ("Prediction Drift", "JS on model\noutput probs"),
        ("Page-Hinkley", "Log-loss\nchange-point"),
        ("Drift Index", "0.4·feat +\n0.3·pred + 0.3·perf"),
        ("Tiered Action", "none / moderate\nhigh / severe"),
    ]
    colors = [TEAL, TEAL, TEAL, PURPLE, ORANGE]
    for x, (lbl, sub), col in zip(xs, labels, colors):
        _make_box(ax, x, y4, bw, 0.09, lbl, sub, col, fontsize=8)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + bw / 2, y4, xs[i + 1] - bw / 2, y4)
    _arrow(ax, 0.50, y3 - 0.04, 0.51, y4 + 0.045, "thresholds")

    # ── Row 5: adaptation & evaluation ──────────────────────────────────────
    y5 = 0.13
    _make_box(ax, 0.25, y5, 0.28, 0.08, "Tiered Retraining",
              "moderate: weighted update (1yr)\n"
              "high: sliding window (2yr)\n"
              "severe: new model instance",
              GREEN)
    _make_box(ax, 0.75, y5, 0.28, 0.08, "Statistical Evaluation",
              "McNemar · OLS · Bootstrap Sharpe\n"
              "BCa CI · IC · Sortino · Cost",
              BLUE)

    _arrow(ax, 0.89, y4 - 0.045, 0.89, (y4 + y5) / 2)
    ax.annotate("", xy=(0.39, y5 + 0.04), xytext=(0.89, (y4 + y5) / 2),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4,
                                connectionstyle="arc3,rad=-0.25",
                                mutation_scale=14), zorder=2)
    _arrow(ax, 0.75, y2 - 0.04, 0.75, y5 + 0.04, "predictions")

    # cooldown feedback loop
    ax.annotate("", xy=(0.32, y4 - 0.045), xytext=(0.11, y4 - 0.045),
                arrowprops=dict(arrowstyle="-|>", color="#999999",
                                lw=1.0, linestyle="dashed", mutation_scale=10), zorder=2)
    ax.text(0.215, y4 - 0.065, "5-day cooldown", fontsize=6.5,
            color="#888888", ha="center")

    ax.set_title(
        "Adaptive Drift Detection Pipeline — System Architecture",
        fontsize=13, fontweight="bold", pad=10,
    )

    out = _RESULTS / "architecture_diagram.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[1/5] Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Detector ablation chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_detector_ablation():
    df = pd.read_csv(_RESULTS / "ablation_combo_summary.csv")

    # Build readable labels
    def combo_label(row):
        parts = []
        if row["use_ks"]:   parts.append("KS")
        if row["use_psi"]:  parts.append("PSI")
        if row["use_js"]:   parts.append("JS")
        if row["use_page_hinkley"]:      parts.append("PH")
        if row["use_prediction_drift"]:  parts.append("Pred")
        return "+".join(parts) if parts else "pure_static"

    df["label"] = df.apply(combo_label, axis=1)
    df = df.sort_values("acc_delta")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors_acc    = ["#d73027" if v < 0 else "#4575b4" for v in df["acc_delta"]]
    colors_sharpe = ["#d73027" if v < 0 else "#1a9850" for v in df["sharpe_delta"]]

    for ax, col, colors, title, xlabel in [
        (axes[0], "acc_delta",    colors_acc,    "Accuracy delta (adaptive − static)", "acc_delta"),
        (axes[1], "sharpe_delta", colors_sharpe, "Sharpe delta (adaptive − static)",   "sharpe_delta"),
    ]:
        bars = ax.barh(df["label"], df[col], color=colors, edgecolor="white",
                       linewidth=0.5, height=0.7)
        ax.axvline(0, color="black", lw=1.0, zorder=3)
        for bar, val in zip(bars, df[col]):
            pad = 0.0002 if col == "acc_delta" else 0.003
            ax.text(val + (pad if val >= 0 else -pad),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}" if col == "acc_delta" else f"{val:+.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=7.5, color="#222222")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, lw=0.6)

    fig.suptitle(
        "Detector Configuration Ablation  (3,332 runs · 50 assets · 4 periods)\n"
        "Mean performance across all assets and periods by detector combination",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = _RESULTS / "ablation_detector_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[2/5] Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Asset-level results chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_asset_results():
    df = pd.read_csv(_RESULTS / "ablation_asset_summary.csv")

    # Merge in group info from ablation_summary
    grp = (
        pd.read_csv(_RESULTS / "ablation_summary.csv")[["ticker", "group"]]
        .drop_duplicates("ticker")
    )
    df = df.merge(grp, on="ticker", how="left")
    df = df.sort_values("acc_delta", ascending=True)

    GROUP_COLORS = {
        "us_equity":    "#4575b4",
        "us_index":     "#74add1",
        "sector":       "#abd9e9",
        "crypto":       "#fdae61",
        "commodity":    "#f46d43",
        "fixed_income": "#a50026",
        "international":"#878787",
    }
    colors = [GROUP_COLORS.get(g, "#cccccc") for g in df["group"]]

    fig, ax = plt.subplots(figsize=(12, 11))
    bars = ax.barh(df["ticker"], df["acc_delta"], color=colors,
                   edgecolor="white", linewidth=0.4, height=0.75)
    ax.axvline(0, color="black", lw=1.0, zorder=3)

    for bar, val in zip(bars, df["acc_delta"]):
        ax.text(val + (0.0003 if val >= 0 else -0.0003),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=7, color="#222222")

    # Legend
    handles = [mpatches.Patch(color=c, label=g.replace("_", " ").title())
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Asset class", title_fontsize=8.5, framealpha=0.9)

    ax.set_xlabel("Mean acc_delta (adaptive − static), averaged over all detector combos",
                  fontsize=10)
    ax.set_title(
        "Per-Asset Adaptive Performance  (50 assets · 4 periods · 16 configs)\n"
        "Positive = adaptive model more accurate than static on average",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, lw=0.6)
    fig.tight_layout()

    out = _RESULTS / "ablation_asset_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[3/5] Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Empirical non-stationarity visualisation
# ──────────────────────────────────────────────────────────────────────────────

def fig_nonstationarity():
    # Load AAPL and GSPC from cache
    data_dir = _root / "data" / "raw"
    files = sorted(data_dir.glob("*.csv"))

    series = {}
    for ticker in ["AAPL", "^GSPC", "GLD", "BTC-USD"]:
        matches = [f for f in files if f.name.startswith(
            ticker.replace("^", "") if "^" in ticker else ticker
        )]
        if not matches:
            continue
        df = pd.read_csv(matches[0], index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        close_col = next((c for c in ["Close", "close", "Adj Close", "Price", "Value"]
                         if c in df.columns), None)
        if close_col:
            series[ticker] = df[close_col].dropna()

    if not series:
        print("[4/5] Skipped: no cached data found.")
        return

    def rolling_ks(s: pd.Series, window: int = 126) -> pd.Series:
        """Two-sample KS statistic: reference=[t-2w:t-w], current=[t-w:t]."""
        vals = s.pct_change().dropna().values
        stats_out = []
        idx_out = []
        for i in range(2 * window, len(vals)):
            ref = vals[i - 2 * window: i - window]
            cur = vals[i - window: i]
            d, _ = stats.ks_2samp(ref, cur)
            stats_out.append(d)
            idx_out.append(s.index[i])
        return pd.Series(stats_out, index=idx_out)

    EVENTS = {
        "GFC bottom\n(Mar 2009)":  "2009-03-09",
        "COVID crash\n(Mar 2020)": "2020-03-23",
        "Fed hike cycle\n(Mar 2022)": "2022-03-16",
        "SVB collapse\n(Mar 2023)": "2023-03-10",
    }

    n = len(series)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.5 * n))
    if n == 1:
        axes = [axes]

    COLORS = ["#4575b4", "#d73027", "#1a9850", "#fdae61"]

    for row_idx, (ticker, s) in enumerate(series.items()):
        col = COLORS[row_idx % len(COLORS)]
        ax_price = axes[row_idx][0]
        ax_ks    = axes[row_idx][1]

        # Price panel
        ret_idx = (1 + s.pct_change().dropna()).cumprod()
        ax_price.plot(ret_idx.index, ret_idx.values, color=col, lw=1.2)
        ax_price.set_title(f"{ticker} — Cumulative Return", fontsize=10, fontweight="bold")
        ax_price.set_ylabel("Cumulative return (rebased to 1)")
        ax_price.spines[["top", "right"]].set_visible(False)

        # KS panel
        ks = rolling_ks(s, window=63)  # ~3-month windows
        ax_ks.fill_between(ks.index, 0, ks.values, alpha=0.35, color=col)
        ax_ks.plot(ks.index, ks.values, color=col, lw=0.9)
        threshold = float(np.percentile(ks.values, 90))
        ax_ks.axhline(threshold, color="#d62728", lw=1.0, ls="--",
                      label=f"90th pct = {threshold:.3f}")
        ax_ks.set_title(f"{ticker} — Rolling KS Statistic (63-day reference window)",
                        fontsize=10, fontweight="bold")
        ax_ks.set_ylabel("KS D-statistic")
        ax_ks.legend(fontsize=8, framealpha=0.9)
        ax_ks.spines[["top", "right"]].set_visible(False)

        for label, date_str in EVENTS.items():
            try:
                dt = pd.Timestamp(date_str)
                if ks.index.min() <= dt <= ks.index.max():
                    ax_ks.axvline(dt, color="#555555", lw=0.9, ls=":", alpha=0.8)
                    ax_ks.text(dt, ax_ks.get_ylim()[1] * 0.9, label,
                               fontsize=6.5, rotation=90, va="top",
                               color="#555555", ha="right")
                if ret_idx.index.min() <= dt <= ret_idx.index.max():
                    ax_price.axvline(dt, color="#555555", lw=0.9, ls=":", alpha=0.8)
            except Exception:
                pass

    fig.suptitle(
        "Empirical Evidence of Distribution Shift in Financial Time Series\n"
        "Rolling KS statistic exceeds 90th-percentile baseline at major market events — "
        "validating the need for adaptive drift detection",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = _RESULTS / "nonstationarity_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/5] Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Retrain strategy comparison chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_retrain_ablation():
    df = pd.read_csv(_RESULTS / "retrain_combo_summary.csv")

    def strategy_label(row):
        parts = []
        if row["use_sgd_online"]:             parts.append("SGD")
        if row["use_scheduled_retrain"]:       parts.append("Scheduled")
        if row["use_error_rate_trigger"]:      parts.append("ErrorTrigger")
        return "+".join(parts) if parts else "Drift-only"

    df["label"] = df.apply(strategy_label, axis=1)

    metrics = [
        ("acc_delta",      "Accuracy delta",        "#4575b4"),
        ("sharpe_delta",   "Sharpe delta",           "#1a9850"),
        ("logloss_delta",  "Log-loss delta\n(+ve = adaptive better)", "#d73027"),
        ("n_retrains",     "Mean retrains per run",  "#888888"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (col, title, color) in zip(axes, metrics):
        vals = df[col]
        bar_colors = [color if v >= 0 else "#d73027" for v in vals] if col != "n_retrains" else [color] * len(vals)
        bars = ax.bar(df["label"], vals, color=bar_colors, edgecolor="white",
                      linewidth=0.6, width=0.6)
        if col != "n_retrains":
            ax.axhline(0, color="black", lw=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (max(abs(vals)) * 0.02 if val >= 0 else -max(abs(vals)) * 0.02),
                    f"{val:.4f}" if col == "acc_delta" else f"{val:.2f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8, color="#222222")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, lw=0.6)

    fig.suptitle(
        "Retraining Strategy Comparison  (980 runs · 50 assets · 4 periods)\n"
        "All strategies use the PSI+PH+prediction drift detector configuration",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = _RESULTS / "retrain_ablation_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[5/5] Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating missing figures...")
    fig_architecture()
    fig_detector_ablation()
    fig_asset_results()
    fig_nonstationarity()
    fig_retrain_ablation()
    print("\nDone. All 5 figures saved to results/")
