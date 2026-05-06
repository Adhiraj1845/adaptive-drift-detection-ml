"""IC market-correlation improvement: adaptive vs static.

Three-panel figure (PH-only configuration throughout):
  A – Scatter of ic_adaptive vs ic_static, coloured by asset group
  B – Grouped bar chart: mean IC by asset group (static vs adaptive)
  C – Mean IC delta by evaluation period with 95% CI
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_IC_CSV = "results/information_coefficient.csv"
_OUT    = "results/ic_correlation_chart.png"
_COMBO  = "noKS_noPSI_noJS_ph"   # PH-only (recommended config)

_TICKER_GROUP: dict[str, str] = {
    "AAPL": "US Equity",  "MSFT": "US Equity",  "GOOGL": "US Equity",
    "AMZN": "US Equity",  "NVDA": "US Equity",  "META": "US Equity",
    "TSLA": "US Equity",  "JPM": "US Equity",   "BAC": "US Equity",
    "JNJ":  "US Equity",  "V":    "US Equity",   "WMT": "US Equity",
    "XOM":  "US Equity",  "CVX":  "US Equity",   "HD":  "US Equity",
    "PG":   "US Equity",  "UNH":  "US Equity",   "MA":  "US Equity",
    "NFLX": "US Equity",  "AMD":  "US Equity",   "INTC":"US Equity",
    "SPY":  "US Index",   "QQQ":  "US Index",    "DIA": "US Index",
    "IWM":  "US Index",   "VTI":  "US Index",    "ARKK":"US Index",
    "XLF":  "Sector",     "XLK":  "Sector",      "XLE": "Sector",
    "XLV":  "Sector",     "XLI":  "Sector",      "SMH": "Sector",
    "EWJ":  "International", "EWZ": "International", "EEM": "International",
    "VEU":  "International", "FXI": "International",
    "TLT":  "Fixed Income",  "AGG": "Fixed Income",  "HYG": "Fixed Income",
    "LQD":  "Fixed Income",  "SHY": "Fixed Income",
    "GLD":  "Commodity",  "SLV":  "Commodity",   "USO": "Commodity",
    "GDX":  "Commodity",  "DBC":  "Commodity",
    "BTC-USD": "Crypto",  "ETH-USD": "Crypto",
}

_GROUP_ORDER = [
    "Crypto", "US Equity", "US Index", "Sector",
    "International", "Fixed Income", "Commodity",
]
_GROUP_COLORS = {
    "Crypto":        "#d7191c",
    "US Equity":     "#2c7bb6",
    "US Index":      "#1a9641",
    "Sector":        "#756bb1",
    "International": "#e08214",
    "Fixed Income":  "#4dac26",
    "Commodity":     "#b2abd2",
}
_PERIOD_ORDER  = ["early", "std", "covid", "recent"]
_PERIOD_LABELS = {
    "early":  "Early\n(2015–19)",
    "std":    "Standard\n(2018–24)",
    "covid":  "COVID\n(2020–24)",
    "recent": "Recent\n(2022–24)",
}


def main() -> None:
    df = pd.read_csv(_IC_CSV)
    df["group"] = df["ticker"].map(_TICKER_GROUP)

    ph = (
        df[df["combo"] == _COMBO]
        .dropna(subset=["ic_static", "ic_adaptive", "group"])
        .copy()
    )
    print(f"PH-only rows after filtering: {len(ph)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ------------------------------------------------------------------ #
    # Panel A: scatter ic_static vs ic_adaptive                           #
    # ------------------------------------------------------------------ #
    ax = axes[0]
    all_vals = pd.concat([ph["ic_static"], ph["ic_adaptive"]])
    pad = 0.03
    lo = all_vals.min() - pad
    hi = all_vals.max() + pad
    lim = (lo, hi)

    ax.plot(lim, lim, color="black", lw=1.0, ls="--", zorder=1)
    ax.axhline(0, color="grey", lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color="grey", lw=0.5, ls=":", zorder=0)

    for grp in _GROUP_ORDER:
        sub = ph[ph["group"] == grp]
        ax.scatter(
            sub["ic_static"], sub["ic_adaptive"],
            c=_GROUP_COLORS[grp], alpha=0.70, s=28,
            zorder=2, edgecolors="none",
        )

    n_above = (ph["ic_adaptive"] > ph["ic_static"]).sum()
    n_total = len(ph)
    ax.text(
        0.03, 0.97,
        f"{n_above / n_total * 100:.0f}% of runs:\nadaptive IC > static IC\n(n = {n_total})",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    # Annotate overall means
    mu_s = ph["ic_static"].mean()
    mu_a = ph["ic_adaptive"].mean()
    ax.axvline(mu_s, color="#2c7bb6", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(mu_a, color="#d7191c", lw=1.0, ls="--", alpha=0.7)
    ax.text(mu_s + 0.003, lo + 0.005, f"μ_static\n={mu_s:+.4f}",
            fontsize=7, color="#2c7bb6", va="bottom")
    ax.text(lo + 0.003, mu_a + 0.003, f"μ_adapt={mu_a:+.4f}",
            fontsize=7, color="#d7191c", va="bottom")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Static IC", fontsize=10)
    ax.set_ylabel("Adaptive IC", fontsize=10)
    ax.set_title("A   Adaptive vs static IC per run\n"
                 "(PH-only; each point = one ticker-period pair)",
                 fontsize=10)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_GROUP_COLORS[g], markersize=7, label=g)
        for g in _GROUP_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", lw=1.0, ls="--", label="y = x")
    )
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
              framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Panel B: grouped bars — mean IC by asset group                      #
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    means_s, means_a, se_s, se_a = [], [], [], []
    for grp in _GROUP_ORDER:
        sub = ph[ph["group"] == grp]
        means_s.append(sub["ic_static"].mean())
        means_a.append(sub["ic_adaptive"].mean())
        se_s.append(sub["ic_static"].sem())
        se_a.append(sub["ic_adaptive"].sem())

    x = np.arange(len(_GROUP_ORDER))
    w = 0.36
    ax2.bar(x - w / 2, means_s, w, yerr=se_s, capsize=3,
            color="#2c7bb6", alpha=0.82, label="Static",
            error_kw={"elinewidth": 1.0, "ecolor": "black"})
    ax2.bar(x + w / 2, means_a, w, yerr=se_a, capsize=3,
            color="#d7191c", alpha=0.82, label="Adaptive",
            error_kw={"elinewidth": 1.0, "ecolor": "black"})

    ax2.axhline(0,    color="black",   lw=0.8, ls="--")
    ax2.axhline(0.02, color="#4dac26", lw=0.8, ls=":", alpha=0.6,
                label="IC = 0.02 (weak)")
    ax2.axhline(0.05, color="#756bb1", lw=0.8, ls=":", alpha=0.6,
                label="IC = 0.05 (useful)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(_GROUP_ORDER, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Mean IC (Spearman rank correlation)", fontsize=9)
    ax2.set_title("B   Mean IC by asset group\n"
                  "(error bars = ±1 SE; Grinold & Kahn 1999 benchmarks shown)",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")

    # ------------------------------------------------------------------ #
    # Panel C: IC delta by evaluation period                              #
    # ------------------------------------------------------------------ #
    ax3 = axes[2]
    rows = []
    for period in _PERIOD_ORDER:
        sub = ph[ph["period"] == period]["ic_delta"].dropna()
        if len(sub) < 3:
            continue
        mean = sub.mean()
        ci95 = 1.96 * sub.sem()
        rows.append({
            "label": _PERIOD_LABELS.get(period, period),
            "mean":  mean,
            "ci":    ci95,
            "n":     len(sub),
        })
    period_df = pd.DataFrame(rows)

    bar_colors = [
        "#1a9641" if m > 0 else "#d7191c"
        for m in period_df["mean"]
    ]
    xpos = np.arange(len(period_df))
    ax3.bar(xpos, period_df["mean"], yerr=period_df["ci"],
            capsize=4, color=bar_colors, alpha=0.82,
            error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax3.axhline(0, color="black", lw=0.8, ls="--")

    for i, (_, row) in enumerate(period_df.iterrows()):
        yoff = row["ci"] + abs(row["mean"]) * 0.05 + 0.0005
        ypos = row["mean"] + yoff if row["mean"] >= 0 else row["mean"] - yoff - 0.002
        ax3.text(i, ypos, f"n={row['n']}", ha="center", fontsize=7.5)

    ax3.set_xticks(xpos)
    ax3.set_xticklabels(period_df["label"], fontsize=9)
    ax3.set_ylabel("Mean IC delta (adaptive − static)", fontsize=9)
    ax3.set_title("C   IC improvement by evaluation period\n"
                  "(bars = mean delta; error bars = 95% CI)",
                  fontsize=10)

    fig.suptitle(
        "Market correlation improvement: adaptive vs static model\n"
        "IC = Spearman rank correlation between predicted probability and realised next-day outcome"
        "  |  PH-only (recommended) configuration",
        fontsize=11,
    )
    fig.tight_layout()
    Path(_OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {_OUT}")


if __name__ == "__main__":
    main()
