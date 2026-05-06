"""Per-run McNemar test comparing static vs adaptive across ablation monitoring CSVs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = "results"
_DEFAULT_ALPHA = 0.05


def _mcnemar_stat(b: int, c: int) -> tuple[float, float]:
    """McNemar test with continuity correction; returns (chi2_stat, p_value)."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def _run_mcnemar(path: Path) -> dict | None:
    try:
        df = pd.read_csv(
            path,
            usecols=["y_true_next", "y_pred_static", "y_pred_adaptive"],
            dtype={"y_true_next": int, "y_pred_static": int, "y_pred_adaptive": int},
        )
    except Exception:
        return None

    y   = df["y_true_next"].values
    ys  = df["y_pred_static"].values
    ya  = df["y_pred_adaptive"].values

    correct_s = (ys == y)
    correct_a = (ya == y)

    n00 = int(np.sum(~correct_s & ~correct_a))
    n01 = int(np.sum(~correct_s &  correct_a))
    n10 = int(np.sum( correct_s & ~correct_a))
    n11 = int(np.sum( correct_s &  correct_a))

    stat, p = _mcnemar_stat(n01, n10)

    acc_s = float(correct_s.mean())
    acc_a = float(correct_a.mean())

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "chi2": round(stat, 4),
        "p_value": round(p, 6),
        "acc_static": round(acc_s, 4),
        "acc_adaptive": round(acc_a, 4),
        "acc_delta": round(acc_a - acc_s, 4),
        "n_obs": len(df),
    }


def _parse_stem(stem: str) -> tuple[str, str, str]:
    """Parse daily_monitoring stem into (ticker, period, combo)."""
    body = stem.replace("daily_monitoring_", "")
    parts = body.split("_")
    ticker = parts[0]
    period = parts[1] if len(parts) > 1 else "unknown"
    combo  = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
    return ticker, period, combo


def run_ablation_mcnemar(
    results_dir: str = _RESULTS_DIR,
    alpha: float = _DEFAULT_ALPHA,
    chart_path: str = "results/mcnemar_chart.png",
    csv_per_run: str = "results/mcnemar_per_run.csv",
    csv_combo: str = "results/mcnemar_combo_summary.csv",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
    print(f"Processing {len(files)} monitoring files …")

    records: list[dict] = []
    for path in files:
        res = _run_mcnemar(path)
        if res is None:
            continue
        ticker, period, combo = _parse_stem(path.stem)
        res["ticker"]  = ticker
        res["period"]  = period
        res["combo"]   = combo
        res["file"]    = path.name
        records.append(res)

    if not records:
        print("No records produced.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    n_tests = len(df)
    bonferroni_alpha = alpha / n_tests
    df["p_bonferroni"] = (df["p_value"] * n_tests).clip(upper=1.0).round(6)

    df["verdict"] = "neutral"
    df.loc[(df["p_value"] < alpha) & (df["acc_delta"] > 0), "verdict"] = "adaptive_wins"
    df.loc[(df["p_value"] < alpha) & (df["acc_delta"] < 0), "verdict"] = "static_wins"

    df["verdict_bonf"] = "neutral"
    df.loc[(df["p_bonferroni"] < alpha) & (df["acc_delta"] > 0), "verdict_bonf"] = "adaptive_wins"
    df.loc[(df["p_bonferroni"] < alpha) & (df["acc_delta"] < 0), "verdict_bonf"] = "static_wins"

    df.to_csv(csv_per_run, index=False)
    print(f"Saved: {csv_per_run}  ({len(df)} rows)")

    vc = df["verdict"].value_counts()
    vc_bonf = df["verdict_bonf"].value_counts()
    total = len(df)
    print(f"\nMcNemar results across {total} runs  (α={alpha}):")
    print(f"  Adaptive wins : {vc.get('adaptive_wins', 0):>5}  ({100*vc.get('adaptive_wins',0)/total:.1f}%)")
    print(f"  Static wins   : {vc.get('static_wins',  0):>5}  ({100*vc.get('static_wins', 0)/total:.1f}%)")
    print(f"  Neutral       : {vc.get('neutral',      0):>5}  ({100*vc.get('neutral',     0)/total:.1f}%)")
    print(f"\nAfter Bonferroni correction (α_adj={bonferroni_alpha:.2e}):")
    print(f"  Adaptive wins : {vc_bonf.get('adaptive_wins', 0):>5}  ({100*vc_bonf.get('adaptive_wins',0)/total:.1f}%)")
    print(f"  Static wins   : {vc_bonf.get('static_wins',  0):>5}  ({100*vc_bonf.get('static_wins', 0)/total:.1f}%)")

    combo_summary = (
        df.groupby("combo")
        .agg(
            n_runs=("ticker", "count"),
            pct_adaptive_wins=(
                "verdict",
                lambda x: (x == "adaptive_wins").mean() * 100,
            ),
            pct_static_wins=(
                "verdict",
                lambda x: (x == "static_wins").mean() * 100,
            ),
            mean_acc_delta=("acc_delta", "mean"),
            mean_chi2=("chi2", "mean"),
            median_p=("p_value", "median"),
        )
        .round(3)
        .reset_index()
        .sort_values("pct_adaptive_wins", ascending=False)
    )
    combo_summary.to_csv(csv_combo, index=False)
    print(f"\nTop combos by adaptive-win rate:")
    print(combo_summary.head(10).to_string(index=False))
    print(f"Saved: {csv_combo}")

    _plot_mcnemar(df, chart_path, alpha)

    return df


def _plot_mcnemar(df: pd.DataFrame, path: str, alpha: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        colours = {
            "adaptive_wins": "#1a9641",
            "static_wins":   "#d7191c",
            "neutral":       "#aaaaaa",
        }
        for verdict, grp in df.groupby("verdict"):
            neg_logp = -np.log10(grp["p_value"].clip(lower=1e-10))
            ax.scatter(grp["acc_delta"], neg_logp,
                       c=colours[verdict], s=10, alpha=0.5, label=verdict)
        ax.axhline(-np.log10(alpha), color="grey", ls="--", lw=1,
                   label=f"α={alpha}")
        ax.axvline(0, color="black", ls=":", lw=0.7)
        ax.set_xlabel("Accuracy delta (adaptive − static)")
        ax.set_ylabel("−log₁₀(p-value)")
        ax.set_title("McNemar volcano plot")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        combo_counts = df.groupby(["combo", "verdict"]).size().unstack(fill_value=0)
        top_combos = df["combo"].value_counts().head(12).index
        combo_counts = combo_counts.reindex(top_combos).fillna(0)
        combo_counts = combo_counts.reindex(
            columns=["adaptive_wins", "neutral", "static_wins"], fill_value=0
        )
        combo_counts = combo_counts.div(combo_counts.sum(axis=1), axis=0) * 100

        x = np.arange(len(combo_counts))
        w = 0.25
        bar_colors = ["#1a9641", "#aaaaaa", "#d7191c"]
        for i, (col, c) in enumerate(zip(combo_counts.columns, bar_colors)):
            ax2.bar(x + i * w, combo_counts[col], w, color=c, label=col, alpha=0.85)

        ax2.set_xticks(x + w)
        ax2.set_xticklabels(combo_counts.index, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("% of runs")
        ax2.set_title("McNemar verdict by detector combo  (top 12)")
        ax2.legend(fontsize=8)

        fig.suptitle(
            f"McNemar adaptive vs static  |  {len(df)} runs  |  α={alpha}",
            fontsize=12,
        )
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [chart skipped] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-run McNemar tests on ablation CSVs")
    parser.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA,
                        help="Significance level (default 0.05)")
    args = parser.parse_args()
    run_ablation_mcnemar(alpha=args.alpha)
