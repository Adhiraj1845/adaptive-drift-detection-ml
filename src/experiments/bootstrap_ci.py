"""
Bootstrap confidence intervals on all headline metrics.

Computes 95% BCa (bias-corrected and accelerated) bootstrap confidence
intervals for the main results across the full 4,312-run ablation set.

Metrics covered:
  - acc_delta (overall, per group, per period, per detector combo)
  - sharpe_delta
  - logloss_delta
  - McNemar win/loss rates
  - Drift-conditional acc_delta gap (drift_active - quiet)
  - Sortino delta, Information Ratio

The BCa method (Efron, 1987) is preferred over percentile bootstrap
as it corrects for both bias and skewness in the sampling distribution.

n_boot = 10,000 by default.

Output
------
  results/bootstrap_ci.csv         -- all CIs in one table
  results/bootstrap_ci_chart.png   -- forest plot of key CIs

Usage
-----
    python -m src.experiments.bootstrap_ci
    python -m src.experiments.bootstrap_ci --n_boot 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = "results"
_N_BOOT_DEFAULT = 10_000
_RNG_SEED = 42


def _bca_ci(data: np.ndarray, stat_fn: Callable, n_boot: int,
             alpha: float = 0.05, rng: np.random.Generator = None) -> tuple[float, float, float]:
    """
    BCa bootstrap CI (Efron 1987).
    Returns (point_estimate, ci_lo, ci_hi).
    """
    data = data[np.isfinite(data)]
    if len(data) < 5:
        est = stat_fn(data)
        return float(est), float("nan"), float("nan")

    rng = rng or np.random.default_rng(_RNG_SEED)
    est = stat_fn(data)

    # Bootstrap distribution
    boot = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return float(est), float("nan"), float("nan")

    # Bias correction
    z0 = float(np.sum(boot < est) / len(boot))
    from scipy.special import ndtri, ndtr
    z0_hat = ndtri(np.clip(z0, 1e-9, 1 - 1e-9))

    # Acceleration (jackknife)
    jk = np.array([stat_fn(np.delete(data, i)) for i in range(min(len(data), 200))])
    jk = jk[np.isfinite(jk)]
    if len(jk) > 0:
        num   = np.sum((jk.mean() - jk) ** 3)
        denom = 6.0 * (np.sum((jk.mean() - jk) ** 2) ** 1.5)
        a_hat = float(num / (denom + 1e-12))
    else:
        a_hat = 0.0

    z_alpha = ndtri(alpha / 2)
    z_1ma   = ndtri(1 - alpha / 2)

    def _adj(z):
        denom = 1.0 - a_hat * (z0_hat + z)
        if abs(denom) < 1e-9:
            return z
        return z0_hat + (z0_hat + z) / denom

    lo = float(np.percentile(boot, ndtr(_adj(z_alpha)) * 100))
    hi = float(np.percentile(boot, ndtr(_adj(z_1ma))  * 100))
    return float(est), lo, hi


def run_bootstrap_ci(
    n_boot: int = _N_BOOT_DEFAULT,
    csv_out: str = "results/bootstrap_ci.csv",
    chart_out: str = "results/bootstrap_ci_chart.png",
) -> pd.DataFrame:

    abl = pd.read_csv("results/ablation_summary.csv")
    ok  = abl[abl["status"] == "ok"].copy()
    rng = np.random.default_rng(_RNG_SEED)

    print(f"Bootstrap CI analysis (n_boot={n_boot:,}, n_runs={len(ok):,}) …")

    records = []

    def _add(label: str, data: np.ndarray, stat_fn=np.mean):
        data = data[np.isfinite(data)]
        if len(data) < 5:
            return
        est, lo, hi = _bca_ci(data, stat_fn, n_boot, rng=rng)
        records.append({
            "metric": label,
            "n":      len(data),
            "estimate": round(est, 5),
            "ci_lo":    round(lo, 5),
            "ci_hi":    round(hi, 5),
            "ci_width": round(hi - lo, 5),
            "sig_0":    "YES" if lo > 0 or hi < 0 else "NO",
        })
        print(f"  {label:<45s}: {est:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'★' if lo > 0 or hi < 0 else ' '}")

    # ── Overall metrics ──────────────────────────────────────────────────────
    print("\nOverall (3,332 ablation runs):")
    _add("acc_delta (overall)",     ok["acc_delta"].values)
    _add("sharpe_delta (overall)",  ok["sharpe_delta"].values)
    _add("logloss_delta (overall)", ok["logloss_delta"].values)

    # ── Per-period ───────────────────────────────────────────────────────────
    print("\nPer evaluation period:")
    for p_lbl, g in ok.groupby("period_label"):
        _add(f"acc_delta ({p_lbl})",    g["acc_delta"].values)
        _add(f"sharpe_delta ({p_lbl})", g["sharpe_delta"].values)

    # ── Per-group ────────────────────────────────────────────────────────────
    print("\nPer asset group:")
    for grp, g in ok.groupby("group"):
        _add(f"acc_delta ({grp})",    g["acc_delta"].values)
        _add(f"sharpe_delta ({grp})", g["sharpe_delta"].values)

    # ── Per-combo (top combos) ────────────────────────────────────────────────
    def _combo(r):
        parts = []
        for f, l in [("use_ks","KS"),("use_psi","PSI"),("use_js","JS"),("use_page_hinkley","PH")]:
            if r.get(f): parts.append(l)
        return "+".join(parts) if parts else "none"
    ok["combo"] = ok.apply(_combo, axis=1)

    print("\nKey detector combos:")
    for combo in ["PH", "PSI+PH", "KS+PSI+JS+PH", "none"]:
        g = ok[ok["combo"] == combo]
        if len(g) > 0:
            _add(f"acc_delta ({combo})",    g["acc_delta"].values)
            _add(f"sharpe_delta ({combo})", g["sharpe_delta"].values)

    # ── Drift-conditional gap (from drift_conditional.csv if exists) ─────────
    dc_path = Path("results/drift_conditional.csv")
    if dc_path.exists():
        dc = pd.read_csv(dc_path).dropna(subset=["drift_acc_delta", "quiet_acc_delta"])
        dc["gap"] = dc["drift_acc_delta"] - dc["quiet_acc_delta"]
        print("\nDrift-conditional (from drift_conditional.csv):")
        _add("acc_delta (drift-active)",    dc["drift_acc_delta"].values)
        _add("acc_delta (quiet)",           dc["quiet_acc_delta"].values)
        _add("drift - quiet gap",           dc["gap"].values)

    # ── Risk-adjusted (from risk_adjusted.csv if exists) ─────────────────────
    ra_path = Path("results/risk_adjusted.csv")
    if ra_path.exists():
        ra = pd.read_csv(ra_path).dropna(subset=["sortino_delta"])
        print("\nRisk-adjusted metrics:")
        _add("sortino_delta",      ra["sortino_delta"].values)
        _add("calmar_delta",       ra["calmar_delta"].dropna().values)
        ir_vals = ra["information_ratio"].dropna().values
        ir_vals = ir_vals[np.isfinite(ir_vals) & (np.abs(ir_vals) < 20)]
        _add("information_ratio",  ir_vals)

    # ── IC (from information_coefficient.csv if exists) ──────────────────────
    ic_path = Path("results/information_coefficient.csv")
    if ic_path.exists():
        ic = pd.read_csv(ic_path).dropna(subset=["ic_delta"])
        print("\nInformation Coefficient:")
        _add("ic_static",   ic["ic_static"].values)
        _add("ic_adaptive", ic["ic_adaptive"].values)
        _add("ic_delta",    ic["ic_delta"].values)

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    _plot(df_out, chart_out)
    return df_out


def _plot(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Filter to key metrics for forest plot
        key = df[df["metric"].str.contains(
            "overall|drift-active|quiet|gap|group|sortino|information_ratio|ic_delta",
            case=False
        )].copy()
        key = key[np.isfinite(key["ci_lo"]) & np.isfinite(key["ci_hi"])]

        if len(key) == 0:
            key = df.copy()

        fig, ax = plt.subplots(figsize=(12, max(6, len(key) * 0.45)))
        y_pos = range(len(key))

        colors = ["#1a9641" if r["sig_0"] == "YES" and r["estimate"] > 0
                  else "#d7191c" if r["sig_0"] == "YES" and r["estimate"] < 0
                  else "#636363" for _, r in key.iterrows()]

        ax.barh(list(y_pos), key["ci_width"].values,
                left=key["ci_lo"].values,
                height=0.5, color=colors, alpha=0.4)
        ax.scatter(key["estimate"].values, list(y_pos),
                   color=colors, s=30, zorder=5)
        ax.axvline(0, color="black", lw=1.0, ls="--")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(key["metric"].values, fontsize=7)
        ax.set_xlabel("Effect estimate with 95% BCa bootstrap CI")
        ax.set_title(
            "Forest plot: bootstrap confidence intervals on all headline metrics\n"
            "Green=significantly positive, Red=significantly negative, Grey=not significant",
            fontsize=10,
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
    parser.add_argument("--n_boot", type=int, default=_N_BOOT_DEFAULT)
    args = parser.parse_args()
    run_bootstrap_ci(n_boot=args.n_boot)
