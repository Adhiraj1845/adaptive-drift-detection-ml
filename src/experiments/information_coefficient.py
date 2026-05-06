"""Information Coefficient (IC) and ICIR: Spearman rank correlation of p1 with next-day returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = "results"
_IC_WINDOW   = 63   # rolling IC window (~1 quarter)


def _ic(signal: np.ndarray, returns: np.ndarray) -> float:
    if len(signal) < 5:
        return float("nan")
    mask = np.isfinite(signal) & np.isfinite(returns)
    if mask.sum() < 5:
        return float("nan")
    corr, _ = sp_stats.spearmanr(signal[mask], returns[mask])
    return float(corr)


def run_ic_analysis(
    results_dir: str = _RESULTS_DIR,
    csv_out:     str = "results/information_coefficient.csv",
    chart_out:   str = "results/ic_charts.png",
) -> pd.DataFrame:

    files = sorted(Path(results_dir).glob("daily_monitoring_*.csv"))
    print(f"Processing {len(files)} files …")

    records = []
    all_ic_s: list[float] = []
    all_ic_a: list[float] = []
    all_ic_drift: list[float] = []
    all_ic_quiet: list[float] = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                usecols=["date", "p1_static", "p1_adaptive",
                         "return_next", "action"],
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "p1_static", "p1_adaptive", "return_next"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            continue

        if len(df) < 50:
            continue

        p_s   = df["p1_static"].values
        p_a   = df["p1_adaptive"].values
        rets  = df["return_next"].values
        drift = (df["action"] != "none").values

        # Full-period IC
        ic_s = _ic(p_s,   rets)
        ic_a = _ic(p_a,   rets)

        # Drift-conditional IC
        ic_drift_a = _ic(p_a[drift],  rets[drift])
        ic_quiet_a = _ic(p_a[~drift], rets[~drift])

        # Rolling IC (quarterly windows) — use p_adaptive
        rolling_ics = []
        for i in range(_IC_WINDOW, len(df)):
            win_p = p_a[i - _IC_WINDOW: i]
            win_r = rets[i - _IC_WINDOW: i]
            rolling_ics.append(_ic(win_p, win_r))

        rolling_ics_arr = np.array([x for x in rolling_ics if not np.isnan(x)])
        icir = float(np.mean(rolling_ics_arr) / (np.std(rolling_ics_arr) + 1e-9)) if len(rolling_ics_arr) > 2 else float("nan")

        # Hit rate: direction accuracy weighted by signal strength
        signal_direction = (p_a >= 0.5).astype(int) * 2 - 1   # +1 or -1
        ret_direction    = (rets  > 0).astype(int) * 2 - 1
        hit_rate_a = float(np.mean(signal_direction == ret_direction))
        signal_direction_s = (p_s >= 0.5).astype(int) * 2 - 1
        hit_rate_s = float(np.mean(signal_direction_s == ret_direction))

        all_ic_s.append(ic_s)
        all_ic_a.append(ic_a)
        if not np.isnan(ic_drift_a):
            all_ic_drift.append(ic_drift_a)
        if not np.isnan(ic_quiet_a):
            all_ic_quiet.append(ic_quiet_a)

        stem = path.stem.replace("daily_monitoring_", "")
        parts = stem.split("_")
        records.append({
            "ticker":       parts[0],
            "period":       parts[1] if len(parts) > 1 else "?",
            "combo":        "_".join(parts[2:]) if len(parts) > 2 else "?",
            "ic_static":    round(ic_s,        4),
            "ic_adaptive":  round(ic_a,        4),
            "ic_delta":     round(ic_a - ic_s, 4),
            "ic_drift":     round(ic_drift_a,  4) if not np.isnan(ic_drift_a) else float("nan"),
            "ic_quiet":     round(ic_quiet_a,  4) if not np.isnan(ic_quiet_a) else float("nan"),
            "icir":         round(icir,         4) if not np.isnan(icir) else float("nan"),
            "hit_rate_static":   round(hit_rate_s, 4),
            "hit_rate_adaptive": round(hit_rate_a, 4),
            "n_days":       len(df),
            "pct_drift":    round(drift.mean() * 100, 1),
        })

    if not records:
        print("No records.")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_out, index=False)

    ok = df_out.dropna(subset=["ic_static", "ic_adaptive"])

    ic_s_arr = np.array(all_ic_s)
    ic_a_arr = np.array(all_ic_a)
    ic_s_arr = ic_s_arr[np.isfinite(ic_s_arr)]
    ic_a_arr = ic_a_arr[np.isfinite(ic_a_arr)]

    print(f"\nInformation Coefficient analysis ({len(ok)} runs):\n")
    print(f"  IC (static):             mean={ic_s_arr.mean():+.4f}  std={ic_s_arr.std():.4f}")
    print(f"  IC (adaptive):           mean={ic_a_arr.mean():+.4f}  std={ic_a_arr.std():.4f}")
    print(f"  IC delta (adapt-static): mean={ok['ic_delta'].mean():+.4f}  "
          f"({(ok['ic_delta']>0).mean()*100:.1f}% positive)")
    print()
    ic_d  = np.array(all_ic_drift); ic_d = ic_d[np.isfinite(ic_d)]
    ic_q  = np.array(all_ic_quiet); ic_q = ic_q[np.isfinite(ic_q)]
    print(f"  IC (adaptive) — drift-active: {ic_d.mean():+.4f}  (n={len(ic_d)})")
    print(f"  IC (adaptive) — quiet days:   {ic_q.mean():+.4f}  (n={len(ic_q)})")
    print()
    icir_vals = ok["icir"].dropna()
    print(f"  ICIR (adaptive rolling):  mean={icir_vals.mean():+.4f}  "
          f"({(icir_vals > 0.5).mean()*100:.1f}% > 0.5 benchmark)")
    print(f"  Hit rate static:    {ok['hit_rate_static'].mean():.4f}")
    print(f"  Hit rate adaptive:  {ok['hit_rate_adaptive'].mean():.4f}")
    print()
    # Industry benchmark comparison
    benchmarks = [(0.02, "Weak"), (0.05, "Useful"), (0.10, "Strong")]
    for thresh, label in benchmarks:
        pct_s = (ic_s_arr > thresh).mean() * 100
        pct_a = (ic_a_arr > thresh).mean() * 100
        print(f"  IC > {thresh:.2f} ({label:6s}): static={pct_s:.1f}%  adaptive={pct_a:.1f}%")
    print()
    print(f"  Grinold & Kahn (1999) benchmarks: IC>0.05='useful', IC>0.10='strong', ICIR>0.50='consistent'")

    t, p = sp_stats.ttest_1samp(ok["ic_delta"].dropna(), 0)
    print(f"\n  t-test IC delta > 0: t={t:.2f}, p={p:.4f}")
    print(f"\nSaved: {csv_out}")
    _plot(ok, all_ic_s, all_ic_a, chart_out)
    return df_out


def _plot(df: pd.DataFrame, ic_s: list, ic_a: list, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        ic_s_clean = [x for x in ic_s if not np.isnan(x)]
        ic_a_clean = [x for x in ic_a if not np.isnan(x)]

        # Panel 1: IC distribution static vs adaptive
        ax = axes[0]
        ax.hist(ic_s_clean, bins=50, alpha=0.6, color="#2c7bb6",
                label=f"Static  μ={np.mean(ic_s_clean):+.4f}", density=True)
        ax.hist(ic_a_clean, bins=50, alpha=0.6, color="#d7191c",
                label=f"Adaptive μ={np.mean(ic_a_clean):+.4f}", density=True)
        ax.axvline(0,    color="black",   lw=0.8, ls="--")
        ax.axvline(0.05, color="#1a9641", lw=1.0, ls=":",
                   label="IC=0.05 (useful)")
        ax.axvline(0.10, color="#756bb1", lw=1.0, ls=":",
                   label="IC=0.10 (strong)")
        ax.set_xlabel("Information Coefficient (Spearman rank corr)")
        ax.set_ylabel("Density")
        ax.set_title("IC distribution: static vs adaptive predictions\n"
                     "(Grinold & Kahn 1999 benchmarks shown)")
        ax.legend(fontsize=8)

        # Panel 2: IC delta scatter by ticker group
        ax2 = axes[1]
        ax2.hist(df["ic_delta"].dropna(), bins=50, color="#756bb1",
                 edgecolor="white", alpha=0.85)
        ax2.axvline(0, color="black", lw=1.0, ls="--")
        ax2.axvline(df["ic_delta"].mean(), color="#d7191c", lw=2, ls="--",
                    label=f"Mean={df['ic_delta'].mean():+.4f}")
        ax2.set_xlabel("IC delta (adaptive − static)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"IC improvement distribution\n"
                      f"{(df['ic_delta']>0).mean()*100:.1f}% of runs show positive IC delta")
        ax2.legend()

        # Panel 3: IC drift-active vs quiet
        ax3 = axes[2]
        ic_drift_clean = df["ic_drift"].dropna().values
        ic_quiet_clean = df["ic_quiet"].dropna().values
        parts = ax3.violinplot([ic_drift_clean, ic_quiet_clean],
                                positions=[1, 2], showmedians=True)
        for pc, col in zip(parts["bodies"], ["#d7191c", "#2c7bb6"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
        ax3.axhline(0,    color="black",   lw=0.8, ls="--")
        ax3.axhline(0.05, color="#1a9641", lw=0.8, ls=":", alpha=0.7)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(["Drift-active days", "Quiet days"])
        ax3.set_ylabel("IC (adaptive)")
        ax3.set_title("Adaptive IC: drift-active vs quiet periods\n"
                      "(higher IC during drift validates conditional effectiveness)")
        ax3.text(1, ic_drift_clean.mean() + 0.005,
                 f"μ={ic_drift_clean.mean():+.4f}", ha="center", fontsize=8, color="#d7191c")
        ax3.text(2, ic_quiet_clean.mean() + 0.005,
                 f"μ={ic_quiet_clean.mean():+.4f}", ha="center", fontsize=8, color="#2c7bb6")

        # Panel 4: ICIR distribution with benchmark line
        ax4 = axes[3]
        icir_clean = df["icir"].dropna().values
        icir_clean = icir_clean[np.isfinite(icir_clean) & (np.abs(icir_clean) < 10)]
        ax4.hist(icir_clean, bins=50, color="#1a9641", edgecolor="white", alpha=0.85)
        ax4.axvline(0,   color="black",   lw=1.0, ls="--")
        ax4.axvline(0.5, color="#fdae61", lw=1.5, ls="--",
                    label="ICIR=0.50 (consistent)")
        ax4.axvline(1.0, color="#d7191c", lw=1.5, ls="--",
                    label="ICIR=1.00 (exceptional)")
        ax4.set_xlabel("IC Information Ratio (ICIR = mean IC / std IC)")
        ax4.set_ylabel("Count")
        ax4.set_title(f"ICIR distribution (adaptive model)\n"
                      f"{(icir_clean > 0.5).mean()*100:.1f}% exceed 0.5 benchmark  "
                      f"(mean={icir_clean.mean():+.3f})")
        ax4.legend(fontsize=8)

        fig.suptitle(
            "Information Coefficient analysis: predictive power of static vs adaptive signals\n"
            "IC and ICIR are standard quant-finance metrics (Grinold & Kahn, 1999)",
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
    run_ic_analysis()
