from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.significance import (
    bootstrap_auc_diff_ci,
    bootstrap_sharpe_diff_ci,
    mcnemar_test_from_preds,
    ols_logloss_on_drift,
)


def _equity_to_returns(eq: np.ndarray) -> np.ndarray:
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 2:
        return np.array([], dtype=float)
    return (eq[1:] / eq[:-1]) - 1.0


def _sig(flag: bool) -> str:
    return "YES *" if flag else "no"


def _sharpe_interp(label: str, sh) -> str:
    ci_lo, ci_hi, pt = sh.ci_low, sh.ci_high, sh.point_estimate
    if ci_lo > 0:
        return f"{label}: SIGNIFICANT improvement — adaptive Sharpe higher (CI entirely above 0)."
    elif ci_hi < 0:
        return f"{label}: SIGNIFICANT degradation — adaptive Sharpe lower (CI entirely below 0)."
    else:
        direction = "higher" if pt > 0 else "lower"
        return (f"{label}: Not significant — CI straddles 0 [{ci_lo:.3f}, {ci_hi:.3f}]. "
                f"Point estimate {direction} for adaptive but insufficient evidence.")


def per_period_breakdown(
    daily: pd.DataFrame,
    eq: pd.DataFrame,
    *,
    period: str = "YE",
) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").set_index("date")

    eq = eq.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").set_index("date")

    if "drift_event" not in daily.columns and "action" in daily.columns:
        daily["drift_event"] = (daily["action"] != "none").astype(int)

    def _sh(r: np.ndarray) -> float:
        mu, sig = float(np.mean(r)), float(np.std(r))
        return float(mu / (sig + 1e-12)) * float(np.sqrt(252.0))

    rows = []
    for label, grp in daily.resample(period):
        if len(grp) == 0:
            continue
        y      = grp["y_true_next"].astype(int).to_numpy()
        yhat_s = grp["y_pred_static"].astype(int).to_numpy()
        yhat_a = grp["y_pred_adaptive"].astype(int).to_numpy()
        acc_s  = float(np.mean(yhat_s == y))
        acc_a  = float(np.mean(yhat_a == y))
        n_drift = int(grp["drift_event"].sum()) if "drift_event" in grp.columns else 0

        eq_slice = eq.loc[grp.index.min():grp.index.max()]
        corr_s_p = corr_a_p = float("nan")
        if len(eq_slice) >= 2:
            r_mk = (eq_slice["equity_market"].astype(float).to_numpy()[1:]
                    / eq_slice["equity_market"].astype(float).to_numpy()[:-1]) - 1.0
            r_s = (eq_slice["equity_longonly_static"].astype(float).to_numpy()[1:]
                   / eq_slice["equity_longonly_static"].astype(float).to_numpy()[:-1]) - 1.0
            r_a = (eq_slice["equity_longonly_adaptive"].astype(float).to_numpy()[1:]
                   / eq_slice["equity_longonly_adaptive"].astype(float).to_numpy()[:-1]) - 1.0
            sharpe_diff = _sh(r_a) - _sh(r_s)
            if len(r_mk) > 1 and r_mk.std() > 0:
                corr_s_p = float(np.corrcoef(r_mk, r_s)[0, 1])
                corr_a_p = float(np.corrcoef(r_mk, r_a)[0, 1])
        else:
            sharpe_diff = float("nan")

        period_label = str(label.year) if period in ("Y", "YE") else f"{label.year}-Q{label.quarter}"
        rows.append({
            "period":         period_label,
            "n_obs":          len(grp),
            "acc_static":     round(acc_s, 4),
            "acc_adaptive":   round(acc_a, 4),
            "n_drift_events": n_drift,
            "sharpe_diff":    round(sharpe_diff, 4),
            "corr_static":    round(corr_s_p, 4),
            "corr_adaptive":  round(corr_a_p, 4),
        })

    out = pd.DataFrame(rows)
    print("\n--- Per-Period Breakdown (yearly) ---")
    print(f"  {'Period':<12} {'N':>6} {'Acc-S':>7} {'Acc-A':>7} {'Drift#':>7} {'Sh-Diff':>9} {'Cor-S':>7} {'Cor-A':>7}")
    print("  " + "-" * 72)
    for _, r in out.iterrows():
        sh = f"{r['sharpe_diff']:+.4f}" if not (isinstance(r["sharpe_diff"], float) and np.isnan(r["sharpe_diff"])) else "   n/a"
        cs = f"{r['corr_static']:.3f}" if not np.isnan(r["corr_static"]) else "  n/a"
        ca = f"{r['corr_adaptive']:.3f}" if not np.isnan(r["corr_adaptive"]) else "  n/a"
        print(f"  {r['period']:<12} {r['n_obs']:>6} {r['acc_static']:>7.4f} "
              f"{r['acc_adaptive']:>7.4f} {r['n_drift_events']:>7} {sh:>9} {cs:>7} {ca:>7}")
    print()
    return out


def run_evaluation_from_results(
    daily_csv: str,
    equity_csv: str,
    *,
    print_head: bool = True,
) -> dict:
    daily = pd.read_csv(daily_csv)
    eq    = pd.read_csv(equity_csv)

    tag = Path(daily_csv).stem.replace("daily_monitoring_", "")

    if print_head:
        print(f"\nLoaded: {daily_csv}  rows={len(daily)}")

    if "drift_event" not in daily.columns and "action" in daily.columns:
        daily["drift_event"] = (daily["action"] != "none").astype(int)

    y      = daily["y_true_next"].astype(int).to_numpy()
    yhat_s = daily["y_pred_static"].astype(int).to_numpy()
    yhat_a = daily["y_pred_adaptive"].astype(int).to_numpy()

    acc_s = float(np.mean(yhat_s == y))
    acc_a = float(np.mean(yhat_a == y))
    drift_events = int(daily["drift_event"].sum()) if "drift_event" in daily.columns else 0

    mc  = mcnemar_test_from_preds(y, yhat_s, yhat_a)
    ols = ols_logloss_on_drift(daily, logloss_col="logloss_adaptive", drift_col="drift_event")

    r_mkt    = _equity_to_returns(eq["equity_market"].astype(float).to_numpy())
    r_long_s = _equity_to_returns(eq["equity_longonly_static"].astype(float).to_numpy())
    r_long_a = _equity_to_returns(eq["equity_longonly_adaptive"].astype(float).to_numpy())
    r_ls_s   = _equity_to_returns(eq["equity_longshort_static"].astype(float).to_numpy())
    r_ls_a   = _equity_to_returns(eq["equity_longshort_adaptive"].astype(float).to_numpy())

    # Market correlations — key requirement: adaptive must exceed static and gap widens
    corr_s = float(np.corrcoef(r_mkt, r_long_s)[0, 1]) if len(r_mkt) > 1 else float("nan")
    corr_a = float(np.corrcoef(r_mkt, r_long_a)[0, 1]) if len(r_mkt) > 1 else float("nan")

    sh_long = bootstrap_sharpe_diff_ci(r_long_s, r_long_a, n_boot=3000, seed=42)
    sh_ls   = bootstrap_sharpe_diff_ci(r_ls_s,   r_ls_a,   n_boot=3000, seed=42)

    auc_ci = None
    if "p1_static" in daily.columns and "p1_adaptive" in daily.columns:
        p_s    = daily["p1_static"].astype(float).to_numpy()
        p_a    = daily["p1_adaptive"].astype(float).to_numpy()
        auc_ci = bootstrap_auc_diff_ci(y, p_s, p_a, n_boot=3000, seed=42)

    # ── Print detailed results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Statistical Evaluation — {tag}")
    print("=" * 60)

    print(f"\n  Observations: {len(daily)}   Drift events flagged: {drift_events}")
    print(f"  Accuracy — static: {acc_s:.3f}   adaptive: {acc_a:.3f}")
    corr_gap = corr_a - corr_s
    corr_flag = "✓" if corr_a > corr_s else "✗"
    print(f"  Market corr — static: {corr_s:.3f}   adaptive: {corr_a:.3f}   gap: {corr_gap:+.3f} {corr_flag}\n")

    print("McNemar test (paired classifier comparison)")
    print(f"  b={mc.b_static_correct_adapt_wrong}  c={mc.c_static_wrong_adapt_correct}"
          f"  statistic={mc.statistic:.4f}  p={mc.pvalue:.4f}")
    if mc.pvalue < 0.05:
        winner = "adaptive" if mc.c_static_wrong_adapt_correct > mc.b_static_correct_adapt_wrong else "static"
        print(f"  → SIGNIFICANT (p<0.05): {winner} model makes fewer errors.\n")
    else:
        print(f"  → Not significant (p={mc.pvalue:.3f}): no reliable difference in error rates.\n")

    print("OLS: logloss_adaptive ~ 1 + drift_event  (HC1 robust SE)")
    print(f"  beta={ols.coef:+.6f}  p={ols.pvalue:.4f}  R²={ols.r2:.4f}")
    if ols.pvalue < 0.05:
        direction = "higher" if ols.coef > 0 else "lower"
        print(f"  → SIGNIFICANT (p={ols.pvalue:.3f}): drift days associated with {direction} log-loss.\n")
    else:
        print(f"  → Not significant (p={ols.pvalue:.3f}): drift does not reliably explain log-loss changes.\n")

    print("Bootstrap Sharpe difference (adaptive − static, 3000 resamples)")
    print(f"  Long-only:  {sh_long.point_estimate:+.4f}  95% CI=[{sh_long.ci_low:.4f}, {sh_long.ci_high:.4f}]")
    print(f"  Long-short: {sh_ls.point_estimate:+.4f}  95% CI=[{sh_ls.ci_low:.4f}, {sh_ls.ci_high:.4f}]")
    print(f"  → {_sharpe_interp('Long-only',  sh_long)}")
    print(f"  → {_sharpe_interp('Long-short', sh_ls)}\n")

    if auc_ci is not None:
        print("Bootstrap AUC difference (adaptive − static)")
        print(f"  {auc_ci.point_estimate:+.4f}  95% CI=[{auc_ci.ci_low:.4f}, {auc_ci.ci_high:.4f}]")
        ci_lo, ci_hi, pt = auc_ci.ci_low, auc_ci.ci_high, auc_ci.point_estimate
        if ci_lo > 0:
            print("  → SIGNIFICANT: adaptive AUC higher (CI entirely above 0).\n")
        elif ci_hi < 0:
            print("  → SIGNIFICANT: adaptive AUC lower (CI entirely below 0).\n")
        else:
            direction = "higher" if pt > 0 else "lower"
            print(f"  → Not significant — CI straddles 0. Point estimate {direction} for adaptive.\n")

    n_tests = 5  # McNemar, OLS, Sharpe-long, Sharpe-LS, AUC
    alpha_bonf = 0.05 / n_tests  # 0.01
    print("Multiple-testing correction (Bonferroni)")
    print(f"  {n_tests} simultaneous tests — FWER-corrected alpha = {alpha_bonf:.4f}  "
          f"(uncorrected alpha = 0.05)")
    mc_bonf  = mc.pvalue  < alpha_bonf
    ols_bonf = ols.pvalue < alpha_bonf
    print(f"  McNemar  p={mc.pvalue:.4f}  corrected-sig={'YES *' if mc_bonf  else 'no'}")
    print(f"  OLS      p={ols.pvalue:.4f}  corrected-sig={'YES *' if ols_bonf else 'no'}")
    print("  Bootstrap CIs are at 95% (not 99%). For FWER-corrected conclusions,")
    print("  bootstrap_*_ci should be re-run with alpha=0.01 (percentile=[0.5, 99.5]).\n")

    # ── Per-period breakdown ──────────────────────────────────────────────────
    try:
        per_period_breakdown(daily, eq, period="YE")
    except Exception as e:
        print(f"  [per-period breakdown skipped] {e}\n")

    return {
        "tag":               tag,
        "n_obs":             len(daily),
        "drift_events":      drift_events,
        "acc_static":        acc_s,
        "acc_adaptive":      acc_a,
        "corr_static":       corr_s,
        "corr_adaptive":     corr_a,
        "mcnemar_p":         mc.pvalue,
        "mcnemar_sig":       mc.pvalue < 0.05,
        "mcnemar_sig_bonf":  mc_bonf,
        "sharpe_long_pt":    sh_long.point_estimate,
        "sharpe_long_lo":    sh_long.ci_low,
        "sharpe_long_hi":    sh_long.ci_high,
        "sharpe_long_sig":   sh_long.ci_low > 0 or sh_long.ci_high < 0,
        "sharpe_ls_pt":      sh_ls.point_estimate,
        "sharpe_ls_lo":      sh_ls.ci_low,
        "sharpe_ls_hi":      sh_ls.ci_high,
        "sharpe_ls_sig":     sh_ls.ci_low > 0 or sh_ls.ci_high < 0,
        "auc_pt":            auc_ci.point_estimate if auc_ci else None,
        "auc_lo":            auc_ci.ci_low if auc_ci else None,
        "auc_hi":            auc_ci.ci_high if auc_ci else None,
        "auc_sig":           (auc_ci.ci_low > 0 or auc_ci.ci_high < 0) if auc_ci else None,
        "ols_beta":          ols.coef,
        "ols_p":             ols.pvalue,
        "ols_sig_bonf":      ols_bonf,
        "alpha_bonferroni":  alpha_bonf,
    }


def scan_and_evaluate_all(results_dir: str = "results") -> None:
    """Auto-discover all run pairs, evaluate each, print comparison table.
    """
    results_path = Path(results_dir)
    daily_files  = sorted(results_path.glob("daily_monitoring_*.csv"))

    if not daily_files:
        print(f"No daily_monitoring_*.csv files found in '{results_dir}'.")
        return

    summaries = []
    for daily_f in daily_files:
        tag       = daily_f.name.replace("daily_monitoring_", "").replace(".csv", "")
        equity_f  = results_path / f"equity_curves_{tag}.csv"
        if not equity_f.exists():
            print(f"\n[skip] No matching equity_curves for tag '{tag}'")
            continue
        try:
            s = run_evaluation_from_results(str(daily_f), str(equity_f), print_head=False)
            summaries.append(s)
        except Exception as e:
            print(f"\n[error] {tag}: {e}")

    if not summaries:
        return

    # ── Summary comparison table ───────────────────────────────────────────────
    col_w = 42
    print("\n" + "=" * 110)
    print("  CROSS-RUN COMPARISON SUMMARY")
    print("=" * 110)

    header = (f"  {'Run':<{col_w}} {'N':>6} {'Drift':>6} "
              f"{'Acc-S':>6} {'Acc-A':>6} "
              f"{'McN-p':>7} {'Sig?':>5} {'Bonf?':>6} "
              f"{'Sh-L':>7} {'Sig?':>5} "
              f"{'Sh-LS':>7} {'Sig?':>5} "
              f"{'AUC-d':>7} {'Sig?':>5}")
    print(header)
    print("-" * 118)

    for s in summaries:
        tag_short = s["tag"][-col_w:] if len(s["tag"]) > col_w else s["tag"]
        auc_pt  = f"{s['auc_pt']:+.3f}" if s["auc_pt"] is not None else "  n/a"
        auc_sig = _sig(s["auc_sig"]) if s["auc_sig"] is not None else "  —"
        bonf_sig = _sig(s.get("mcnemar_sig_bonf", False))
        row = (f"  {tag_short:<{col_w}} {s['n_obs']:>6} {s['drift_events']:>6} "
               f"{s['acc_static']:>6.3f} {s['acc_adaptive']:>6.3f} "
               f"{s['mcnemar_p']:>7.3f} {_sig(s['mcnemar_sig']):>5} {bonf_sig:>6} "
               f"{s['sharpe_long_pt']:>+7.3f} {_sig(s['sharpe_long_sig']):>5} "
               f"{s['sharpe_ls_pt']:>+7.3f} {_sig(s['sharpe_ls_sig']):>5} "
               f"{auc_pt:>7} {auc_sig:>5}")
        print(row)

    print("-" * 118)
    print("  Sig?  = YES * if p<0.05 or CI entirely above/below 0")
    print("  Bonf? = YES * if McNemar p < 0.01 (Bonferroni-corrected, 5 tests)")
    print("  Sh-L  = Sharpe diff long-only (adaptive−static)   Sh-LS = long-short")
    print("  AUC-d = AUC diff (adaptive−static)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    scan_and_evaluate_all(results_dir)
