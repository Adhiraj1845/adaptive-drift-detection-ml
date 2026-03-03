# src/evaluation/run_report.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def run_evaluation_from_results(
    daily_csv: str,
    equity_csv: str,
    *,
    print_head: bool = True,
) -> None:
    daily = pd.read_csv(daily_csv)
    eq = pd.read_csv(equity_csv)

    if print_head:
        print("Loaded:")
        print(f"  daily:  {daily_csv}  rows={len(daily)}")
        print(f"  equity: {equity_csv} rows={len(eq)}")

    # 1) McNemar test: adaptive vs static classification difference
    y = daily["y_true_next"].astype(int).to_numpy()
    yhat_s = daily["y_pred_static"].astype(int).to_numpy()
    yhat_a = daily["y_pred_adaptive"].astype(int).to_numpy()

    mc = mcnemar_test_from_preds(y, yhat_s, yhat_a)

    # 2) Regression: logloss vs drift indicator
    ols = ols_logloss_on_drift(daily, logloss_col="logloss_adaptive", drift_col="drift_event")

    # 3) Economic significance: bootstrap Sharpe diff on long-only and long-short
    eq_long_s = eq["equity_longonly_static"].astype(float).to_numpy()
    eq_long_a = eq["equity_longonly_adaptive"].astype(float).to_numpy()
    r_long_s = _equity_to_returns(eq_long_s)
    r_long_a = _equity_to_returns(eq_long_a)

    eq_ls_s = eq["equity_longshort_static"].astype(float).to_numpy()
    eq_ls_a = eq["equity_longshort_adaptive"].astype(float).to_numpy()
    r_ls_s = _equity_to_returns(eq_ls_s)
    r_ls_a = _equity_to_returns(eq_ls_a)

    sh_long = bootstrap_sharpe_diff_ci(r_long_s, r_long_a, n_boot=3000, seed=42)
    sh_ls = bootstrap_sharpe_diff_ci(r_ls_s, r_ls_a, n_boot=3000, seed=42)

    # 4) Optional AUC significance if probabilities are present
    auc_line = None
    if "p1_static" in daily.columns and "p1_adaptive" in daily.columns:
        p_s = daily["p1_static"].astype(float).to_numpy()
        p_a = daily["p1_adaptive"].astype(float).to_numpy()
        auc_ci = bootstrap_auc_diff_ci(y, p_s, p_a, n_boot=3000, seed=42)
        auc_line = auc_ci

    print("\n====================")
    print("Statistical Evaluation")
    print("====================\n")

    print("McNemar test (paired classifier comparison)")
    print(f"  b (static correct, adapt wrong): {mc.b_static_correct_adapt_wrong}")
    print(f"  c (static wrong, adapt correct): {mc.c_static_wrong_adapt_correct}")
    print(f"  statistic={mc.statistic:.4f}  p-value={mc.pvalue:.6f}")
    print("  Interpretation: p < 0.05 suggests adaptive and static differ significantly.\n")

    print("OLS regression: logloss_adaptive ~ 1 + drift_event (robust SE)")
    print(f"  beta(drift_event)={ols.coef:.6f}  p-value={ols.pvalue:.6f}  R^2={ols.r2:.4f}")
    print("  Interpretation: beta > 0 and significant means drift days have worse loss.\n")

    print("Bootstrap Sharpe difference (adaptive - static)")
    print(f"  Long-only: point={sh_long.point_estimate:.4f}  CI=[{sh_long.ci_low:.4f}, {sh_long.ci_high:.4f}]")
    print(f"  Long-short: point={sh_ls.point_estimate:.4f}  CI=[{sh_ls.ci_low:.4f}, {sh_ls.ci_high:.4f}]")
    print("  Interpretation: CI excluding 0 suggests significant Sharpe difference.\n")

    if auc_line is not None:
        print("Bootstrap AUC difference (adaptive - static)")
        print(f"  point={auc_line.point_estimate:.4f}  CI=[{auc_line.ci_low:.4f}, {auc_line.ci_high:.4f}]")
        print("  Interpretation: CI excluding 0 suggests significant AUC difference.\n")
    else:
        print("AUC significance skipped (p1_static/p1_adaptive not found in daily CSV).")
        print("If you keep those columns in daily output, you get AUC significance too.\n")


if __name__ == "__main__":
    # Default to your current outputs
    daily_path = "results/daily_monitoring_2016_2020.csv"
    equity_path = "results/equity_curves_2016_2020.csv"
    run_evaluation_from_results(daily_path, equity_path)