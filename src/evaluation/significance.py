from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class McNemarResult:
    b_static_correct_adapt_wrong: int
    c_static_wrong_adapt_correct: int
    statistic: float
    pvalue: float


@dataclass(frozen=True)
class OLSResult:
    coef: float
    pvalue: float
    r2: float


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float


def mcnemar_test_from_preds(
    y_true: Iterable[int],
    y_pred_static: Iterable[int],
    y_pred_adapt: Iterable[int],
) -> McNemarResult:
    """McNemar test on paired classifier correctness (equal error rates null)."""
    y_true = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(y_pred_static), dtype=int)
    b = np.asarray(list(y_pred_adapt), dtype=int)

    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("y_true, y_pred_static, y_pred_adapt must have same length")

    static_ok = (a == y_true)
    adapt_ok = (b == y_true)

    both_ok = int(np.sum(static_ok & adapt_ok))
    s_ok_a_bad = int(np.sum(static_ok & (~adapt_ok)))   # b in docs
    s_bad_a_ok = int(np.sum((~static_ok) & adapt_ok))   # c in docs
    both_bad = int(np.sum((~static_ok) & (~adapt_ok)))

    table = [[both_ok, s_ok_a_bad], [s_bad_a_ok, both_bad]]

    res = mcnemar(table, exact=False, correction=True)
    return McNemarResult(
        b_static_correct_adapt_wrong=s_ok_a_bad,
        c_static_wrong_adapt_correct=s_bad_a_ok,
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
    )


def ols_logloss_on_drift(
    daily_df: pd.DataFrame,
    *,
    logloss_col: str = "logloss_adaptive",
    drift_col: str = "drift_event",
) -> OLSResult:
    """Regress logloss on drift indicator: logloss_t = alpha + beta * drift_t + eps_t."""
    df = daily_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    y = df[logloss_col].astype(float).to_numpy()
    x = (df[drift_col].astype(int)).to_numpy()
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit(cov_type="HC1")  # robust SEs
    beta = float(model.params[1])
    pval = float(model.pvalues[1])
    r2 = float(model.rsquared)
    return OLSResult(coef=beta, pvalue=pval, r2=r2)


def _safe_sharpe(x: np.ndarray, eps: float = 1e-12) -> float:
    mu = float(np.mean(x))
    sig = float(np.std(x))
    return float(mu / (sig + eps)) * float(np.sqrt(252.0))


def bootstrap_sharpe_diff_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for Sharpe(b) - Sharpe(a)."""
    rng = np.random.default_rng(seed)
    ra = np.asarray(returns_a, dtype=float)
    rb = np.asarray(returns_b, dtype=float)
    if len(ra) != len(rb):
        raise ValueError("returns_a and returns_b must have same length")

    n = len(ra)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _safe_sharpe(rb[idx]) - _safe_sharpe(ra[idx])

    point = _safe_sharpe(rb) - _safe_sharpe(ra)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return BootstrapCI(point_estimate=float(point), ci_low=lo, ci_high=hi)


def bootstrap_auc_diff_ci(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for AUC(b) - AUC(a) using rank-based (Mann-Whitney) AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(p_a, dtype=float)
    pb = np.asarray(p_b, dtype=float)
    if not (len(y) == len(pa) == len(pb)):
        raise ValueError("y_true, p_a, p_b must have same length")

    def auc_rank(yv: np.ndarray, pv: np.ndarray) -> float:
        # AUC via Mann–Whitney U
        order = np.argsort(pv)
        y_sorted = yv[order]
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(pv) + 1, dtype=float)

        pos = (yv == 1)
        n_pos = int(np.sum(pos))
        n_neg = len(yv) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        rank_sum_pos = float(np.sum(ranks[pos]))
        u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))

    n = len(y)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = auc_rank(y[idx], pb[idx]) - auc_rank(y[idx], pa[idx])

    point = auc_rank(y, pb) - auc_rank(y, pa)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return BootstrapCI(point_estimate=float(point), ci_low=lo, ci_high=hi)