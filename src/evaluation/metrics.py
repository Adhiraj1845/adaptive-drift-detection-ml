from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationSummary:
    accuracy: float
    precision_1: float
    recall_1: float
    f1_1: float
    support_0: int
    support_1: int


@dataclass(frozen=True)
class DriftDetectionSummary:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    mean_detection_delay_days: Optional[float]
    median_detection_delay_days: Optional[float]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def classification_summary_from_daily(
    daily_df: pd.DataFrame,
    *,
    y_col: str = "y_true_next",
    yhat_col: str = "y_pred_adaptive",
) -> ClassificationSummary:
    """Expects columns y_true_next, y_pred_adaptive as written by main.py."""
    y = daily_df[y_col].astype(int).to_numpy()
    yhat = daily_df[yhat_col].astype(int).to_numpy()

    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec1 = _safe_div(tp, tp + fp)
    rec1 = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec1 * rec1, prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    support0 = int(np.sum(y == 0))
    support1 = int(np.sum(y == 1))

    return ClassificationSummary(
        accuracy=float(acc),
        precision_1=float(prec1),
        recall_1=float(rec1),
        f1_1=float(f1),
        support_0=support0,
        support_1=support1,
    )


def _to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    return x.astype(int) != 0


def _build_drift_mask(
    index: pd.DatetimeIndex,
    drift_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.Series:
    mask = pd.Series(False, index=index)
    for (a, b) in drift_windows:
        a = pd.Timestamp(a)
        b = pd.Timestamp(b)
        mask.loc[(mask.index >= a) & (mask.index <= b)] = True
    return mask


def drift_detection_metrics(
    daily_df: pd.DataFrame,
    *,
    drift_windows: List[Tuple[str | pd.Timestamp, str | pd.Timestamp]],
    detected_col: str = "drift_event",
) -> DriftDetectionSummary:
    """Evaluate daily drift flags vs ground-truth drift windows (per-day binary classification)."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    gt = _build_drift_mask(df.index, [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in drift_windows])
    det = _to_bool_series(df[detected_col])

    tp = int(np.sum(gt & det))
    fp = int(np.sum((~gt) & det))
    fn = int(np.sum(gt & (~det)))
    tn = int(np.sum((~gt) & (~det)))

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    far = _safe_div(fp, fp + tn)

    delays = []
    for (start, end) in [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in drift_windows]:
        window_idx = df.index[(df.index >= start) & (df.index <= end)]
        if len(window_idx) == 0:
            continue
        window_det = det.loc[window_idx]
        if window_det.any():
            first_det_date = window_det[window_det].index[0]
            delays.append((first_det_date - start).days)

    mean_delay = float(np.mean(delays)) if delays else None
    median_delay = float(np.median(delays)) if delays else None

    return DriftDetectionSummary(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        false_alarm_rate=float(far),
        mean_detection_delay_days=mean_delay,
        median_detection_delay_days=median_delay,
    )


def episode_summary_from_daily(
    daily_df: pd.DataFrame,
    *,
    drift_level_col: Optional[str] = "drift_level",
    detected_col: str = "drift_event",
) -> Dict[str, float]:
    """Summarise detection episodes from a daily detected flag."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    det = _to_bool_series(df[detected_col])

    episodes = []
    in_ep = False
    start_i = 0
    for i, d in enumerate(det.tolist()):
        if d and not in_ep:
            in_ep = True
            start_i = i
        if (not d) and in_ep:
            in_ep = False
            episodes.append((start_i, i - 1))
    if in_ep:
        episodes.append((start_i, len(det) - 1))

    ep_lens = [(b - a + 1) for a, b in episodes]
    pct_days = float(np.mean(det.astype(int))) if len(det) else 0.0

    out: Dict[str, float] = {
        "episode_count": float(len(episodes)),
        "mean_episode_len": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "median_episode_len": float(np.median(ep_lens)) if ep_lens else 0.0,
        "pct_days_detected": float(pct_days),
    }

    if drift_level_col is not None and drift_level_col in df.columns:
        for lvl in ["low", "moderate", "high", "severe"]:
            out[f"pct_days_level_{lvl}"] = float(np.mean((df[drift_level_col] == lvl).astype(int)))

    return out


def runtime_summary_from_daily(daily_df: pd.DataFrame, *, step_ms_col: str = "step_ms") -> Dict[str, float]:
    if step_ms_col not in daily_df.columns:
        return {"mean_step_ms": 0.0, "p95_step_ms": 0.0, "p99_step_ms": 0.0}

    x = daily_df[step_ms_col].astype(float).to_numpy()
    if len(x) == 0:
        return {"mean_step_ms": 0.0, "p95_step_ms": 0.0, "p99_step_ms": 0.0}

    return {
        "mean_step_ms": float(np.mean(x)),
        "p95_step_ms": float(np.quantile(x, 0.95)),
        "p99_step_ms": float(np.quantile(x, 0.99)),
    }
