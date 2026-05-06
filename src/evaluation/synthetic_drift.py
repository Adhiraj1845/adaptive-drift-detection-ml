from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import add_features
from src.model.random_forest_model import RandomForestModel


@dataclass(frozen=True)
class SyntheticDriftConfig:
    # drift windows (inclusive) as offsets in days from stream start
    drift_start_day: int
    drift_end_day: int

    # drift types
    mean_shift: float = 0.0         # applied to chosen feature
    volatility_scale: float = 1.0   # multiply returns in window
    correlation_flip: bool = False  # flip sign of chosen correlated feature proxy

    # which feature to perturb
    feature_name: str = "Return"


@dataclass(frozen=True)
class SyntheticRunResult:
    df: pd.DataFrame
    drift_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Synthetic drift expects a DatetimeIndex.")
    out = out.sort_index()
    return out


def inject_synthetic_drift(
    df: pd.DataFrame,
    *,
    cfg: SyntheticDriftConfig,
    stream_start: pd.Timestamp,
) -> SyntheticRunResult:
    """Inject synthetic drift into an already-featured dataframe (call after add_features)."""
    out = _ensure_datetime_index(df)

    start = pd.Timestamp(stream_start) + pd.Timedelta(days=int(cfg.drift_start_day))
    end = pd.Timestamp(stream_start) + pd.Timedelta(days=int(cfg.drift_end_day))

    mask = (out.index >= start) & (out.index <= end)
    if mask.sum() == 0:
        return SyntheticRunResult(df=out, drift_windows=[(start, end)])

    if cfg.mean_shift != 0.0:
        if cfg.feature_name not in out.columns:
            raise KeyError(f"Feature '{cfg.feature_name}' not in df columns.")
        out.loc[mask, cfg.feature_name] = out.loc[mask, cfg.feature_name].astype(float) + float(cfg.mean_shift)

    if cfg.volatility_scale != 1.0:
        if "Return" in out.columns:
            out.loc[mask, "Return"] = out.loc[mask, "Return"].astype(float) * float(cfg.volatility_scale)
        if "LogReturn" in out.columns:
            out.loc[mask, "LogReturn"] = out.loc[mask, "LogReturn"].astype(float) * float(cfg.volatility_scale)

    if cfg.correlation_flip and "CO_Return" in out.columns:
        out.loc[mask, "CO_Return"] = -out.loc[mask, "CO_Return"].astype(float)

    return SyntheticRunResult(df=out, drift_windows=[(start, end)])


def make_synthetic_experiment(
    base_market_df: pd.DataFrame,
    *,
    cfg: SyntheticDriftConfig,
    train_start: str,
    train_end: str,
    stream_start: str,
    stream_end: str,
) -> SyntheticRunResult:
    """Feature, slice into train/stream, inject drift, and return a combined df."""
    df = _ensure_datetime_index(base_market_df)
    df = add_features(df)

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    stream_start_ts = pd.Timestamp(stream_start)
    stream_end_ts = pd.Timestamp(stream_end)

    df_train = df.loc[train_start_ts:train_end_ts].dropna().copy()
    df_stream = df.loc[stream_start_ts:stream_end_ts].dropna().copy()

    if len(df_train) < 300 or len(df_stream) < 300:
        raise ValueError("Not enough rows in train/stream slices for synthetic experiment.")

    injected = inject_synthetic_drift(df_stream, cfg=cfg, stream_start=stream_start_ts)

    full = pd.concat([df_train, injected.df], axis=0).sort_index()
    return SyntheticRunResult(df=full, drift_windows=injected.drift_windows)


def baseline_train_rf(
    df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    features: List[str],
) -> RandomForestModel:
    train_df = df.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].dropna().copy()
    train_df = train_df.iloc[:-1].copy()  # leakage-safe
    m = RandomForestModel(n_estimators=300, random_state=42)
    m.train(train_df[features], train_df["Target"])
    return m
