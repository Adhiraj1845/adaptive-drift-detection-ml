# src/data_loader.py
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data using yfinance.

    Returns a DataFrame indexed by DatetimeIndex with columns:
    Open, High, Low, Close, Volume

    Notes for dissertation:
    - Ensures deterministic column shape (handles MultiIndex)
    - Removes duplicates
    - Sorts index
    - Drops rows with missing required fields
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # yfinance sometimes returns MultiIndex columns (e.g., ('Close', '^GSPC'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns from yfinance: {missing}. Got={df.columns.tolist()}")

    df = df[REQUIRED_COLS].copy()

    # Clean index and rows
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Drop rows with any missing OHLCV
    df = df.dropna(subset=REQUIRED_COLS)

    return df


def add_features(
    df: pd.DataFrame,
    *,
    dropna: bool = True,
    feature_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add leakage-safe, time-t only features for next-day direction prediction.

    Labeling:
    - Return: r_t = Close_t / Close_{t-1} - 1
    - NextReturn: r_{t+1}
    - Target: 1 if NextReturn > 0 else 0

    Feature set (all computed using information available up to time t):
    - Return (r_t)
    - LogReturn
    - Intraday range features: (High-Low)/Close, (Close-Open)/Open
    - Rolling momentum: sum of returns over windows
    - Rolling volatility: std of returns over windows
    - Rolling mean return
    - Volume change and volume z-score
    - Rolling max drawdown proxy (based on Close rolling max)
    """
    if feature_windows is None:
        feature_windows = [5, 10, 20, 60]

    _validate_ohlcv(df)

    out = df.copy()
    out = out.sort_index()

    # Basic returns
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = np.log(out["Close"]).diff()

    # Next-day return for economic evaluation and label
    out["NextReturn"] = out["Return"].shift(-1)
    out["Target"] = (out["NextReturn"] > 0).astype(int)

    # Price action features (t-only)
    out["HL_Range"] = (out["High"] - out["Low"]) / out["Close"]
    out["CO_Return"] = (out["Close"] - out["Open"]) / out["Open"]

    # Volume features
    out["Vol_Change"] = out["Volume"].pct_change()

    # Rolling features (t-only)
    for w in feature_windows:
        w = int(w)
        if w <= 1:
            continue

        out[f"Ret_Mean_{w}"] = out["Return"].rolling(w).mean()
        out[f"Ret_Vol_{w}"] = out["Return"].rolling(w).std()
        out[f"Mom_Sum_{w}"] = out["Return"].rolling(w).sum()

        # Volume z-score over rolling window
        vol_mean = out["Volume"].rolling(w).mean()
        vol_std = out["Volume"].rolling(w).std()
        out[f"Vol_Z_{w}"] = (out["Volume"] - vol_mean) / (vol_std.replace(0.0, np.nan))

        # Drawdown proxy: (Close / rolling_max_close) - 1
        roll_max = out["Close"].rolling(w).max()
        out[f"DD_{w}"] = (out["Close"] / roll_max) - 1.0

    # Final cleanup
    if dropna:
        # Important: do NOT drop the last row solely because NextReturn is NaN,
        # since streaming slice may already exclude it. But for training it is fine.
        out = out.dropna()

    return out


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _validate_ohlcv(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"OHLCV missing required columns: {missing}. Got={df.columns.tolist()}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")

    if len(df) < 300:
        raise ValueError(f"df too small for reliable rolling features. Rows={len(df)}")
