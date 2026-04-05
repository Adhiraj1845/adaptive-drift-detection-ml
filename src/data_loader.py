# src/data_loader.py
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class Schema:
    open: Optional[str]
    high: Optional[str]
    low: Optional[str]
    close: Optional[str]
    volume: Optional[str]
    value: Optional[str]


def download_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def download_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web
    df = web.DataReader(series_id, "fred", start=start, end=end)
    df.columns = ["Value"]
    df.index = pd.to_datetime(df.index)
    # Reindex to full business-day range and forward-fill so sparse series
    # (monthly/quarterly) become dense; this prevents massive row loss in
    # add_features after pct_change + dropna.
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bdays).ffill().dropna()
    df.index.name = "Date"
    return df


def infer_schema(df: pd.DataFrame) -> Schema:
    col_map = {c.lower(): c for c in df.columns}
    open_col = col_map.get("open")
    high_col = col_map.get("high")
    low_col = col_map.get("low")
    close_col = col_map.get("close") or col_map.get("adj close") or col_map.get("adj_close")
    volume_col = col_map.get("volume") or col_map.get("vol")

    if close_col and open_col and high_col and low_col:
        return Schema(open=open_col, high=high_col, low=low_col, close=close_col, volume=volume_col, value=None)
    elif close_col:
        return Schema(open=None, high=None, low=None, close=close_col, volume=volume_col, value=None)
    else:
        value_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        return Schema(open=None, high=None, low=None, close=value_col, volume=None, value=value_col)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def load_csv(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    if date_col:
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.set_index(date_col)
    else:
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
    if len(df) < 60:
        raise ValueError(f"df too small for rolling features. Rows={len(df)}")


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b.replace(0.0, np.nan) + eps)


def add_features(
    df: pd.DataFrame,
    *,
    schema: Optional[Schema] = None,
    dropna: bool = True,
    feature_windows: Optional[List[int]] = None,
    clip_z: float = 20.0,
) -> pd.DataFrame:
    if feature_windows is None:
        feature_windows = [5, 10, 20, 60]

    out = df.copy().sort_index()

    # Rename columns to standard names based on schema
    if schema is not None:
        rename = {}
        for src, dst in [
            (schema.open, "Open"), (schema.high, "High"), (schema.low, "Low"),
            (schema.close, "Close"), (schema.volume, "Volume"),
        ]:
            if src and src != dst and src in out.columns:
                rename[src] = dst
        if rename:
            out = out.rename(columns=rename)
        has_ohlcv = bool(schema.open and schema.high and schema.low)
    else:
        has_ohlcv = all(c in out.columns for c in ["Open", "High", "Low", "Volume"])

    if has_ohlcv:
        _validate_ohlcv(out)
        for c in REQUIRED_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    else:
        if "Close" not in out.columns:
            raise KeyError("add_features requires at least a 'Close' column.")
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    # Returns and target
    out["Return"] = out["Close"].pct_change(fill_method=None)
    close_safe = out["Close"].where(out["Close"] > 0, np.nan)
    out["LogReturn"] = np.log(close_safe).diff()
    out["NextReturn"] = out["Return"].shift(-1)
    out["Target"] = (out["NextReturn"] > 0).astype(int)

    # OHLCV-only features
    if has_ohlcv:
        out["HL_Range"] = _safe_div(out["High"] - out["Low"], out["Close"])
        out["CO_Return"] = _safe_div(out["Close"] - out["Open"], out["Open"])
        out["Vol_Change"] = out["Volume"].pct_change(fill_method=None)

    # Rolling features
    for w in feature_windows:
        w = int(w)
        if w <= 1:
            continue
        out[f"Ret_Mean_{w}"] = out["Return"].rolling(w).mean()
        out[f"Ret_Vol_{w}"] = out["Return"].rolling(w).std()
        out[f"Mom_Sum_{w}"] = out["Return"].rolling(w).sum()

        if has_ohlcv:
            vol_mean = out["Volume"].rolling(w).mean()
            vol_std = out["Volume"].rolling(w).std().replace(0.0, np.nan)
            vol_z = (out["Volume"] - vol_mean) / vol_std
            if clip_z is not None:
                vol_z = vol_z.clip(lower=-float(clip_z), upper=float(clip_z))
            out[f"Vol_Z_{w}"] = vol_z

        roll_max = out["Close"].rolling(w).max()
        out[f"DD_{w}"] = _safe_div(out["Close"], roll_max) - 1.0

    # Trend regime features — price relative to moving averages.
    for ma_w in [20, 60]:
        ma  = out["Close"].rolling(ma_w).mean()
        std = out["Close"].rolling(ma_w).std().replace(0.0, np.nan)
        z   = (out["Close"] - ma) / std
        if clip_z is not None:
            z = z.clip(lower=-float(clip_z), upper=float(clip_z))
        out[f"Price_Z_{ma_w}"] = z

    ma_20 = out["Close"].rolling(20).mean()
    ma_60 = out["Close"].rolling(60).mean()
    out["MA_cross_20_60"] = _safe_div(ma_20 - ma_60, ma_60) * 100

    out = out.replace([np.inf, -np.inf], np.nan)

    if dropna:
        out = out.dropna()

    return out
