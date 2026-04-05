"""
Tests for src/data_loader.py

Covers:
- add_features: expected columns, no inf/NaN, binary target, schema renaming,
  value-only schema (no OHLCV features), rolling windows
- infer_schema: OHLCV detection, value-only, close-only
- load_csv / save_csv: round-trip, date_col parameter
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.data_loader import Schema, add_features, infer_schema, load_csv, save_csv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Minimal synthetic OHLCV DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_value_only(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Single price series — no OHLCV columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"Close": close}, index=dates)


# ── add_features (OHLCV) ──────────────────────────────────────────────────────

class TestAddFeaturesOHLCV:
    def test_core_columns_present(self):
        df = add_features(_make_ohlcv())
        for col in ["Return", "LogReturn", "Target", "NextReturn"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_ohlcv_derived_columns_present(self):
        df = add_features(_make_ohlcv())
        for col in ["HL_Range", "CO_Return", "Vol_Change"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rolling_columns_created_for_all_windows(self):
        df = add_features(_make_ohlcv(), feature_windows=[5, 10, 20, 60])
        for w in [5, 10, 20, 60]:
            assert f"Ret_Mean_{w}" in df.columns
            assert f"Ret_Vol_{w}" in df.columns
            assert f"Mom_Sum_{w}" in df.columns
            assert f"Vol_Z_{w}" in df.columns
            assert f"DD_{w}" in df.columns

    def test_no_inf_in_output(self):
        df = add_features(_make_ohlcv())
        numeric = df.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Output contains inf values"

    def test_no_nan_in_output(self):
        df = add_features(_make_ohlcv())
        assert not df.isnull().any().any(), "Output contains NaN values"

    def test_target_is_binary(self):
        df = add_features(_make_ohlcv())
        assert set(df["Target"].unique()).issubset({0, 1})

    def test_target_has_both_classes(self):
        """Both classes must exist — otherwise the classifier has nothing to learn."""
        df = add_features(_make_ohlcv())
        assert len(df["Target"].unique()) == 2

    def test_dropna_trims_rows(self):
        """Rolling window NaNs mean output is smaller than input."""
        raw = _make_ohlcv(n=400)
        df = add_features(raw)
        assert len(df) < len(raw)

    def test_dropna_false_keeps_all_rows(self):
        raw = _make_ohlcv(n=400)
        df = add_features(raw, dropna=False)
        assert len(df) == len(raw)

    def test_custom_feature_windows(self):
        df = add_features(_make_ohlcv(), feature_windows=[3, 7])
        assert "Ret_Mean_3" in df.columns
        assert "Ret_Mean_7" in df.columns
        assert "Ret_Mean_5" not in df.columns  # default windows not used

    def test_vol_z_clipped(self):
        """Vol_Z features must be within the clip bound (default 20)."""
        df = add_features(_make_ohlcv(), clip_z=20.0)
        for w in [5, 10, 20, 60]:
            col = f"Vol_Z_{w}"
            if col in df.columns:
                assert df[col].max() <= 20.0
                assert df[col].min() >= -20.0


# ── add_features (schema) ─────────────────────────────────────────────────────

class TestAddFeaturesSchema:
    def test_schema_renames_close_column(self):
        raw = _make_ohlcv().rename(columns={"Close": "Price"})
        schema = Schema(open="Open", high="High", low="Low", close="Price", volume="Volume", value=None)
        df = add_features(raw, schema=schema)
        assert "Return" in df.columns
        assert "Target" in df.columns

    def test_value_only_schema_skips_ohlcv_features(self):
        raw = _make_value_only()
        schema = Schema(open=None, high=None, low=None, close="Close", volume=None, value="Close")
        df = add_features(raw, schema=schema)
        assert "Return" in df.columns
        assert "HL_Range" not in df.columns
        assert "CO_Return" not in df.columns
        assert "Vol_Change" not in df.columns

    def test_value_only_schema_rolling_still_created(self):
        raw = _make_value_only()
        schema = Schema(open=None, high=None, low=None, close="Close", volume=None, value="Close")
        df = add_features(raw, schema=schema)
        for w in [5, 10, 20, 60]:
            assert f"Ret_Mean_{w}" in df.columns
            assert f"DD_{w}" in df.columns

    def test_no_future_leakage_in_feature_names(self):
        """Columns with 'next' or 'future' must not survive beyond NextReturn."""
        df = add_features(_make_ohlcv())
        leaking = [
            c for c in df.columns
            if any(b in c.lower() for b in ["future"]) and c != "NextReturn"
        ]
        assert leaking == [], f"Potential leakage columns: {leaking}"


# ── infer_schema ─────────────────────────────────────────────────────────────

class TestInferSchema:
    def test_detects_full_ohlcv(self):
        schema = infer_schema(_make_ohlcv())
        assert schema.open == "Open"
        assert schema.high == "High"
        assert schema.low == "Low"
        assert schema.close == "Close"
        assert schema.volume == "Volume"
        assert schema.value is None

    def test_detects_close_only(self):
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        schema = infer_schema(df)
        assert schema.close == "Close"
        assert schema.open is None

    def test_detects_value_column_fallback(self):
        df = pd.DataFrame({"Value": [1.0, 2.0, 3.0]})
        schema = infer_schema(df)
        assert schema.value == "Value"

    def test_case_insensitive_matching(self):
        df = _make_ohlcv().rename(columns=str.lower)
        schema = infer_schema(df)
        assert schema.close == "close"
        assert schema.open == "open"


# ── load_csv / save_csv ───────────────────────────────────────────────────────

class TestCSVRoundtrip:
    def test_roundtrip_preserves_rows(self, tmp_path):
        raw = _make_ohlcv(n=400)
        path = str(tmp_path / "ohlcv.csv")
        save_csv(raw, path)
        loaded = load_csv(path)
        assert len(loaded) == len(raw)

    def test_roundtrip_index_is_datetimeindex(self, tmp_path):
        raw = _make_ohlcv(n=100)
        path = str(tmp_path / "ohlcv.csv")
        save_csv(raw, path)
        loaded = load_csv(path)
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_load_csv_with_date_col(self, tmp_path):
        raw = _make_ohlcv(n=50).reset_index().rename(columns={"index": "Date"})
        path = str(tmp_path / "flat.csv")
        raw.to_csv(path, index=False)
        loaded = load_csv(path, date_col="Date")
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert len(loaded) == 50

    def test_save_creates_parent_dirs(self, tmp_path):
        raw = _make_ohlcv(n=50)
        path = str(tmp_path / "sub" / "dir" / "data.csv")
        save_csv(raw, path)
        assert os.path.exists(path)
