# src/utils/cli.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd


def _enable_colour() -> bool:
    if sys.platform == "win32":
        os.system("")  # enables VT100 processing in Windows Terminal / cmd
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOUR = _enable_colour()

def _c(code: str, text: str) -> str:
    if not _COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _bold(t: str)   -> str: return _c("1", t)
def _cyan(t: str)   -> str: return _c("96", t)
def _green(t: str)  -> str: return _c("92", t)
def _yellow(t: str) -> str: return _c("93", t)
def _red(t: str)    -> str: return _c("91", t)
def _dim(t: str)    -> str: return _c("2", t)


def _hr(char: str = "─", width: int = 54) -> str:
    return _dim(char * width)


def _section(title: str, step: Optional[str] = None) -> None:
    prefix = _dim(f"[{step}] ") if step else ""
    print(f"\n{prefix}{_bold(_cyan(title))}")
    print(_hr())


@dataclass(frozen=True)
class RunConfig:
    # data
    source: str                   # "yahoo", "csv", "fred"
    ticker_or_series: str
    csv_path: Optional[str]
    date_col: Optional[str]
    schema_mode: str              # "auto", "ohlcv", "value"
    close_col: Optional[str]
    open_col: Optional[str]
    high_col: Optional[str]
    low_col: Optional[str]
    volume_col: Optional[str]

    # periods
    data_start: str
    data_end: str
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str

    # adaptation
    retrain_lookback_years: int
    min_retrain_rows: int

    # model
    model_name: str               # "random_forest", "logistic_regression", "gradient_boosting"

    # outputs
    run_tag: str

    # detector toggles (default all on for backwards-compat)
    use_ks: bool = True
    use_psi: bool = True
    use_js: bool = True
    use_prediction_drift: bool = True
    use_page_hinkley: bool = True

    # adaptation mechanism toggles
    use_sgd_online: bool = True          # daily SGD partial_fit updates
    use_scheduled_retrain: bool = True   # periodic batch retrain every 10 days
    use_error_rate_trigger: bool = True  # retrain when static acc drops below 0.45


def _ask(label: str, default: Optional[str] = None, hint: str = "") -> str:
    hint_str = _dim(f"  {hint}") if hint else ""
    if hint_str:
        print(hint_str)
    default_str = _dim(f" [{_green(default)}]") if default is not None else ""
    raw = input(f"  {_cyan('›')} {label}{default_str}: ").strip()
    return raw if raw else (default or "")


def _ask_int(label: str, default: int, hint: str = "") -> int:
    while True:
        raw = _ask(label, str(default), hint)
        try:
            return int(raw)
        except ValueError:
            print(_red(f"  ✗ '{raw}' is not a valid integer. Try again."))


def _ask_date(label: str, default: str) -> str:
    while True:
        raw = _ask(label, default, hint="format: YYYY-MM-DD")
        try:
            pd.Timestamp(raw)
            return raw
        except Exception:
            print(_red(f"  ✗ '{raw}' is not a valid date. Use YYYY-MM-DD."))


def _ask_menu(title: str, options: Sequence[tuple[str, str]], default: int = 1) -> int:
    print()
    for i, (key, desc) in enumerate(options, 1):
        marker = _green("▶") if i == default else " "
        print(f"  {marker} {_bold(str(i))}.  {_cyan(key):<28} {_dim(desc)}")
    while True:
        raw = _ask(f"Select", str(default))
        try:
            n = int(raw)
            if 1 <= n <= len(options):
                return n
        except ValueError:
            pass
        print(_red(f"  ✗ Enter a number between 1 and {len(options)}."))


def _collect_config() -> RunConfig:
    _section("Data Source", "Step 1/4")

    source_options = [
        ("Yahoo Finance",  "stocks, crypto, indices  (e.g. ^GSPC, BTC-USD)"),
        ("FRED",           "economic series  (e.g. SP500, DGS10, UNRATE)"),
        ("CSV file",       "local dataset"),
    ]
    src_idx = _ask_menu("Data source", source_options, default=1)
    source_keys = ["yahoo", "fred", "csv"]
    source = source_keys[src_idx - 1]

    csv_path = date_col = None
    ticker_or_series = ""

    if source == "yahoo":
        ticker_or_series = _ask("Ticker symbol", "^GSPC",
                                hint="e.g. ^GSPC  AAPL  BTC-USD  TSLA  GLD")
        data_start_default = "2010-01-01"
        data_end_default   = "2024-12-31"
    elif source == "fred":
        ticker_or_series = _ask("FRED series ID", "SP500",
                                hint="e.g. SP500  UNRATE  DGS10  FEDFUNDS  CPIAUCSL")
        data_start_default = "2000-01-01"
        data_end_default   = "2024-12-31"
    else:
        csv_path           = _ask("CSV path", "data/raw/mydata.csv")
        date_col           = _ask("Date column name", "Date")
        ticker_or_series   = _ask("Dataset label (used in output filenames)", "custom")
        data_start_default = "2010-01-01"
        data_end_default   = "2024-12-31"

    schema_options = [
        ("auto",  "detect OHLCV or value-only automatically  (recommended)"),
        ("ohlcv", "force full OHLCV feature set"),
        ("value", "single price/value column only"),
    ]
    print()
    schema_idx = _ask_menu("Column schema", schema_options, default=1)
    schema_mode = ["auto", "ohlcv", "value"][schema_idx - 1]

    close_col = open_col = high_col = low_col = volume_col = None
    if schema_mode in {"ohlcv", "value"}:
        close_col = _ask("Close/value column name", "",
                         hint="leave blank to auto-detect")
        close_col = close_col or None
        if schema_mode == "ohlcv":
            open_col   = (_ask("Open column name",   "", hint="blank = auto") or None)
            high_col   = (_ask("High column name",   "", hint="blank = auto") or None)
            low_col    = (_ask("Low column name",    "", hint="blank = auto") or None)
            volume_col = (_ask("Volume column name", "", hint="blank = auto") or None)

    _section("Date Ranges", "Step 2/4")
    print(_dim("  Tip: training end must be before evaluation start to avoid leakage."))

    data_start  = _ask_date("Full data  start", data_start_default)
    data_end    = _ask_date("Full data  end  ", data_end_default)
    print()
    train_start = _ask_date("Training   start", data_start_default)
    train_end   = _ask_date("Training   end  ", "2015-12-31")
    print()
    eval_start  = _ask_date("Evaluation start", "2016-01-01")
    eval_end    = _ask_date("Evaluation end  ", data_end_default)

    _section("Adaptation Settings", "Step 3/4")

    retrain_lookback_years = _ask_int(
        "Lookback window for retraining (years)", 5,
        hint="how many years of history to use when the model retrains")
    min_retrain_rows = _ask_int(
        "Minimum rows required before retraining", 400,
        hint="lower for short/sparse datasets (e.g. 75); higher = more stable retrains")

    _section("Model & Output", "Step 4/4")

    model_options = [
        ("random_forest",        "robust ensemble, handles non-linearity well"),
        ("logistic_regression",  "fast, interpretable linear baseline"),
        ("gradient_boosting",    "high accuracy sequential ensemble"),
    ]
    model_idx  = _ask_menu("Base model", model_options, default=1)
    model_name = ["random_forest", "logistic_regression", "gradient_boosting"][model_idx - 1]

    run_tag_default = (
        f"{ticker_or_series}_{train_start}_{train_end}__{eval_start}_{eval_end}"
    ).replace(":", "")
    run_tag = _ask("Run tag (output filename suffix)", run_tag_default)

    return RunConfig(
        source=source,
        ticker_or_series=ticker_or_series,
        csv_path=csv_path,
        date_col=date_col,
        schema_mode=schema_mode,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        data_start=data_start,
        data_end=data_end,
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        retrain_lookback_years=retrain_lookback_years,
        min_retrain_rows=min_retrain_rows,
        model_name=model_name,
        run_tag=run_tag,
    )


def _print_summary(cfg: RunConfig) -> None:
    _section("Configuration Summary")

    source_label = {
        "yahoo": f"Yahoo Finance  ({cfg.ticker_or_series})",
        "fred":  f"FRED           ({cfg.ticker_or_series})",
        "csv":   f"CSV file       ({cfg.csv_path})",
    }[cfg.source]

    rows = [
        ("Source",       source_label),
        ("Schema",       cfg.schema_mode),
        ("",             ""),
        ("Data window",  f"{cfg.data_start}  →  {cfg.data_end}"),
        ("Training",     f"{cfg.train_start}  →  {cfg.train_end}"),
        ("Evaluation",   f"{cfg.eval_start}  →  {cfg.eval_end}"),
        ("",             ""),
        ("Model",        cfg.model_name),
        ("Lookback",     f"{cfg.retrain_lookback_years} yr   |   min retrain rows: {cfg.min_retrain_rows}"),
        ("Tag",          cfg.run_tag[:60] + ("…" if len(cfg.run_tag) > 60 else "")),
    ]

    for label, value in rows:
        if label == "":
            print()
        else:
            print(f"  {_cyan(f'{label:<14}')}{value}")

    print()
    print(_hr())


def prompt_run_config() -> RunConfig:
    width = 54
    print()
    print(_bold(_cyan("╔" + "═" * (width - 2) + "╗")))
    print(_bold(_cyan("║")) + _bold(f"{'Adaptive Drift Detection Framework':^{width-2}}") + _bold(_cyan("║")))
    print(_bold(_cyan("║")) + _bold(f"{'Run Configuration':^{width-2}}") + _bold(_cyan("║")))
    print(_bold(_cyan("╚" + "═" * (width - 2) + "╝")))

    while True:
        cfg = _collect_config()
        _print_summary(cfg)

        ans = input(f"  {_cyan('›')} {_bold('Proceed?')} {_dim('[Y/n]')}: ").strip().lower()
        if ans in ("", "y", "yes"):
            print()
            return cfg

        print(_yellow("\n  ↺ Restarting setup...\n"))
