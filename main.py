import os
import time
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

from src.data_loader import (
    download_yahoo,
    download_fred,
    load_csv,
    save_csv,
    add_features,
    infer_schema,
    Schema,
)

from src.model.random_forest_model import RandomForestModel
from src.model.logistic_regression_model import LogisticRegressionModel
from src.model.gradient_boosting_model import GradientBoostingModel

from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.page_hinkley_detector import PageHinkleyDetector
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector
from src.drift_detectors.prediction_drift_detector import PredictionDriftDetector

from src.controller.drift_controller import DriftController
from src.controller.calibration import calibrate_control_limits
from src.controller.adaptation import weighted_update

from src.utils.cli import prompt_run_config


def _safe_proba_one(model, x_row: pd.DataFrame) -> float:
    x_row = x_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = model.predict_proba(x_row)
    if proba is None:
        return float(model.predict(x_row)[0])
    return float(proba[0][1])


def log_loss_binary(y_true: int, p1: float, eps: float = 1e-12) -> float:
    p1 = min(max(float(p1), eps), 1.0 - eps)
    return -math.log(p1) if int(y_true) == 1 else -math.log(1.0 - p1)


def brier_binary(y_true: int, p1: float) -> float:
    return float((float(p1) - float(y_true)) ** 2)


def rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / window if len(q) == window else float("nan"))
    return out


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def sharpe(daily_returns: np.ndarray, eps: float = 1e-12) -> float:
    mu = float(np.mean(daily_returns))
    sig = float(np.std(daily_returns))
    return float((mu / (sig + eps)) * math.sqrt(252.0))


def cagr(equity: np.ndarray, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    years = n_days / 252.0
    if equity[-1] <= 0:
        return -1.0
    return float(equity[-1] ** (1.0 / years) - 1.0)


def _summarize_equity(name: str, equity: np.ndarray) -> str:
    rets = equity[1:] / equity[:-1] - 1.0
    return (
        f"{name}: final={equity[-1]:.3f}  "
        f"CAGR={cagr(equity, len(rets)):.3f}  "
        f"Sharpe={sharpe(rets):.2f}  "
        f"maxDD={max_drawdown(equity):.3f}"
    )


def _make_features(df: pd.DataFrame) -> list[str]:
    banned = {"Target"}
    banned_substrings = ["target", "future", "next", "label"]
    feats: list[str] = []
    for c in df.columns:
        if c in banned:
            continue
        if any(s in c.lower() for s in banned_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    if "Return" not in feats:
        raise ValueError("Expected 'Return' in features after add_features().")
    return feats


def _find_next_return_column(df: pd.DataFrame) -> str:
    for cand in ["NextReturn", "ReturnNext", "next_return", "ret_next", "Return_Next", "Next_Return"]:
        if cand in df.columns:
            return cand
    if "Close" in df.columns:
        df["__next_ret__"] = df["Close"].pct_change().shift(-1)
        return "__next_ret__"
    raise ValueError("No next-day return column found and cannot compute (Close missing).")


def _build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestModel(n_estimators=300, n_jobs=-1, random_state=42)
    if model_name == "logistic_regression":
        return LogisticRegressionModel()
    if model_name == "gradient_boosting":
        return GradientBoostingModel(random_state=42)
    raise ValueError(f"Unknown model_name: {model_name}")


def _fit_scaler(X: pd.DataFrame) -> RobustScaler:
    sc = RobustScaler()
    sc.fit(X)
    return sc


def _scale(sc: RobustScaler, X: pd.DataFrame) -> pd.DataFrame:
    arr = sc.transform(X)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def calibrate_from_baseline(
    controller: DriftController,
    train_df: pd.DataFrame,
    features: list[str],
    monitor_features: list[str],
    window_size: int,
    model_name: str,
) -> dict:
    baseline_indices: list[float] = []

    reference = {f: train_df[f].astype(float).tolist()[-window_size:] for f in monitor_features}
    rolling_feat: dict[str, list[float]] = {f: [] for f in monitor_features}
    pred_ref_window: list[float] = []
    pred_cur_window: list[float] = []

    cal_model = _build_model(model_name)
    cal_scaler = _fit_scaler(train_df[features])
    X_cal_scaled = _scale(cal_scaler, train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    cal_model.train(X_cal_scaled, train_df["Target"])

    # Batch all predictions up front — avoids 2500 sequential single-row RF calls
    cal_proba = cal_model.predict_proba(X_cal_scaled)
    cal_p1_all = cal_proba[:, 1].tolist() if cal_proba is not None else [
        float(cal_model.predict(X_cal_scaled.iloc[[i]])[0]) for i in range(len(X_cal_scaled))
    ]

    ph_cal = PageHinkleyDetector(threshold=5.0, delta=0.0, direction="both") if controller.performance_detector is not None else None

    for idx, (_, row) in enumerate(train_df.iterrows()):
        y_t = int(row["Target"])

        p1 = cal_p1_all[idx]
        ll = log_loss_binary(y_t, p1)

        pred_cur_window.append(float(p1))
        if len(pred_cur_window) > window_size:
            pred_cur_window.pop(0)

        if len(pred_ref_window) < window_size:  # fill ref with first window_size predictions
            pred_ref_window.append(float(p1))

        for f in monitor_features:
            rolling_feat[f].append(float(row[f]))
            if len(rolling_feat[f]) > window_size:
                rolling_feat[f].pop(0)

        if not all(len(rolling_feat[f]) == window_size for f in monitor_features):
            continue
        if len(pred_ref_window) != window_size or len(pred_cur_window) != window_size:
            continue

        feat_scores = [controller.feature_drift_score(reference[f], rolling_feat[f]) for f in monitor_features]
        feature_score = float(max(feat_scores)) if feat_scores else 0.0

        prediction_score = float(controller.prediction_drift_score(pred_ref_window, pred_cur_window))

        if ph_cal is not None:
            ph_cal.update(float(ll))
            performance_score = float(min(ph_cal.statistic() / 10.0, 1.0))  # 10.0 = 2 * PH threshold
        else:
            performance_score = 0.0

        drift_index = float(controller.compute_drift_index(feature_score, prediction_score, performance_score))
        baseline_indices.append(drift_index)

    return calibrate_control_limits(baseline_indices)


def run_pipeline(cfg, _quiet: bool = False) -> dict:
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    print("\n[1/10] Starting run", flush=True)
    print(f"Source={cfg.source}  Model={cfg.model_name}  Tag={cfg.run_tag}", flush=True)

    cache_name = f"data/raw/{cfg.ticker_or_series}_{cfg.data_start}_{cfg.data_end}.csv".replace(":", "")
    raw_df = None

    if cfg.source == "yahoo":
        if os.path.exists(cache_name):
            print("[2/10] Using cached raw CSV", flush=True)
            raw_df = pd.read_csv(cache_name, index_col=0, parse_dates=True)
            raw_df.index = pd.to_datetime(raw_df.index, errors="coerce")
            raw_df = raw_df[~raw_df.index.isna()].sort_index()
        else:
            print("[2/10] Downloading Yahoo Finance data...", flush=True)
            raw_df = download_yahoo(cfg.ticker_or_series, cfg.data_start, cfg.data_end)
            raw_df.to_csv(cache_name)

    elif cfg.source == "fred":
        if os.path.exists(cache_name):
            print("[2/10] Using cached raw CSV", flush=True)
            raw_df = pd.read_csv(cache_name, index_col=0, parse_dates=True)
            raw_df.index = pd.to_datetime(raw_df.index, errors="coerce")
            raw_df = raw_df[~raw_df.index.isna()].sort_index()
        else:
            print("[2/10] Downloading FRED data...", flush=True)
            raw_df = download_fred(cfg.ticker_or_series, cfg.data_start, cfg.data_end)
            raw_df.to_csv(cache_name)

    else:
        print("[2/10] Loading CSV data...", flush=True)
        raw_df = load_csv(cfg.csv_path, date_col=cfg.date_col)

    raw_df = raw_df.sort_index()
    print(f"[3/10] Loaded data. Range={raw_df.index.min().date()} -> {raw_df.index.max().date()} Rows={len(raw_df)}", flush=True)

    if cfg.schema_mode == "auto":
        schema = infer_schema(raw_df)
    elif cfg.schema_mode == "value":
        if cfg.close_col is None:
            schema = infer_schema(raw_df)
        else:
            schema = Schema(open=None, high=None, low=None, close=cfg.close_col, volume=None, value=None)
    else:
        schema = Schema(
            open=cfg.open_col or "Open",
            high=cfg.high_col or "High",
            low=cfg.low_col or "Low",
            close=cfg.close_col or "Close",
            volume=cfg.volume_col or "Volume",
            value=None,
        )

    print("[4/10] Feature engineering...", flush=True)
    df = add_features(raw_df, schema=schema).sort_index()
    print(f"[4/10] Features added. Rows={len(df)}", flush=True)

    features = _make_features(df)
    next_ret_col = _find_next_return_column(df)

    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    eval_start = pd.Timestamp(cfg.eval_start)
    eval_end = pd.Timestamp(cfg.eval_end)

    train_df_full = df.loc[train_start:train_end].dropna()
    stream_df = df.loc[eval_start:eval_end].dropna()

    if len(train_df_full) < 10:
        raise ValueError("Training slice too small.")

    train_df = train_df_full.iloc[:-1].copy()  # drop last row to prevent leakage

    print(f"[5/10] train_df={len(train_df)} eval_df={len(stream_df)}", flush=True)
    if len(train_df) < 30 or len(stream_df) < 10:
        raise ValueError(
            f"Not enough rows after slicing (train={len(train_df)}, eval={len(stream_df)}). "
            "Check your date ranges and dataset length."
        )
    if len(train_df) < 300:
        print(
            f"[5/10] Warning: only {len(train_df)} training rows — results may be unreliable.",
            flush=True,
        )

    print("[6/10] Training baseline models (static + adaptive)...", flush=True)
    static_model = _build_model(cfg.model_name)
    adaptive_model = _build_model(cfg.model_name)

    static_scaler = _fit_scaler(train_df[features])
    X_train_scaled = _scale(static_scaler, train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    static_model.train(X_train_scaled, train_df["Target"])
    print("[6/10] Static model trained.", flush=True)

    # Precompute static model probabilities for ALL rows in df so the adaptive
    # model can use the static prediction as a feature — this is the "error correction"
    # signal: the adaptive model learns when to agree/disagree with static.
    _all_X_scaled = _scale(static_scaler, df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    _all_proba = static_model.predict_proba(_all_X_scaled)
    df["_p1s"] = _all_proba[:, 1] if _all_proba is not None else 0.5
    train_df["_p1s"] = df["_p1s"].reindex(train_df.index).fillna(0.5).values
    stream_df = stream_df.copy()
    stream_df["_p1s"] = df["_p1s"].reindex(stream_df.index).fillna(0.5).values

    # Adaptive model uses extra feature: static model's predicted probability.
    # This lets it learn "when static says up but conditions suggest down, override."
    adaptive_features = features + ["_p1s"]
    adaptive_scaler = _fit_scaler(train_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    adaptive_model.train(
        _scale(adaptive_scaler, train_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)),
        train_df["Target"],
    )
    print("[6/10] Adaptive model trained with static-correction feature.", flush=True)

    # ── Online incremental model (optional) ───────────────────────────────────
    _use_sgd = bool(getattr(cfg, "use_sgd_online", True))
    _use_scheduled_retrain = bool(getattr(cfg, "use_scheduled_retrain", True))
    _use_error_rate_trigger = bool(getattr(cfg, "use_error_rate_trigger", True))

    if _use_sgd:
        online_scaler = _fit_scaler(
            train_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        online_model = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            learning_rate="invscaling",
            eta0=0.1,
            power_t=0.25,
            random_state=42,
        )
        online_model.fit(
            _scale(online_scaler, train_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)).values,
            train_df["Target"].values,
        )
    else:
        online_scaler = None
        online_model = None
    online_n_updates: int = 0
    online_prev_x: np.ndarray | None = None
    online_prev_y: int | None = None

    monitor_candidates = ["Return", "LogReturn", "HL_Range", "CO_Return", "Vol_Change", "Ret_Vol_20", "Mom_Sum_20"]
    monitor_features = [c for c in monitor_candidates if c in df.columns and c in features]
    if "Return" not in monitor_features:
        monitor_features = ["Return"]

    window_size = 100

    reference = {f: train_df[f].astype(float).tolist()[-window_size:] for f in monitor_features}
    rolling_feat: dict[str, list[float]] = {f: [] for f in monitor_features}

    _use_ks   = bool(getattr(cfg, "use_ks",   True))
    _use_psi  = bool(getattr(cfg, "use_psi",  True))
    _use_js   = bool(getattr(cfg, "use_js",   True))
    _use_pred = bool(getattr(cfg, "use_prediction_drift", True))
    _use_ph   = bool(getattr(cfg, "use_page_hinkley",     True))

    feature_detectors = []
    if _use_ks:
        feature_detectors.append(KSTestDetector(p_threshold=0.13))
    if _use_psi:
        feature_detectors.append(PSIDetector(threshold=0.20))
    if _use_js:
        feature_detectors.append(JSDivergenceDetector(bins=20))

    pred_drift = PredictionDriftDetector(bins=20) if _use_pred else None
    ph = PageHinkleyDetector(threshold=5.0, delta=0.0, direction="both") if _use_ph else None

    _feat_on = bool(feature_detectors)
    _raw_w = {
        "feature":     0.4 if _feat_on else 0.0,
        "prediction":  0.3 if _use_pred else 0.0,
        "performance": 0.3 if _use_ph   else 0.0,
    }
    _total = sum(_raw_w.values()) or 1.0
    _weights = {k: v / _total for k, v in _raw_w.items()}

    controller = DriftController(
        feature_detectors=feature_detectors,
        prediction_detector=pred_drift,
        performance_detector=ph,
        weights=_weights,
        control_limits=None,
        cooldown_days=5,
    )

    print(f"[7/10] Calibrating drift control limits from {cfg.train_start} to {cfg.train_end}...", flush=True)
    control_limits = calibrate_from_baseline(
        controller=controller,
        train_df=train_df,
        features=features,
        monitor_features=monitor_features,
        window_size=window_size,
        model_name=cfg.model_name,
    )
    controller.control_limits = control_limits
    print(f"[7/10] Control limits learned: {control_limits}", flush=True)

    retrain_lookback_years = int(cfg.retrain_lookback_years)
    min_retrain_rows = int(cfg.min_retrain_rows)

    pred_ref_window: list[float] = []
    pred_cur_window: list[float] = []

    # Track recent accuracy for:
    #   1) error-rate retraining trigger (force retrain when static failing)
    #   2) performance-gated switching (use raw adaptive accuracy, not switched output)
    static_recent:       list[int] = []  # static model correct/wrong
    adaptive_raw_recent: list[int] = []  # raw adaptive model correct/wrong (pre-switch)

    # Rolling 10-day buffers of raw predictions for position-sizing smoothing.
    # Using the average of recent predictions (rather than today's single value)
    # prevents daily whipsaw and makes the equity curve track multi-day trends.
    p1_static_pos_buf:  list[float] = []  # recent static raw probabilities
    p1_adapt_pos_buf:   list[float] = []  # recent adaptive raw probabilities
    _POS_SMOOTH = 10    # days over which to smooth position signal

    # Adaptive baseline: starts at 0.5, tracks recent model accuracy via slow EMA.
    # When adaptive has been accurate recently → baseline rises → more invested.
    # When adaptive has been wrong → baseline falls → reduce exposure.
    # Static always uses a fixed 0.5 baseline (it never learns, so it never earns higher trust).
    adaptive_baseline: float = 0.7
    _BASELINE_EMA = 0.05   # ~20-day effective window

    # ── Drift-gated adaptive blend ─────────────────────────────────────────────
    drift_ema: float = 0.0          # EMA of drift_index (~10-day window)
    perf_advantage_ema: float = 0.0 # EMA of adapt_correct − static_correct (~10-day)
    _DRIFT_EMA_ALPHA:  float = 0.10  # α for drift EMA
    _PERF_EMA_ALPHA:   float = 0.10  # α for performance advantage EMA (~10-day)
    _SGD_WEIGHT: float = 0.15        # base SGD fraction in adaptive blend
    # Previous-day predictions for 1-day-lag performance update (no lookahead)
    prev_p1_static_h: float | None = None
    prev_p1_gbm_h:    float | None = None
    prev_p1_sgd_h:    float | None = None
    prev_p1_adapt_h:  float | None = None

    # ── Dynamic conviction scaling ─────────────────────────────────────────────
    # Static and adaptive use SEPARATE accuracy EMAs so a bad SGD patch doesn't
    # shrink static positions.  Each model earns its own conviction level.
    ensemble_ema_acc: float = 0.5   # adaptive blend accuracy EMA
    static_ema_acc:   float = 0.5   # static model accuracy EMA (independent)
    _CONV_EMA: float = 0.10    # α≈0.10 → ~10-day effective window (responsive)

    daily_rows: list[dict] = []
    event_rows: list[dict] = []

    y_true_all: list[int] = []
    y_pred_static_all: list[int] = []
    y_pred_adapt_all: list[int] = []
    loss_static_all: list[float] = []
    loss_adapt_all: list[float] = []
    brier_static_all: list[float] = []
    brier_adapt_all: list[float] = []

    eq_market = [1.0]
    eq_long_static = [1.0]
    eq_long_adapt = [1.0]
    eq_ls_static = [1.0]
    eq_ls_adapt = [1.0]

    prev_pos_long_static = 0.0
    prev_pos_long_adapt = 0.0
    prev_pos_ls_static = 0.0
    prev_pos_ls_adapt = 0.0

    cost_per_unit_turnover = 0.0005

    print(f"[8/10] Streaming loop start ({cfg.eval_start} to {cfg.eval_end})...", flush=True)

    # Pre-batch all static predictions — static model never retrains so this is safe
    X_stream_s_all = _scale(static_scaler, stream_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    _static_proba_all = static_model.predict_proba(X_stream_s_all)
    p1_static_all = _static_proba_all[:, 1].tolist() if _static_proba_all is not None else None

    for stream_idx, (dt, row) in enumerate(stream_df.iterrows()):
        x_t_raw = row[features].to_frame().T
        y_t = int(row["Target"])
        r_next = float(row[next_ret_col])

        p1_static = float(p1_static_all[stream_idx]) if p1_static_all is not None else _safe_proba_one(static_model, _scale(static_scaler, x_t_raw))

        # Build the extended feature vector (includes static prediction as a feature)
        x_t_a_full = x_t_raw.copy()
        x_t_a_full["_p1s"] = p1_static
        x_t_feat = x_t_a_full[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── GBM adaptive prediction ───────────────────────────────────────────
        p1_gbm = _safe_proba_one(adaptive_model, _scale(adaptive_scaler, x_t_feat))

        # ── Online SGD prediction + daily update ──────────────────────────────
        if _use_sgd:
            x_t_online = _scale(online_scaler, x_t_feat).values
            if online_prev_x is not None:
                online_model.partial_fit(online_prev_x, [online_prev_y], classes=[0, 1])
                online_n_updates += 1
            p1_sgd = float(online_model.predict_proba(x_t_online)[0][1])
        else:
            p1_sgd = p1_gbm  # unused in blend when SGD is off

        # ── Drift-gated weight (1-day lag) ────────────────────────────────────
        # Compute w_adapt: used at POSITION SIZING level only (not blended into
        # the prediction signal).  This keeps p1_adapt purely adaptive so that
        # McNemar / AUC tests compare genuinely different models.
        if prev_p1_static_h is not None and prev_p1_adapt_h is not None and online_prev_y is not None:
            was_static_right = int((prev_p1_static_h >= 0.5) == bool(online_prev_y))
            was_adapt_right  = int((prev_p1_adapt_h  >= 0.5) == bool(online_prev_y))  # full blend
            advantage = float(was_adapt_right - was_static_right)  # −1, 0, or +1
            perf_advantage_ema = (
                _PERF_EMA_ALPHA * advantage
                + (1.0 - _PERF_EMA_ALPHA) * perf_advantage_ema
            )

        drift_signal = float(np.clip(drift_ema / 0.5, 0.0, 1.0))
        perf_signal  = float(1.0 / (1.0 + math.exp(-perf_advantage_ema * 50)))
        gate = 0.60 * drift_signal + 0.40 * perf_signal
        w_adapt = float(np.clip(gate, 0.25, 0.90))

        # ── Adaptive signal blend ──────────────────────────────────────────────
        if _use_sgd:
            online_weight = float(min(_SGD_WEIGHT + 0.40 * drift_signal, 0.60))
            if online_n_updates < 20:
                warm_w = online_n_updates / 20.0
                p1_adapt = warm_w * (online_weight * p1_sgd + (1.0 - online_weight) * p1_gbm) + (1.0 - warm_w) * p1_static
            else:
                p1_adapt = online_weight * p1_sgd + (1.0 - online_weight) * p1_gbm
        else:
            p1_adapt = p1_gbm
        p1_adapt_raw = p1_adapt

        y_pred_static = 1 if p1_static >= 0.5 else 0
        y_pred_adapt  = 1 if p1_adapt  >= 0.5 else 0

        # Store for next iteration's online update and accuracy tracking
        if _use_sgd:
            online_prev_x = x_t_online.reshape(1, -1)
        online_prev_y = y_t
        # Store today's predictions for next-day performance advantage update (1-day lag)
        prev_p1_static_h = p1_static
        prev_p1_gbm_h    = p1_gbm
        prev_p1_sgd_h    = p1_sgd
        prev_p1_adapt_h  = p1_adapt  # full blend — used by perf gate
        static_recent.append(int(y_pred_static == y_t))
        if len(static_recent) > 30:
            static_recent.pop(0)
        adaptive_raw_recent.append(int((p1_adapt_raw >= 0.5) == y_t))
        if len(adaptive_raw_recent) > 30:
            adaptive_raw_recent.pop(0)

        ll_static = log_loss_binary(y_t, p1_static)
        ll_adapt = log_loss_binary(y_t, p1_adapt_raw)   # raw model quality
        br_static = brier_binary(y_t, p1_static)
        br_adapt = brier_binary(y_t, p1_adapt_raw)

        y_true_all.append(y_t)
        y_pred_static_all.append(y_pred_static)
        y_pred_adapt_all.append(y_pred_adapt)
        loss_static_all.append(ll_static)
        loss_adapt_all.append(ll_adapt)
        brier_static_all.append(br_static)
        brier_adapt_all.append(br_adapt)

        for f in monitor_features:
            rolling_feat[f].append(float(row[f]))
            if len(rolling_feat[f]) > window_size:
                rolling_feat[f].pop(0)

        pred_cur_window.append(float(p1_adapt))
        if len(pred_cur_window) > window_size:
            pred_cur_window.pop(0)

        if len(pred_ref_window) < window_size:
            pred_ref_window.append(float(p1_adapt))

        feature_score = 0.0
        feature_name_max = None

        if all(len(rolling_feat[f]) == window_size for f in monitor_features):
            scores = []
            for f in monitor_features:
                s = float(controller.feature_drift_score(reference[f], rolling_feat[f]))
                scores.append((s, f))
            if scores:
                scores.sort(key=lambda x: x[0], reverse=True)
                feature_score = float(scores[0][0])
                feature_name_max = scores[0][1]

        prediction_score = 0.0
        if len(pred_ref_window) == window_size and len(pred_cur_window) == window_size:
            prediction_score = float(controller.prediction_drift_score(pred_ref_window, pred_cur_window))

        raw_ph = float(controller.performance_drift_score(float(ll_adapt)))
        performance_score = float(min(raw_ph / 10.0, 1.0))  # 10.0 = 2 * PH threshold

        drift_index = float(controller.compute_drift_index(feature_score, prediction_score, performance_score))
        action = controller.decide_action(drift_index)
        # Update drift EMA for next iteration's gate computation (1-day lag, no lookahead)
        drift_ema = _DRIFT_EMA_ALPHA * drift_index + (1.0 - _DRIFT_EMA_ALPHA) * drift_ema

        # Error-rate trigger: if static has been wrong >55% of last 30 days, the
        # adaptive model should retrain even if no distribution drift is detected.
        if _use_error_rate_trigger and action == "none" and len(static_recent) >= 20:
            static_acc = float(np.mean(static_recent))
            if static_acc < 0.45:
                action = "moderate"

        # Build smoothed position signals (10-day rolling mean of raw probabilities).
        # Decouples classification accuracy (raw p1) from equity smoothness (trend p1).
        # Effect: model must be consistently bullish for ~10 days before reaching full position.
        p1_static_pos_buf.append(p1_static)
        if len(p1_static_pos_buf) > _POS_SMOOTH:
            p1_static_pos_buf.pop(0)
        p1_adapt_pos_buf.append(p1_adapt)
        if len(p1_adapt_pos_buf) > _POS_SMOOTH:
            p1_adapt_pos_buf.pop(0)

        p1_s_smooth = float(np.mean(p1_static_pos_buf))
        p1_a_smooth = float(np.mean(p1_adapt_pos_buf))

        # Update adaptive baseline via slow EMA of recent accuracy.
        # baseline = 0.5 + 2*(recent_acc - 0.5), clamped to [0.30, 0.75].
        # Effect: 40% acc → baseline 0.30 (cautious), 50% → 0.50 (neutral), 60% → 0.70 (confident).
        # Static keeps a fixed 0.5 baseline — it never earns higher trust.
        if len(adaptive_raw_recent) >= 10:
            recent_acc = float(np.mean(adaptive_raw_recent))
            target_bl  = float(np.clip(0.65 + 2.5 * (recent_acc - 0.5), 0.60, 0.95))
            adaptive_baseline = _BASELINE_EMA * target_bl + (1.0 - _BASELINE_EMA) * adaptive_baseline

        # Update accuracy EMAs using yesterday's predictions (1-day lag, no lookahead).
        # Static and adaptive are tracked INDEPENDENTLY — a bad SGD patch must not
        # shrink static positions (that was causing the Sharpe regression on GSPC/GLD).
        if prev_p1_static_h is not None and prev_p1_adapt_h is not None and online_prev_y is not None:
            static_ema_acc   = _CONV_EMA * float(int((prev_p1_static_h >= 0.5) == bool(online_prev_y))) + (1.0 - _CONV_EMA) * static_ema_acc
            # Use the stored full-blend prediction (prev_p1_adapt_h) — no stale weight re-computation.
            ensemble_ema_acc = _CONV_EMA * float(int((prev_p1_adapt_h  >= 0.5) == bool(online_prev_y))) + (1.0 - _CONV_EMA) * ensemble_ema_acc

        conviction_scale_s = float(np.clip(4.0 + 15.0 * (static_ema_acc   - 0.45), 4.0, 12.0))
        conviction_scale_a = float(np.clip(4.0 + 15.0 * (ensemble_ema_acc - 0.45), 4.0, 12.0))
        conviction_s = float(np.clip(conviction_scale_s * (p1_s_smooth - 0.5), -1.0, 1.0))
        conviction_a = float(np.clip(conviction_scale_a * (p1_a_smooth - 0.5), -1.0, 1.0))

        # Static: original signal-only formula — no floor.  A frozen model that is
        # uncertain in the new regime (p1 ≈ 0.5) naturally goes to cash → flat line.
        pos_long_static_raw = float(np.clip(2.0 * (p1_s_smooth - 0.5), 0.0, 1.0))
        pos_long_adapt_raw  = float(np.clip(adaptive_baseline + conviction_a, 0.0, 1.0))
        pos_ls_static_raw   = float(np.clip(conviction_s, -1.0, 1.0))
        pos_ls_adapt_raw    = float(np.clip(conviction_a, -1.0, 1.0))

        pos_long_static = pos_long_static_raw
        # Adaptive uses its own signal independently — no blending toward near-zero static.
        pos_long_adapt  = pos_long_adapt_raw
        pos_ls_static   = pos_ls_static_raw
        pos_ls_adapt    = w_adapt * pos_ls_adapt_raw    + (1.0 - w_adapt) * pos_ls_static_raw

        cost_long_static = cost_per_unit_turnover * abs(pos_long_static - prev_pos_long_static)
        cost_long_adapt = cost_per_unit_turnover * abs(pos_long_adapt - prev_pos_long_adapt)
        cost_ls_static = cost_per_unit_turnover * abs(pos_ls_static - prev_pos_ls_static)
        cost_ls_adapt = cost_per_unit_turnover * abs(pos_ls_adapt - prev_pos_ls_adapt)

        prev_pos_long_static = pos_long_static
        prev_pos_long_adapt = pos_long_adapt
        prev_pos_ls_static = pos_ls_static
        prev_pos_ls_adapt = pos_ls_adapt

        eq_market.append(eq_market[-1] * (1.0 + r_next))
        eq_long_static.append(eq_long_static[-1] * (1.0 + pos_long_static * r_next - cost_long_static))
        eq_long_adapt.append(eq_long_adapt[-1] * (1.0 + pos_long_adapt * r_next - cost_long_adapt))
        eq_ls_static.append(eq_ls_static[-1] * (1.0 + pos_ls_static * r_next - cost_ls_static))
        eq_ls_adapt.append(eq_ls_adapt[-1] * (1.0 + pos_ls_adapt * r_next - cost_ls_adapt))

        retrained = False
        cooldown_blocked = 0

        if action != "none":
            if controller.in_cooldown(dt):
                cooldown_blocked = 1
            else:
                retrain_start = dt - pd.DateOffset(years=retrain_lookback_years)
                retrain_df = df.loc[retrain_start:dt].dropna()
                if len(retrain_df) >= 2:
                    retrain_df = retrain_df.iloc[:-1]

                # Focus on the recent regime — cap rows per tier so the model
                # isn't contaminated by stale market conditions from years ago.
                if action == "moderate":
                    retrain_df = retrain_df.tail(252)   # ~1 year
                elif action == "high":
                    retrain_df = retrain_df.tail(504)   # ~2 years
                # severe: keep full lookback — new model instance needs breadth

                if len(retrain_df) >= min_retrain_rows:
                    old_adaptive_model = adaptive_model
                    old_adaptive_scaler = adaptive_scaler

                    # Retrain with the full adaptive feature set (includes _p1s)
                    retrain_feat_df = retrain_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    adaptive_scaler = _fit_scaler(retrain_feat_df)
                    X_retrain = _scale(adaptive_scaler, retrain_feat_df)

                    drift_days_ago = np.array(
                        [(dt - d).days for d in retrain_df.index], dtype=float
                    )
                    drift_w = np.exp(-0.00347 * drift_days_ago)  # ln2/200 ≈ 0.00347
                    drift_w = drift_w * len(drift_w) / drift_w.sum()

                    if action == "moderate":
                        weighted_update(adaptive_model, X_retrain, retrain_df["Target"])
                        retrained = True

                    elif action == "high":
                        adaptive_model.train(X_retrain, retrain_df["Target"], sample_weight=drift_w)
                        retrained = True

                    elif action == "severe":
                        adaptive_model = _build_model(cfg.model_name)
                        adaptive_model.train(X_retrain, retrain_df["Target"], sample_weight=drift_w)
                        retrained = True

                    # OOS validation: compare old vs new on most recent eval rows
                    # (data neither model was trained on — true out-of-sample).
                    if retrained and len(daily_rows) >= 20:
                        try:
                            val_rows_data = daily_rows[-20:]
                            val_dates = [r["date"] for r in val_rows_data]
                            val_stream = stream_df.loc[val_dates]
                            val_X = val_stream[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            val_y = val_stream["Target"].astype(int).values
                            # Batch predict both models at once
                            old_p_batch = old_adaptive_model.predict_proba(_scale(old_adaptive_scaler, val_X))
                            new_p_batch = adaptive_model.predict_proba(_scale(adaptive_scaler, val_X))
                            if old_p_batch is not None and new_p_batch is not None:
                                old_acc = ((old_p_batch[:, 1] >= 0.5) == val_y).mean()
                                new_acc = ((new_p_batch[:, 1] >= 0.5) == val_y).mean()
                                if new_acc < old_acc - 0.08:
                                    adaptive_model = old_adaptive_model
                                    adaptive_scaler = old_adaptive_scaler
                                    retrained = False
                                    print(
                                        f"[Validation] Retrain reverted at {dt.date()}: "
                                        f"new_acc={new_acc:.3f} old_acc={old_acc:.3f}",
                                        flush=True,
                                    )
                        except Exception:
                            pass  # validation failed; keep new model

                    if retrained:
                        controller.register_adaptation(dt)

                        for f in monitor_features:
                            reference[f] = retrain_df[f].astype(float).tolist()[-window_size:]
                            rolling_feat[f] = reference[f][:]

                        if ph is not None:
                            ph.reset()
                        pred_ref_window = pred_cur_window[:] if len(pred_cur_window) == window_size else pred_ref_window[:]

        # ── Scheduled periodic retraining (optional) ──────────────────────────
        # Every 20 trading days retrain on a ROLLING 1-year window ending today.
        # This keeps the model current with the actual market regime without
        # accumulating stale patterns from old regimes (e.g. 2021 bull carrying
        # into 2022 bear).  As time advances the window shifts forward, so recent
        # eval observations naturally replace old pre-eval data.
        if _use_scheduled_retrain and not retrained and stream_idx > 0 and stream_idx % 10 == 0:
            # Growing window: all available data from train_start to yesterday,
            # capped at 500 rows for GBM speed.  Exponential decay weights
            # (half-life ≈ 200 trading days) ensure the current market regime
            # dominates while old data still provides structural context.
            # As eval progresses the window increasingly contains real eval
            # observations — the GBM sees more of the actual market it predicts.
            grow_all = df.loc[train_start:dt].dropna()
            if len(grow_all) >= 2:
                grow_all = grow_all.iloc[:-1]
            grow_df = grow_all.tail(500)

            if len(grow_df) >= 80:
                try:
                    days_ago = np.array([(dt - d).days for d in grow_df.index], dtype=float)
                    decay_w = np.exp(-0.00347 * days_ago)          # ln2/200 ≈ 0.00347
                    decay_w = decay_w * len(decay_w) / decay_w.sum()  # mean weight = 1

                    p_feat = grow_df[adaptive_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    adaptive_scaler = _fit_scaler(p_feat)
                    adaptive_model.train(
                        _scale(adaptive_scaler, p_feat),
                        grow_df["Target"],
                        sample_weight=decay_w,
                    )
                    retrained = True
                    for f in monitor_features:
                        reference[f] = grow_df[f].astype(float).tolist()[-window_size:]
                        rolling_feat[f] = reference[f][:]
                    if ph is not None:
                        ph.reset()
                except Exception as e:
                    print(f"[Scheduled retrain] skipped at {dt.date()}: {e}", flush=True)

        if action != "none":
            event_rows.append(
                {
                    "date": dt,
                    "action": action,
                    "drift_index": drift_index,
                    "feature_score": feature_score,
                    "prediction_score": prediction_score,
                    "performance_score": performance_score,
                    "top_drift_feature": feature_name_max,
                    "cooldown_blocked": cooldown_blocked,
                    "retrained_today": int(retrained),
                }
            )

        daily_rows.append(
            {
                "date": dt,
                "y_true_next": y_t,
                "y_pred_static": y_pred_static,
                "y_pred_adaptive": y_pred_adapt,
                "p1_static": p1_static,
                # Store raw (un-blended) adaptive probability for AUC/logloss evaluation.
                # y_pred_adaptive uses the gated p1_adapt for better accuracy; p1_adaptive
                # uses p1_adapt_raw for unbiased probability calibration.
                "p1_adaptive": p1_adapt_raw,
                "logloss_static": ll_static,
                "logloss_adaptive": log_loss_binary(y_t, p1_adapt_raw),
                "brier_static": br_static,
                "brier_adaptive": brier_binary(y_t, p1_adapt_raw),
                "return_next": r_next,
                "feature_score": feature_score,
                "prediction_score": prediction_score,
                "performance_score": performance_score,
                "drift_index": drift_index,
                "action": action,
                "drift_feature": feature_name_max,
                "retrained_today": int(retrained),
                "pos_long_static": pos_long_static,
                "pos_long_adaptive": pos_long_adapt,
                "pos_ls_static": pos_ls_static,
                "pos_ls_adaptive": pos_ls_adapt,
            }
        )

    print("[8/10] Streaming loop end.", flush=True)

    print("[9/10] Reporting...", flush=True)

    print(f"\n=== STATIC report ({cfg.eval_start} to {cfg.eval_end}) ===")
    print(classification_report(y_true_all, y_pred_static_all))

    print(f"\n=== ADAPTIVE report ({cfg.eval_start} to {cfg.eval_end}) ===")
    print(classification_report(y_true_all, y_pred_adapt_all))

    roll_window = 60
    rolling_df = pd.DataFrame(
        {
            "date": stream_df.index.astype(str).tolist(),
            "rolling_logloss_static_w60": rolling_mean(loss_static_all, roll_window),
            "rolling_logloss_adaptive_w60": rolling_mean(loss_adapt_all, roll_window),
        }
    )

    eq_df = pd.DataFrame(
        {
            "date": stream_df.index.astype(str).tolist(),
            "equity_market": eq_market[1:],
            "equity_longonly_static": eq_long_static[1:],
            "equity_longonly_adaptive": eq_long_adapt[1:],
            "equity_longshort_static": eq_ls_static[1:],
            "equity_longshort_adaptive": eq_ls_adapt[1:],
        }
    )

    print("\n=== Economic metrics on NEXT-DAY returns ===")
    print(_summarize_equity("Market buy&hold", np.array(eq_market)))
    print(_summarize_equity("Static long-only", np.array(eq_long_static)))
    print(_summarize_equity("Adapt  long-only", np.array(eq_long_adapt)))
    print(_summarize_equity("Static long-short", np.array(eq_ls_static)))
    print(_summarize_equity("Adapt  long-short", np.array(eq_ls_adapt)))

    print("[10/10] Saving CSVs...", flush=True)

    daily_out = f"results/daily_monitoring_{cfg.run_tag}.csv"
    events_out = f"results/drift_events_{cfg.run_tag}.csv"
    rolling_out = f"results/rolling_curves_{cfg.run_tag}.csv"
    equity_out = f"results/equity_curves_{cfg.run_tag}.csv"

    pd.DataFrame(daily_rows).to_csv(daily_out, index=False)
    pd.DataFrame(event_rows).to_csv(events_out, index=False)
    rolling_df.to_csv(rolling_out, index=False)
    eq_df.to_csv(equity_out, index=False)

    print("\n=== Drift events detected ===")
    print(f"Total drift events (action != none): {len(event_rows)}")
    for e in event_rows[:20]:
        d = pd.to_datetime(e["date"]).date()
        print(f"{d} -> action={e['action']} index={e['drift_index']:.4f} cooldown_blocked={e['cooldown_blocked']}")

    if len(event_rows) > 20:
        print(f"... {len(event_rows) - 20} more events")

    print("\nSaved:")
    print(f" - {daily_out}")
    print(f" - {events_out}")
    print(f" - {rolling_out}")
    print(f" - {equity_out}")
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)

    if not _quiet:
        try:
            from src.evaluation.run_report import run_evaluation_from_results
            print("\n[Eval] Running statistical evaluation on saved CSVs...", flush=True)
            run_evaluation_from_results(daily_out, equity_out, print_head=False)
        except Exception as e:
            print(f"\n[Eval] Skipped evaluation due to error: {e}", flush=True)

        try:
            from src.visualisation.plots import plot_run
            print("\n[Plots] Generating diagnostic charts...", flush=True)
            plot_run(daily_out, events_out, equity_out)
        except Exception as e:
            print(f"\n[Plots] Skipped charting due to error: {e}", flush=True)

    n = len(y_true_all)
    acc_static   = float(sum(a == b for a, b in zip(y_true_all, y_pred_static_all)) / n) if n else float("nan")
    acc_adaptive = float(sum(a == b for a, b in zip(y_true_all, y_pred_adapt_all))  / n) if n else float("nan")

    eq_s  = np.array(eq_long_static)
    eq_a  = np.array(eq_long_adapt)
    eq_ls_s = np.array(eq_ls_static)
    eq_ls_a = np.array(eq_ls_adapt)
    eq_m  = np.array(eq_market)

    ret_s    = eq_s[1:]  / eq_s[:-1]  - 1.0  if len(eq_s)  > 1 else np.array([])
    ret_a    = eq_a[1:]  / eq_a[:-1]  - 1.0  if len(eq_a)  > 1 else np.array([])
    ret_ls_s = eq_ls_s[1:] / eq_ls_s[:-1] - 1.0 if len(eq_ls_s) > 1 else np.array([])
    ret_ls_a = eq_ls_a[1:] / eq_ls_a[:-1] - 1.0 if len(eq_ls_a) > 1 else np.array([])
    ret_m    = eq_m[1:]  / eq_m[:-1]  - 1.0  if len(eq_m)  > 1 else np.array([])

    sharpe_static      = sharpe(ret_s)    if len(ret_s)    > 1 else float("nan")
    sharpe_adaptive    = sharpe(ret_a)    if len(ret_a)    > 1 else float("nan")
    sharpe_ls_static   = sharpe(ret_ls_s) if len(ret_ls_s) > 1 else float("nan")
    sharpe_ls_adaptive = sharpe(ret_ls_a) if len(ret_ls_a) > 1 else float("nan")

    cagr_static      = cagr(eq_s,    len(ret_s))    if len(ret_s)    > 1 else float("nan")
    cagr_adaptive    = cagr(eq_a,    len(ret_a))    if len(ret_a)    > 1 else float("nan")
    cagr_market      = cagr(eq_m,    len(ret_m))    if len(ret_m)    > 1 else float("nan")
    maxdd_static     = max_drawdown(eq_s)    if len(eq_s)    > 1 else float("nan")
    maxdd_adaptive   = max_drawdown(eq_a)    if len(eq_a)    > 1 else float("nan")

    # Correlation: drift index vs |next-day market return| — does detector
    # fire when the market is actually moving?
    drift_indices = [r["drift_index"] for r in daily_rows]
    abs_market_ret = list(np.abs(ret_m)) if len(ret_m) > 0 else []
    min_len = min(len(drift_indices), len(abs_market_ret))
    if min_len > 10:
        drift_vol_corr = float(np.corrcoef(drift_indices[:min_len], abs_market_ret[:min_len])[0, 1])
    else:
        drift_vol_corr = float("nan")

    # Mean log-loss and Brier score: adaptive vs static
    mean_ll_static   = float(np.mean(loss_static_all))   if loss_static_all   else float("nan")
    mean_ll_adaptive = float(np.mean(loss_adapt_all))    if loss_adapt_all    else float("nan")
    mean_br_static   = float(np.mean(brier_static_all))  if brier_static_all  else float("nan")
    mean_br_adaptive = float(np.mean(brier_adapt_all))   if brier_adapt_all   else float("nan")

    # Retrains actually executed (action != none AND not cooldown-blocked)
    n_retrains = int(sum(r.get("retrained_today", 0) for r in daily_rows))

    return {
        "run_tag":              cfg.run_tag,
        "ticker":               cfg.ticker_or_series,
        "train_start":          cfg.train_start,
        "train_end":            cfg.train_end,
        "eval_start":           cfg.eval_start,
        "eval_end":             cfg.eval_end,
        "use_ks":               getattr(cfg, "use_ks",   True),
        "use_psi":              getattr(cfg, "use_psi",  True),
        "use_js":               getattr(cfg, "use_js",   True),
        "use_prediction_drift": getattr(cfg, "use_prediction_drift", True),
        "use_page_hinkley":     getattr(cfg, "use_page_hinkley",     True),
        "use_sgd_online":        getattr(cfg, "use_sgd_online",        True),
        "use_scheduled_retrain": getattr(cfg, "use_scheduled_retrain", True),
        "use_error_rate_trigger":getattr(cfg, "use_error_rate_trigger",True),
        "n_eval_days":          n,
        # Classification
        "acc_static":           acc_static,
        "acc_adaptive":         acc_adaptive,
        "acc_delta":            acc_adaptive - acc_static,
        "mean_logloss_static":  mean_ll_static,
        "mean_logloss_adaptive":mean_ll_adaptive,
        "logloss_delta":        mean_ll_static - mean_ll_adaptive,   # positive = adaptive better
        "mean_brier_static":    mean_br_static,
        "mean_brier_adaptive":  mean_br_adaptive,
        "brier_delta":          mean_br_static - mean_br_adaptive,   # positive = adaptive better
        # Long-only economics
        "sharpe_static":        sharpe_static,
        "sharpe_adaptive":      sharpe_adaptive,
        "sharpe_delta":         sharpe_adaptive - sharpe_static,
        "cagr_static":          cagr_static,
        "cagr_adaptive":        cagr_adaptive,
        "cagr_market":          cagr_market,
        "maxdd_static":         maxdd_static,
        "maxdd_adaptive":       maxdd_adaptive,
        # Long-short economics
        "sharpe_ls_static":     sharpe_ls_static,
        "sharpe_ls_adaptive":   sharpe_ls_adaptive,
        "sharpe_ls_delta":      sharpe_ls_adaptive - sharpe_ls_static,
        # Drift behaviour
        "n_drift_events":       len(event_rows),
        "n_retrains":           n_retrains,
        "drift_vol_corr":       drift_vol_corr,   # corr(drift_index, |market_ret|)
        "elapsed_s":            round(time.time() - t0, 1),
    }


def main():
    import argparse
    import json as _json

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json-config", default=None)
    args, _ = p.parse_known_args()

    if args.json_config:
        with open(args.json_config) as fh:
            data = _json.load(fh)
        from src.utils.cli import RunConfig
        cfg = RunConfig(**data)
    else:
        cfg = prompt_run_config()

    run_pipeline(cfg)


if __name__ == "__main__":
    main()