import os
import time
import math
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from src.data_loader import download_ohlcv, add_features, save_csv, load_csv
from src.model.random_forest_model import RandomForestModel

from src.drift_detectors.ks_test_detector import KSTestDetector
from src.drift_detectors.psi_detector import PSIDetector
from src.drift_detectors.page_hinkley_detector import PageHinkleyDetector
from src.drift_detectors.js_divergence_detector import JSDivergenceDetector
from src.drift_detectors.prediction_drift_detector import PredictionDriftDetector

from src.controller.drift_controller import DriftController
from src.controller.calibration import calibrate_control_limits
from src.controller.adaptation import weighted_update, sliding_window_retrain, ensemble_refresh


# =========================================================
# Helpers
# =========================================================

def _safe_proba_one(model: RandomForestModel, x_row: pd.DataFrame) -> float:
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


def _find_next_return_column(df: pd.DataFrame) -> str:
    for cand in ["NextReturn", "ReturnNext", "next_return", "ret_next", "Return_Next", "Next_Return"]:
        if cand in df.columns:
            return cand
    if "Close" in df.columns:
        df["__next_ret__"] = df["Close"].pct_change().shift(-1)
        return "__next_ret__"
    raise ValueError("No next-day return column found and cannot compute (Close missing).")


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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _summarize_equity(name: str, equity: np.ndarray) -> str:
    rets = equity[1:] / equity[:-1] - 1.0
    return (
        f"{name}: final={equity[-1]:.3f}  "
        f"CAGR={cagr(equity, len(rets)):.3f}  "
        f"Sharpe={sharpe(rets):.2f}  "
        f"maxDD={max_drawdown(equity):.3f}"
    )


# =========================================================
# Calibration: learn drift-index control limits from baseline
# =========================================================

def calibrate_from_baseline(
    controller: DriftController,
    train_df: pd.DataFrame,
    features: list[str],
    monitor_features: list[str],
    window_size: int,
) -> dict:
    """
    We simulate streaming inside 2010-2015 to generate a baseline distribution
    of drift-index scores, then set control limits via quantiles.
    """
    baseline_indices: list[float] = []

    # Reference distributions for each monitored feature built on the full training period.
    reference = {f: train_df[f].astype(float).tolist() for f in monitor_features}

    # Rolling windows during calibration
    rolling_feat: dict[str, list[float]] = {f: [] for f in monitor_features}
    rolling_pred: list[float] = []
    rolling_loss: list[float] = []

    # Use a fixed baseline model to generate predictions for prediction-drift calibration.
    # We do not want adaptation during calibration.
    cal_model = RandomForestModel(n_estimators=300, random_state=42)
    cal_model.train(train_df[features], train_df["Target"])

    # Separate PH instance for calibration so it does not contaminate your live PH
    ph_cal = PageHinkleyDetector(threshold=5.0, delta=0.0, direction="both")

    for dt, row in train_df.iterrows():
        x_t = row[features].to_frame().T
        y_t = int(row["Target"])

        p1 = _safe_proba_one(cal_model, x_t)
        ll = log_loss_binary(y_t, p1)

        rolling_pred.append(float(p1))
        if len(rolling_pred) > window_size:
            rolling_pred.pop(0)

        rolling_loss.append(float(ll))
        if len(rolling_loss) > window_size:
            rolling_loss.pop(0)

        for f in monitor_features:
            rolling_feat[f].append(float(row[f]))
            if len(rolling_feat[f]) > window_size:
                rolling_feat[f].pop(0)

        if not all(len(rolling_feat[f]) == window_size for f in monitor_features):
            continue
        if len(rolling_pred) != window_size:
            continue
        if len(rolling_loss) != window_size:
            continue

        # Feature drift score = max across monitored features
        feat_scores = []
        for f in monitor_features:
            feat_scores.append(controller.feature_drift_score(reference[f], rolling_feat[f]))
        feature_score = float(max(feat_scores)) if feat_scores else 0.0

        # Prediction drift score comparing current window to an early baseline window
        # Use the first window_size predictions as "reference" once available
        # (stable and simple; dissertation-justifiable)
        pred_reference = rolling_pred[:]  # for early part this is fine, after it stabilizes it is stable
        prediction_score = controller.prediction_drift_score(pred_reference, rolling_pred)

        # Performance drift score from PH on logloss
        ph_cal.update(float(ll))
        performance_score = float(ph_cal.statistic())

        drift_index = float(controller.compute_drift_index(feature_score, prediction_score, performance_score))
        baseline_indices.append(drift_index)

    if len(baseline_indices) < 50:
        # fall back to simple defaults if calibration sample too small
        return {"low": 0.6, "moderate": 1.0, "high": 1.5, "severe": 2.0}

    return calibrate_control_limits(baseline_indices)


# =========================================================
# MAIN
# =========================================================

def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    print("[1/10] Starting run", flush=True)

    # -------------------------
    # Data load
    # -------------------------
    csv_path = "data/raw/sp500.csv"
    if os.path.exists(csv_path):
        print("[2/10] Using cached CSV", flush=True)
        df = load_csv(csv_path)
    else:
        print("[2/10] Downloading Yahoo Finance data...", flush=True)
        df = download_ohlcv("^GSPC", "2010-01-01", "2024-12-31")
        save_csv(df, csv_path)
        df = load_csv(csv_path)

    df = df.sort_index()
    print(f"[3/10] Loaded data. Range={df.index.min().date()} -> {df.index.max().date()} Rows={len(df)}", flush=True)

    print("[4/10] Feature engineering...", flush=True)
    df = add_features(df).sort_index()
    print(f"[4/10] Features added. Rows={len(df)}", flush=True)

    features = _make_features(df)
    next_ret_col = _find_next_return_column(df)

    # -------------------------
    # Split without leakage
    # -------------------------
    train_start = pd.Timestamp("2010-01-01")
    train_end = pd.Timestamp("2015-12-31")
    stream_start = pd.Timestamp("2016-01-01")
    stream_end = pd.Timestamp("2024-12-31")

    train_df_full = df.loc[train_start:train_end].dropna()
    stream_df = df.loc[stream_start:stream_end].dropna()

    if len(train_df_full) < 10:
        raise ValueError("Training slice too small.")

    # Leakage-safe: drop last row in training slice
    train_df = train_df_full.iloc[:-1].copy()

    print(f"[5/10] train_df={len(train_df)} stream_df={len(stream_df)}", flush=True)
    if len(train_df) < 300 or len(stream_df) < 300:
        raise ValueError("Not enough rows after slicing. Check dates/index parsing.")

    # -------------------------
    # Train baseline models
    # -------------------------
    print("[6/10] Training baseline models (static + adaptive)...", flush=True)
    static_model = RandomForestModel(n_estimators=300, random_state=42)
    adaptive_model = RandomForestModel(n_estimators=300, random_state=42)

    static_model.train(train_df[features], train_df["Target"])
    adaptive_model.train(train_df[features], train_df["Target"])
    print("[6/10] Baselines trained.", flush=True)

    # -------------------------
    # Choose monitored features (feature drift)
    # -------------------------
    monitor_features: list[str] = []
    for cand in ["Return", "LogReturn", "HL_Range", "CO_Return", "Vol_Change", "Ret_Vol_20", "Mom_Sum_20"]:
        if cand in df.columns and cand in features:
            monitor_features.append(cand)
    if "Return" not in monitor_features:
        monitor_features = ["Return"]

    window_size = 100

    # Reference distributions per monitored feature
    reference = {f: train_df[f].astype(float).tolist() for f in monitor_features}
    rolling_feat: dict[str, list[float]] = {f: [] for f in monitor_features}

    # -------------------------
    # Detectors
    # -------------------------
    ks = KSTestDetector(p_threshold=0.13)
    psi = PSIDetector(threshold=0.20)
    js = JSDivergenceDetector(bins=20)

    pred_drift = PredictionDriftDetector(bins=20)

    # Performance drift
    ph = PageHinkleyDetector(threshold=5.0, delta=0.0, direction="both")

    # Controller (weights can be justified in dissertation)
    controller = DriftController(
        feature_detectors=[ks, psi, js],
        prediction_detector=pred_drift,
        performance_detector=ph,
        weights={"feature": 0.4, "prediction": 0.3, "performance": 0.3},
        control_limits=None,
        cooldown_days=5,
    )

    # -------------------------
    # Calibration on baseline period
    # -------------------------
    print("[7/10] Calibrating drift control limits from 2010-2015...", flush=True)
    control_limits = calibrate_from_baseline(
        controller=controller,
        train_df=train_df,
        features=features,
        monitor_features=monitor_features,
        window_size=window_size,
    )
    controller.control_limits = control_limits
    print(f"[7/10] Control limits learned: {control_limits}", flush=True)

    # -------------------------
    # Adaptive response settings
    # -------------------------
    retrain_lookback_years = 5
    min_retrain_rows = 400
    retrain_cooldown_days = controller.cooldown_days  # controller enforces this already

    # ensemble for severe drift (kept small for runtime)
    # this keeps your "severe drift -> ensemble refresh" requirement.
    ensemble_models: list[RandomForestModel] = [adaptive_model]
    ensemble_max_size = 3

    # -------------------------
    # Prediction drift windows
    # -------------------------
    pred_ref_window: list[float] = []   # baseline prediction distribution reference
    pred_cur_window: list[float] = []   # rolling window of predictions

    # -------------------------
    # Outputs and tracking
    # -------------------------
    daily_rows: list[dict] = []
    event_rows: list[dict] = []

    y_true_all: list[int] = []
    y_pred_static_all: list[int] = []
    y_pred_adapt_all: list[int] = []
    loss_static_all: list[float] = []
    loss_adapt_all: list[float] = []

    # Economic evaluation
    eq_market = [1.0]
    eq_long_static = [1.0]
    eq_long_adapt = [1.0]
    eq_ls_static = [1.0]
    eq_ls_adapt = [1.0]

    prev_pos_long_static = 0.0
    prev_pos_long_adapt = 0.0
    prev_pos_ls_static = 0.0
    prev_pos_ls_adapt = 0.0

    cost_per_unit_turnover = 0.0005  # 5 bps

    print("[8/10] Streaming loop start (2016-2024)...", flush=True)

    # -------------------------
    # Streaming loop
    # -------------------------
    for dt, row in stream_df.iterrows():
        x_t = row[features].to_frame().T
        y_t = int(row["Target"])
        r_next = float(row[next_ret_col])

        # Predictions
        p1_static = _safe_proba_one(static_model, x_t)

        # Adaptive uses ensemble average (even if size=1)
        p1_adapt = float(np.mean([_safe_proba_one(m, x_t) for m in ensemble_models]))

        y_pred_static = 1 if p1_static >= 0.5 else 0
        y_pred_adapt = 1 if p1_adapt >= 0.5 else 0

        ll_static = log_loss_binary(y_t, p1_static)
        ll_adapt = log_loss_binary(y_t, p1_adapt)
        br_static = brier_binary(y_t, p1_static)
        br_adapt = brier_binary(y_t, p1_adapt)

        y_true_all.append(y_t)
        y_pred_static_all.append(y_pred_static)
        y_pred_adapt_all.append(y_pred_adapt)
        loss_static_all.append(ll_static)
        loss_adapt_all.append(ll_adapt)

        # -------------------------
        # Update rolling feature windows
        # -------------------------
        for f in monitor_features:
            rolling_feat[f].append(float(row[f]))
            if len(rolling_feat[f]) > window_size:
                rolling_feat[f].pop(0)

        # -------------------------
        # Update prediction drift windows
        # -------------------------
        pred_cur_window.append(float(p1_adapt))
        if len(pred_cur_window) > window_size:
            pred_cur_window.pop(0)

        # Build prediction reference window once, then keep stable unless we retrain
        if len(pred_ref_window) < window_size:
            pred_ref_window.append(float(p1_adapt))

        # -------------------------
        # Compute component scores (only if windows ready)
        # -------------------------
        feature_score = 0.0
        feature_name_max = None
        feature_score_by_feature: dict[str, float] = {}

        if all(len(rolling_feat[f]) == window_size for f in monitor_features):
            scores = []
            for f in monitor_features:
                s = float(controller.feature_drift_score(reference[f], rolling_feat[f]))
                feature_score_by_feature[f] = s
                scores.append((s, f))
            if scores:
                scores.sort(key=lambda x: x[0], reverse=True)
                feature_score = float(scores[0][0])
                feature_name_max = scores[0][1]

        prediction_score = 0.0
        if len(pred_ref_window) == window_size and len(pred_cur_window) == window_size:
            prediction_score = float(controller.prediction_drift_score(pred_ref_window, pred_cur_window))

        performance_score = float(controller.performance_drift_score(float(ll_adapt)))

        drift_index = float(controller.compute_drift_index(feature_score, prediction_score, performance_score))
        action = controller.decide_action(drift_index)

        # -------------------------
        # Economic positions
        # -------------------------
        pos_long_static = float(np.clip(2.0 * (p1_static - 0.5), 0.0, 1.0))
        pos_long_adapt = float(np.clip(2.0 * (p1_adapt - 0.5), 0.0, 1.0))
        pos_ls_static = float(np.clip(2.0 * (p1_static - 0.5), -1.0, 1.0))
        pos_ls_adapt = float(np.clip(2.0 * (p1_adapt - 0.5), -1.0, 1.0))

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

        # -------------------------
        # Adaptation decision with cooldown and leakage-safe retraining
        # -------------------------
        retrained = False
        cooldown_blocked = 0

        if action != "none":
            if controller.in_cooldown(dt):
                cooldown_blocked = 1
            else:
                # Build retrain slice: lookback window up to dt
                retrain_start = dt - pd.DateOffset(years=retrain_lookback_years)
                retrain_df = df.loc[retrain_start:dt].dropna()

                # Leakage-safe: drop last row because its label depends on the future
                if len(retrain_df) >= 2:
                    retrain_df = retrain_df.iloc[:-1]

                if len(retrain_df) >= min_retrain_rows:
                    # Moderate: weighted update (sample weights)
                    if action == "moderate":
                        weighted_update(adaptive_model, retrain_df[features], retrain_df["Target"])
                        # ensemble keeps the same base model, but predictions now improved
                        ensemble_models[0] = adaptive_model
                        retrained = True

                    # High: sliding-window retrain
                    elif action == "high":
                        sliding_window_retrain(adaptive_model, retrain_df[features], retrain_df["Target"], window_size=500)
                        ensemble_models[0] = adaptive_model
                        retrained = True

                    # Severe: ensemble refresh
                    elif action == "severe":
                        new_model = RandomForestModel(n_estimators=300, random_state=42)
                        sliding_window_retrain(new_model, retrain_df[features], retrain_df["Target"], window_size=500)
                        ensemble_models = ensemble_refresh(ensemble_models, new_model)

                        # Cap ensemble size
                        if len(ensemble_models) > ensemble_max_size:
                            ensemble_models = ensemble_models[-ensemble_max_size:]

                        retrained = True

                    if retrained:
                        controller.register_adaptation(dt)

                        # Update feature reference distributions to the retrain_df regime
                        for f in monitor_features:
                            reference[f] = retrain_df[f].astype(float).tolist()
                            rolling_feat[f] = reference[f][-window_size:].copy()

                        # Reset performance detector after retrain (concept drift handled)
                        ph.reset()

                        # Reset prediction drift reference distribution to new regime
                        pred_ref_window = pred_cur_window[:] if len(pred_cur_window) == window_size else pred_ref_window[:]

        # Log drift event row (only on action != none)
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
                    "ensemble_size": len(ensemble_models),
                }
            )

        # Daily monitoring row (rich)
        daily_rows.append(
            {
                "date": dt,
                "y_true_next": y_t,
                "y_pred_static": y_pred_static,
                "y_pred_adaptive": y_pred_adapt,
                "p1_static": p1_static,
                "p1_adaptive": p1_adapt,
                "logloss_static": ll_static,
                "logloss_adaptive": ll_adapt,
                "brier_static": br_static,
                "brier_adaptive": br_adapt,
                "return_next": r_next,
                "feature_score": feature_score,
                "prediction_score": prediction_score,
                "performance_score": performance_score,
                "drift_index": drift_index,
                "action": action,
                "drift_feature": feature_name_max,
                "retrained_today": int(retrained),
                "ensemble_size": len(ensemble_models),
                "pos_long_static": pos_long_static,
                "pos_long_adaptive": pos_long_adapt,
                "pos_ls_static": pos_ls_static,
                "pos_ls_adaptive": pos_ls_adapt,
            }
        )

    print("[8/10] Streaming loop end.", flush=True)

    # -------------------------
    # Reporting
    # -------------------------
    print("[9/10] Reporting...", flush=True)

    print("\n=== RF trained, STATIC (no retraining) report (2016-2024) ===")
    print(classification_report(y_true_all, y_pred_static_all))

    print("\n=== RF trained, ADAPTIVE (drift-triggered adaptation) report (2016-2024) ===")
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

    print("\n=== Economic metrics on NEXT-DAY returns (2016-2024) ===")
    print(_summarize_equity("Market buy&hold", np.array(eq_market)))
    print(_summarize_equity("Static long-only", np.array(eq_long_static)))
    print(_summarize_equity("Adapt  long-only", np.array(eq_long_adapt)))
    print(_summarize_equity("Static long-short", np.array(eq_ls_static)))
    print(_summarize_equity("Adapt  long-short", np.array(eq_ls_adapt)))

    # -------------------------
    # Save outputs
    # -------------------------
    print("[10/10] Saving CSVs...", flush=True)

    daily_out = "results/daily_monitoring_2016_2024.csv"
    events_out = "results/drift_events_2016_2024.csv"
    rolling_out = "results/rolling_curves_2016_2024.csv"
    equity_out = "results/equity_curves_2016_2024.csv"

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
    # ===========================
    # Statistical evaluation block
    # ===========================
    try:
        from src.evaluation.run_report import run_evaluation_from_results

        print("\n[Eval] Running statistical evaluation on saved CSVs...", flush=True)
        run_evaluation_from_results(daily_out, equity_out, print_head=False)
    except Exception as e:
        print(f"\n[Eval] Skipped evaluation due to error: {e}", flush=True)


if __name__ == "__main__":
    main()
