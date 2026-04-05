# Adaptive Drift Detection for Financial Time Series

A framework for detecting and adapting to distribution shift in financial time-series binary classification tasks. The system trains a static baseline model alongside an adaptive model that monitors incoming data, detects covariate and concept drift, and triggers tiered retraining when the data distribution departs from the training distribution.

---

## Problem Statement

A classifier trained on historical financial data is deployed in a non-stationary environment. Market regimes, volatility clusters, and macro-economic shifts cause the input distribution P(X) and the conditional P(Y|X) to change over time, degrading model performance. The goal is to detect these changes as quickly as possible and retrain the model before degradation compounds, while keeping false alarm rates low.

---

## Overview

The pipeline runs in four stages:

1. **Feature engineering** — price data (OHLCV or single-value) is transformed into a feature matrix with rolling return statistics, volatility estimates, momentum, and drawdown features. The prediction target is next-day return direction (binary).

2. **Baseline training** — both a static model (frozen for the entire evaluation period) and an adaptive model (subject to retraining) are trained on the same training window and scaled with a `RobustScaler` fitted on that window.

3. **Calibration** — the drift control limits (alarm thresholds) are learned by replaying the training period through the full detector stack. The 90th / 95th / 99th / 99.9th percentiles of drift indices observed in-sample become the `low / moderate / high / severe` limits. A `max_ph_stat` normalization factor for the Page-Hinkley detector is extracted at this stage.

4. **Streaming evaluation** — each day in the evaluation period is processed sequentially. Drift scores are computed, a composite drift index is formed, an action tier is selected, and (if not in cooldown) the adaptive model is retrained. Predictions from both models are logged alongside all drift diagnostics.

---

## Architecture

```
main.py
├── data_loader.py                     download / load / feature-engineer data
├── model/
│   ├── random_forest_model.py         sklearn RF wrapper
│   ├── logistic_regression_model.py   sklearn LR wrapper
│   └── gradient_boosting_model.py     sklearn GBM wrapper
├── drift_detectors/
│   ├── ks_test_detector.py            Kolmogorov-Smirnov D-statistic
│   ├── psi_detector.py                Population Stability Index
│   ├── js_divergence_detector.py      Jensen-Shannon divergence
│   ├── page_hinkley_detector.py       sequential mean-shift monitor
│   ├── prediction_drift_detector.py   JS on model output probabilities
│   └── drift_index.py                 voting-based index (utility)
├── controller/
│   ├── drift_controller.py            composite drift index + action dispatch
│   ├── calibration.py                 percentile-based threshold learning
│   └── adaptation.py                  weighted update / sliding-window / ensemble refresh
├── evaluation/
│   ├── significance.py                McNemar, OLS, bootstrap Sharpe & AUC CI
│   └── run_report.py                  full statistical report + per-period breakdown
├── visualisation/
│   └── plots.py                       all diagnostic charts
├── experiments/
│   ├── synthetic_benchmark.py         controlled drift benchmarks
│   └── ablation.py                    per-detector ablation study
└── utils/
    └── cli.py                         interactive run-configuration wizard
```

---

## Drift Detection

Three complementary detectors operate on a rolling window of `window_size = 100` observations compared against a reference window of equal size.

### Kolmogorov-Smirnov Test

Computes the two-sample KS statistic D = max|F_ref(x) − F_cur(x)|. The implementation walks both sorted arrays in O((n+m) log(n+m)) time without constructing explicit CDFs. The critical value formula D_α ≈ c(α) · √((n+m)/(nm)) with c = {1.22, 1.36, 1.63} for α = {0.10, 0.05, 0.01} is available but the primary output is the raw D statistic, which enters the composite index as a score in [0, 1].

### Population Stability Index

PSI = Σᵢ (Pᵢ − Qᵢ) · ln(Pᵢ / Qᵢ) over histogram bins, measuring how much a distribution has shifted relative to a reference. Bins are defined by quantiles of the reference distribution to ensure uniform expected counts. Standard interpretation: PSI < 0.1 stable, 0.1–0.2 minor shift, > 0.2 significant shift.

### Jensen-Shannon Divergence

JS(P‖Q) = (KL(P‖M) + KL(Q‖M)) / 2 where M = (P+Q)/2. Bounded in [0, ln 2]. Bin edges span the combined reference + current range so non-overlapping distributions score near ln(2).

### Page-Hinkley Detector

Sequential change-point detector applied to the adaptive model's streaming log-loss. Tracks cumulative deviations from the online mean:

```
m_t = Σ(xᵢ − μᵢ − δ)
stat_increase = m_t − min(m_t)
stat_decrease = max(m_t) − m_t
```

Direction can be `increase`, `decrease`, or `both`. Because the statistic is unbounded (it grows as a cumulative sum), it is normalized at runtime: `performance_score = min(ph.statistic() / max_ph_stat, 3.0)`, where `max_ph_stat` is the maximum PH statistic observed during the calibration period.

### Composite Drift Index

```
drift_index = 0.4 · feature_score
            + 0.3 · prediction_score
            + 0.3 · performance_score
```

`feature_score` is the maximum KS/PSI/JS score across all monitored features. `prediction_score` is the JS divergence between the reference probability window and the current probability window. `performance_score` is the normalized Page-Hinkley statistic on adaptive model log-loss.

---

## Calibration

Calibration replays the training set through the detector stack to learn data-driven thresholds. A separate model is trained on the training data (also RobustScaler-scaled), and per-row predictions are generated to simulate what the streaming loop will see. The resulting sequence of drift indices is used to compute:

| Tier     | Percentile     | Action                                       |
|----------|----------------|----------------------------------------------|
| none     | < 90th         | No action                                    |
| moderate | 90th – 95th    | Weighted update (recent rows upweighted)     |
| high     | 95th – 99th    | Sliding-window retrain (last 500 rows)       |
| severe   | 99th – 99.9th  | Full ensemble refresh with new model         |

A 5-day cooldown prevents repeated retraining on the same drift event.

---

## Feature Engineering

`add_features()` constructs the following from raw price data. Features requiring OHLCV are only computed when high, low, and volume columns are present.

| Feature | Description |
|---------|-------------|
| `Return` | Daily percentage return |
| `LogReturn` | Log return ln(Pₜ / Pₜ₋₁) |
| `Target` | Binary label: 1 if next-day return > 0 |
| `HL_Range` | (High − Low) / Close — OHLCV only |
| `CO_Return` | (Close − Open) / Open — OHLCV only |
| `Vol_Change` | Volume percentage change — OHLCV only |
| `Ret_Mean_{w}` | Rolling mean return, w ∈ {5, 10, 20, 60} |
| `Ret_Vol_{w}` | Rolling return volatility, w ∈ {5, 10, 20, 60} |
| `Mom_Sum_{w}` | Rolling return sum (momentum), w ∈ {5, 10, 20, 60} |
| `Vol_Z_{w}` | Volume z-score clipped to ±20 — OHLCV only |
| `DD_{w}` | Rolling drawdown from rolling max, w ∈ {5, 10, 20, 60} |

All features are scaled with a `RobustScaler` (median + IQR normalization) fitted on the training window before entering any model. The scaler is re-fitted on each retrain window so the adaptive model always works in a consistent feature space regardless of the current distribution.

---

## Adaptation Strategies

Three retraining strategies are dispatched based on severity:

**Weighted update** (`moderate`): Re-fits the current adaptive model on all available lookback data with linearly increasing sample weights (0.5 at the oldest row, 1.5 at the most recent). This biases the model toward current regime behavior without discarding history.

**Sliding-window retrain** (`high`): Re-fits on the most recent 500 rows of the lookback window with uniform weights. Useful when the shift is large enough that older data is misleading.

**Ensemble refresh** (`severe`): A new model is trained from scratch and added to a rolling ensemble (max size 3, FIFO eviction). Predictions are the mean probability across all ensemble members. This handles structural breaks where no prior model is reliable.

After any retrain, the reference window resets to the last 100 rows of the retrain dataset, and the Page-Hinkley detector resets to prevent stale cumulative statistics.

---

## Statistical Evaluation

`run_evaluation_from_results()` runs a battery of statistical tests on any completed run's output CSVs.

### McNemar Test
Paired test on whether static and adaptive models make errors on the same observations. Uses the continuity-corrected chi-squared statistic. Tests H₀: equal error rates.

### OLS Regression
`logloss_adaptive ~ 1 + drift_event` with HC1 (heteroskedasticity-robust) standard errors. Tests whether days flagged as drift are associated with significantly higher log-loss, validating that the drift detector fires on genuinely difficult days.

### Bootstrap Sharpe Difference
3000-resample bootstrap confidence interval for Sharpe(adaptive) − Sharpe(static) on both long-only and long-short equity curves. If the 95% CI excludes 0, the Sharpe difference is statistically significant.

### Bootstrap AUC Difference
3000-resample bootstrap CI for AUC(adaptive) − AUC(static) using a rank-based AUC (Mann-Whitney U statistic).

### Multiple-Testing Correction
All 5 tests are run simultaneously. A Bonferroni-corrected significance level of α = 0.05 / 5 = 0.01 is reported. The bootstrap CIs are at 95%; FWER-corrected conclusions require 99% CIs.

### Per-Period Breakdown
The evaluation window is sliced into calendar years. Per year: observation count, static accuracy, adaptive accuracy, number of drift events, and Sharpe difference (adaptive − static long-only). This guards against aggregate metrics being driven by a single lucky period.

---

## Visualisations

`plot_run()` generates the following charts automatically after each run:

| Chart | Filename | Description |
|-------|----------|-------------|
| Rolling log-loss | `chart_rolling_logloss_*.png` | 60-day rolling log-loss, both models, drift events as vertical lines |
| Rolling accuracy | `chart_rolling_accuracy_*.png` | 60-day rolling accuracy with 50% baseline |
| Cumulative advantage | `chart_cumulative_advantage_*.png` | Cumulative (static − adaptive) log-loss; positive = adaptive winning |
| Drift index | `chart_drift_index_*.png` | Composite drift index with colour-coded event markers |
| Equity curves | `chart_equity_*.png` | All 4 strategies + market buy-and-hold |
| Model quality | `chart_model_quality_*.png` | 2×2 grid: ROC curve, reliability diagram, precision-recall curve, 30-day rolling log-loss |

---

## Synthetic Experiments

### Benchmark (`src/experiments/synthetic_benchmark.py`)

Three synthetic drift scenarios with controlled ground truth:

**Abrupt drift** — pre-drift X ~ N(0,1), post-drift X ~ N(Δμ, σ_vol). Mean shift is instantaneous. Tests detection sensitivity as a function of drift magnitude Δμ.

**Gradual drift** — mean shifts linearly from 0 to Δμ over 200 days then holds flat. Harder to detect early; tests whether detectors can accumulate evidence over a ramp.

**Concept drift** — X ~ N(0,1) throughout (feature marginal is identical pre and post-drift). Pre-drift: P(Y=1|X) = σ(X). Post-drift: P(Y=1|X) = σ(−X) (label sign flip). Feature-based detectors (KS, PSI, JS) should be blind to this and return TPR ≈ FPR, demonstrating why performance monitoring via Page-Hinkley is necessary.

Detection uses a sliding window of 100 observations versus a 250-observation reference. Threshold is calibrated at a 5% FPR from the pre-drift scores. Metrics: TPR, FPR, median detection latency (days after drift), and fraction of seeds where drift was never detected.

### Ablation (`src/experiments/ablation.py`)

Runs the benchmark across 7 drift magnitudes (Δμ ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}) for each of the three scenarios, with 20 seeds per magnitude. Evaluates 4 detector configurations: KS alone, PSI alone, JS alone, and Composite (max of all three). Outputs TPR, latency, and never-detected tables plus a 2-panel chart per scenario.

---

## Data Sources

| Source | CLI selection | Notes |
|--------|---------------|-------|
| Yahoo Finance | `Yahoo Finance` | Use `^GSPC`, `AAPL`, `SPY`, `GLD`, etc. |
| FRED | `FRED` | Forward-filled to business-day frequency. E.g. `SP500`, `DGS10`, `FEDFUNDS`. |
| CSV | `CSV file` | Needs at least a date index and a close/price column. |

Downloaded data is cached at `data/raw/<ticker>_<start>_<end>.csv` and reused on subsequent runs.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive configuration wizard
python main.py

# Run synthetic benchmark (all three drift scenarios)
python -m src.experiments.synthetic_benchmark

# Run ablation study (all detectors × all scenarios)
python -m src.experiments.ablation

# Regenerate charts for existing results
python -m src.visualisation.plots results/

# Run statistical evaluation on existing results
python -m src.evaluation.run_report results/
```

### Recommended minimum run parameters

| Parameter | Recommended minimum |
|-----------|-------------------|
| Training rows | 300 (warning issued below this) |
| Evaluation rows | 252 (1 year) |
| `min_retrain_rows` | 400 for daily equity data; 75 for short/sparse series |
| `retrain_lookback_years` | 5 |

---

## Output Files

All outputs are written to `results/`:

| File | Contents |
|------|----------|
| `daily_monitoring_<tag>.csv` | Per-day predictions, probabilities, loss values, drift scores, action taken |
| `drift_events_<tag>.csv` | Days where action ≠ none: tier, scores, top drifting feature, cooldown flag |
| `rolling_curves_<tag>.csv` | 60-day rolling log-loss for both models |
| `equity_curves_<tag>.csv` | Equity curves for market, long-only and long-short static/adaptive |
| `chart_*.png` | All diagnostic charts |

The `<tag>` defaults to `<ticker>_<train_start>_<train_end>__<eval_start>_<eval_end>` and is overridable at the CLI prompt.

---

## Implementation Notes

**Reference window symmetry** — the reference window and detection window are both 100 rows. Asymmetric sizes (e.g. reference = full training set vs window = 100) bias two-sample statistics toward smaller values, reducing sensitivity. Both windows are kept equal on initialisation, after retraining, and inside the calibration loop.

**PH normalization** — the Page-Hinkley statistic is a cumulative sum and grows without bound in long runs. It is normalized by `max_ph_stat` (the maximum PH value observed during calibration) so it enters the composite index on the same scale as the bounded [0,1] feature and prediction scores. The normalized value is capped at 3.0.

**Scaler re-fitting on retrain** — the `RobustScaler` is re-fitted on each retrain window. Using a scaler fitted on the original training data for post-drift data introduces systematic bias when the distribution has shifted.

**Leakage prevention** — the training dataframe drops its final row before training. The last row's `Target` uses the subsequent row's return; including it would leak a future observation. The same convention applies to retrain windows.

**Cooldown** — a 5-day minimum gap between retraining events prevents thrashing. Events that occur during cooldown are logged (`cooldown_blocked=1`) but trigger no retrain.

---

## Directory Structure

```
adaptive-drift-detection-ml/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/                   cached downloaded data
├── results/                   all run outputs (CSVs + charts)
└── src/
    ├── data_loader.py
    ├── model/
    ├── drift_detectors/
    ├── controller/
    ├── evaluation/
    ├── visualisation/
    ├── experiments/
    └── utils/
```
