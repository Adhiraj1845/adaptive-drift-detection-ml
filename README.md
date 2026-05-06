# Adaptive Drift Detection for Financial Time Series

**COMP3931 Individual Project — Adhiraj Kumar — University of Leeds 2025/26**

An adaptive machine learning framework that detects distributional shift in financial time series and triggers tiered model retraining before accuracy degrades. Evaluated across 4,312 runs on 50 instruments spanning ten years and six asset classes.

---

## Quick Start (for assessors)

**Prerequisites:** Python 3.12, Node.js 18+ (only needed to rebuild the frontend — `frontend/dist/` is pre-built and committed).

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

# 2. Run the test suite (121 tests, all passing)
python -m pytest tests/ -v

# 3. Launch the monitoring dashboard
python api.py
# Open http://localhost:8000
```

All experiment results (CSVs and figures) are pre-committed to `results/`. No re-running is required to view them.

---

## What the Framework Does

A static classifier trained on historical data assumes the future resembles the past. In financial markets that assumption breaks repeatedly: regimes change, correlations flip, and volatility clusters. This framework detects those changes early and adapts before accuracy collapses.

The pipeline runs a static model and an adaptive model on the same feature matrix in parallel. During evaluation, four detectors monitor for distributional shift:

| Detector | Signal | Complexity | Role |
|---|---|---|---|
| Page-Hinkley (PH) | Rolling log-loss (CUSUM) | O(1) | Primary — minimum sufficient |
| Kolmogorov-Smirnov | Feature distribution P(X) | O(n log n) | Supplementary |
| Population Stability Index | Feature distribution P(X) | O(n) | Supplementary |
| Jensen-Shannon divergence | Prediction distribution P(Ŷ\|X) | O(n) | Supplementary |

Scores are aggregated into a composite drift index (weights: 0.4 feature / 0.3 prediction / 0.3 performance), EMA-smoothed at α = 0.10, and compared against asset-adaptive thresholds calibrated from each instrument's reference window at the 60th, 75th, 90th, and 95th percentiles.

When drift is detected, the controller dispatches one of four tiered actions — position weight update, SGD weighted blend, sliding-window retrain, or ensemble refresh — subject to a 5-day cooldown. A position gate blends the two model outputs in proportion to the composite drift index.

---

## Repository Structure

```
adaptive-drift-detection-ml/
├── src/
│   ├── data_loader.py              OHLCV feature engineering (33 features, look-ahead safe)
│   ├── model/                      RF, GBM, LR wrappers with common interface
│   ├── drift_detectors/            KS, PSI, JS, PH, prediction drift monitor
│   ├── controller/
│   │   ├── drift_controller.py     composite drift index, EMA smoothing, action dispatch
│   │   ├── calibration.py          asset-adaptive threshold learning (60/75/90/95th pct)
│   │   └── adaptation.py           SGD blend / sliding-window retrain / ensemble refresh
│   ├── evaluation/                 BCa bootstrap, McNemar, IC, Sharpe, risk metrics
│   └── experiments/                23 experiment scripts (results pre-committed)
├── tests/                          121 unit tests across 5 modules
├── configs/                        JSON run configurations per experiment
├── results/                        all CSVs and figures (~4,312 runs, pre-committed)
├── frontend/dist/                  pre-built React monitoring dashboard
├── api.py                          FastAPI backend (5 OpenAPI endpoints)
├── main.py                         interactive single-run entry point
└── requirements.txt
```

---

## Key Results

Evaluated across 4,312 runs: 3,332 detector ablation runs and 980 retrain-strategy runs. 50 instruments across 6 asset classes (large-cap US equities, index funds, commodity ETFs, fixed income, international equity, cryptocurrency). Four sub-periods covering distinct macroeconomic regimes: 2015–2019, 2018–2024, 2020–2024, 2022–2024.

| Metric | Result |
|---|---|
| Accuracy delta BCa 95% CI | [+0.0035, +0.0051] ★ |
| Sharpe delta BCa 95% CI | [+0.0842, +0.1098] ★ |
| Sortino delta | +0.1519 (BCa significant, 68.4% positive) |
| IC delta | +0.0023 (t = 3.58, p = 0.0003, Bonferroni-significant) |
| McNemar win ratio | 3.1:1 (25 Bonferroni wins vs 0 static wins) |
| Drift-conditional accuracy gap | 4.6× larger on alarm-active days (BCa [+0.0082, +0.0116]) |
| Break-even transaction cost | 310 bps median; 98.4% of positive-delta runs viable at 5 bps |

★ excludes zero at the 95% BCa level.

**RQ3 — minimum sufficient configuration:** PH alone is Pareto-optimal (+0.0068 Δacc, 82 s). No other configuration achieves both higher accuracy and lower runtime simultaneously. The lead-time hypothesis — that upstream feature detectors would fire before PH — is rejected: PH fired first in 74.1% of matched alarm events with a median lead time of 0 days across 47,845 events.

**Documented limitations:**
- Log-loss degrades by −0.20 in 97.5% of runs (calibration-discrimination trade-off)
- Maximum drawdown is ~6× worse in median terms (conviction gate amplification)
- Near-zero or negative delta for international equity ETFs and fixed income
- 2022 Federal Reserve tightening cycle: architectural ceiling of batch retraining

---

## Test Suite

```bash
python -m pytest tests/ -v
```

121 tests across 5 modules. Key property categories:

- **Look-ahead bias prevention**: feature value on day t is unchanged when days t+1 through t+5 are perturbed, verified for all 33 features
- **PH detector**: zero at initialisation, no false alarm on stationary Gaussian, alarm within bounded steps after Δμ ≥ 0.5σ
- **Calibration**: thresholds satisfy τ₁ ≤ τ₂ ≤ τ₃ ≤ τ₄; fallback triggers correctly when reference window is too short
- **Position gate**: blend weight w_adapt ∈ [0.25, 0.90] for all inputs; blended probability p̂_t ∈ [0, 1]

---

## Reproducing Experiments

Results are pre-committed. Re-running is optional. All scripts are in `src/experiments/`.

```bash
# Detector ablation (3,332 runs — ~2 hours)
python -m src.experiments.detector_ablation

# Retrain strategy ablation (980 runs — ~45 minutes)
python -m src.experiments.retrain_ablation

# BCa bootstrap confidence intervals (~5 minutes)
python -m src.experiments.bootstrap_ci

# Sensitivity analysis over 25-point grid (~30 minutes)
python -m src.experiments.sensitivity_analysis

# Synthetic ground-truth benchmark (~420 runs, ~10 minutes)
python -m src.experiments.synthetic_benchmark

# Lead-time analysis (fast, reads committed ablation results)
python -m src.experiments.detector_lead_time_analysis
```

Full list with estimated runtimes: `results/EXPERIMENTS.md`.

---

## Monitoring Dashboard

```bash
python api.py
```

FastAPI backend with 5 OpenAPI/Swagger-documented REST endpoints. Pre-built React/TypeScript frontend served from `frontend/dist/`. The dashboard exposes drift scores, retrain history, and rolling performance for all committed runs. Examiners can inspect results interactively without re-running any experiment.

---

## Parameter Summary

| Parameter | Value | Justification |
|---|---|---|
| window_size | 100 days | KS power: detects 0.5σ shifts at n=100, α=0.05; within Hamilton regime duration |
| cooldown_days | 5 | MiFID II T+2 settlement buffer; 43% of error days fall within 5 days of prior alarm |
| Composite weights | 0.4 / 0.3 / 0.3 | Causal ordering: feature shift upstream of prediction shift, upstream of loss |
| EMA α | 0.10 | Half-life ~6.6 days; consistent with 5-day cooldown |
| SGD blend γ | 0.15 | 85% historical / 15% recent; consistent with Hamilton regime persistence |
| Calibration percentiles | 60 / 75 / 90 / 95 | Asset-adaptive; 60th conservative floor to limit false-alarm retraining |
| Position gate clip | [0.25, 0.90] | Prevents degenerate all-adaptive or all-static blends |
| OOS rollback threshold | 8 pp | 3–8 σ below pre-retrain accuracy; reverts degrading retrains |
| min_retrain_rows | 400 | ~19 months; architectural floor of batch paradigm |

---

## Disclaimer

This framework produces model outputs on historical data for academic research purposes only. It does not constitute investment advice and is not connected to any live trading system. Data sourced from Yahoo Finance (personal/research use) and the Federal Reserve Economic Data repository (FRED, public domain).
