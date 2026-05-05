# Adaptive Drift Detection for Financial Time Series
**COMP3931 Individual Project — Adhiraj Kumar — University of Leeds 2025/26**

An adaptive machine learning framework that detects distribution shift in financial time-series data and triggers tiered model retraining before accuracy degrades. Evaluated across 4,312 runs on 50 instruments over ten years.

---

## For Assessors: Getting Started

### 1. Prerequisites

- Python 3.12 (tested on 3.12.4)
- Node.js 18+ (only needed to rebuild the frontend — the pre-built `frontend/dist/` is already committed)

### 2. Install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the test suite

```bash
python -m pytest tests/ -v
```

121 tests across 5 modules, all passing. Covers: KS/PSI/JS/PH detector correctness, controller composite index, calibration monotonicity, position gate clip bounds, and the data-layer look-ahead bias prevention check.

### 4. View all experiment results (dashboard)

```bash
python api.py
```

Then open [http://localhost:8000](http://localhost:8000) in a browser. The dashboard shows drift scores, retrain history, and rolling performance for all completed runs. All experiment results are pre-committed to `results/` — no re-running is needed to view them.

### 5. View results directly

All CSVs and figures are in `results/`. The file `results/EXPERIMENTS.md` is an index of all 23 experiments with their key findings.

---

## Repository Structure

```
adaptive-drift-detection-ml/
├── src/
│   ├── data_loader.py              OHLCV feature engineering (33 features)
│   ├── model/                      RF, GBM, LR wrappers
│   ├── drift_detectors/            KS, PSI, JS, PH, prediction drift monitor
│   ├── controller/
│   │   ├── drift_controller.py     composite drift index + action dispatch
│   │   ├── calibration.py          asset-adaptive threshold learning
│   │   └── adaptation.py          SGD blend / sliding-window / ensemble refresh
│   ├── evaluation/                 McNemar, BCa bootstrap, IC, Sharpe tests
│   └── experiments/                23 experiment scripts (all results pre-committed)
├── results/                        all CSVs and figures (pre-committed, ~4,312 runs)
│   └── EXPERIMENTS.md              experiment index with findings
├── tests/                          121 unit tests
├── configs/                        per-experiment JSON run configurations
├── api.py                          FastAPI backend
├── frontend/dist/                  pre-built React dashboard
├── main.py                         interactive single-run entry point
└── requirements.txt
```

---

## Reproducing Experiments

All 23 experiment scripts are in `src/experiments/`. Results are already committed to `results/`, so re-running is optional. To reproduce a specific experiment:

```bash
# Example: reproduce the detector ablation (3,332 runs — takes ~2 hours)
python -m src.experiments.detector_ablation

# Example: reproduce BCa bootstrap CIs (fast, ~5 minutes)
python -m src.experiments.bootstrap_ci

# Example: reproduce sensitivity analysis (575 runs — takes ~30 minutes)
python -m src.experiments.sensitivity_analysis
```

Full list of experiment scripts with estimated runtimes is in `results/EXPERIMENTS.md`.

---

## Framework Overview

The pipeline trains a static baseline model alongside an adaptive model on the same training window. During the evaluation period, four detectors monitor for drift:

| Detector | Signal monitored | Complexity |
|---|---|---|
| Kolmogorov-Smirnov | Feature distribution P(X) | O(n log n) |
| Population Stability Index | Feature distribution P(X) | O(n) |
| Jensen-Shannon divergence | Prediction distribution P(Ŷ\|X) | O(n) |
| Page-Hinkley | Rolling log-loss stream | O(1) |

Scores are aggregated into a composite drift index (weights: 0.4 feature / 0.3 prediction / 0.3 performance), EMA-smoothed at α=0.10, and compared against asset-adaptive thresholds calibrated from each instrument's reference window (60th/75th/90th/95th percentiles).

When drift is detected, the controller dispatches one of four actions (none / SGD weighted update / sliding-window retrain / ensemble refresh) subject to a 5-day cooldown. A position gate blends the two model outputs proportionally to the drift index.

---

## Key Results (4,312 runs)

| Metric | Result |
|---|---|
| Accuracy delta BCa 95% CI | [+0.0035, +0.0051]★ |
| Sharpe delta BCa 95% CI | [+0.0842, +0.1098]★ |
| McNemar win ratio | 3.1:1 (25 Bonferroni wins, 0 static wins) |
| IC delta | +0.0023 (t=3.58, p=0.0003) |
| Drift-conditional accuracy gap | 4.6× larger on alarm-active days |
| Break-even transaction cost | 310 bps median (98.4% viable at 5 bps) |

★ excludes zero at the 95% BCa level.

Documented limitations: log-loss degradation (−0.20, all runs), maximum drawdown amplification (6× worse median), near-zero delta for fixed income and international equity, 2022 tightening cycle failure (architectural ceiling of batch retraining).

---

## Disclaimer

This framework produces model outputs on historical data for academic research purposes only. It does not constitute investment advice and is not connected to any live execution system. Data sourced from Yahoo Finance (personal/research use) and FRED (public domain).
