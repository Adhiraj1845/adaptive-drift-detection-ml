# Dissertation Writing Context
# Upload this file to Claude web when writing the dissertation.
# This is the single source of truth for all project details.

---

## Project in One Sentence
An adaptive drift detection framework for financial time-series binary classification (next-day return direction) that trains a static baseline model alongside an adaptive model, monitors for covariate/concept/performance drift using a 4-detector ensemble, and triggers tiered retraining when drift is detected.

## University / Module
University of Leeds — COMP3931 Individual Project (or COMP3932 Synoptic Project)
Session: 2025/26

## Repository
Branch: `framework-refactor`
GitHub: Adhiraj1845/adaptive-drift-detection-ml
121 tests passing. All results committed.

---

## LaTeX Template File Structure (Overleaf)
```
finalReport.tex        — master file: \include{prelude}, chapter1–4, \bibliography{refs}, \include{appendices}
config.tex             — \newcommand{\fulltitle}{...}, \fullname, \session, \degree, \module
prelude.tex            — deliverables table, declaration, summary, acknowledgements, TOC
summary.tex            — max 1 A4 page abstract
acknowledge.tex        — acknowledgements
chapters/chapter1.tex  — Introduction + Background Research
chapters/chapter2.tex  — Methodology
chapters/chapter3.tex  — Implementation and Validation
chapters/chapter4.tex  — Results, Evaluation and Discussion
appendices.tex         — Appendices A through G
refs.bib               — BibTeX bibliography (72 references)
```

config.tex title: `Adaptive Drift Detection for Financial Time Series Classification`

Formatting: body text 11pt, 1.5 line spacing. Figures and tables captioned below. All figures cited from text before appearing. One citation style throughout (IEEE recommended).

---

## Key Parameters (all confirmed in code)
- `window_size = 100` trading days (≈5 months)
- `cooldown_days = 5` (T+2 equity settlement, MiFID II)
- `min_retrain_rows = 400`
- Detector weights: feature=0.4, prediction=0.3, performance=0.3
- Calibration thresholds: 60th/75th/90th/95th percentiles (asset-adaptive)
- Controller default limits: low=0.6, moderate=1.0, high=1.5, severe=2.0 (overridden by calibration at runtime)
- `_SGD_WEIGHT = 0.15`
- `_DRIFT_EMA_ALPHA = 0.10`, `_PERF_EMA_ALPHA = 0.10` (half-life ≈6.6 days)
- OOS rollback threshold: 8%
- Position gate: w_adapt = 0.60×drift_signal + 0.40×perf_signal, clipped [0.25, 0.90]
- Static base position: 0.60 + 0.40×conviction_s; adaptive baseline floor: 0.45

---

## System Architecture (5 components)
1. **Data layer** (`src/data_loader.py`): Yahoo Finance / FRED / CSV → feature matrix. Features: returns, log-returns, volatility 5/10/20/60d, momentum, drawdown, MACD-family (OHLCV only). Target = binary next-day return direction.
2. **Model layer** (`src/model/`): RF (Breiman 2001), LR, GBM (Friedman 2001). Thin sklearn wrappers. RobustScaler applied (Huber 1981).
3. **Detector stack** (`src/drift_detectors/`): KS test O(n log n), PSI quantile bins, JS divergence bounded [0,ln2], PageHinkley O(1) per update, PredictionDrift (JS on output probabilities).
4. **Controller** (`src/controller/`): DriftController computes composite drift index = 0.4×feature + 0.3×prediction + 0.3×performance. calibration.py learns asset-adaptive thresholds. adaptation.py: weighted_update (moderate), sliding_window_retrain (high), ensemble_refresh (severe).
5. **Evaluation** (`src/evaluation/`): McNemar, BCa bootstrap, OLS drift-conditional, per-period breakdown. Dashboard: FastAPI backend (api.py) + React/TypeScript frontend.

---

## Three Research Questions and Verified Answers

### RQ1: Do distribution-based detectors identify genuine market regime changes?
- Synthetic benchmark: TPR 97–100% at Δμ≥1.0σ vs ADWIN 0.2% (ADWIN blind to feature drift)
- Market event alignment: COVID 71.7% detection rate, SVB 55%, median alarm lag = 0 days across 10 events
- Drift-conditional: acc_delta 4.5× higher on drift-active days vs quiet periods; BCa CI [+0.0059, +0.0096]★
- Volatility regime: Kruskal-Wallis H=64.22, p<0.0001 — regime modulates adaptive gain

### RQ2: Does adaptive retraining improve prediction quality?
- Overall: acc_delta=+0.0043 (53.5% positive), sharpe_delta=+0.097 (61.7% positive)
- BCa CIs (n=10,000): acc_delta=[+0.0035,+0.0051]★, sharpe_delta=[+0.0842,+0.1098]★
- McNemar per-run: 3:1 adaptive:static win ratio, 25 Bonferroni wins (0 static wins)
- IC: IC_adaptive=+0.0173 vs IC_static=+0.0150; delta t=3.58, p=0.0003★; 20.3% exceed ICIR=0.50 (Grinold & Kahn benchmark)
- Sortino delta: +0.1519 (68.4% positive); Information Ratio: +0.4778 (89.2% positive)
- Transaction cost: break-even 310.3bps median; 98.4% runs viable at 5bps retail; 94.4% at 200bps
- Beats SMA crossover: t=2.06, p=0.039★; beats buy-and-hold: t=2.05, p=0.040★
- Sharpe decomposition: 6% from prediction quality alone, 21% from conviction sizing

### RQ3: What is the minimum sufficient detector configuration?
- Best combo: PSI+PH+prediction_drift (acc_delta=+0.0068, only 14 retrains, 82s runtime)
- KS fully redundant when PSI active (identical metrics across all PSI+KS combos)
- error_trigger = drift_only on 100% of runs (accuracy lags drift by mean 11.1 days)
- SGD wins Sharpe efficiency: sharpe_delta=+0.128, 31 retrains vs 35 for drift_only
- PSI+PH = full 5-detector ensemble on accuracy, 10.7% cheaper

---

## All 19 Experiments — Findings and Output Files

| # | Script | Runs | Key finding | Output files |
|---|---|---|---|---|
| 1 | detector_ablation.py | 3,332 | PSI+PH best; KS redundant | ablation_summary.csv, ablation_asset_summary.csv (50 rows), ablation_combo_summary.csv (16 rows) |
| 2 | retrain_ablation.py | 980 | SGD wins Sharpe; error_trigger=drift_only | retrain_summary.csv, retrain_combo_summary.csv |
| 3 | ablation_mcnemar.py | 4,312 | 3:1 win ratio, 25 Bonferroni wins | mcnemar_per_run.csv, mcnemar_combo_summary.csv, mcnemar_chart.png |
| 4 | bootstrap_ci.py | — | All 3 headline metrics significant | bootstrap_ci.csv (42 rows), bootstrap_ci_chart.png |
| 5 | sensitivity_analysis.py | 575 | CV<3 for 13/16 tickers | sensitivity_analysis.csv, sensitivity_charts.png |
| 6 | drift_conditional_analysis.py | 4,312 | 4.5× improvement during drift | drift_conditional.csv, drift_conditional_charts.png |
| 7 | period_deepdive.py | 4,312 | 2022 bad (Fed hike); 2023-24 positive | period_deepdive.csv, period_deepdive_charts.png |
| 8 | regime_analysis.py | 4,312 | K-W H=64.22, p<0.0001 | regime_analysis.csv, regime_analysis_charts.png |
| 9 | sharpe_decomposition.py | 196 | 6% prediction, 21% sizing | sharpe_decomposition.csv, sharpe_decomposition.png |
| 10 | error_trigger_analysis.py | 196 | 11.1-day accuracy lag after drift | error_trigger_analysis.csv, error_trigger_lag.png |
| 11 | risk_adjusted_metrics.py | 4,312 | Sortino+0.15★, CVaR honestly worse | risk_adjusted.csv, risk_adjusted_charts.png |
| 12 | information_coefficient.py | 4,312 | IC delta t=3.58 p=0.0003★ | information_coefficient.csv, ic_charts.png |
| 13 | transaction_cost_analysis.py | 4,312 | Break-even 310bps, 98.4% viable at 5bps | transaction_cost.csv, transaction_cost_charts.png |
| 14 | financial_baselines.py | 4,312 | Beats SMA p=0.039, B&H p=0.040 | financial_baselines.csv, financial_baselines_charts.png |
| 15 | hybrid_switching_strategy.py | 4,312 | 71% logloss improvement | hybrid_strategy.csv, hybrid_strategy_charts.png |
| 16 | synthetic_benchmark.py | ~420 | TPR 97-100% vs ADWIN 0.2% | chart_synthetic_benchmark.png |
| 17 | feature_importance_analysis.py | 594 rows | Ret_Vol_20 dominant in crises | feature_importance.csv, feature_importance_bars.png, feature_importance_heatmap.png |
| 18 | market_event_analysis.py | 1,240 rows | COVID 71.7%, lag=0 days | market_event_proximity.csv, market_event_heatmap.png, market_event_timeline.png |
| 19 | computational_cost_analysis.py | 16 rows | PH alone most efficient | cost_analysis.csv, cost_analysis_charts.png |

---

## All 24 Figures (results/ directory)

### Recommended for main body (6)
1. `architecture_diagram.png` — full pipeline flowchart → Chapter 3
2. `chart_synthetic_benchmark.png` — TPR/FPR/latency, 3 drift scenarios → Chapter 3
3. `bootstrap_ci_chart.png` — BCa CI forest plot → Chapter 4
4. `drift_conditional_charts.png` — drift-active vs quiet acc_delta → Chapter 4
5. `mcnemar_chart.png` — per-run McNemar wins/neutral/losses → Chapter 4
6. `nonstationarity_chart.png` — rolling KS on 4 assets with event annotations → Chapter 1

### To Appendix F (remaining 18)
- `ablation_detector_chart.png` — 17 combos ranked by acc_delta
- `ablation_asset_chart.png` — 50 assets ranked, colour-coded by asset class
- `retrain_ablation_chart.png` — 5 strategies × 4 metrics
- `sensitivity_charts.png` — window × cooldown heatmap
- `period_deepdive_charts.png` — per-year with 2022 highlighted
- `regime_analysis_charts.png` — low/medium/high vol regime
- `sharpe_decomposition.png` — prediction vs sizing waterfall
- `error_trigger_lag.png` — lag distribution
- `risk_adjusted_charts.png` — Sortino/Calmar/IR/CVaR
- `ic_charts.png` — IC distribution and ICIR scatter
- `transaction_cost_charts.png` — break-even curves
- `financial_baselines_charts.png` — adaptive vs baselines
- `hybrid_strategy_charts.png` — switching strategy
- `feature_importance_bars.png` and `feature_importance_heatmap.png`
- `market_event_heatmap.png` and `market_event_timeline.png`
- `cost_analysis_charts.png`

---

## Honest Negatives (address directly in Section 4.5)
1. **logloss_delta = −0.2036** — adaptive logloss WORSE in 97.5% of runs. Sign convention: logloss_delta = static − adaptive, so negative = adaptive worse. Frame as calibration-discrimination trade-off: adaptive outputs more extreme probabilities (better discrimination), worse calibration. Hybrid switching reduces penalty by 71%.
2. **MaxDD 4× worse** (−31.7% vs −7.7%) — conviction sizing artefact, not signal failure. Sortino (+0.1519★) and Calmar (+0.0102) remain positive.
3. **International/bonds negative** — international assets acc_delta=−0.5%, bonds −0.2%. Framework does not help stable low-vol assets (no regime shift = no signal). Consistent with theory.
4. **2022 bad year** — acc_delta=−0.0034. Fed rate hike cycle hurt all 16 configurations equally. Not a framework failure; framed as the only period where even the best configs underperform.
5. **20-day momentum dominates** financial baselines (Sharpe=+2.324 vs adaptive +0.507) — 2016–2024 sustained bull market biases momentum. Note honestly.

---

## Design Decisions and Citations
- **KS test**: Kolmogorov (1933), Smirnov (1948), Rabanser et al. (2019). Non-parametric, bounded [0,1], detects tail shifts. Redundant when PSI active (confirmed by ablation).
- **PSI**: Siddiqi (2006), Basel II (2005). Industry standard. Thresholds: <0.10 stable, <0.25 moderate, ≥0.25 significant.
- **JS divergence**: Lin (1991), Shannon (1948), Kullback & Leibler (1951). Symmetric and bounded [0,ln2] — used for prediction space where KL divergence is inapplicable (no natural reference direction).
- **Page-Hinkley**: Page (1954), Mouss et al. (2004). O(1) time and space per update. Applied to log-loss stream.
- **Why not ADWIN**: Bifet & Gavaldà (2007) — requires labelled accuracy feedback, blind to feature drift. Synthetic benchmark empirically confirms: ADWIN TPR=0.2% vs stack 97–100%.
- **Why not DDM**: Gama et al. (2004) — error-based only, equivalent to error_trigger which is 100% redundant (accuracy lags drift 11.1 days).
- **Weights 0.4/0.3/0.3**: Causal ordering — feature drift leads prediction drift leads performance drift. 11.1-day lag confirms hierarchy. Equal 0.3/0.3 = no a priori preference between two lagging signals. Sensitivity CV<3 confirms robustness.
- **Window=100**: KS power: D_0.05≈0.192 at n=100, sufficient for Δμ≥0.3σ. Hamilton (1989) regimes last 1–3 years; 5 months stays within one regime.
- **Cooldown=5**: T+2 equity settlement cycle, MiFID II Art. 5. 57% of low-acc days fall within 5-day window of prior drift event (error_trigger_analysis).
- **Calibration 60/75/90/95th**: Bifet & Gavaldà (2007) principle: data-driven thresholds outperform fixed ones. Asset-adaptive.
- **RobustScaler**: Huber (1981), Rousseeuw & Croux (1993). Financial returns leptokurtic (Engle 1982). Median/IQR resistant to crash outliers.
- **SGD weight 0.15**: Retrain ablation — pure SGD over-adapts, 0.15 blend preserves stability. Herbster & Warmuth (1998), Losing et al. (2018).
- **Position gate 0.60/0.40**: Sharpe decomp: sizing=21% vs prediction=6% of Sharpe gain. 3:2 reflects drift fires earlier (lag=0) vs performance lags 11.1 days.
- **Sequential split**: Rapach & Zhou (2013). Look-ahead bias prevention — random split on time series induces temporal leakage.
- **RF+LR+GBM**: Spans bias-variance frontier. Krauss et al. (2017) validates RF+GBM for S&P 500. No LSTM: simpler models outperform OOS (Rapach & Zhou 2013).

---

## Citations (72 total — BibTeX keys for refs.bib)

### Concept Drift Surveys
- widmer1996: Widmer & Kubat (1996) Machine Learning 23(1)
- tsymbal2004: Tsymbal (2004) TCD Technical Report
- gama2014survey: Gama et al. (2014) ACM Computing Surveys 46(4)
- lu2019review: Lu et al. (2019) IEEE TKDE 31(12)
- zliobaite2010: Žliobaitė (2010) arXiv:1010.4784
- webb2016: Webb et al. (2016) Data Mining and Knowledge Discovery 30(4)

### Drift Detectors
- kolmogorov1933: Kolmogorov (1933) Giornale dell'Istituto Italiano degli Attuari 4
- smirnov1948: Smirnov (1948) Annals of Mathematical Statistics 19(2)
- page1954: Page (1954) Biometrika 41(1–2)
- mouss2004: Mouss et al. (2004) Asian Control Conference
- shannon1948: Shannon (1948) Bell System Technical Journal 27(3)
- kullback1951: Kullback & Leibler (1951) Annals of Mathematical Statistics 22(1)
- lin1991: Lin (1991) IEEE Transactions on Information Theory 37(1)
- bifet2007adwin: Bifet & Gavaldà (2007) SIAM Data Mining
- gama2004ddm: Gama et al. (2004) SBIA, LNCS 3171
- baenagarcia2006eddm: Baena-García et al. (2006) ECML/PKDD Workshop
- kifer2004: Kifer et al. (2004) VLDB
- rabanser2019: Rabanser et al. (2019) NeurIPS 32
- nishida2007: Nishida & Yamauchi (2007) Discovery Science, LNCS 4755

### PSI / Credit Risk
- siddiqi2006: Siddiqi (2006) Credit Risk Scorecards, Wiley
- basel2005: Basel Committee (2005) Working Paper No. 14

### Online and Adaptive Learning
- bottou1998: Bottou (1998) On-Line Learning in Neural Networks, Cambridge UP
- herbster1998: Herbster & Warmuth (1998) Machine Learning 32(2)
- klinkenberg2000: Klinkenberg & Joachims (2000) ICML
- losing2018: Losing et al. (2018) Neurocomputing 275
- bifet2010moa: Bifet et al. (2010) JMLR 11
- gomes2017arf: Gomes et al. (2017) Machine Learning 106(9–10)

### Ensemble Methods Under Drift
- minku2012ddd: Minku & Yao (2012) IEEE TKDE 24(4)
- kolter2007dwm: Kolter & Maloof (2007) JMLR 8
- brzezinski2014: Brzezinski & Stefanowski (2014) IEEE TNNLS 25(1)
- street2001sea: Street & Kim (2001) KDD

### Financial Time Series
- hamilton1989: Hamilton (1989) Econometrica 57(2)
- ang2012regime: Ang & Timmermann (2012) Annual Review of Financial Economics 4
- fama1970: Fama (1970) Journal of Finance 25(2)
- lo2004amh: Lo (2004) Journal of Portfolio Management 30(5)
- engle1982: Engle (1982) Econometrica 50(4)
- bollerslev1986: Bollerslev (1986) Journal of Econometrics 31(3)
- rapach2013: Rapach & Zhou (2013) Handbook of Economic Forecasting Vol. 2A

### ML in Finance
- gu2020: Gu, Kelly & Xiu (2020) Review of Financial Studies 33(5)
- fischer2018: Fischer & Krauss (2018) European Journal of Operational Research 270(2)
- krauss2017: Krauss et al. (2017) European Journal of Operational Research 259(2)
- patel2015: Patel et al. (2015) Expert Systems with Applications 42(1)
- breiman2001: Breiman (2001) Machine Learning 45(1)
- friedman2001: Friedman (2001) Annals of Statistics 29(5)

### Statistical Methods
- mcnemar1947: McNemar (1947) Psychometrika 12(2)
- white1980: White (1980) Econometrica 48(4)
- efron1987: Efron (1987) JASA 82(397)
- efron1993: Efron & Tibshirani (1993) Introduction to the Bootstrap, Chapman & Hall
- dietterich1998: Dietterich (1998) Neural Computation 10(7)
- demsar2006: Demšar (2006) JMLR 7
- dunn1961: Dunn (1961) JASA 56(293)

### IC and Portfolio Management
- grinold1989: Grinold (1989) Journal of Portfolio Management 15(3)
- grinold2000: Grinold & Kahn (2000) Active Portfolio Management, 2nd ed., McGraw-Hill
- sharpe1966: Sharpe (1966) Journal of Business 39(1)
- sortino1991: Sortino & van der Meer (1991) Journal of Portfolio Management 17(4)
- young1991: Young (1991) Futures 20(1)

### Risk Measures
- markowitz1952: Markowitz (1952) Journal of Finance 7(1)
- rockafellar2000: Rockafellar & Uryasev (2000) Journal of Risk 2(3)

### Feature Engineering
- murphy1999: Murphy (1999) Technical Analysis of Financial Markets, NYIF
- brock1992: Brock et al. (1992) Journal of Finance 47(5)

### Probability Calibration
- platt1999: Platt (1999) Advances in Large Margin Classifiers, MIT Press
- niculescumizil2005: Niculescu-Mizil & Caruana (2005) ICML
- degroot1983: DeGroot & Fienberg (1983) Journal of the Royal Statistical Society D 32

### Legal, Ethical, Professional
- fca2018: FCA (2018) Algorithmic Trading Compliance in Wholesale Markets
- mifid2014: European Parliament (2014) Directive 2014/65/EU (MiFID II)
- kirilenko2017: Kirilenko et al. (2017) Journal of Finance 72(3)
- acm2018: ACM (2018) Code of Ethics and Professional Conduct
- bcs2022: BCS (2022) Code of Conduct
- pasquale2015: Pasquale (2015) The Black Box Society, Harvard UP

### Exponential Smoothing
- brown1959: Brown (1959) Statistical Forecasting for Inventory Control, McGraw-Hill
- gardner1985: Gardner (1985) Journal of Forecasting 4(1)

### Robust Preprocessing
- huber1981: Huber (1981) Robust Statistics, Wiley
- rousseeuw1993: Rousseeuw & Croux (1993) JASA 88(424)

### Hyperparameter Sensitivity
- bergstra2012: Bergstra & Bengio (2012) JMLR 13
- probst2019: Probst et al. (2019) WIREs Data Mining 9(3)

---

## Dissertation Chapter Structure

### Chapter 1 — Introduction and Background Research
1.1 Introduction (non-specialist motivation)
1.2 Financial time series non-stationarity (Hamilton 1989, Lo 2004 AMH, Engle 1982)
1.3 Concept drift taxonomy: covariate shift P(X), concept drift P(Y|X), label shift P(Y) — Gama et al. (2014)
1.4 Drift detection methods — include comparison table [Method | Type | Requires labels | Bounded | Complexity] + explain why ADWIN and DDM were rejected
1.5 Adaptive learning under drift (Bottou 1998, Herbster & Warmuth 1998, Gomes et al. 2017)
1.6 ML in finance (Gu et al. 2020, Krauss et al. 2017, Breiman 2001, Friedman 2001)
1.7 Research questions and contributions

### Chapter 2 — Methodology
2.1 Problem formulation (binary classification, non-stationarity, sequential evaluation protocol)
2.2 Data collection and preparation (Yahoo Finance / FRED, look-ahead bias, sequential split, feature engineering)
2.3 Detector design rationale (why 4 detectors covering 3 drift types; rejection of ADWIN/DDM)
2.4 Parameter justification — one subsection each:
    2.4.1 Window size = 100 (KS power + Hamilton regime duration)
    2.4.2 Cooldown = 5 days (T+2 MiFID II)
    2.4.3 Weights 0.4/0.3/0.3 (causal ordering, 11.1-day lag confirmation)
    2.4.4 Calibration percentiles 60/75/90/95 (Bifet & Gavaldà asset-adaptive)
    2.4.5 RobustScaler (Huber 1981, fat-tailed returns)
    2.4.6 SGD weight 0.15 (retrain ablation validation)
    2.4.7 Position gate 0.60/0.40 (Sharpe decomposition validation)
2.5 Model selection (RF, LR, GBM — spans bias-variance frontier)
2.6 Version control and project management (50+ commits, clean narrative on GitHub)

### Chapter 3 — Implementation and Validation
3.1 System overview + architecture diagram (Figure 3.1)
3.2 Detector implementations with complexity: KS O(n log n), PSI, JS, PH O(1), PredictionDrift
3.3 Controller and calibration (composite drift index formula, tiered action dispatch)
3.4 Adaptation strategies (weighted update / sliding window / ensemble refresh)
3.5 Position-gated blend (mathematical formulation)
3.6 Testing and validation:
    3.6.1 Unit tests: 121 tests, mathematical properties verified
    3.6.2 Synthetic benchmark: TPR/FPR/latency on controlled ground truth (Figure 3.2)
    3.6.3 End-to-end: AAPL 2018-2022 → acc_delta=+0.011, sharpe_delta=+0.996
3.7 Dashboard and API (FastAPI + React — demonstrates substantial complexity)

### Chapter 4 — Results, Evaluation and Discussion
4.1 Experimental setup (50 assets, 4 periods, 4,312 ablation runs, what static vs adaptive means)
4.2 RQ1: Detector validity
    - Drift-conditional analysis (Figure 4.1 drift_conditional_charts.png)
    - Volatility regime analysis (K-W H=64.22)
    - Market event alignment (COVID 71.7%, lag=0)
4.3 RQ2: Prediction quality improvement
    - Headline metrics + BCa CIs (Figure 4.2 bootstrap_ci_chart.png)
    - McNemar per-run (Figure 4.3 mcnemar_chart.png)
    - Risk-adjusted (Sortino, Calmar, IR, CVaR)
    - Financial baselines (SMA, B&H, momentum)
    - IC/ICIR (Grinold & Kahn framework)
    - Transaction cost break-even
4.4 RQ3: Minimum sufficient configuration
    - Detector ablation (Table 4.1 — inline summary; full Table B.1 in Appendix B)
    - Retrain strategy ablation
    - Computational cost frontier
4.5 Limitations and honest discussion (all 5 honest negatives)
4.6 Hyperparameter sensitivity (CV<3 for 13/16 tickers)
4.7 Future work (5 concrete directions)
4.8 Conclusions

### Appendix A — Self-appraisal
A.1 Critical self-evaluation of project process
A.2 Personal reflection and lessons learned (specific: reference window bug, calibration complexity, position-gated blend decision, end-to-end bugs missed by unit tests)
A.3 Legal, Social, Ethical, Professional issues (all 4 must be addressed even if not applicable):
    A.3.1 Legal: data terms of use, no personal data, not investment advice (FCA)
    A.3.2 Social: algorithmic trading volatility (Kirilenko 2017), wealth concentration, arms-race
    A.3.3 Ethical: survivorship bias, look-ahead bias prevention, model transparency, honest negatives
    A.3.4 Professional: BCS Code of Conduct, reproducibility (code public on GitHub), ACM Code of Ethics

### Appendix B — Ablation Results Tables
Table B.1: All 16 detector combos (from ablation_combo_summary.csv)
Table B.2: 5 retrain strategies (from retrain_combo_summary.csv)
Table B.3: McNemar by asset group and period (from mcnemar_combo_summary.csv)
Table B.4: Top 10 and bottom 10 assets by acc_delta (from ablation_asset_summary.csv)

### Appendix C — Statistical Validation
Table C.1: Full BCa bootstrap CIs, all 42 metric/subset combinations (from bootstrap_ci.csv)
Table C.2: Sensitivity analysis by ticker — mean acc_delta, CV (from sensitivity_analysis.csv)
Figure C.1: Sensitivity heatmap (sensitivity_charts.png)

### Appendix D — Performance Deep-Dives
Table D.1: Per-year breakdown (from period_deepdive.csv aggregated)
Table D.2: Volatility regime summary (from regime_analysis.csv aggregated)
Figure D.1: period_deepdive_charts.png
Figure D.2: regime_analysis_charts.png
Figure D.3: feature_importance_heatmap.png

### Appendix E — Financial Evaluation
Table E.1: Risk-adjusted metrics summary (from risk_adjusted.csv)
Table E.2: Transaction cost by asset class (from transaction_cost.csv)
Table E.3: Financial baselines t-stats and p-values (from financial_baselines.csv)
Figure E.1: risk_adjusted_charts.png
Figure E.2: transaction_cost_charts.png

### Appendix F — Supplementary Figures
Figure F.1: architecture_diagram.png
Figure F.2: nonstationarity_chart.png
Figure F.3: ablation_asset_chart.png
Figure F.4: ablation_detector_chart.png
Figure F.5: retrain_ablation_chart.png
Figure F.6: sharpe_decomposition.png
Figure F.7: error_trigger_lag.png
Figure F.8: ic_charts.png
Figure F.9: financial_baselines_charts.png
Figure F.10: hybrid_strategy_charts.png
Figure F.11: market_event_heatmap.png + market_event_timeline.png
Figure F.12: cost_analysis_charts.png

### Appendix G — External Materials
Libraries used (all open-source, no code copied — all implementations are original):
yfinance (data download), fredapi (FRED data), scikit-learn (model base classes and scalers),
scipy/statsmodels (statistical tests), matplotlib/seaborn (visualisation), pandas/numpy (data processing),
FastAPI (dashboard backend), React/TypeScript/Vite/Tailwind (dashboard frontend).
Data sourced from Yahoo Finance (research use) and FRED (public domain).

---

## Summary (summary.tex) — Ready to use

Financial time-series classification models trained on historical data are deployed into non-stationary environments where market regimes, volatility clusters, and macroeconomic structural breaks cause the input distribution P(X) and conditional P(Y|X) to evolve over time. This project designs, implements, and evaluates an adaptive drift detection framework that monitors an ensemble of four complementary detectors — the Kolmogorov-Smirnov test, Population Stability Index, Jensen-Shannon divergence, and Page-Hinkley sequential change-point detector — and triggers tiered model retraining when distributional shift is identified.

The framework was evaluated across 19 experiments totalling over 10,000 runs on 50 financial instruments spanning 2016–2024. Key findings: the detector stack achieves 97–100\% true positive rate on abrupt synthetic drift (compared to 0.2\% for ADWIN under identical conditions); adaptive accuracy improvement is 4.5$\times$ higher on drift-active days than quiet periods (BCa CI: $[{+}0.0059, {+}0.0096]$); and headline metrics --- accuracy delta $[{+}0.0035, {+}0.0051]$ and Sharpe delta $[{+}0.0842, {+}0.1098]$ --- are statistically significant at 95\% BCa across all runs. McNemar per-run tests show a 3:1 adaptive-to-static win ratio with 25 Bonferroni-significant wins. The adaptive edge survives transaction costs up to 310 basis points and beats both a simple moving average crossover strategy ($p=0.039$) and buy-and-hold ($p=0.040$). Limitations are reported honestly: adaptive log-loss is higher in 97.5\% of runs (calibration--discrimination trade-off), maximum drawdown is 4$\times$ worse as a conviction sizing artefact, and the framework provides no benefit for stable low-volatility assets such as bonds and international equities.

---

## Instructions for Claude web
- All numbers above are verified from actual experiment output files. Do not invent numbers.
- When writing LaTeX, use \cite{bibtexkey} with the keys listed above.
- Figures use \includegraphics{results/filename.png} — all figures are in the results/ directory.
- Write in full academic prose, not bullet points. Chapter 4 recommended to write first.
- If you need to verify a number or check a file, say so — the user has a Claude Code instance with direct access to the codebase and all CSVs.
- Do not add the Co-Authored-By Claude line to anything.
