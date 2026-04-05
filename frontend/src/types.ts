export interface RunMeta {
  tag: string;
  n_obs?: number;
  n_drift_events?: number;
  acc_static?: number;
  acc_adaptive?: number;
  date_min?: string;
  date_max?: string;
}

export interface DailyRow {
  date: string;
  y_true_next?: number;
  y_pred_static?: number;
  y_pred_adaptive?: number;
  p1_static?: number;
  p1_adaptive?: number;
  logloss_static?: number;
  logloss_adaptive?: number;
  brier_static?: number;
  brier_adaptive?: number;
  feature_score?: number;
  prediction_score?: number;
  performance_score?: number;
  drift_index?: number;
  action?: string;
  drift_event?: number;
  return_next?: number;
  pos_long_static?: number;
  pos_long_adaptive?: number;
  [key: string]: unknown;
}

export interface EquityRow {
  date: string;
  equity_market?: number;
  equity_longonly_static?: number;
  equity_longonly_adaptive?: number;
  equity_longshort_static?: number;
  equity_longshort_adaptive?: number;
  [key: string]: unknown;
}

export interface DriftEvent {
  date: string;
  action?: string;
  drift_index?: number;
  feature_score?: number;
  prediction_score?: number;
  performance_score?: number;
  cooldown_blocked?: number;
  [key: string]: unknown;
}

export interface CorrelationData {
  columns: string[];
  matrix: (number | null)[][];
}

export interface RocPrData {
  auc_static: number;
  auc_adaptive: number;
  roc_static: { fpr: number; tpr: number }[];
  roc_adaptive: { fpr: number; tpr: number }[];
  pr_static: { rec: number; prec: number }[];
  pr_adaptive: { rec: number; prec: number }[];
}

export interface EvalResult {
  tag: string;
  n_obs: number;
  drift_events: number;
  acc_static: number;
  acc_adaptive: number;
  mcnemar_p: number;
  mcnemar_sig: boolean;
  mcnemar_sig_bonf: boolean;
  sharpe_long_pt: number;
  sharpe_long_lo?: number;
  sharpe_long_hi?: number;
  sharpe_long_sig: boolean;
  sharpe_ls_pt: number;
  sharpe_ls_sig: boolean;
  auc_pt: number | null;
  auc_lo?: number | null;
  auc_hi?: number | null;
  auc_sig: boolean | null;
  ols_beta: number;
  ols_p: number;
  ols_sig_bonf: boolean;
  alpha_bonferroni: number;
  per_period: PerPeriodRow[];
}

export interface PerPeriodRow {
  period: string;
  n_obs: number;
  acc_static: number;
  acc_adaptive: number;
  n_drift_events: number;
  sharpe_diff: number | null;
}

export interface ActiveStream {
  runId: string;
  tag: string;
  logs: string[];
  progress: number;
  elapsed: number;
  done: boolean;
  error: string | null;
}

export type Page = "home" | "new_run" | "dashboard" | "equity" | "drift" | "correlations" | "quality" | "stats";
