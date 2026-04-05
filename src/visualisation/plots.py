from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


_STATIC   = "#2c7bb6"
_ADAPTIVE = "#d7191c"
_ACTUAL   = "#1a9641"
_DRIFT    = "#fdae61"


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _rolling_logloss_chart(daily: pd.DataFrame, events: pd.DataFrame,
                            tag: str, out_path: str, window: int = 60) -> None:
    dates = pd.to_datetime(daily["date"])
    loss_s = daily["logloss_static"].rolling(window, min_periods=window // 2).mean()
    loss_a = daily["logloss_adaptive"].rolling(window, min_periods=window // 2).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, loss_s, color=_STATIC,   lw=1.8, label=f"Static model ({window}-day rolling loss)")
    ax.plot(dates, loss_a, color=_ADAPTIVE, lw=1.8, label=f"Adaptive model ({window}-day rolling loss)")

    if events is not None and len(events) > 0:
        ev_dates = pd.to_datetime(events["date"])
        actions  = events["action"] if "action" in events.columns else None
        for i, ed in enumerate(ev_dates):
            act = actions.iloc[i] if actions is not None else "drift"
            alpha = 0.6 if act == "severe" else 0.3
            lw    = 1.5 if act == "severe" else 0.8
            ax.axvline(ed, color=_DRIFT, alpha=alpha, lw=lw, zorder=0)
        ax.plot([], [], color=_DRIFT, alpha=0.5, lw=1.2, label="Drift event (amber)")

    ax.set_title(f"Rolling Log-Loss — Static vs Adaptive\n{tag}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-day rolling log-loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _rolling_accuracy_chart(daily: pd.DataFrame, events: pd.DataFrame,
                             tag: str, out_path: str, window: int = 60) -> None:
    dates  = pd.to_datetime(daily["date"])
    y      = daily["y_true_next"]
    corr_s = (daily["y_pred_static"]   == y).astype(float).rolling(window, min_periods=window // 2).mean()
    corr_a = (daily["y_pred_adaptive"] == y).astype(float).rolling(window, min_periods=window // 2).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, corr_s, color=_STATIC,   lw=1.8, label=f"Static ({window}-day accuracy)")
    ax.plot(dates, corr_a, color=_ADAPTIVE, lw=1.8, label=f"Adaptive ({window}-day accuracy)")
    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.6, label="50% (random)")

    if events is not None and len(events) > 0:
        ev_dates = pd.to_datetime(events["date"])
        for ed in ev_dates:
            ax.axvline(ed, color=_DRIFT, alpha=0.3, lw=0.8, zorder=0)
        ax.plot([], [], color=_DRIFT, alpha=0.5, lw=1.2, label="Drift event")

    ax.set_title(f"Rolling Accuracy — Static vs Adaptive\n{tag}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-day rolling accuracy")
    ax.set_ylim(0.3, 0.75)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _cumulative_advantage_chart(daily: pd.DataFrame, tag: str, out_path: str) -> None:
    dates     = pd.to_datetime(daily["date"])
    advantage = (daily["logloss_static"] - daily["logloss_adaptive"]).cumsum()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, advantage, color=_ADAPTIVE, lw=1.8, label="Cumulative log-loss saved by adaptive model")
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.fill_between(dates, advantage, 0,
                    where=(advantage >= 0), alpha=0.15, color=_ADAPTIVE, label="Adaptive better")
    ax.fill_between(dates, advantage, 0,
                    where=(advantage < 0),  alpha=0.15, color=_STATIC,   label="Static better")

    ax.set_title(f"Cumulative Log-Loss Advantage (Static − Adaptive)\n{tag}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative loss saved")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _equity_chart(equity: pd.DataFrame, tag: str, out_path: str) -> None:
    dates = pd.to_datetime(equity["date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, equity["equity_market"],            color="grey",    lw=1.4, ls="--", label="Buy & Hold")
    ax.plot(dates, equity["equity_longonly_static"],   color=_STATIC,   lw=1.6, ls="--", label="Long-only static")
    ax.plot(dates, equity["equity_longonly_adaptive"], color=_ADAPTIVE, lw=1.8,          label="Long-only adaptive")
    ax.plot(dates, equity["equity_longshort_static"],  color=_STATIC,   lw=1.4, ls=":",  label="Long-short static")
    ax.plot(dates, equity["equity_longshort_adaptive"],color=_ADAPTIVE, lw=1.6, ls=":",  label="Long-short adaptive")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    ax.set_title(f"Equity Curves — All Strategies\n{tag}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (starts at 1.0)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _drift_index_chart(daily: pd.DataFrame, events: pd.DataFrame,
                       tag: str, out_path: str) -> None:
    dates = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, daily["drift_index"], color="#555555", lw=1.2, label="Drift index")

    if events is not None and len(events) > 0:
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"])
        colours = {"moderate": "#fdae61", "high": "#f46d43", "severe": "#d73027"}
        for _, row in ev.iterrows():
            c = colours.get(row.get("action", "moderate"), _DRIFT)
            ax.axvline(row["date"], color=c, alpha=0.7, lw=1.2)

        for action, colour in colours.items():
            ax.plot([], [], color=colour, lw=1.5, label=f"{action.capitalize()} drift")

    ax.set_title(f"Drift Index Over Time\n{tag}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Composite drift index")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _model_quality_chart(daily: pd.DataFrame, tag: str, out_path: str) -> None:
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

    y   = daily["y_true_next"].astype(int).to_numpy()
    p_s = daily["p1_static"].astype(float).to_numpy()
    p_a = daily["p1_adaptive"].astype(float).to_numpy()
    dates = pd.to_datetime(daily["date"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_roc, ax_rel, ax_pr, ax_ll = axes.flatten()

    fpr_s, tpr_s, _ = roc_curve(y, p_s)
    fpr_a, tpr_a, _ = roc_curve(y, p_a)
    auc_s = sk_auc(fpr_s, tpr_s)
    auc_a = sk_auc(fpr_a, tpr_a)
    ax_roc.plot(fpr_s, tpr_s, color=_STATIC,   lw=1.8, label=f"Static   (AUC={auc_s:.3f})")
    ax_roc.plot(fpr_a, tpr_a, color=_ADAPTIVE, lw=1.8, label=f"Adaptive (AUC={auc_a:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=9)

    bin_edges = np.linspace(0.0, 1.0, 11)

    def _reliability(p, y_):
        mean_pred, frac_pos = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            mean_pred.append(float(np.mean(p[mask])))
            frac_pos.append(float(np.mean(y_[mask])))
        return np.array(mean_pred), np.array(frac_pos)

    mp_s, fp_s = _reliability(p_s, y)
    mp_a, fp_a = _reliability(p_a, y)
    ax_rel.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax_rel.scatter(mp_s, fp_s, color=_STATIC,   s=40, zorder=5, label="Static")
    ax_rel.plot(mp_s, fp_s, color=_STATIC, lw=1.2)
    ax_rel.scatter(mp_a, fp_a, color=_ADAPTIVE, s=40, zorder=5, marker="^", label="Adaptive")
    ax_rel.plot(mp_a, fp_a, color=_ADAPTIVE, lw=1.2)
    ax_rel.set_xlabel("Mean predicted probability")
    ax_rel.set_ylabel("Fraction of positives")
    ax_rel.set_title("Reliability Diagram")
    ax_rel.legend(fontsize=9)
    ax_rel.set_xlim(-0.02, 1.02)
    ax_rel.set_ylim(-0.02, 1.02)

    prec_s, rec_s, _ = precision_recall_curve(y, p_s)
    prec_a, rec_a, _ = precision_recall_curve(y, p_a)
    prevalence = float(np.mean(y))
    ax_pr.plot(rec_s, prec_s, color=_STATIC,   lw=1.8, label="Static")
    ax_pr.plot(rec_a, prec_a, color=_ADAPTIVE, lw=1.8, label="Adaptive")
    ax_pr.axhline(prevalence, color="grey", lw=0.8, ls="--", alpha=0.6,
                  label=f"Baseline (prevalence={prevalence:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=9)

    w = 30
    roll_s = daily["logloss_static"].rolling(w, min_periods=w // 2).mean()
    roll_a = daily["logloss_adaptive"].rolling(w, min_periods=w // 2).mean()
    ax_ll.plot(dates, roll_s, color=_STATIC,   lw=1.6, label=f"Static ({w}-day rolling)")
    ax_ll.plot(dates, roll_a, color=_ADAPTIVE, lw=1.6, label=f"Adaptive ({w}-day rolling)")
    ax_ll.set_xlabel("Date")
    ax_ll.set_ylabel("Log-loss")
    ax_ll.set_title("Log-Loss Over Time")
    ax_ll.legend(fontsize=9)
    ax_ll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ll.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    fig.suptitle(f"Model Quality Diagnostics — {tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_run(
    daily_csv: str,
    events_csv: str,
    equity_csv: str,
    *,
    out_dir: str | None = None,
    window: int = 60,
) -> None:
    tag = Path(daily_csv).stem.replace("daily_monitoring_", "")
    save_dir = Path(out_dir) if out_dir else Path(daily_csv).parent

    print(f"\n[Plots] Generating charts for: {tag}")

    daily  = _load(daily_csv)
    events = _load(events_csv) if Path(events_csv).exists() else None
    equity = _load(equity_csv) if Path(equity_csv).exists() else None

    _rolling_logloss_chart(
        daily, events, tag,
        str(save_dir / f"chart_rolling_logloss_{tag}.png"), window)

    _rolling_accuracy_chart(
        daily, events, tag,
        str(save_dir / f"chart_rolling_accuracy_{tag}.png"), window)

    _cumulative_advantage_chart(
        daily, tag,
        str(save_dir / f"chart_cumulative_advantage_{tag}.png"))

    _drift_index_chart(
        daily, events, tag,
        str(save_dir / f"chart_drift_index_{tag}.png"))

    if equity is not None:
        _equity_chart(
            equity, tag,
            str(save_dir / f"chart_equity_{tag}.png"))

    if "p1_static" in daily.columns and "p1_adaptive" in daily.columns:
        try:
            _model_quality_chart(
                daily, tag,
                str(save_dir / f"chart_model_quality_{tag}.png"))
        except Exception as e:
            print(f"  [model quality chart skipped] {e}")


def plot_all(results_dir: str = "results") -> None:
    results = Path(results_dir)
    daily_files = sorted(results.glob("daily_monitoring_*.csv"))

    if not daily_files:
        print(f"No daily_monitoring_*.csv files found in '{results_dir}'.")
        return

    for daily_f in daily_files:
        tag      = daily_f.name.replace("daily_monitoring_", "").replace(".csv", "")
        events_f = results / f"drift_events_{tag}.csv"
        equity_f = results / f"equity_curves_{tag}.csv"
        try:
            plot_run(str(daily_f), str(events_f), str(equity_f))
        except Exception as e:
            print(f"  [error] {tag}: {e}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    plot_all(results_dir)
