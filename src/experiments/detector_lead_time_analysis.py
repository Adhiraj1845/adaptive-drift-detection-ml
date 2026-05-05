"""
Detector lead-time analysis: KS+PSI vs PH alarm timing.

For each matched (ticker, period) pair, compares when the KS+PSI-only
configuration first raises an alarm vs when the PH-only configuration
first raises an alarm.

lead_time = t_first_alarm(KS+PSI) - t_first_alarm(PH), in trading days.
Negative  = KS+PSI fires first (before PH).
Positive  = PH fires first.
Zero      = same day.

Repeated across all alarm events (not just first) using nearest-match
pairing within a ±20-day window.

Output
------
  results/detector_lead_time.csv       -- per-pair summary
  results/detector_lead_time_chart.png -- histogram of lead times

Usage
-----
    python -m src.experiments.detector_lead_time_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_RESULTS_DIR = Path("results")
_PH_SUFFIX   = "_noKS_noPSI_noJS_ph.csv"
_FEAT_SUFFIX = "_ks_psi_noJS_noPH.csv"
_MATCH_WINDOW = 20  # trading days: alarms within ±20 days are "the same event"


def _alarm_dates(csv_path: Path) -> list[pd.Timestamp]:
    """Return sorted list of dates where action != 'none'."""
    try:
        df = pd.read_csv(csv_path, usecols=["date", "action"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df.loc[df["action"] != "none", "date"].tolist()
    except Exception:
        return []


def _match_alarms(
    feat_dates: list[pd.Timestamp],
    ph_dates: list[pd.Timestamp],
    window: int,
) -> list[int]:
    """
    For each KS+PSI alarm, find the nearest PH alarm within ±window days.
    Returns list of lead times (feat_date - ph_date).days for each matched pair.
    Negative = feature detector fires first.
    """
    if not feat_dates or not ph_dates:
        return []
    leads = []
    ph_arr = np.array([d.value for d in ph_dates])
    ns_per_day = int(pd.Timedelta("1D").value)
    for fd in feat_dates:
        diffs = (ph_arr - fd.value) / ns_per_day  # positive = ph is later
        abs_diffs = np.abs(diffs)
        if abs_diffs.min() <= window:
            idx = abs_diffs.argmin()
            lead = int(fd.value / ns_per_day - ph_arr[idx] / ns_per_day)
            leads.append(lead)  # negative = feat fires before ph
    return leads


def run(
    results_dir: Path = _RESULTS_DIR,
    csv_out: str = "results/detector_lead_time.csv",
    chart_path: str = "results/detector_lead_time_chart.png",
) -> pd.DataFrame:

    ph_files = sorted(results_dir.glob(f"*{_PH_SUFFIX}"))
    records = []
    all_leads: list[int] = []

    for ph_path in ph_files:
        stem = ph_path.name[len("daily_monitoring_"):-len(_PH_SUFFIX)]
        feat_path = results_dir / f"daily_monitoring_{stem}{_FEAT_SUFFIX}"
        if not feat_path.exists():
            continue

        ph_dates   = _alarm_dates(ph_path)
        feat_dates = _alarm_dates(feat_path)

        if not ph_dates and not feat_dates:
            continue

        leads = _match_alarms(feat_dates, ph_dates, _MATCH_WINDOW)
        all_leads.extend(leads)

        # First-alarm lead time (simpler headline stat)
        first_feat = feat_dates[0]  if feat_dates  else None
        first_ph   = ph_dates[0]    if ph_dates    else None
        if first_feat and first_ph:
            first_lead = (first_feat - first_ph).days
        else:
            first_lead = None

        parts = stem.split("_")
        ticker = parts[0]
        period = parts[1] if len(parts) > 1 else "unknown"

        records.append({
            "ticker":           ticker,
            "period":           period,
            "n_feat_alarms":    len(feat_dates),
            "n_ph_alarms":      len(ph_dates),
            "n_matched_pairs":  len(leads),
            "mean_lead_days":   round(np.mean(leads), 1) if leads else float("nan"),
            "median_lead_days": int(np.median(leads))    if leads else float("nan"),
            "pct_feat_first":   round(sum(1 for l in leads if l < 0) / len(leads) * 100, 1) if leads else float("nan"),
            "first_alarm_lead": first_lead,
        })

    df_out = pd.DataFrame(records)
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False)

    print(f"\nDetector lead-time analysis ({len(df_out)} ticker-period pairs):")
    if all_leads:
        print(f"  Matched alarm pairs:        {len(all_leads)}")
        print(f"  Mean lead time (days):      {np.mean(all_leads):.1f}  "
              f"[negative = KS+PSI fires first]")
        print(f"  Median lead time (days):    {np.median(all_leads):.1f}")
        p25, p75 = np.percentile(all_leads, [25, 75])
        print(f"  IQR:                        [{p25:.0f}, {p75:.0f}]")
        pct_feat_first = sum(1 for l in all_leads if l < 0) / len(all_leads) * 100
        print(f"  % events where KS+PSI first:{pct_feat_first:.1f}%")
        print(f"  % events where PH first:    {100 - pct_feat_first:.1f}%")
    else:
        print("  No matched alarm pairs found.")

    _plot(all_leads, chart_path)
    print(f"Saved: {csv_out}")
    return df_out


def _plot(leads: list[int], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        if not leads:
            ax.text(0.5, 0.5, "No matched alarm pairs", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            bins = range(min(leads) - 1, max(leads) + 2)
            ax.hist(leads, bins=list(bins), color="#4393c3", edgecolor="white",
                    alpha=0.85, label="Matched alarm pairs")
            ax.axvline(0, color="black", lw=1.5, ls="--", label="Simultaneous (0 days)")
            ax.axvline(np.mean(leads), color="#d73027", lw=2, ls="--",
                       label=f"Mean = {np.mean(leads):.1f} days")
            ax.axvline(np.median(leads), color="#fc8d59", lw=1.5, ls=":",
                       label=f"Median = {int(np.median(leads))} days")
            pct = sum(1 for l in leads if l < 0) / len(leads) * 100
            ax.set_xlabel(
                "Lead time: t(KS+PSI) $-$ t(PH), trading days\n"
                "Negative = KS+PSI fires first;  Positive = PH fires first",
                fontsize=10,
            )
            ax.set_ylabel("Number of matched alarm pairs", fontsize=10)
            ax.set_title(
                f"KS+PSI vs PH alarm lead time across {len(leads)} matched events\n"
                f"KS+PSI fires first in {pct:.0f}% of events",
                fontsize=11,
            )
            ax.legend(fontsize=9)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  [chart skipped] {e}")


if __name__ == "__main__":
    run()
