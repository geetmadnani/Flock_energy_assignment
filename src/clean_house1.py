"""Part 1: Inspection + cleaning pipeline for RAW REFIT House 1.

Findings grounding the thresholds below (see src/inspect_house1.py output,
also written to DATA.md):
  - 4,140,886 raw rows, 2014-06-27 to 2015-07-10, already time-sorted, 0 exact
    duplicate timestamps in this file (dedupe step kept anyway, as a no-op
    safety net, since the pipeline must not assume this holds for other
    houses/extracts).
  - Nominal polling interval is 6-8s (97.9% of gaps fall in that range).
    3,603 gaps > 60s, 59 gaps > 10 min, 19 gaps > 1 hour, largest ~8.65 days
    (a genuine sensor/logger outage, not something to interpolate over).
  - Aggregate has 4.6% NaN rows; appliance channels 5.6%-46% NaN (sensors
    added/removed/offline at different times - consistent with the official
    REFIT notes on hardware/network outages).
  - Appliance (IAM) readings up to ~98,301 W are physically impossible for a
    single UK 13A socket (~3,000 W max); the official REFIT clean pipeline
    removes IAM spikes > 4,000 W, so we adopt the same, documented threshold.
  - Aggregate readings are plausible up to ~20-25 kW for a 4-bed detached
    house running shower+cooker+kettle simultaneously, so we only null out
    Aggregate < 0 or > 20,000 W (74 rows, ~0.002%) rather than deleting real
    peak-usage events.
  - 2.8% of rows have sum(appliances) > aggregate (+50W tolerance) - flagged
    and reported, not corrected (mirrors the official "Issues" column logic,
    but we don't attempt the household-level IAM re-matching the official
    pipeline does - out of scope per the assignment brief).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
from data_utils import (  # noqa: E402
    load_raw_house,
    APPLIANCE_COLS,
    resample_1min,
    fill_short_gaps,
    outage_table,
)

RAW_PATH = f"{_ROOT}/data/raw_extract/RAW_House1_Part2.csv"
OUT_DIR = f"{_ROOT}/data/processed"
FIG_DIR = f"{_ROOT}/figures"

AGGREGATE_MIN, AGGREGATE_MAX = 0, 20000
APPLIANCE_MIN, APPLIANCE_MAX = 0, 4000
SHORT_GAP_MAX_MIN = 3

FRIDGE_COL = "Appliance1"
WASHER_COL = "Appliance5"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    summary = {}

    df = load_raw_house(RAW_PATH)
    summary["raw_rows"] = len(df)
    summary["raw_time_range"] = [str(df["Time"].min()), str(df["Time"].max())]

    # 1. sort
    df = df.sort_values("Time").reset_index(drop=True)

    # 2. dedupe duplicate timestamps (mean of numeric cols) - documented rule
    n_dupe_timestamps = df["Time"].duplicated().sum()
    summary["duplicate_timestamp_rows_removed"] = int(n_dupe_timestamps)
    df = df.groupby("Time", as_index=False).mean(numeric_only=True)

    # keep an unclean-but-sorted/deduped copy for the "before" plots, so the
    # before/after comparison isolates the effect of outlier + gap handling
    df_before = df.copy()

    # 3. flag/null obvious impossible values
    n_agg_bad = ((df["Aggregate"] < AGGREGATE_MIN) | (df["Aggregate"] > AGGREGATE_MAX)).sum()
    df.loc[(df["Aggregate"] < AGGREGATE_MIN) | (df["Aggregate"] > AGGREGATE_MAX), "Aggregate"] = np.nan
    summary["aggregate_impossible_values_nulled"] = int(n_agg_bad)

    n_app_bad = {}
    app_sum_before = df[APPLIANCE_COLS].fillna(0).sum(axis=1)
    for c in APPLIANCE_COLS:
        bad = (df[c] < APPLIANCE_MIN) | (df[c] > APPLIANCE_MAX)
        n_app_bad[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    summary["appliance_impossible_values_nulled"] = n_app_bad

    app_sum = df[APPLIANCE_COLS].fillna(0).sum(axis=1)
    n_inconsistent = int((app_sum > df["Aggregate"].fillna(0) + 50).sum())
    summary["rows_sum_appliances_gt_aggregate"] = n_inconsistent
    summary["rows_sum_appliances_gt_aggregate_pct"] = round(n_inconsistent / len(df) * 100, 3)

    # 4. resample to 1-minute grid
    res_after = resample_1min(df)
    res_before = resample_1min(df_before)  # same grid, no outlier removal, for plotting only

    n_missing_bins_before_fill = int(res_after.isna().all(axis=1).sum())
    summary["resampled_1min_bins_total"] = len(res_after)
    summary["resampled_1min_bins_fully_missing_before_fill"] = n_missing_bins_before_fill

    # 5. fill short gaps, leave longer outages as NaN
    res_filled = fill_short_gaps(res_after, max_gap_minutes=SHORT_GAP_MAX_MIN)
    gaps = outage_table(res_filled)
    gaps = gaps.sort_values("duration_min", ascending=False)
    summary["remaining_outage_count"] = len(gaps)
    summary["remaining_outage_total_minutes"] = int(gaps["duration_min"].sum()) if len(gaps) else 0
    summary["largest_outages_minutes"] = gaps["duration_min"].head(10).tolist()

    gaps.to_csv(f"{OUT_DIR}/house1_outages.csv", index=False)
    res_filled.to_parquet(f"{OUT_DIR}/house1_clean_1min.parquet")

    with open(f"{OUT_DIR}/house1_cleaning_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Plots ----
    # (a) full-period daily mean overview, before vs after
    daily_before = res_before.resample("1D").mean()
    daily_after = res_filled.resample("1D").mean()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, col, label in zip(
        axes, ["Aggregate", FRIDGE_COL, WASHER_COL], ["Aggregate", "Fridge (Appliance1)", "Washing Machine (Appliance5)"]
    ):
        ax.plot(daily_before.index, daily_before[col], label="before cleaning", alpha=0.6, lw=1)
        ax.plot(daily_after.index, daily_after[col], label="after cleaning", alpha=0.9, lw=1)
        ax.set_ylabel("Watts (daily mean)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("House 1: full-period daily-mean power, before vs after cleaning")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/house1_overview_before_after.png", dpi=130)
    plt.close(fig)

    # (b) zoomed representative day, 1-min resolution, before vs after
    # pick a day with washing-machine activity in the raw (before) signal
    active_days = res_before[res_before[WASHER_COL].fillna(0) > 200].index.normalize().unique()
    day = pd.Timestamp(active_days[len(active_days) // 2]) if len(active_days) else res_before.index[1000].normalize()
    day_start, day_end = day, day + pd.Timedelta(days=1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, col, label in zip(
        axes, ["Aggregate", FRIDGE_COL, WASHER_COL], ["Aggregate", "Fridge (Appliance1)", "Washing Machine (Appliance5)"]
    ):
        b = res_before.loc[day_start:day_end, col]
        a = res_filled.loc[day_start:day_end, col]
        ax.plot(b.index, b.values, label="before cleaning", alpha=0.6, lw=1)
        ax.plot(a.index, a.values, label="after cleaning", alpha=0.9, lw=1)
        ax.set_ylabel("Watts")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"House 1: representative day {day.date()}, 1-min resolution, before vs after cleaning")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/house1_day_before_after.png", dpi=130)
    plt.close(fig)

    print(json.dumps(summary, indent=2, default=str))
    print("\nSaved:")
    print(f"  {OUT_DIR}/house1_clean_1min.parquet")
    print(f"  {OUT_DIR}/house1_outages.csv")
    print(f"  {OUT_DIR}/house1_cleaning_summary.json")
    print(f"  {FIG_DIR}/house1_overview_before_after.png")
    print(f"  {FIG_DIR}/house1_day_before_after.png")


if __name__ == "__main__":
    main()
