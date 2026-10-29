"""Part 2: Washing-machine EDA across households using OFFICIAL CLEANED REFIT data.

Cycle-detection rule (plain language): the washing machine is considered
"on" in any 1-minute bin where its power exceeds 10 W (above typical standby
/ sensor-noise levels). A run of "on" minutes is grouped into a single wash
cycle together with any pause of up to 10 minutes of "off" time inside it,
since real wash cycles pause for filling/heating/spin breaks and we don't
want to split one physical cycle into several. A candidate cycle must last
at least 5 minutes in total (on+internal pauses) to filter out noise blips.
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
from data_utils import load_clean_house, resample_1min, fill_short_gaps, WASHING_MACHINE_CHANNEL  # noqa: E402

HOUSES = [1, 2, 5, 6, 7, 10, 19, 20]
DATA_DIR = f"{_ROOT}/data/clean_extract"
OUT_DIR = f"{_ROOT}/data/processed"
FIG_DIR = f"{_ROOT}/figures"

ON_THRESHOLD_W = 10
MERGE_GAP_MIN = 10
MIN_CYCLE_MIN = 5


def detect_cycles(power_1min: pd.Series):
    """power_1min: 1-min resolution Series (W), DatetimeIndex, may contain NaN."""
    on = (power_1min.fillna(0) > ON_THRESHOLD_W).astype(int)
    # merge gaps of up to MERGE_GAP_MIN minutes: dilate "on" runs
    off_run_id = (on != on.shift()).cumsum()
    off_lengths = (~on.astype(bool)).groupby(off_run_id).transform("sum")
    bridgeable = (on == 0) & (off_lengths <= MERGE_GAP_MIN)
    merged_on = on | bridgeable.astype(int)

    run_id = (merged_on != merged_on.shift()).cumsum()
    cycles = []
    for rid, grp in power_1min.groupby(run_id):
        if merged_on.loc[grp.index].iloc[0] == 1 and len(grp) >= MIN_CYCLE_MIN:
            vals = grp.fillna(0)
            cycles.append(
                {
                    "start": grp.index[0],
                    "end": grp.index[-1],
                    "duration_min": len(grp),
                    "peak_w": float(vals.max()),
                    "energy_wh": float(vals.sum() / 60.0),  # W * (1/60 h) per minute bin
                    "hour": grp.index[0].hour,
                    "dayofweek": grp.index[0].dayofweek,  # 0=Mon
                }
            )
    return pd.DataFrame(cycles)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    availability = []
    all_cycles = []
    series_by_house = {}

    for h in HOUSES:
        path = f"{DATA_DIR}/CLEAN_House{h}.csv"
        df = load_clean_house(path)
        df = df.sort_values("Time")
        # drop rows flagged by REFIT as inconsistent (sum(IAMs) > aggregate),
        # without touching the Time index column
        power_cols = ["Aggregate"] + [c for c in df.columns if c.startswith("Appliance")]
        df.loc[df["Issues"] == 1, power_cols] = np.nan
        ch = f"Appliance{WASHING_MACHINE_CHANNEL[h]}"
        sub = df[["Time", "Aggregate", ch]].rename(columns={ch: "WM"})
        res = resample_1min(sub, time_col="Time")
        res_filled = fill_short_gaps(res, max_gap_minutes=3)
        series_by_house[h] = res_filled

        total_min = len(res_filled)
        missing_min = int(res_filled["WM"].isna().sum())
        avail_min = total_min - missing_min
        off_min = int((res_filled["WM"].fillna(0) <= ON_THRESHOLD_W).sum())
        cycles = detect_cycles(res_filled["WM"])
        cycles["house"] = h
        all_cycles.append(cycles)

        availability.append(
            {
                "house": h,
                "wm_channel": WASHING_MACHINE_CHANNEL[h],
                "start": str(res_filled.index.min()),
                "end": str(res_filled.index.max()),
                "total_days": round(total_min / 1440, 1),
                "total_1min_bins": total_min,
                "missing_pct": round(missing_min / total_min * 100, 2),
                "off_pct": round(off_min / total_min * 100, 2),
                "n_cycles_detected": len(cycles),
                # normalised by AVAILABLE (non-missing) days, not calendar days -
                # otherwise a house with more missing WM data looks artificially
                # under-used relative to a house with better sensor coverage
                "available_days": round(avail_min / 1440, 1),
                "cycles_per_week": round(len(cycles) / (avail_min / 1440 / 7), 2),
            }
        )
        print(f"house {h}: {len(cycles)} cycles, {total_min} bins, missing {missing_min/total_min*100:.2f}%")

    os.makedirs(f"{OUT_DIR}/house_1min", exist_ok=True)
    for h, s in series_by_house.items():
        s.to_parquet(f"{OUT_DIR}/house_1min/house{h}_agg_wm_1min.parquet")

    avail_df = pd.DataFrame(availability)
    cycles_df = pd.concat(all_cycles, ignore_index=True)
    avail_df.to_csv(f"{OUT_DIR}/wm_availability.csv", index=False)
    cycles_df.to_csv(f"{OUT_DIR}/wm_cycles.csv", index=False)

    print("\n=== Availability ===")
    print(avail_df.to_string(index=False))

    # ---- Plot (i): representative 24h aggregate + WM trace ----
    h0 = 6  # House 6 - clear single washing machine channel, good example
    s = series_by_house[h0]
    cyc0 = cycles_df[cycles_df["house"] == h0]
    day = cyc0["start"].dt.normalize().mode()[0] if len(cyc0) else s.index[1000].normalize()
    day_s = s.loc[day: day + pd.Timedelta(days=1)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(day_s.index, day_s["Aggregate"], lw=1, color="tab:blue")
    axes[0].set_ylabel("Aggregate (W)")
    axes[0].set_title(f"House {h0}: representative 24h trace, {day.date()}")
    axes[1].plot(day_s.index, day_s["WM"], lw=1, color="tab:orange")
    axes[1].axhline(ON_THRESHOLD_W, color="gray", ls="--", lw=0.8, label=f"on-threshold ({ON_THRESHOLD_W}W)")
    axes[1].set_ylabel("Washing machine (W)")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/wm_24h_trace_house{h0}.png", dpi=130)
    plt.close(fig)

    # ---- Plot (ii): one or more detected cycles, zoomed ----
    cyc_sorted = cycles_df[cycles_df["house"] == h0].sort_values("duration_min", ascending=False)
    example = cyc_sorted.iloc[len(cyc_sorted) // 3]  # a representative (not the single biggest outlier) cycle
    pad = pd.Timedelta(minutes=15)
    zoom = s.loc[example["start"] - pad: example["end"] + pad]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(zoom.index, zoom["WM"], lw=1.2, color="tab:orange", label="Washing machine power")
    ax.axvspan(example["start"], example["end"], color="tab:orange", alpha=0.15, label="detected cycle")
    ax.axhline(ON_THRESHOLD_W, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Watts")
    ax.set_xlabel("Time")
    ax.set_title(
        f"House {h0}: detected cycle on {example['start'].date()} "
        f"(duration {example['duration_min']:.0f} min, peak {example['peak_w']:.0f} W, "
        f"energy {example['energy_wh']:.0f} Wh)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/wm_detected_cycle_house{h0}.png", dpi=130)
    plt.close(fig)

    # ---- Plot (iii): cross-house comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    order = sorted(cycles_df["house"].unique())
    data_dur = [cycles_df.loc[cycles_df.house == h, "duration_min"] for h in order]
    axes[0].boxplot(data_dur, tick_labels=[f"H{h}" for h in order], showfliers=False)
    axes[0].set_ylabel("Cycle duration (min)")
    axes[0].set_title("Cycle duration by house")

    data_peak = [cycles_df.loc[cycles_df.house == h, "peak_w"] for h in order]
    axes[1].boxplot(data_peak, tick_labels=[f"H{h}" for h in order], showfliers=False)
    axes[1].set_ylabel("Peak power (W)")
    axes[1].set_title("Cycle peak power by house")

    data_energy = [cycles_df.loc[cycles_df.house == h, "energy_wh"] for h in order]
    axes[2].boxplot(data_energy, tick_labels=[f"H{h}" for h in order], showfliers=False)
    axes[2].set_ylabel("Energy per cycle (Wh)")
    axes[2].set_title("Energy per cycle by house")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/wm_cross_house_comparison.png", dpi=130)
    plt.close(fig)

    # ---- Usage patterns: hour of day / day of week ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cycles_df["hour"].value_counts().sort_index().reindex(range(24), fill_value=0).plot(
        kind="bar", ax=axes[0], color="tab:blue"
    )
    axes[0].set_xlabel("Hour of day (cycle start)")
    axes[0].set_ylabel("Number of cycles (all houses)")
    axes[0].set_title("Washing-machine starts by hour of day")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cycles_df["dayofweek"].value_counts().sort_index().reindex(range(7), fill_value=0).plot(
        kind="bar", ax=axes[1], color="tab:green"
    )
    axes[1].set_xticklabels(dow_labels, rotation=0)
    axes[1].set_xlabel("Day of week (cycle start)")
    axes[1].set_ylabel("Number of cycles (all houses)")
    axes[1].set_title("Washing-machine starts by day of week")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/wm_usage_patterns.png", dpi=130)
    plt.close(fig)

    print("\nSaved availability/cycles tables and 5 figures to figures/.")


if __name__ == "__main__":
    main()
