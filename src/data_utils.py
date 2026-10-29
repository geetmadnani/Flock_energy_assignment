"""Shared helpers for loading and resampling REFIT house data."""

import numpy as np
import pandas as pd

RAW_COLS = ["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
CLEAN_COLS = RAW_COLS + ["Issues"]

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]

# Washing-machine channel per house, taken from the official REFIT appliance
# list (CLEAN_READ_ME_081116.txt). Houses with a dedicated "Washing Machine"
# IAM only (excluding combo Washer-Dryer units, which have a different duty
# cycle/power signature and are tracked separately where present).
WASHING_MACHINE_CHANNEL = {
    1: 5,   # Washing Machine, Beko WMC6140 (House 1 also has a separate Washer Dryer on ch4)
    2: 2,   # Washing Machine, LG F1289TD
    3: 6,   # Washing Machine
    5: 3,   # Washing Machine, AEG L99695HWD
    6: 2,   # Washing Machine, Bosch Classixx 1200 Express
    7: 5,   # Washing Machine, Bosch
    10: 5,  # Washing Machine, Beko WI1382
    11: 3,  # Washing Machine
    13: 3,  # Washing Machine
    15: 3,  # Washing Machine, Beko WMB91242LB
    16: 5,  # Washing Machine, Bosch WAB24262GB/01
    17: 4,  # Washing Machine, Bosch Exxcel 8
    18: 5,  # Washing Machine
    19: 2,  # Washing Machine, Bosch WAE24060GB/03
    20: 4,  # Washing Machine
    21: 3,  # Washing Machine, Beko WMB81241LW
}

FRIDGE_CHANNEL = {1: 1, 2: 1, 5: 1}


def load_raw_house(path, nrows=None):
    df = pd.read_csv(
        path,
        usecols=RAW_COLS,
        dtype={"Unix": "int64"},
        parse_dates=["Time"],
        nrows=nrows,
    )
    return df


def load_clean_house(path, nrows=None):
    df = pd.read_csv(
        path,
        usecols=CLEAN_COLS,
        dtype={"Unix": "int64", "Issues": "int8"},
        parse_dates=["Time"],
        nrows=nrows,
    )
    return df


def resample_1min(df, time_col="Time", agg="mean"):
    """Resample an irregular-interval reading series to a regular 1-minute grid."""
    s = df.set_index(time_col).sort_index()
    numeric_cols = [c for c in s.columns if c not in ("Unix",)]
    out = s[numeric_cols].resample("1min").agg(agg)
    return out


def fill_short_gaps(df, max_gap_minutes=3):
    """Linearly interpolate runs of missing 1-min bins up to max_gap_minutes long.
    Longer runs are left as NaN (flagged as outages elsewhere) rather than filled.
    """
    filled = df.copy()
    is_na = filled.isna().all(axis=1)
    # identify runs of consecutive all-NaN rows
    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")
    short_gap_mask = is_na & (run_lengths <= max_gap_minutes)
    interpolated = filled.interpolate(method="time", limit_area="inside")
    filled.loc[short_gap_mask] = interpolated.loc[short_gap_mask]
    return filled


def outage_table(df):
    """Return start/end/duration of remaining (unfilled, longer) gaps."""
    is_na = df.isna().all(axis=1)
    run_id = (is_na != is_na.shift()).cumsum()
    gaps = []
    for rid, grp in df.groupby(run_id):
        if is_na.loc[grp.index].iloc[0]:
            gaps.append(
                {
                    "start": grp.index[0],
                    "end": grp.index[-1],
                    "duration_min": len(grp),
                }
            )
    return pd.DataFrame(gaps)
