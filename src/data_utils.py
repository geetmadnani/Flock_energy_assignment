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
    """Linearly interpolate runs of missing 1-min bins up to max_gap_minutes long,
    per column. REFIT's IAMs can drop out independently of each other and of the
    aggregate meter (one appliance channel can go dark for hours while every
    other column keeps reporting), so gaps are filled column-by-column rather
    than only where every column is NaN at once. Longer runs are left as NaN
    (flagged as outages elsewhere) rather than filled.
    """
    filled = df.copy()
    for col in filled.columns:
        filled[col] = filled[col].interpolate(
            method="time", limit_area="inside", limit=max_gap_minutes
        )
    return filled


def outage_table(df):
    """Return start/end/duration of remaining (unfilled, longer) gaps, per column.
    Computed independently per column for the same reason fill_short_gaps is:
    a column can be missing while the rest of the row is fine.
    """
    gaps = []
    for col in df.columns:
        is_na = df[col].isna()
        if not is_na.any():
            continue
        run_id = (is_na != is_na.shift()).cumsum()
        for _, grp in is_na.groupby(run_id):
            if grp.iloc[0]:
                gaps.append(
                    {
                        "column": col,
                        "start": grp.index[0],
                        "end": grp.index[-1],
                        "duration_min": len(grp),
                    }
                )
    return pd.DataFrame(gaps, columns=["column", "start", "end", "duration_min"])


def per_column_missingness(df):
    """% of rows NaN per column (after gap-filling) - surfaces channels that go
    missing independently of the rest, which a row-wise missingness check would
    hide (a row with 8 good columns and 1 NaN column is still "missing")."""
    pct = (df.isna().mean() * 100).round(3)
    return pct.rename("pct_missing").rename_axis("column").reset_index()
