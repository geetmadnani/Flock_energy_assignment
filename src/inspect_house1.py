"""Quick inspection of RAW House 1 data to ground cleaning-rule thresholds in
actual observed values rather than guesses. Prints a findings summary."""

import os
import sys
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
from data_utils import load_raw_house, APPLIANCE_COLS  # noqa: E402

PATH = f"{_ROOT}/data/raw_extract/RAW_House1_Part2.csv"


def main():
    df = load_raw_house(PATH)
    print("rows:", len(df))
    print("time range:", df["Time"].min(), "to", df["Time"].max())

    # ordering
    is_sorted = df["Time"].is_monotonic_increasing
    print("already time-sorted:", is_sorted)
    n_out_of_order = (df["Time"].diff().dt.total_seconds() < 0).sum()
    print("rows where time goes backwards vs previous row:", n_out_of_order)

    # duplicates
    dup_mask = df["Time"].duplicated(keep=False)
    print("duplicate timestamp rows:", dup_mask.sum(), "=", dup_mask.sum() / len(df) * 100, "%")
    print("unique duplicated timestamps:", df.loc[dup_mask, "Time"].nunique())

    # intervals (on sorted, deduped by first)
    s = df.sort_values("Time")
    diffs = s["Time"].diff().dt.total_seconds().dropna()
    print("\ninterval stats (seconds):")
    print(diffs.describe())
    print("median interval:", diffs.median())
    print("% of intervals == 6,7,8s (nominal):", (diffs.isin([6, 7, 8])).mean() * 100)
    print("count of gaps > 60s:", (diffs > 60).sum())
    print("count of gaps > 600s (10min):", (diffs > 600).sum())
    print("count of gaps > 3600s (1hr):", (diffs > 3600).sum())
    print("largest 10 gaps (s):")
    print(diffs.sort_values(ascending=False).head(10))

    # missing values
    print("\nNaN counts per column:")
    print(df.isna().sum())

    # extreme / impossible values
    print("\nAggregate stats:")
    print(df["Aggregate"].describe())
    print("Aggregate negative count:", (df["Aggregate"] < 0).sum())
    print("Aggregate > 20000W count:", (df["Aggregate"] > 20000).sum())
    print("Aggregate > 15000W count:", (df["Aggregate"] > 15000).sum())
    print("Aggregate > 10000W count:", (df["Aggregate"] > 10000).sum())
    print("Aggregate top 10 values:", df["Aggregate"].sort_values(ascending=False).head(10).tolist())

    for c in APPLIANCE_COLS:
        neg = (df[c] < 0).sum()
        over4000 = (df[c] > 4000).sum()
        mx = df[c].max()
        print(f"{c}: max={mx}, negative={neg}, >4000W={over4000}")

    # sum of appliances vs aggregate
    app_sum = df[APPLIANCE_COLS].fillna(0).sum(axis=1)
    over = (app_sum > df["Aggregate"].fillna(0) + 50).sum()  # small tolerance
    print("\nrows where sum(appliances) > aggregate (+50W tolerance):", over, "=", over / len(df) * 100, "%")


if __name__ == "__main__":
    main()
