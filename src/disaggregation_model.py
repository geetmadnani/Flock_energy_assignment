"""Part 3: Washing-machine disaggregation model from household aggregate power.

Household split (no time-window leakage, since each house's entire timeline
goes to exactly one split):
  train = houses 1, 2, 5, 6
  val   = house 7           (used only for early-stopping / threshold choice)
  test  = houses 10, 19, 20 (fully held out, never touched before final eval)

This is a cross-house generalisation setup: the model must disaggregate a
washing machine it has never seen wired into a house it has never seen,
using only the aggregate signal plus time-of-day context. This is the
realistic deployment scenario for a utility that wants one model it can roll
out to many homes without bespoke per-home training/calibration.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = f"{_ROOT}/data/processed/house_1min"
OUT_DIR = f"{_ROOT}/data/processed"
FIG_DIR = f"{_ROOT}/figures"

TRAIN_HOUSES = [1, 2, 5, 6]
VAL_HOUSE = 7
TEST_HOUSES = [10, 19, 20]
ALL_HOUSES = [1, 2, 5, 6, 7, 10, 19, 20]
ON_THRESHOLD_W = 10

ROLL_WINDOWS = [5, 15, 30, 60]
LAGS = [1, 2, 5, 15]


def load_house_series(h):
    df = pd.read_parquet(f"{DATA_DIR}/house{h}_agg_wm_1min.parquet")
    return df


def build_features(df):
    f = pd.DataFrame(index=df.index)
    agg = df["Aggregate"]
    f["agg"] = agg
    for w in ROLL_WINDOWS:
        r = agg.rolling(f"{w}min", min_periods=1)
        f[f"agg_roll_mean_{w}"] = r.mean()
        f[f"agg_roll_std_{w}"] = r.std()
        f[f"agg_roll_min_{w}"] = r.min()
        f[f"agg_roll_max_{w}"] = r.max()
    for lag in LAGS:
        f[f"agg_lag_{lag}"] = agg.shift(lag, freq="min").reindex(f.index)
        f[f"agg_diff_{lag}"] = agg - f[f"agg_lag_{lag}"]
    hour = df.index.hour + df.index.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["target"] = df["WM"]
    f["house"] = df.get("house")
    return f


def assemble(houses):
    parts = []
    for h in houses:
        df = load_house_series(h)
        feat = build_features(df)
        feat["house"] = h
        parts.append(feat)
    full = pd.concat(parts)
    full = full.dropna(subset=["target", "agg"])
    return full


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sae = abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-6)
    on_true = (y_true > ON_THRESHOLD_W).astype(int)
    on_pred = (y_pred > ON_THRESHOLD_W).astype(int)
    f1 = f1_score(on_true, on_pred, zero_division=0)
    prec = precision_score(on_true, on_pred, zero_division=0)
    rec = recall_score(on_true, on_pred, zero_division=0)
    return {
        "label": label,
        "MAE_W": round(float(mae), 2),
        "RMSE_W": round(float(rmse), 2),
        "SAE": round(float(sae), 4),
        "on_off_F1": round(float(f1), 4),
        "on_off_precision": round(float(prec), 4),
        "on_off_recall": round(float(rec), 4),
        "n_rows": int(len(y_true)),
    }


def assemble_time_split(houses, train_frac=0.7, val_frac=0.15):
    """Per house, split its OWN timeline chronologically into train/val/test
    (e.g. first 70% of its days -> train, next 15% -> val, last 15% -> test),
    then concatenate each split across houses. This avoids leakage (test
    periods are strictly later in time than anything trained on) while
    letting the model see each appliance's signature at least once -- the
    contrast case to the fully-unseen-household split above.
    """
    train_parts, val_parts, test_parts = [], [], []
    for h in houses:
        df = load_house_series(h)
        feat = build_features(df)
        feat["house"] = h
        feat = feat.dropna(subset=["target", "agg"])
        n = len(feat)
        i_train = int(n * train_frac)
        i_val = int(n * (train_frac + val_frac))
        train_parts.append(feat.iloc[:i_train])
        val_parts.append(feat.iloc[i_train:i_val])
        test_parts.append(feat.iloc[i_val:])
    return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts)


def run_time_split_experiment(feature_cols):
    print("\n=== Experiment B: time-based split within known households ===")
    train_df, val_df, test_df = assemble_time_split(ALL_HOUSES)
    print(f"train rows={len(train_df)}, val rows={len(val_df)}, test rows={len(test_df)}")

    results = []
    always_off = np.zeros(len(test_df))
    results.append(evaluate(test_df["target"].values, always_off, "timesplit_baseline_always_off"))

    profile_lookup = train_df.groupby(train_df.index.hour)["target"].mean()
    profile_pred = pd.Series(test_df.index.hour, index=test_df.index).map(profile_lookup).fillna(
        train_df["target"].mean()
    ).values
    results.append(evaluate(test_df["target"].values, profile_pred, "timesplit_baseline_hourly_profile"))

    train_set = lgb.Dataset(train_df[feature_cols], label=train_df["target"])
    val_set = lgb.Dataset(val_df[feature_cols], label=val_df["target"], reference=train_set)
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    test_pred = np.clip(model.predict(test_df[feature_cols], num_iteration=model.best_iteration), 0, None)
    results.append(evaluate(test_df["target"].values, test_pred, "timesplit_lightgbm_model"))
    active_mask = test_df["target"].values > ON_THRESHOLD_W
    results.append(
        evaluate(test_df["target"].values[active_mask], test_pred[active_mask], "timesplit_lightgbm_active_only")
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/model_results_timesplit.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"best_iteration={model.best_iteration}")

    # actual vs predicted plot for one house in the time-split test tail
    h_plot = ALL_HOUSES[0]
    m = test_df["house"].values == h_plot
    sub = test_df.loc[m].sort_index()
    pred_h = pd.Series(test_pred, index=test_df.index).loc[m].sort_index()
    if len(sub):
        active_days = sub.index[sub["target"] > 200].normalize().unique()
        day = active_days[len(active_days) // 2] if len(active_days) else sub.index[0].normalize()
        day_end = day + pd.Timedelta(days=1)
        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(sub.loc[day:day_end].index, sub.loc[day:day_end, "target"].values, label="actual WM power", lw=1.3)
        ax.plot(pred_h.loc[day:day_end].index, pred_h.loc[day:day_end].values, label="predicted (LightGBM)", lw=1.1, alpha=0.8)
        ax.set_ylabel("Watts")
        ax.set_xlabel("Time")
        ax.set_title(f"House {h_plot} (time-split test tail, seen household): actual vs predicted, {day.date()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/model_actual_vs_predicted_timesplit_house{h_plot}.png", dpi=130)
        plt.close(fig)

    return results_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading + building features for train/val/test houses...")
    train_df = assemble(TRAIN_HOUSES)
    val_df = assemble([VAL_HOUSE])
    test_df = assemble(TEST_HOUSES)

    feature_cols = [c for c in train_df.columns if c not in ("target", "house")]
    print(f"train rows={len(train_df)}, val rows={len(val_df)}, test rows={len(test_df)}")
    print(f"n features={len(feature_cols)}")

    # ---- baselines ----
    results = []

    # 1. always-off
    always_off = np.zeros(len(test_df))
    results.append(evaluate(test_df["target"].values, always_off, "baseline_always_off"))

    # 2. hour-of-day usage-profile baseline (built from TRAIN only)
    profile_lookup = train_df.groupby(train_df.index.hour)["target"].mean()
    profile_pred = pd.Series(test_df.index.hour, index=test_df.index).map(profile_lookup).fillna(train_df["target"].mean()).values
    results.append(evaluate(test_df["target"].values, profile_pred, "baseline_hourly_profile"))

    # ---- model: LightGBM regression ----
    train_set = lgb.Dataset(train_df[feature_cols], label=train_df["target"])
    val_set = lgb.Dataset(val_df[feature_cols], label=val_df["target"], reference=train_set)

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    test_pred = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    test_pred = np.clip(test_pred, 0, None)
    results.append(evaluate(test_df["target"].values, test_pred, "lightgbm_model"))

    # also evaluate model restricted to rows where target is "active" (>threshold) -- failure-mode insight
    active_mask = test_df["target"].values > ON_THRESHOLD_W
    results.append(
        evaluate(test_df["target"].values[active_mask], test_pred[active_mask], "lightgbm_model_active_only")
    )

    # per-house breakdown on test set
    per_house = []
    for h in TEST_HOUSES:
        m = test_df["house"].values == h
        yh_true = test_df["target"].values[m]
        yh_pred = test_pred[m]
        r = evaluate(yh_true, yh_pred, f"lightgbm_house{h}")
        per_house.append(r)

    results_df = pd.DataFrame(results + per_house)
    results_df.to_csv(f"{OUT_DIR}/model_results.csv", index=False)
    print(results_df.to_string(index=False))

    # feature importance
    imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importance(importance_type="gain")}
    ).sort_values("importance", ascending=False)
    imp.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
    print("\nTop 10 features by gain:")
    print(imp.head(10).to_string(index=False))

    # ---- plot: actual vs predicted on a representative test-house day ----
    test_df_sorted = test_df.sort_index()
    h_plot = TEST_HOUSES[0]
    hmask = test_df_sorted["house"] == h_plot
    sub_idx = test_df_sorted.index[hmask]
    pred_full = pd.Series(np.nan, index=test_df.index)
    pred_full.loc[:] = test_pred
    pred_series = pred_full.loc[sub_idx].sort_index()
    true_series = test_df.loc[sub_idx, "target"].sort_index()

    active_days = true_series[true_series > 200].index.normalize().unique()
    day = active_days[len(active_days) // 2] if len(active_days) else sub_idx[1000].normalize()
    day_end = day + pd.Timedelta(days=1)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(true_series.loc[day:day_end].index, true_series.loc[day:day_end].values, label="actual WM power", lw=1.3)
    ax.plot(
        pred_series.loc[day:day_end].index,
        pred_series.loc[day:day_end].values,
        label="predicted (LightGBM)",
        lw=1.1,
        alpha=0.8,
    )
    ax.set_ylabel("Watts")
    ax.set_xlabel("Time")
    ax.set_title(f"House {h_plot} (held-out test house): actual vs predicted washing-machine power, {day.date()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/model_actual_vs_predicted_house{h_plot}.png", dpi=130)
    plt.close(fig)

    with open(f"{OUT_DIR}/model_summary.json", "w") as f:
        json.dump(
            {
                "train_houses": TRAIN_HOUSES,
                "val_house": VAL_HOUSE,
                "test_houses": TEST_HOUSES,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "best_iteration": model.best_iteration,
            },
            f,
            indent=2,
        )

    print("\nSaved model_results.csv, feature_importance.csv, model_summary.json, and plot.")

    run_time_split_experiment(feature_cols)


if __name__ == "__main__":
    main()
