from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# feature_minutes first: it imports config, which puts the repo root on sys.path
from feature_minutes import build_minutes_features
from data.sqlite_helpers import connect

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "minutes.json"

# same shape as the rate models: small trees, heavy subsampling, ~6k rows
PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "seed": 0,
}

NUM_ROUNDS = 400
EARLY_STOPPING = 30
VALIDATION_SEASONS = 1

TARGET = "mpg"

# ids, the target, and anything derived from the season being predicted
DROP_COLS = ("player_id", "season", "team", "pos", "exp", "birth_date",
             "overall_pick", "draft_tier", "mpg", "gp", "total_min",
             "has_history", "career_seasons")


def feature_columns(df):
    return [c for c in df.columns if c not in DROP_COLS]


# xgboost needs numbers, and there are only four tiers plus three positions
def encode_categoricals(df):
    tiers = pd.get_dummies(df["draft_tier"], prefix="tier").astype(float)
    pos = pd.get_dummies(df["pos"], prefix="pos").astype(float)
    return pd.concat([tiers, pos], axis=1)


# rookies have no prior minutes, so they are not what this model learns from --
# their minutes come from the rookie prior instead
def training_rows(df, train_seasons):
    keep = df["has_history"] & df[TARGET].notna() & df["season"].isin(train_seasons)
    return df[keep]


# one model, one target, one file
def train_minutes_model(feature_df, train_seasons, save=True):
    df = training_rows(feature_df, train_seasons)

    if df.empty:
        raise ValueError("no training rows for minutes")

    X = pd.concat([df[feature_columns(df)], encode_categoricals(df)], axis=1)
    y = df[TARGET]

    # split by season, never at random: the job is predicting forward in time
    seasons = sorted(df["season"].unique())
    is_val = df["season"].isin(seasons[-VALIDATION_SEASONS:]).to_numpy()

    dtrain = xgb.DMatrix(X[~is_val], y[~is_val], feature_names=list(X.columns))
    dvalid = xgb.DMatrix(X[is_val], y[is_val], feature_names=list(X.columns))

    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )

    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        booster.save_model(MODEL_PATH)

    return booster


# rmse from just repeating last season's minutes, which is the bar to beat
def baseline_rmse(df):
    sub = df[df[TARGET].notna() & df["mpg_lag1"].notna()]
    return float(np.sqrt(((sub[TARGET] - sub["mpg_lag1"]) ** 2).mean()))


if __name__ == "__main__":
    conn = connect()
    feats = build_minutes_features(conn, "2026-27")
    train_seasons = sorted(feats.loc[feats.has_history, "season"].unique())

    booster = train_minutes_model(feats, train_seasons)
    rows = training_rows(feats, train_seasons)

    rmse = float(booster.best_score)
    base = baseline_rmse(rows)
    print(f"rows {len(rows)}  best_iter {booster.best_iteration}")
    print(f"valid rmse {rmse:.3f} mpg   lag1 baseline {base:.3f}   {(base - rmse) / base * 100:+.1f}%")

    gain = booster.get_score(importance_type="gain")
    print("\ntop features by gain")
    for name, score in sorted(gain.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:26} {score:8.1f}")

    print(f"\nsaved to {MODEL_PATH}")
