from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import CATEGORIES
from data.sqlite_helpers import connect
from feature_rates import build_rate_features

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# small trees and heavy subsampling because 6k rows overfits fast
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

# the last season of training data is held out to stop the boosting
VALIDATION_SEASONS = 1

# columns that are ids, targets, or leftovers -- never features
DROP_COLS = ("player_id", "season", "birth_date", "overall_pick", "gp", "mpg",
             "total_min", "usage_rate", "career_seasons", "has_history")


# everything that survives the drop list, plus one-hot draft tier
def feature_columns(df):
    raw = [c + "_rate" for c in CATEGORIES]
    return [c for c in df.columns if c not in DROP_COLS and c not in raw and c != "draft_tier"]


# xgboost wants numbers, and there are only four tiers
def encode_tiers(df):
    return pd.get_dummies(df["draft_tier"], prefix="tier").astype(float)


# rows a model can actually learn from: has a past and has a target
def training_rows(df, target_category, train_seasons):
    target = target_category + "_rate"
    keep = df["has_history"] & df[target].notna() & df["season"].isin(train_seasons)
    return df[keep]


# one category, one model. Nine separate files so a bad category can be refit
# without touching the other eight.
def train_rate_model(feature_df, target_category, train_seasons, save=True):
    target = target_category + "_rate"
    df = training_rows(feature_df, target_category, train_seasons)

    if df.empty:
        raise ValueError(f"no training rows for {target_category}")

    feats = feature_columns(df)
    X = pd.concat([df[feats], encode_tiers(df)], axis=1)
    y = df[target]

    # split by season, not at random: predicting the future from the past is the
    # actual job, and a random split lets a player's own later season leak in
    seasons = sorted(df["season"].unique())
    holdout = seasons[-VALIDATION_SEASONS:]
    is_val = df["season"].isin(holdout).to_numpy()

    # NaN is passed through on purpose -- xgboost learns a direction for missing
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
        booster.save_model(ARTIFACT_DIR / f"rate_{target_category}.json")

    return booster


# nine categories, nine models, nine files
def train_all(feature_df, train_seasons, save=True):
    models = {}
    for cat in CATEGORIES:
        booster = train_rate_model(feature_df, cat, train_seasons, save=save)
        models[cat] = booster
    return models


# rmse against just predicting the previous season, which is the bar to beat
def baseline_rmse(df, target_category):
    target, lag = target_category + "_rate", target_category + "_rate_lag1"
    sub = df[df[target].notna() & df[lag].notna()]
    return float(np.sqrt(((sub[target] - sub[lag]) ** 2).mean()))


if __name__ == "__main__":
    conn = connect()
    feats = build_rate_features(conn, "2026-27")
    train_seasons = sorted(feats.loc[feats.has_history, "season"].unique())

    print(f"training on {len(train_seasons)} seasons: {train_seasons[0]} .. {train_seasons[-1]}")
    print(f"{'category':10} {'rows':>6} {'best_it':>8} {'valid_rmse':>11} {'lag1_rmse':>10}  vs_baseline")

    for cat in CATEGORIES:
        booster = train_rate_model(feats, cat, train_seasons)
        rows = len(training_rows(feats, cat, train_seasons))
        rmse = float(booster.best_score)
        base = baseline_rmse(training_rows(feats, cat, train_seasons), cat)
        edge = (base - rmse) / base * 100.0
        print(f"{cat:10} {rows:6} {booster.best_iteration:8} {rmse:11.5f} {base:10.5f}  {edge:+6.1f}%")

    print(f"\nsaved to {ARTIFACT_DIR}")