# Games played, the availability half of a projection

import numpy as np
import pandas as pd
import xgboost as xgb

# config first so the repo root lands on sys.path before any data.* import
from config import DB_PATH
from data.sqlite_helpers import connect
from feature_minutes import GAMES_IN_SEASON, build_minutes_features
from train_minutes import ARTIFACT_DIR, encode_categoricals

MODEL_PATH = ARTIFACT_DIR / "games.json"

PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.04,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "seed": 17,
}

NUM_ROUNDS = 400
EARLY_STOPPING = 30
VALIDATION_SEASONS = 1

TARGET = "gp"

# same exclusions as the minutes model: ids, and anything derived from the season
# being predicted. gp is the target here rather than a leak, so it is dropped from
# the features and read back as y.
DROP_COLS = ("player_id", "season", "team", "pos", "exp", "birth_date",
             "overall_pick", "draft_tier", "mpg", "gp", "total_min",
             "has_history", "career_seasons")


def feature_columns(df):
    return [c for c in df.columns if c not in DROP_COLS]


# rookies have no prior games, so they learn nothing here -- their availability
# comes from the rookie prior, the same split the minutes model makes
def training_rows(df, train_seasons):
    keep = df["has_history"] & df[TARGET].notna() & df["season"].isin(train_seasons)
    return df[keep]


def train_games_model(feature_df, train_seasons, save=True):
    df = training_rows(feature_df, train_seasons)

    if df.empty:
        raise ValueError("no training rows for games")

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


# rmse from repeating last season's games, which is the bar to beat
def baseline_rmse(df):
    sub = df[df[TARGET].notna() & df["gp_lag1"].notna()]
    return float(np.sqrt(((sub[TARGET] - sub["gp_lag1"]) ** 2).mean()))


# rmse from assuming everyone plays every game, which is what predict.py did
def full_season_rmse(df):
    sub = df[df[TARGET].notna()]
    return float(np.sqrt(((sub[TARGET] - GAMES_IN_SEASON) ** 2).mean()))


# a projection cannot be negative games or more than a full season
def clip_games(games):
    return np.clip(games, 0.0, float(GAMES_IN_SEASON))


if __name__ == "__main__":
    conn = connect()
    feats = build_minutes_features(conn, "2026-27")
    train_seasons = sorted(feats.loc[feats.has_history, "season"].unique())

    booster = train_games_model(feats, train_seasons)
    rows = training_rows(feats, train_seasons)

    rmse = float(booster.best_score)
    base = baseline_rmse(rows)
    naive = full_season_rmse(rows)

    print(f"rows {len(rows)}  best_iter {booster.best_iteration}")
    print(f"valid rmse       {rmse:6.3f} games")
    print(f"lag1 baseline    {base:6.3f}   {(base - rmse) / base * 100:+.1f}%")
    print(f"assume-82 (old)  {naive:6.3f}   {(naive - rmse) / naive * 100:+.1f}%")

    gain = booster.get_score(importance_type="gain")
    print("\ntop features by gain")
    for name, score in sorted(gain.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:26} {score:8.1f}")

    print(f"\nsaved to {MODEL_PATH}")
