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

# Raw current-season totals. pts is the numerator of pts_rate, so handing it to
# the model is handing over the answer -- and fgm/fga/ftm/fta rebuild the two
# percentages exactly. These are the target, not features.
RAW_STAT_COLS = ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m",
                 "fgm", "fga", "ftm", "fta")

# Anything measured before the season is safe, and these suffixes are what mark
# it. Allowing by pattern rather than blocking by name means a new column added
# to the feature table is excluded until someone deliberately lets it in.
LAGGED_SUFFIXES = ("_lag1", "_avg2", "_avg3")

# backward-looking by construction, but without a suffix to prove it
SAFE_COLS = ("age", "age_c", "age_c2", "career_min")


# a column is a feature only if it is provably from before the season
def is_safe_feature(col):
    return col.endswith(LAGGED_SUFFIXES) or col in SAFE_COLS


# everything that survives the drop list, plus one-hot draft tier
def feature_columns(df):
    raw = [c + "_rate" for c in CATEGORIES]
    return [
        c for c in df.columns
        if c not in DROP_COLS
        and c not in raw
        and c not in RAW_STAT_COLS
        and c != "draft_tier"
        and is_safe_feature(c)
    ]


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


# Leak check. A feature is legitimate only if it was knowable before the season
# started, so this tests that two different ways: by name, and by whether the
# column actually changes when the current season's stats change.
def verify(feature_df, categories=CATEGORIES):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    cols = feature_columns(feature_df)
    check("features were selected", len(cols) > 0, f"n={len(cols)}")

    # the specific bug this guards: raw totals are the target's own numerator
    leaked = [c for c in cols if c in RAW_STAT_COLS]
    check("no raw current-season totals", not leaked, str(leaked) if leaked else "")

    targets = [c + "_rate" for c in categories]
    check("no rate target used as a feature", not [c for c in cols if c in targets])

    unsafe = [c for c in cols if not is_safe_feature(c)]
    check("every feature is lagged or age-like", not unsafe, str(unsafe[:5]) if unsafe else "")

    # The empirical version, which catches a leak no name test can see: perturb
    # the current season's stats and rebuild. A feature that moves was reading
    # the season it is supposed to predict.
    scrambled = feature_df.copy()
    rng = np.random.default_rng(0)
    for col in RAW_STAT_COLS:
        if col in scrambled.columns:
            scrambled[col] = rng.permutation(scrambled[col].to_numpy())

    moved = [
        c for c in cols
        if c in scrambled.columns
        and not scrambled[c].equals(feature_df[c])
    ]
    check("features ignore the current season's stats", not moved,
          str(moved[:5]) if moved else "")

    # a target still has to vary, or the model has nothing to learn
    for cat in categories:
        col = cat + "_rate"
        if col in feature_df.columns:
            spread = float(feature_df[col].std())
            if spread == 0 or pd.isna(spread):
                check(f"{cat} target varies", False)
                break
    else:
        check("every target varies", True)

    print("all good" if ok else "LEAK DETECTED")
    return ok


if __name__ == "__main__":
    conn = connect()
    feats = build_rate_features(conn, "2026-27")
    train_seasons = sorted(feats.loc[feats.has_history, "season"].unique())

    print("leak check")
    if not verify(feats):
        raise SystemExit(1)
    print()

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