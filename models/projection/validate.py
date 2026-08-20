from pathlib import Path

import pandas as pd
import xgboost as xgb

# config first so the repo root lands on sys.path before any data.* import
from config import CATEGORIES
from data.sqlite_helpers import connect
from feature_minutes import build_minutes_features
from feature_rates import build_rate_features
import train_minutes
import train_rates

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
OUT_CSV = ARTIFACT_DIR / "validation.csv"

# minutes plus the nine categories, in the order the columns come out
TARGETS = ("mpg",) + CATEGORIES

# the season being scored, and the one no model is allowed to see
TEST_SEASON = "2025-26"


# every season before the test one, in order
def train_seasons_before(feature_df, test_season):
    seasons = feature_df.loc[feature_df["has_history"], "season"].unique()
    return sorted(s for s in seasons if s < test_season)


# refit on the training seasons only. The saved artifacts in artifacts/ were fit
# through 2025-26, so scoring with them would be scoring a season the model
# already saw -- this fits fresh boosters that have never met the test season.
def fit_booster(feature_df, target, train_seasons):
    if target == "mpg":
        return train_minutes.train_minutes_model(feature_df, train_seasons, save=False)
    return train_rates.train_rate_model(feature_df, target, train_seasons, save=False)


# the design matrix for one slice, built by whichever trainer owns this target
def design(feature_df, target, seasons):
    if target == "mpg":
        rows = train_minutes.training_rows(feature_df, seasons)
        X = pd.concat([rows[train_minutes.feature_columns(rows)],
                       train_minutes.encode_categoricals(rows)], axis=1)
        return rows, X, rows[train_minutes.TARGET]

    rows = train_rates.training_rows(feature_df, target, seasons)
    X = pd.concat([rows[train_rates.feature_columns(rows)],
                   train_rates.encode_tiers(rows)], axis=1)
    return rows, X, rows[target + "_rate"]


# a season can be missing a dummy level, so line the test columns up with train
def align(X, columns):
    return X.reindex(columns=columns, fill_value=0.0)


# fit on the past, score the test season, return three columns
def score_target(feature_df, target, test_season, train_seasons):
    booster = fit_booster(feature_df, target, train_seasons)

    _, X_train, _ = design(feature_df, target, train_seasons)
    rows, X_test, actual = design(feature_df, target, [test_season])

    dm = xgb.DMatrix(align(X_test, X_train.columns),
                     feature_names=list(X_train.columns))
    pred = booster.predict(dm, iteration_range=(0, booster.best_iteration + 1))

    return pd.DataFrame({
        "player_id": rows["player_id"].to_numpy(),
        f"{target}_pred": pred,
        f"{target}_actual": actual.to_numpy(),
        f"{target}_error": pred - actual.to_numpy(),
    })


# one row per player, three columns per target, all out-of-sample
def validate(conn, test_season=TEST_SEASON, out_csv=OUT_CSV):
    rate_feats = build_rate_features(conn, "2026-27")
    min_feats = build_minutes_features(conn, "2026-27")

    out = None
    for target in TARGETS:
        source = min_feats if target == "mpg" else rate_feats
        seasons = train_seasons_before(source, test_season)
        part = score_target(source, target, test_season, seasons)
        out = part if out is None else out.merge(part, on="player_id", how="outer")

    out = out.sort_values("player_id").reset_index(drop=True)

    if out_csv:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    conn = connect()
    df = validate(conn)

    rate_feats = build_rate_features(conn, "2026-27")
    seasons = train_seasons_before(rate_feats, TEST_SEASON)

    print(f"trained {seasons[0]}..{seasons[-1]} ({len(seasons)} seasons), "
          f"scored {TEST_SEASON}: {len(df)} players\n")

    print(f"\n{'target':10} {'n':>5} {'MAE':>10} {'bias':>11} {'mean_actual':>12}")
    for target in TARGETS:
        err = df[f"{target}_error"].dropna()
        actual = df[f"{target}_actual"].dropna()
        print(f"{target:10} {len(err):5} {err.abs().mean():10.5f} "
              f"{err.mean():+11.5f} {actual.mean():12.5f}")

    print(f"\nwrote {OUT_CSV}")
