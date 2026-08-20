from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# config first so the repo root lands on sys.path before any data.* import
from config import CATEGORIES, DEFAULT_SHRINKAGE_K, SHRINKAGE_K, TARGET_SEASON
from data.sqlite_helpers import connect
from feature_minutes import GAMES_IN_SEASON, build_minutes_features
from feature_rates import build_rate_features
from rookie_priors import compute_draft_tier_priors, norm_pos, shrinkage_weight, tier_for_pick
import train_games
import train_minutes
import train_rates

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
OUT_CSV = ARTIFACT_DIR / "projections.csv"

# minutes gets its own k because it is measured in minutes per game, not in the
# per-minute units the rate categories share
MINUTES_SHRINKAGE_K = 500.0

# games is measured in games, so it needs its own k too
GAMES_SHRINKAGE_K = 500.0

# who to project: everyone on an NBA roster for the season being drafted
POOL_SQL = """
SELECT DISTINCT ro.player_id,
       ro.team,
       COALESCE(p.position, ro.position) AS position,
       ro.exp,
       d.overall_pick
FROM rosters ro
LEFT JOIN players p   ON p.player_id = ro.player_id
LEFT JOIN nba_draft d ON d.player_id = ro.player_id
WHERE ro.season = ?
"""


def model_path(target):
    if target == "mpg":
        return ARTIFACT_DIR / "minutes.json"
    if target == "gp":
        return ARTIFACT_DIR / "games.json"
    return ARTIFACT_DIR / f"rate_{target}.json"


def load_booster(target):
    booster = xgb.Booster()
    booster.load_model(model_path(target))
    return booster


# Blend a model's raw output toward the tier prior at w = n / (n + k). A rookie
# has career_minutes = 0, so w = 0 and the prior wins outright -- which is the
# point: his lag features were NaN, and whatever the model made of that is not
# worth trusting.
def apply_shrinkage(raw_rate_hat, position, draft_tier, career_minutes,
                    tier_priors, k, column=None):
    column = column or "rate"
    prior = lookup_prior(tier_priors, position, draft_tier, column)

    if prior is None or pd.isna(prior):
        return raw_rate_hat

    if raw_rate_hat is None or pd.isna(raw_rate_hat):
        return prior

    w = shrinkage_weight(career_minutes, k)
    return w * raw_rate_hat + (1.0 - w) * prior


# the (position, tier) cell, falling back to the tier when the cell is missing
def lookup_prior(tier_priors, position, draft_tier, column):
    cell = tier_priors[
        (tier_priors["position"] == position) & (tier_priors["draft_tier"] == draft_tier)
    ]
    if not cell.empty:
        return float(cell.iloc[0][column])

    tier = tier_priors[tier_priors["draft_tier"] == draft_tier]
    if not tier.empty:
        # weight by players so a thin cell does not dominate the tier average
        return float(np.average(tier[column], weights=tier["n_players"]))

    return None


# everyone on a roster for the season being drafted, with tier and position set
def load_pool(conn, as_of_season):
    pool = pd.read_sql_query(POOL_SQL, conn, params=(as_of_season,))
    pool = pool.drop_duplicates("player_id", keep="last")
    pool["position"] = pool["position"].map(norm_pos)
    pool["draft_tier"] = pool["overall_pick"].map(tier_for_pick)
    return pool


# the most recent feature row per player, which is what a projection extends
def latest_features(feature_df):
    return (feature_df.sort_values("season")
            .drop_duplicates("player_id", keep="last")
            .set_index("player_id"))


# raw model output for one target, over whatever rows are handed in
def predict_raw(rows, target, booster):
    if rows.empty:
        return pd.Series(dtype=float)

    if target == "mpg":
        X = pd.concat([rows[train_minutes.feature_columns(rows)],
                       train_minutes.encode_categoricals(rows)], axis=1)
    elif target == "gp":
        X = pd.concat([rows[train_games.feature_columns(rows)],
                       train_games.encode_categoricals(rows)], axis=1)
    else:
        X = pd.concat([rows[train_rates.feature_columns(rows)],
                       train_rates.encode_tiers(rows)], axis=1)

    dm = xgb.DMatrix(X.reindex(columns=booster.feature_names, fill_value=0.0),
                     feature_names=booster.feature_names)
    return pd.Series(booster.predict(dm), index=rows.index)


# Projections for the season being drafted. Shrinkage happens here and only
# here: everything upstream produces raw numbers, and this is where a player's
# own history gets weighed against what his draft slot says about him.
def project_season(conn, as_of_season=TARGET_SEASON, games=GAMES_IN_SEASON):
    pool = load_pool(conn, as_of_season)
    tier_priors = compute_draft_tier_priors(conn, as_of_season)

    rate_feats = latest_features(build_rate_features(conn, as_of_season))
    min_feats = latest_features(build_minutes_features(conn, as_of_season))

    # a player with no feature row is a rookie by definition: nothing to lag
    rate_rows = rate_feats.reindex(pool["player_id"]).reset_index(drop=True)
    min_rows = min_feats.reindex(pool["player_id"]).reset_index(drop=True)
    known = min_rows["career_min"].notna()

    out = pool[["player_id", "team", "position", "draft_tier"]].copy()
    out["career_min"] = min_rows["career_min"].fillna(0.0).to_numpy()
    out["is_rookie"] = (~known).astype(int).to_numpy()

    # minutes first, since every counting stat is a rate multiplied by them
    raw_mpg = pd.Series(np.nan, index=min_rows.index)
    if known.any():
        raw_mpg.loc[known] = predict_raw(min_rows[known], "mpg", load_booster("mpg"))

    out["mpg"] = [
        apply_shrinkage(raw, pos, tier, mins, tier_priors, MINUTES_SHRINKAGE_K,
                        column="mpg")
        for raw, pos, tier, mins in zip(raw_mpg, out["position"], out["draft_tier"],
                                        out["career_min"])
    ]
    # Projected games, not a flat 82. Assuming every player plays every game
    # inflated every season total by whatever a player missed -- league mean GP is
    # ~46 -- and it was what put four players who logged zero games near the top of
    # the 2025-26 board. Rookies fall back to the tier prior, same as minutes.
    raw_gp = pd.Series(np.nan, index=min_rows.index)
    if known.any():
        raw_gp.loc[known] = predict_raw(min_rows[known], "gp", load_booster("gp"))

    out["gp"] = np.clip([
        apply_shrinkage(raw, pos, tier, mins, tier_priors, GAMES_SHRINKAGE_K,
                        column="gp_prior")
        for raw, pos, tier, mins in zip(raw_gp, out["position"], out["draft_tier"],
                                        out["career_min"])
    ], 0.0, float(games))

    out["total_min"] = out["mpg"] * out["gp"]

    for cat in CATEGORIES:
        raw = pd.Series(np.nan, index=rate_rows.index)
        if known.any():
            raw.loc[known] = predict_raw(rate_rows[known], cat, load_booster(cat))

        k = SHRINKAGE_K.get(cat, DEFAULT_SHRINKAGE_K)
        blended = [
            apply_shrinkage(r, pos, tier, mins, tier_priors, k, column=cat)
            for r, pos, tier, mins in zip(raw, out["position"], out["draft_tier"],
                                          out["career_min"])
        ]

        # a regressor has no idea a rate cannot go below zero, and for a center
        # who never shoots threes it will happily predict a small negative
        out[cat + "_rate"] = np.clip(blended, 0.0, None)

        # a season total is the rate carried across the minutes it applies to.
        # The two percentages are already ratios, so they stay as they are.
        if cat in ("fg_pct", "ft_pct"):
            out[cat] = out[cat + "_rate"]
        else:
            out[cat] = out[cat + "_rate"] * out["total_min"]

    return out.sort_values("mpg", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    conn = connect()
    proj = project_season(conn)
    priors = compute_draft_tier_priors(conn, TARGET_SEASON)

    print(f"{TARGET_SEASON}: {len(proj)} players, {int(proj.is_rookie.sum())} rookies\n")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    proj.to_csv(OUT_CSV, index=False)

    cols = ["player_id", "team", "position", "draft_tier", "is_rookie", "mpg",
            "pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "fg_pct", "ft_pct"]
    print()
    print(proj[cols].head(12).round(3).to_string(index=False))
    print(f"\nwrote {OUT_CSV}")
