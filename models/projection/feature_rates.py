import numpy as np
import pandas as pd
from config import CATEGORIES, COUNTING_CATEGORIES, RATIO_CATEGORIES
from data.sqlite_helpers import connect
from rookie_priors import tier_for_pick

# players peak around here, so age is centered on it
PEAK_AGE = 27.0

# a season starts in October, close enough for an age in years
SEASON_START_MONTH_DAY = "-10-01"

# trailing windows for the lag features, in seasons
LAG_WINDOWS = (1, 2, 3)

# roughly the share of free throws that end a possession
FTA_POSSESSION_WEIGHT = 0.44

# season labels sort the same lexically as chronologically, so SQLite can cut
# here. LEFT JOIN on nba_draft so undrafted players keep their rows.
STATS_SQL = """
SELECT s.player_id,
       s.season,
       s.gp,
       s.mpg,
       s.pts, s.reb, s.ast, s.stl, s.blk, s.tov, s.fg3m,
       s.fgm, s.fga, s.ftm, s.fta,
       p.birth_date,
       d.overall_pick
FROM season_stats s
JOIN players p     ON p.player_id = s.player_id
LEFT JOIN nba_draft d ON d.player_id = s.player_id
WHERE s.season < ?
  AND s.gp > 0
"""


# "2015-16" -> 2015
def season_start_year(season):
    return int(str(season)[:4])


# sort matters: a shift on an unsorted frame pairs the wrong seasons together
def load_stats(conn, as_of_season):
    df = pd.read_sql_query(STATS_SQL, conn, params=(as_of_season,))
    return df.sort_values(["player_id", "season"]).reset_index(drop=True)


# turn totals into rates first, so what gets lagged later is a rate
def add_rates(df):
    df = df.copy()
    df["total_min"] = df["mpg"] * df["gp"]

    minutes = df["total_min"].replace(0, np.nan)
    for cat in COUNTING_CATEGORIES:
        df[cat + "_rate"] = df[cat] / minutes

    # no attempts gives NaN, not a fake zero
    for cat, (makes, attempts) in RATIO_CATEGORIES.items():
        df[cat + "_rate"] = df[makes] / df[attempts].replace(0, np.nan)

    # usage without team possessions: how much offense ran through this player
    df["usage_rate"] = (df["fga"] + FTA_POSSESSION_WEIGHT * df["fta"] + df["tov"]) / minutes

    return df


# shift first, then roll - rolling first would average a season into itself
def add_lags(df, windows=LAG_WINDOWS):
    df = df.copy()
    rate_cols = [c + "_rate" for c in CATEGORIES] + ["usage_rate", "mpg", "total_min"]
    grouped = df.groupby("player_id", sort=False)

    for col in rate_cols:
        shifted = grouped[col].shift(1)

        for w in windows:
            if w == 1:
                df[f"{col}_lag1"] = shifted
                continue

            # min_periods=w so a 3-year average needs three real seasons
            df[f"{col}_avg{w}"] = shifted.rolling(w, min_periods=w).mean()

    return df


# age at the season start, plus the squared term for the aging curve
def add_age(df):
    df = df.copy()
    born = pd.to_datetime(df["birth_date"], errors="coerce")
    season_start = pd.to_datetime(df["season"].map(season_start_year).astype(str) + SEASON_START_MONTH_DAY)

    df["age"] = (season_start - born).dt.days / 365.25
    df["age_c"] = df["age"] - PEAK_AGE
    df["age_c2"] = df["age_c"] ** 2
    return df


# career minutes before this season - the n that shrinkage weighs against k
def add_career_and_tier(df):
    df = df.copy()
    grouped = df.groupby("player_id", sort=False)

    # shift before cumsum, or a season counts toward its own prior
    df["career_min"] = grouped["total_min"].shift(1).groupby(df["player_id"]).cumsum()
    df["career_seasons"] = grouped.cumcount()

    # imported so the pick cutoffs are defined in one place
    df["draft_tier"] = df["overall_pick"].map(tier_for_pick)

    # so predict.py can route rookies on a column instead of checking NaNs
    df["has_history"] = df["career_seasons"] > 0
    return df


# one row per player-season, features from prior seasons, current rates as target
def build_rate_features(conn, as_of_season):
    df = load_stats(conn, as_of_season)
    df = add_rates(df)
    df = add_lags(df)
    df = add_age(df)
    df = add_career_and_tier(df)
    return df


# checks the features are sane, mostly that nothing leaked from the target season
def verify(df, as_of_season):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    g = df.sort_values(["player_id", "season"]).groupby("player_id")

    # the big one: lag1 must be the previous season's rate, nothing else
    manual = g["pts_rate"].shift(1)
    check("lag1 equals a manual shift",
          bool(np.isclose(df["pts_rate_lag1"], manual, equal_nan=True).all()))

    # career_min[t+1] - career_min[t] should be exactly the minutes played at t
    drift = g.apply(
        lambda d: (d["career_min"].shift(-1) - d["career_min"] - d["total_min"]).abs().max(),
        include_groups=False,
    )
    check("career_min excludes current season", np.nanmax(drift) < 1e-6)

    # a feature that matched its target too well would mean the season leaked
    corr = df["pts_rate"].corr(df["pts_rate_lag1"])
    check("lag1 correlates but is not the target", 0.3 < corr < 0.95, f"r={corr:.3f}")

    check("every season is before the cutoff", bool((df["season"] < as_of_season).all()))
    check("one row per player-season", not df.duplicated(["player_id", "season"]).any())

    # rookies have no history to summarise, so their lags are NaN by design
    rookies = ~df["has_history"]
    check("rookie rows equal player count", int(rookies.sum()) == df["player_id"].nunique())
    check("rookie lags are all NaN", bool(df.loc[rookies, "pts_rate_lag1"].isna().all()))
    check("veteran lags are filled", bool(df.loc[~rookies, "pts_rate_lag1"].notna().all()))

    check("ages are plausible", bool(df["age"].between(17, 46).all()),
          f"{df['age'].min():.1f}-{df['age'].max():.1f}")
    # dropna first: a player who never attempted a shot has a NaN percentage,
    # which is right, and NaN fails every comparison it is given
    pct = df[["fg_pct_rate", "ft_pct_rate"]].to_numpy().ravel()
    check("percentages in range", bool(((pct >= 0) & (pct <= 1))[~np.isnan(pct)].all()))

    rates_all = df[[c + "_rate" for c in CATEGORIES]].to_numpy().ravel()
    check("no negative rates", bool((rates_all[~np.isnan(rates_all)] >= 0).all()))

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    conn = connect()
    feats = build_rate_features(conn, "2026-27")

    print(f"rows {len(feats)}  players {feats.player_id.nunique()}")
    print(f"rookie rows (no history) {int((~feats.has_history).sum())}")
    ok = verify(feats, "2026-27")
    print(feats[["player_id", "season", "age", "career_min", "draft_tier",
                 "pts_rate", "pts_rate_lag1", "pts_rate_avg3"]].head(12).to_string(index=False))

    raise SystemExit(0 if ok else 1)