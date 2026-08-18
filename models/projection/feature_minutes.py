import numpy as np
import pandas as pd

# config comes first: importing it puts the repo root on sys.path, which is what
# makes the data.* imports below resolve when this file is run directly
from config import TARGET_SEASON
from data.config import CURRENT_SEASON
from data.sqlite_helpers import connect
from feature_rates import PEAK_AGE, SEASON_START_MONTH_DAY, season_start_year
from rookie_priors import norm_pos, tier_for_pick

# a full season of games, for turning total minutes back into a per-game rate
GAMES_IN_SEASON = 82

# one row per player-season with the minutes actually played
MINUTES_SQL = """
SELECT s.player_id,
       s.season,
       s.gp,
       s.mpg,
       s.mpg * s.gp AS total_min,
       p.birth_date,
       d.overall_pick
FROM season_stats s
JOIN players p        ON p.player_id = s.player_id
LEFT JOIN nba_draft d ON d.player_id = s.player_id
WHERE s.season < ?
  AND s.gp > 0
"""

# team and position per player-season, deduped because a traded player appears
# on two rosters and we only want his last stop
ROSTER_SQL = """
SELECT ro.player_id,
       ro.season,
       ro.team,
       COALESCE(p.position, ro.position) AS position,
       ro.exp
FROM rosters ro
LEFT JOIN players p ON p.player_id = ro.player_id
WHERE ro.season < ?
"""

# current-state only: no season column, so this is the live snapshot and nothing
# else. Joining it to a past season would leak the future backwards.
STATUS_SQL = "SELECT player_id, status FROM player_status"


# "2015-16" -> "2016-17"
def next_season(season):
    y = season_start_year(season) + 1
    return f"{y}-{str(y + 1)[-2:]}"


# minutes played per player-season, with the roster he ended on attached
def load_minutes(conn, as_of_season):
    stats = pd.read_sql_query(MINUTES_SQL, conn, params=(as_of_season,))
    rosters = pd.read_sql_query(ROSTER_SQL, conn, params=(as_of_season,))

    # keep one team per player-season, the last one listed
    rosters = rosters.drop_duplicates(["player_id", "season"], keep="last")
    rosters["pos"] = rosters["position"].map(norm_pos)

    df = stats.merge(
        rosters[["player_id", "season", "team", "pos", "exp"]],
        on=["player_id", "season"],
        how="left",
    )
    return df.sort_values(["player_id", "season"]).reset_index(drop=True)


# Turnover, counted from the point of view of the season being predicted: who
# was on this team last year and is not back. Both the head count and the
# minutes they took with them, because losing a starter and losing a 12th man
# are not the same opening.
def team_turnover(df):
    prev = df[["player_id", "season", "team", "pos", "total_min"]].copy()
    prev["season"] = prev["season"].map(next_season)
    prev = prev.rename(columns={"total_min": "prev_min"})

    # left side is last season's roster, right side is this season's
    current = df[["player_id", "season", "team"]].assign(returned=1)
    merged = prev.merge(current, on=["player_id", "season", "team"], how="left")
    merged["returned"] = merged["returned"].fillna(0)
    merged["left"] = 1 - merged["returned"]

    by_team = merged.groupby(["season", "team"], as_index=False).agg(
        team_departures=("left", "sum"),
        team_min_vacated=("prev_min", lambda s: s[merged.loc[s.index, "left"] == 1].sum()),
        team_prev_min=("prev_min", "sum"),
    )
    # a share is comparable across teams in a way a raw count is not
    by_team["team_min_vacated_pct"] = (
        by_team["team_min_vacated"] / by_team["team_prev_min"].replace(0, np.nan)
    )

    # Same at the position group. Cells are small -- centers average about 2.4
    # players -- so only the share is kept, not the count.
    by_pos = merged.groupby(["season", "team", "pos"], as_index=False).agg(
        pos_min_vacated=("prev_min", lambda s: s[merged.loc[s.index, "left"] == 1].sum()),
        pos_prev_min=("prev_min", "sum"),
    )
    by_pos["pos_min_vacated_pct"] = (
        by_pos["pos_min_vacated"] / by_pos["pos_prev_min"].replace(0, np.nan)
    )

    return by_team, by_pos


# prior-season minutes, lagged the same way the rate features are
def add_minute_lags(df):
    df = df.copy()
    grouped = df.groupby("player_id", sort=False)

    for col in ("mpg", "total_min", "gp"):
        df[f"{col}_lag1"] = grouped[col].shift(1)

    # shift before rolling, or the season averages itself in
    shifted_mpg = grouped["mpg"].shift(1)
    df["mpg_avg2"] = shifted_mpg.rolling(2, min_periods=2).mean()
    df["mpg_avg3"] = shifted_mpg.rolling(3, min_periods=3).mean()

    # a jump or a collapse in minutes carries more signal than the level alone
    df["mpg_delta"] = grouped["mpg"].shift(1) - grouped["mpg"].shift(2)

    df["career_min"] = grouped["total_min"].shift(1).groupby(df["player_id"]).cumsum()
    df["career_seasons"] = grouped.cumcount()
    df["has_history"] = df["career_seasons"] > 0
    return df


def add_age(df):
    df = df.copy()
    born = pd.to_datetime(df["birth_date"], errors="coerce")
    start = pd.to_datetime(
        df["season"].map(season_start_year).astype(str) + SEASON_START_MONTH_DAY
    )
    df["age"] = (start - born).dt.days / 365.25
    df["age_c"] = df["age"] - PEAK_AGE
    df["age_c2"] = df["age_c"] ** 2
    return df


# Injury designation exists only as a live snapshot, so it is attached to the
# season being drafted and left NaN everywhere else. A backtest that saw today's
# status would be reading the future.
def add_status(df, conn, as_of_season):
    df = df.copy()
    df["is_inactive"] = np.nan

    if as_of_season != CURRENT_SEASON:
        return df

    status = pd.read_sql_query(STATUS_SQL, conn)
    flag = status.set_index("player_id")["status"].eq("inactive").astype(float)
    latest = df["season"] == df["season"].max()
    df.loc[latest, "is_inactive"] = df.loc[latest, "player_id"].map(flag)
    return df


# one row per player-season: prior minutes, age, turnover, status, rookie flag
def build_minutes_features(conn, as_of_season):
    df = load_minutes(conn, as_of_season)
    by_team, by_pos = team_turnover(df)

    df = add_minute_lags(df)
    df = add_age(df)

    df = df.merge(
        by_team[["season", "team", "team_departures", "team_min_vacated",
                 "team_min_vacated_pct"]],
        on=["season", "team"], how="left",
    )
    df = df.merge(
        by_pos[["season", "team", "pos", "pos_min_vacated", "pos_min_vacated_pct"]],
        on=["season", "team", "pos"], how="left",
    )

    df = add_status(df, conn, as_of_season)

    df["is_rookie"] = (~df["has_history"]).astype(int)
    df["draft_tier"] = df["overall_pick"].map(tier_for_pick)
    return df


# checks the minutes features are sane, mostly that nothing leaked
def verify(df, as_of_season):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    g = df.sort_values(["player_id", "season"]).groupby("player_id")
    check("mpg_lag1 equals a manual shift",
          bool(np.isclose(df["mpg_lag1"], g["mpg"].shift(1), equal_nan=True).all()))
    check("every season is before the cutoff", bool((df["season"] < as_of_season).all()))
    check("one row per player-season", not df.duplicated(["player_id", "season"]).any())
    check("rookie rows equal player count",
          int(df["is_rookie"].sum()) == df["player_id"].nunique())
    check("rookie minutes are NaN", bool(df.loc[df.is_rookie == 1, "mpg_lag1"].isna().all()))

    vac = df["team_min_vacated_pct"].dropna()
    check("vacated share is a share", bool(((vac >= 0) & (vac <= 1)).all()),
          f"{vac.min():.2f}-{vac.max():.2f}")
    check("turnover is mostly populated",
          df["team_departures"].notna().mean() > 0.85,
          f"{df['team_departures'].notna().mean():.1%}")

    # the leak that matters here: today's injury list must not reach a backtest
    if as_of_season != CURRENT_SEASON:
        check("no status leak into backtest", bool(df["is_inactive"].isna().all()))

    check("ages are plausible", bool(df["age"].between(17, 46).all()),
          f"{df['age'].min():.1f}-{df['age'].max():.1f}")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    conn = connect()
    feats = build_minutes_features(conn, "2026-27")

    print(f"rows {len(feats)}  players {feats.player_id.nunique()}")
    ok = verify(feats, "2026-27")
    print()
    print(feats[["player_id", "season", "team", "pos", "age", "mpg", "mpg_lag1",
                 "mpg_delta", "team_departures", "team_min_vacated_pct",
                 "pos_min_vacated_pct"]].tail(8).to_string(index=False))

    raise SystemExit(0 if ok else 1)
