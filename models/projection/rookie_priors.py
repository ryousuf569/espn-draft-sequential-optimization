import pandas as pd
from config import CATEGORIES, COUNTING_CATEGORIES, RATIO_CATEGORIES, DRAFT_TIERS, TIER_NAMES, UNDRAFTED_TIER, DB_PATH, FIRST_SEASON_YEAR
from data.sqlite_helpers import connect

FIRST_PRIOR_DRAFT_YEAR = FIRST_SEASON_YEAR
MIN_MIN_FOR_RATE = 100.0
MIN_CELL_N = 10
UNKNOWN_POSITION = "UNK"
POSITION_MAP = {
    "Guard": "G", "G": "G",
    "Forward": "F", "F": "F",
    "Center": "C", "C": "C",
    "Guard-Forward": "G", "G-F": "G",
    "Forward-Guard": "F", "F-G": "F",
    "Forward-Center": "F", "F-C": "F",
    "Center-Forward": "C", "C-F": "C",
}

DRAFTED_SQL = """
SELECT d.player_id,
       d.overall_pick,
       p.position,
       COALESCE(r.gp, 0)        AS gp,
       COALESCE(r.total_min, 0) AS total_min,
       r.pts, r.reb, r.ast, r.stl, r.blk, r.tov, r.fg3m,
       r.fgm, r.fga, r.ftm, r.fta
FROM nba_draft d
LEFT JOIN rookie_outcomes r ON r.player_id = d.player_id
LEFT JOIN players p         ON p.player_id = d.player_id
WHERE d.draft_year >= ? AND d.draft_year < ?
"""


UNDRAFTED_SQL = """
SELECT ro.player_id,
       NULL AS overall_pick,
       COALESCE(p.position, ro.position) AS position,
       COALESCE(s.gp, 0)         AS gp,
       COALESCE(s.mpg * s.gp, 0) AS total_min,
       s.pts, s.reb, s.ast, s.stl, s.blk, s.tov, s.fg3m,
       s.fgm, s.fga, s.ftm, s.fta
FROM rosters ro
LEFT JOIN nba_draft d    ON d.player_id = ro.player_id
LEFT JOIN players p      ON p.player_id = ro.player_id
LEFT JOIN season_stats s ON s.player_id = ro.player_id AND s.season = ro.season
WHERE ro.exp = 'R'
  AND d.player_id IS NULL
  AND CAST(substr(ro.season, 1, 4) AS INTEGER) >= ?
  AND CAST(substr(ro.season, 1, 4) AS INTEGER) < ?
GROUP BY ro.player_id
"""

def norm_pos(pos):
    if pos is None or pd.isna(pos):
        return UNKNOWN_POSITION
    return POSITION_MAP.get(str(pos).strip(), UNKNOWN_POSITION)

def tier_for_pick(overall_pick):
    if overall_pick is None or pd.isna(overall_pick):
        return UNDRAFTED_TIER

    for name, lo, hi in DRAFT_TIERS:
        if lo <= overall_pick <= hi:
            return name

    return UNDRAFTED_TIER

def load_rookies(conn, as_of_year, first_year=FIRST_PRIOR_DRAFT_YEAR):

    drafted = pd.read_sql_query(DRAFTED_SQL, conn, params=(first_year, as_of_year))
    undrafted = pd.read_sql_query(UNDRAFTED_SQL, conn, params=(first_year, as_of_year))
    df = pd.concat([drafted, undrafted], ignore_index=True)
    # drafted rows come first, so a player in both frames stays drafted
    df = df.drop_duplicates(subset="player_id", keep="first")

    df["position"] = df["position"].map(norm_pos)
    df["draft_tier"] = df["overall_pick"].map(tier_for_pick)
    df[['total_min', 'gp', 'pts',
       'reb', 'ast', 'stl', 'blk', 'tov', 'fg3m', 'fgm', 'fga', 'ftm', 'fta']] = df[['total_min', 'gp', 'pts',
       'reb', 'ast', 'stl', 'blk', 'tov', 'fg3m', 'fgm', 'fga', 'ftm', 'fta']].fillna(0)
    df['gp'] = df['gp'].astype(int)

    return df


# playing time counts everyone in the cell, never-played included, because a
# zero-minute rookie is a real outcome for his slot and not a missing one
def playing_time(group):
    played = group["total_min"] > 0
    n = len(group)

    minutes_played = group.loc[played, "total_min"].sum()
    games_played = group.loc[played, "gp"].sum()

    return pd.Series(
        {
            "n_players": n,
            "n_played": int(played.sum()),
            # the share who never saw the floor: the bias this module corrects for
            "never_played_rate": float((~played).mean()) if n else float("nan"),
            # minutes per game once he is actually in the rotation
            "mpg_if_played": minutes_played / games_played if games_played else float("nan"),
            # what a random pick at this slot plays per team game, zeros folded in.
            # This is the one that multiplies the rates downstream.
            "mpg": minutes_played / n / 82.0 if n else float("nan"),
        }
    )


# rates are minute-weighted: total stat over total minutes, not the mean of each
# player's own rate, or 12 garbage-time minutes would count like a full season
def rates(group):
    rated = group[group["total_min"] >= MIN_MIN_FOR_RATE]
    minutes = rated["total_min"].sum()

    out = {"n_rate": len(rated)}

    for cat in COUNTING_CATEGORIES:
        out[cat] = rated[cat].sum() / minutes if minutes else float("nan")

    # a ratio is made over attempted, never the average of per-player ratios
    for cat, (makes, attempts) in RATIO_CATEGORIES.items():
        attempted = rated[attempts].sum()
        out[cat] = rated[makes].sum() / attempted if attempted else float("nan")

    return pd.Series(out)


def shrinkage_weight(career_minutes, k):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    n = max(0.0, float(career_minutes))
    return n / (n + k)


def _summarize(df, keys):
    grouped = df.groupby(keys, dropna=False)
    stats = grouped.apply(playing_time, include_groups=False)
    rate = grouped.apply(rates, include_groups=False)
    return stats.join(rate).reset_index()


# a thin cell borrows its rates from the whole tier, and says so
def _fill_thin_cells(cells, tier_level):
    tiers = tier_level.set_index("draft_tier")
    filled = []

    for row in cells.to_dict("records"):
        thin = row["n_rate"] < MIN_CELL_N and row["draft_tier"] in tiers.index
        if thin:
            fallback = tiers.loc[row["draft_tier"]]
            for cat in CATEGORIES:
                row[cat] = fallback[cat]
        row["borrowed_rates"] = thin
        filled.append(row)

    return pd.DataFrame(filled)


# one row per (position, draft_tier): the per-minute rate for each category,
# plus the mpg a random pick at that slot actually plays
def compute_draft_tier_priors(conn, as_of_season):
    df = load_rookies(conn, int(as_of_season[:4]))

    tier_level = _summarize(df, ["draft_tier"])
    priors = _fill_thin_cells(_summarize(df, ["position", "draft_tier"]), tier_level)

    cols = ["position", "draft_tier", "n_players", "n_played", "n_rate",
            "never_played_rate", "mpg", "mpg_if_played", *CATEGORIES,
            "borrowed_rates"]
    return priors[cols].sort_values(["draft_tier", "position"]).reset_index(drop=True)


# simple tests to catch bugs
def self_test(conn, as_of_year=2026):
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("shrinkage_weight")
    check("w = 0.5 at n = k", shrinkage_weight(500, 500) == 0.5)
    check("no minutes means all prior", shrinkage_weight(0, 500) == 0.0)
    check("w rises with minutes", shrinkage_weight(2000, 500) > shrinkage_weight(200, 500))
    check("w stays below 1", shrinkage_weight(10**9, 500) < 1.0)
    check("smaller k trusts the player more", shrinkage_weight(300, 150) > shrinkage_weight(300, 500))
    try:
        shrinkage_weight(100, 0)
        check("k <= 0 rejected", False)
    except ValueError:
        check("k <= 0 rejected", True)

    print("norm_pos / tier_for_pick")
    check("both vocabularies fold to G", norm_pos("Guard") == "G" and norm_pos("G-F") == "G")
    check("unknown position is UNK", norm_pos(None) == UNKNOWN_POSITION and norm_pos("Wing") == UNKNOWN_POSITION)
    edges = [(1, "lottery"), (14, "lottery"), (15, "late_first"), (30, "late_first"),
             (31, "second_round"), (60, "second_round")]
    check("tier boundaries", all(tier_for_pick(p) == t for p, t in edges))
    # the real-world case: pandas upcasts a NULL pick column to float64
    check("nan pick is undrafted", tier_for_pick(float("nan")) == UNDRAFTED_TIER)

    print("load_rookies")
    df = load_rookies(conn, as_of_year)
    check("rows returned", len(df) > 0, f"n={len(df)}")
    check("no duplicate players", df["player_id"].duplicated().sum() == 0)
    check("all four tiers present", set(df["draft_tier"]) == set(TIER_NAMES),
          str(sorted(set(df["draft_tier"]))))
    # the LEFT JOIN is the whole point: lose it and the zeros disappear
    never = int((df["total_min"] == 0).sum())
    check("never-played kept as zeros", never > 0, f"n={never}")
    check("every stat column filled", not df[list(COUNTING_CATEGORIES)].isna().any().any())

    print("playing_time / rates")
    pt = df.groupby("draft_tier").apply(playing_time, include_groups=False)
    rt = df.groupby("draft_tier").apply(rates, include_groups=False)

    check("mpg <= mpg_if_played", bool((pt["mpg"] <= pt["mpg_if_played"]).all()),
          "zeros must drag the average down")
    check("counts add up", bool((pt["n_played"] <= pt["n_players"]).all()))
    check("missingness rises with tier",
          pt.loc["lottery", "never_played_rate"] < pt.loc["second_round", "never_played_rate"],
          f"{pt.loc['lottery', 'never_played_rate']:.3f} < {pt.loc['second_round', 'never_played_rate']:.3f}")
    check("rate floor drops players", bool((rt["n_rate"] <= pt["n_players"]).all()))
    check("no rate is NaN", not rt[list(CATEGORIES)].isna().any().any())
    check("percentages in range",
          bool(rt[list(RATIO_CATEGORIES)].ge(0).all().all() and rt[list(RATIO_CATEGORIES)].le(1).all().all()))
    check("lottery outscores second round per minute",
          rt.loc["lottery", "pts"] > rt.loc["second_round", "pts"],
          f"{rt.loc['lottery', 'pts']:.4f} > {rt.loc['second_round', 'pts']:.4f}")
    # a mean-of-ratios would sit near the unweighted average instead
    weighted = df[df.total_min >= MIN_MIN_FOR_RATE]
    check("fg_pct is minute-weighted, not a mean of ratios",
          abs(rt.loc["lottery", "fg_pct"] - (weighted[weighted.overall_pick <= 14].fgm.sum()
              / weighted[weighted.overall_pick <= 14].fga.sum())) < 1e-9)

    print(f"\n{'all checks passed' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
    return not failures


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    conn = connect()
    ok = self_test(conn)

    raise SystemExit(0 if ok else 1)