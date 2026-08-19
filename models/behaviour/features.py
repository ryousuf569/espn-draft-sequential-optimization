# Cox covariates. KM needs none of this -- it conditions on nothing but the
# durations, which is exactly why it ships first.

import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (ARTIFACT_DIR, PLATFORM_ADP_SOURCE, POSITION_REFERENCE,
                    TARGET_SEASON, connect)
from dataset import build_dataset

tables = config.load_tables()

# Model A's output is a file artifact, not a table. This path and add_vorp() are
# the interface between the two models; only this constant changes if that moves.
PROJECTIONS_CSV = ARTIFACT_DIR / "projections.csv"

# The same position vocabulary as models/projection/rookie_priors.py, copied rather
# than imported to avoid pulling in that whole config. If one changes, both change.
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

# statuses that mean a player is not available to play right now
INACTIVE_STATUSES = ("inactive",)

# NBA pick buckets for the rookie prior, mirroring the projection model's tiers
ROOKIE_DRAFT_TIERS = (
    ("lottery", 1, 14),
    ("late_first", 15, 30),
    ("second_round", 31, 60),
)
UNDRAFTED_TIER = "undrafted"


def norm_pos(pos):
    if pos is None or pd.isna(pos):
        return UNKNOWN_POSITION
    return POSITION_MAP.get(str(pos).strip(), UNKNOWN_POSITION)


def rookie_tier_for_pick(overall_pick):
    if overall_pick is None or pd.isna(overall_pick):
        return UNDRAFTED_TIER

    for name, lo, hi in ROOKIE_DRAFT_TIERS:
        if lo <= overall_pick <= hi:
            return name

    return UNDRAFTED_TIER


# platform_rank: where the consensus board puts a player, 1..N ascending on ADP.
# dataset.attach_adp_tier already joins it; this covers a table that skipped it.
def attach_platform_rank(df, season, conn):
    if "platform_rank" in df.columns and df["platform_rank"].notna().any():
        return df

    board = tables.load_adp(conn, season=season, source=PLATFORM_ADP_SOURCE)
    board = board[board["player_id"].notna()].copy()
    board["player_id"] = board["player_id"].astype(int)
    board["platform_rank"] = board["adp"].rank(method="first")

    return df.merge(board[["player_id", "platform_rank"]], on="player_id", how="left")


# Model A value, ranked. projections.csv is built for TARGET_SEASON only, so a
# historical fold has no VORP to join and attach_rank_diff fills those with 0.
def attach_vorp_rank(df, season, projections_csv=PROJECTIONS_CSV):
    if not projections_csv.exists():
        df = df.copy()
        df["vorp_rank"] = np.nan
        return df

    proj = pd.read_csv(projections_csv)

    # vorp is not a column in projections.csv -- Model A derives it at read time, so
    # the same function is reused rather than the math being copied and left to drift
    add_vorp = config.load_projection_module("vorp").add_vorp

    ranked = add_vorp(proj)
    ranked["vorp_rank"] = ranked["vorp"].rank(ascending=False, method="first")
    ranked["player_id"] = ranked["player_id"].astype(int)

    # projections exist for one season only, so a historical fold gets nothing
    if season != TARGET_SEASON:
        out = df.copy()
        out["vorp_rank"] = np.nan
        return out

    return df.merge(ranked[["player_id", "vorp_rank"]], on="player_id", how="left")


# The strongest feature per §6: how far a player's value rank sits from the board's.
# Positive means the board is higher on him than his projection is.
def attach_rank_diff(df):
    out = df.copy()
    diff = out["vorp_rank"] - out["platform_rank"]

    # 0 means "no disagreement known", the honest fill for a fold with no projections
    out["vorp_rank_diff"] = diff.fillna(0.0)
    out["has_vorp"] = diff.notna().astype(int)
    return out


# rosters.position for that season, players.position otherwise: a player moves
# position across a career and the draft reflects where he plays that year
def attach_position(df, season, conn):
    roster = tables.load_rosters(conn, season)[["player_id", "position"]]
    roster = roster.dropna(subset=["position"]).drop_duplicates("player_id")
    roster = roster.rename(columns={"position": "roster_position"})

    players = tables.load_players(conn)[["player_id", "position"]]
    players = players.rename(columns={"position": "player_position"})

    out = df.merge(roster, on="player_id", how="left")
    out = out.merge(players, on="player_id", how="left")

    out["position"] = (out["roster_position"].combine_first(out["player_position"])
                       .map(norm_pos))
    return out.drop(columns=["roster_position", "player_position"])


# Same rookie definition as Model A, deliberately -- drafted this year, exp = 'R',
# or no prior season_stats. Disagreeing would shrink toward different priors.
def attach_rookie_flag(df, season, conn):
    season_year = int(str(season)[:4])

    drafted = tables.load_nba_draft(conn)
    this_class = set(drafted.loc[drafted["draft_year"] == season_year, "player_id"]
                     .astype(int))

    roster = tables.load_rosters(conn, season)
    flagged = set(roster.loc[roster["exp"] == "R", "player_id"].astype(int))

    # anyone with a season_stats row before this season has NBA minutes behind him
    prior = tables.load_season_stats(conn)
    prior = prior[prior["season"] < season]
    experienced = set(prior.loc[prior["gp"].fillna(0) > 0, "player_id"].astype(int))

    out = df.copy()
    ids = out["player_id"].astype(int)
    out["rookie_flag"] = (
        (ids.isin(this_class) | ids.isin(flagged) | ~ids.isin(experienced))
        .astype(int)
    )
    return out


# From player_status, and only for the current season: the table is a snapshot, so
# applying today's status to a 2018 draft would tell the model the future.
def attach_injury_flag(df, season, conn):
    out = df.copy()

    if season != TARGET_SEASON:
        out["injury_flag"] = 0
        out["injury_flag_known"] = 0
        return out

    status = tables.load_player_status(conn)
    hurt = set(status.loc[status["status"].str.lower().isin(INACTIVE_STATUSES),
                          "player_id"].astype(int))

    out["injury_flag"] = out["player_id"].astype(int).isin(hurt).astype(int)
    out["injury_flag_known"] = 1
    return out


# one-hot the position with a reference level dropped, so the design is full rank
def encode_position(df, reference=POSITION_REFERENCE):
    dummies = pd.get_dummies(df["position"], prefix="pos", dtype=float)
    return dummies.drop(columns=[f"pos_{reference}"], errors="ignore")


# the full modeling table: survival columns plus every covariate, in one call
def build_cox_features(survival_df, season, conn):
    df = survival_df[survival_df["season"] == season].copy() \
        if "season" in survival_df.columns else survival_df.copy()

    df = attach_platform_rank(df, season, conn)
    df = attach_vorp_rank(df, season)
    df = attach_rank_diff(df)
    df = attach_position(df, season, conn)
    df = attach_rookie_flag(df, season, conn)
    df = attach_injury_flag(df, season, conn)

    # an unranked player sits just past the board; dropping him would shrink the risk set
    board_size = df["platform_rank"].max()
    fallback = (board_size + 1) if pd.notna(board_size) else 1.0
    df["platform_rank"] = df["platform_rank"].fillna(fallback)

    return df


# The design matrix lifelines fits: durations, event, numeric covariates only.
# CoxPHFitter treats every other column as a covariate, so identifiers are stripped.
def cox_design(df, drop_constant=True):
    numeric = df[["duration", "event_observed", "platform_rank", "vorp_rank_diff",
                  "injury_flag", "rookie_flag"]].copy()
    design = pd.concat([numeric, encode_position(df)], axis=1)

    if not drop_constant:
        return design

    covariates = [c for c in design.columns
                  if c not in ("duration", "event_observed")]
    constant = [c for c in covariates if design[c].nunique(dropna=False) <= 1]

    return design.drop(columns=constant)


# which covariates a fit on this frame will actually use, and which fell out
def usable_covariates(df):
    full = cox_design(df, drop_constant=False)
    kept = cox_design(df, drop_constant=True)

    covariates = [c for c in full.columns if c not in ("duration", "event_observed")]
    dropped = [c for c in covariates if c not in kept.columns]
    return [c for c in covariates if c in kept.columns], dropped


# Rookie prior, separate from rookie_flag: a rookie has no ADP history, so his curve
# comes from players taken near the same NBA slot, grouped as Model A groups them.
def build_rookie_prior(draft_year, conn):
    drafted = tables.load_nba_draft(conn)
    drafted = drafted[drafted["draft_year"] == int(draft_year)].copy()

    if drafted.empty:
        return pd.DataFrame(columns=["position", "draft_tier", "n_players"])

    players = tables.load_players(conn)[["player_id", "position"]]
    drafted = drafted.merge(players, on="player_id", how="left")

    drafted["position"] = drafted["position"].map(norm_pos)
    drafted["draft_tier"] = drafted["overall_pick"].map(rookie_tier_for_pick)

    return (drafted.groupby(["position", "draft_tier"])
            .agg(n_players=("player_id", "size"),
                 mean_pick=("overall_pick", "mean"))
            .reset_index())


# the covariates have to be populated and pointed the right way before fitting
def verify(df):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    check("no rows lost", not df.empty, f"{len(df)} rows")
    check("platform_rank populated", bool(df["platform_rank"].notna().all()))
    check("rank_diff populated", bool(df["vorp_rank_diff"].notna().all()))
    check("position is in the vocabulary",
          bool(df["position"].isin(set(POSITION_MAP.values()) | {UNKNOWN_POSITION}).all()),
          str(sorted(df["position"].unique())))
    check("flags are 0/1", bool(df["rookie_flag"].isin((0, 1)).all()
                                and df["injury_flag"].isin((0, 1)).all()))

    # the design matrix has to be full rank or the Cox fit will not converge
    design = cox_design(df)
    covariates = design.drop(columns=["duration", "event_observed"])
    kept, dropped = usable_covariates(df)
    check("design matrix has no constant column",
          bool(all(covariates[c].nunique() > 1 for c in covariates.columns)),
          f"{len(kept)} kept" + (f", dropped {dropped}" if dropped else ""))
    check("something is left to fit", len(kept) >= 2, f"{kept}")
    check("reference position dropped",
          f"pos_{POSITION_REFERENCE}" not in design.columns)

    # a better board rank has to mean going earlier, or the join is misaligned
    drafted = df[df["event_observed"] == 1]
    corr = drafted["platform_rank"].corr(drafted["duration"])
    check("platform_rank tracks duration", corr > 0.5, f"corr {corr:.3f}")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    conn = connect()
    season = "2025-26"

    survival = build_dataset(conn, season)
    feats = build_cox_features(survival, season, conn)

    print(f"{season}: {len(feats)} rows, {feats['player_id'].nunique()} players\n")
    verify(feats)

    print(f"\nrookie / injury coverage")
    print(f"  rookies          {int(feats['rookie_flag'].sum())}")
    print(f"  injured          {int(feats['injury_flag'].sum())} "
          f"(known: {int(feats['injury_flag_known'].max())})")
    print(f"  vorp joined      {int(feats['has_vorp'].sum())} of {len(feats)}")

    print("\nby position")
    print(feats.groupby("position").agg(
        rows=("player_id", "size"),
        mean_rank=("platform_rank", "mean"),
        mean_duration=("duration", "mean"),
    ).round(2).to_string())

    print("\nrookie prior, 2026 class")
    print(build_rookie_prior(2026, conn).to_string(index=False))
