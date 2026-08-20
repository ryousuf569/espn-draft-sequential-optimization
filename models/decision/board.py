# The draftable pool for one season: who is on the board, what each is worth, and
# what he did. Every policy reads from here so the value definitions cannot drift.

import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (ARTIFACT_DIR, POSITIONS, ROSTER_SLOTS, UNKNOWN_POSITION,
                    connect)
from roster import position_index

tables = config.load_behaviour_module("data")
_features = config.load_behaviour_module("features")
_vorp = config.load_projection_module("vorp")

PROJECTIONS_CSV = ARTIFACT_DIR / "projections.csv"

# the 9 ESPN roto categories and the one that hurts, taken from Model A's config
# rather than restated, so a category change reaches this file automatically
_projection_config = config.load_projection_module("config")
CATEGORIES = _projection_config.CATEGORIES
NEGATIVE_CATEGORIES = _projection_config.NEGATIVE_CATEGORIES


# Realized end-of-season production, which §8.2 scores on: not what a projection
# said, what the player did.
def load_realized(conn, season):
    sql = """
    SELECT s.player_id, s.gp, s.mpg,
           s.pts, s.reb, s.ast, s.stl, s.blk, s.tov, s.fg3m,
           s.fgm, s.fga, s.ftm, s.fta
    FROM season_stats s
    WHERE s.season = ? AND s.gp > 0
    """
    df = pd.read_sql_query(sql, conn, params=(season,))

    # the two ratio categories are derived, never averaged from per-player ratios
    df["fg_pct"] = np.where(df["fga"] > 0, df["fgm"] / df["fga"], np.nan)
    df["ft_pct"] = np.where(df["fta"] > 0, df["ftm"] / df["fta"], np.nan)
    return df


# Model A's own functions, reused rather than reimplemented: rungs 2 and 4 have to
# mean the same thing by VORP or the ablation compares definitions, not policies.
def add_value_and_vorp(df, n_teams, slots=None):
    slots = slots or ROSTER_SLOTS
    return _vorp.add_vorp(df, n_teams=n_teams, slots=slots)


# The ADP board for a season, with the consensus rank every rung 0 pick follows.
def load_board(conn, season):
    board = tables.load_adp(conn, season=season)
    board = board[board["player_id"].notna()].copy()
    board["player_id"] = board["player_id"].astype(int)
    board["platform_rank"] = board["adp"].rank(method="first").astype(int)
    return board.sort_values("platform_rank").reset_index(drop=True)


# That season's roster over the career listing, the same precedence
# features.attach_position uses so the two models agree.
def load_positions(conn, season):
    roster = tables.load_rosters(conn, season)[["player_id", "position"]]
    roster = roster.dropna(subset=["position"]).drop_duplicates("player_id")
    roster = roster.rename(columns={"position": "roster_position"})

    players = tables.load_players(conn)[["player_id", "position"]]
    players = players.rename(columns={"position": "player_position"})

    merged = roster.merge(players, on="player_id", how="outer")
    merged["position"] = (merged["roster_position"]
                          .combine_first(merged["player_position"])
                          .map(_features.norm_pos))
    return merged[["player_id", "position"]]


# Only rung 4 uses these, and they exist for one season -- backtest.py reports the
# rung unavailable rather than quietly falling back to the board.
def load_projections(conn, season, projections_csv=PROJECTIONS_CSV):
    if not projections_csv.exists():
        return None

    proj = pd.read_csv(projections_csv)
    target = _features.TARGET_SEASON

    if season != target:
        return None

    return proj


# Value implied by the board alone, for rungs 0-3. It has to come from the board
# rather than Model A or the outcome, or rung 3 is not sequencing on fixed inputs.
def board_implied_value(conn, season, n_teams, rank_curve=None):
    board = load_board(conn, season)
    positions = load_positions(conn, season)

    df = board.merge(positions, on="player_id", how="left")
    df["position"] = df["position"].fillna(UNKNOWN_POSITION)

    # a rank-to-value curve fit on OTHER seasons, so this season's outcome never
    # leaks into the value the policy drafts on
    if rank_curve is None:
        rank_curve = fit_rank_value_curve(conn, season, n_teams)

    # Rank alone would make value monotone in rank, collapsing rung 1 into rung 0, so
    # the board's own detail (adp_sd, source count) is blended in -- all of it visible.
    df["value"] = np.interp(df["platform_rank"], rank_curve["rank"],
                            rank_curve["value"])
    df["value"] = df["value"] + _board_detail(df)

    # replacement level per position, so rung 2 can subtract it
    levels = {}
    for pos in df["position"].unique():
        pool = np.sort(df.loc[df["position"] == pos, "value"].to_numpy())[::-1]
        if len(pool) == 0:
            levels[pos] = 0.0
            continue
        idx = int(n_teams * ROSTER_SLOTS.get(pos, 1))
        levels[pos] = float(pool[min(idx, len(pool) - 1)])

    df["replacement"] = df["position"].map(levels)
    df["vorp"] = df["value"] - df["replacement"]
    return df


# Consensus is information: a player every site agrees on is a safer hold than one
# at the same rank they disagree about, and adp_sd is that disagreement.
def _board_detail(df, sd_weight=0.35):
    sd = df["adp_sd"].to_numpy(dtype=float)
    n_obs = df["n_observations"].to_numpy(dtype=float)

    # a missing sd means one source and no disagreement measured, which is not the
    # same as agreement -- it gets the median rather than a confident zero
    finite = sd[np.isfinite(sd)]
    sd = np.where(np.isfinite(sd), sd, np.median(finite) if len(finite) else 0.0)

    # scaled by rank, since two picks of disagreement means far more at pick 3 than
    # at pick 120
    relative = sd / np.maximum(df["platform_rank"].to_numpy(dtype=float), 1.0)

    # more sites backing a row makes it more trustworthy, so agreement counts double
    # when several sources produced it
    weight = np.where(np.isfinite(n_obs) & (n_obs > 1), 1.0, 0.5)

    return -sd_weight * relative * weight


# Pooled over every season EXCEPT this one -- the leave-one-out step that keeps
# this season's outcome out of the value a policy drafts on.
def fit_rank_value_curve(conn, season, n_teams, seasons=None, smooth=15):
    seasons = seasons or [s for s in config.BACKTEST_SEASONS if s != season]

    rows = []
    for s in seasons:
        board = load_board(conn, s)
        realized = load_realized(conn, s)
        if realized.empty:
            continue

        valued = add_value_and_vorp(
            realized.merge(load_positions(conn, s), on="player_id", how="left")
                    .assign(position=lambda d: d["position"].fillna(UNKNOWN_POSITION)),
            n_teams)

        merged = board.merge(valued[["player_id", "value"]], on="player_id",
                            how="left")
        # an undrafted-in-reality player still occupied a board slot, and his value
        # is genuinely low rather than missing
        merged["value"] = merged["value"].fillna(valued["value"].min())
        rows.append(merged[["platform_rank", "value"]])

    pooled = pd.concat(rows, ignore_index=True)
    curve = (pooled.groupby("platform_rank")["value"].mean()
             .rolling(smooth, center=True, min_periods=1).mean()
             .reset_index())
    curve.columns = ["rank", "value"]

    # value has to fall with rank or every policy inherits a scrambled board
    curve["value"] = np.maximum.accumulate(curve["value"].to_numpy()[::-1])[::-1]
    return curve


# The pool as parallel arrays, one row per player, so a simulation can slice
# without a groupby.
class Board:
    def __init__(self, conn, season, n_teams, use_projections=False):
        self.season = season
        self.n_teams = n_teams

        implied = board_implied_value(conn, season, n_teams)
        self.projected = None

        if use_projections:
            proj = load_projections(conn, season)
            if proj is not None:
                valued = add_value_and_vorp(proj, n_teams)
                self.projected = valued[["player_id", "value", "vorp"]]

        # realized production is what every roster is finally scored on
        realized = load_realized(conn, season)
        realized_valued = add_value_and_vorp(
            realized.merge(load_positions(conn, season), on="player_id", how="left")
                    .assign(position=lambda d: d["position"].fillna(UNKNOWN_POSITION)),
            n_teams)

        df = implied.merge(
            realized_valued[["player_id", "value"]].rename(
                columns={"value": "realized_value"}),
            on="player_id", how="left")

        # a player who never played earns the floor, not NaN: he was a real pick
        # that returned nothing, and dropping him would flatter whoever took him
        floor = float(realized_valued["value"].min()) if not realized_valued.empty else 0.0
        df["realized_value"] = df["realized_value"].fillna(floor)

        if self.projected is not None:
            df = df.merge(self.projected.rename(
                columns={"value": "proj_value", "vorp": "proj_vorp"}),
                on="player_id", how="left")
            df["proj_value"] = df["proj_value"].fillna(df["value"])
            df["proj_vorp"] = df["proj_vorp"].fillna(df["vorp"])

        self.df = df.sort_values("platform_rank").reset_index(drop=True)

        self.player_ids = self.df["player_id"].to_numpy(dtype=np.int64)
        self.ranks = self.df["platform_rank"].to_numpy(dtype=np.int32)
        self.values = self.df["value"].to_numpy(dtype=np.float64)
        self.vorps = self.df["vorp"].to_numpy(dtype=np.float64)
        self.realized = self.df["realized_value"].to_numpy(dtype=np.float64)
        self.positions = np.array(
            [position_index(p) for p in self.df["position"]], dtype=np.int32)

        self.index_of = {int(pid): i for i, pid in enumerate(self.player_ids)}

    def __len__(self):
        return len(self.df)

    # value under whichever inputs a rung is allowed to see
    def value_for(self, use_projections=False):
        if use_projections and "proj_value" in self.df.columns:
            return self.df["proj_value"].to_numpy(dtype=np.float64)
        return self.values

    def vorp_for(self, use_projections=False):
        if use_projections and "proj_vorp" in self.df.columns:
            return self.df["proj_vorp"].to_numpy(dtype=np.float64)
        return self.vorps


# rank correlation without scipy, since Pearson understates a convex curve
def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    conn = connect()
    season = "2024-25"

    board = Board(conn, season, n_teams=config.N_TEAMS)

    print(f"{season}: {len(board)} players, {config.N_TEAMS} teams\n")

    names = dict(conn.execute("SELECT player_id, name FROM players"))
    print("\ntop of the board")
    print(f"{'player':<26}{'rank':>6}{'pos':>5}{'value':>9}{'vorp':>9}{'realized':>10}")
    for row in board.df.head(12).itertuples():
        name = str(names.get(int(row.player_id), row.player_id)).encode(
            "ascii", "replace").decode()
        print(f"{name:<26}{row.platform_rank:>6}{row.position:>5}"
              f"{row.value:>9.2f}{row.vorp:>9.2f}{row.realized_value:>10.2f}")

    print("\nvalue by position")
    print(board.df.groupby("position").agg(
        n=("player_id", "size"),
        mean_value=("value", "mean"),
        mean_vorp=("vorp", "mean"),
        replacement=("replacement", "first"),
    ).round(3).to_string())
