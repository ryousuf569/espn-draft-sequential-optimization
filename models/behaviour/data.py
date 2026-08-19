import pandas as pd

# config first for the repo root, and connect comes through it too -- see the
# note there on this file's name clash with the repo's data/ package.
from config import PLATFORM_ADP_SOURCE, connect


# season is TEXT shaped "2015-16", so equality is the only filter needed
def _season_clause(season, column="season"):
    if season is None:
        return "", ()
    return f" WHERE {column} = ?", (season,)


def load_draft_results(conn, season=None):
    where, params = _season_clause(season)
    sql = ("SELECT draft_id, season, pick_number, player_id, team_slot "
           "FROM draft_results" + where + " ORDER BY draft_id, pick_number")
    return pd.read_sql_query(sql, conn, params=params)


# source defaults to what config calls platform rank, so a caller wanting every
# source has to ask for it rather than get it by forgetting
def load_adp(conn, season=None, source=PLATFORM_ADP_SOURCE):
    clauses, params = [], []

    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = ("SELECT player_id, adp_name, season, source, adp, adp_sd, n_observations "
           "FROM adp" + where + " ORDER BY season, adp")
    return pd.read_sql_query(sql, conn, params=tuple(params))


def load_players(conn):
    return pd.read_sql_query(
        "SELECT player_id, name, position, birth_date FROM players", conn)


def load_rosters(conn, season):
    return pd.read_sql_query(
        "SELECT player_id, season, team, position, exp FROM rosters WHERE season = ?",
        conn, params=(season,))


# DATA GAP: one distinct updated_at across every row, so this is a snapshot with
# no history -- a past draft's injury designations cannot be reconstructed from it.
def load_player_status(conn):
    return pd.read_sql_query(
        "SELECT player_id, team, status, updated_at FROM player_status", conn)


def load_nba_draft(conn):
    return pd.read_sql_query(
        "SELECT player_id, draft_year, round_number, overall_pick, team, "
        "organization, age_at_draft FROM nba_draft", conn)


def load_rookie_outcomes(conn):
    return pd.read_sql_query(
        "SELECT player_id, draft_year, season, overall_pick, gp, mpg, total_min "
        "FROM rookie_outcomes", conn)


# not one of the tables above, but the rookie flag needs to know whether a player
# had logged minutes before a season -- the same definition Model A uses
def load_season_stats(conn, season=None):
    where, params = _season_clause(season)
    sql = ("SELECT player_id, season, gp, mpg FROM season_stats" + where)
    return pd.read_sql_query(sql, conn, params=params)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    conn = connect()

    loaders = {
        "draft_results": lambda: load_draft_results(conn),
        "adp": lambda: load_adp(conn),
        "players": lambda: load_players(conn),
        "rosters": lambda: load_rosters(conn, "2025-26"),
        "player_status": lambda: load_player_status(conn),
        "nba_draft": lambda: load_nba_draft(conn),
        "rookie_outcomes": lambda: load_rookie_outcomes(conn),
        "season_stats": lambda: load_season_stats(conn),
    }

    for name, fn in loaders.items():
        df = fn()
        print(f"{name:16} {len(df):7} rows  {list(df.columns)}")
