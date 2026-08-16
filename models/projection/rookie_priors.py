import pandas as pd
from config import CATEGORIES, COUNTING_CATEGORIES, RATIO_CATEGORIES, DRAFT_TIERS, UNDRAFTED_TIER, DB_PATH, FIRST_SEASON_YEAR
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
    None


def compute_draft_tier(conn, as_of_season_str, as_of_season_int):

    rookie_query = "SELECT * FROM rookie_outcomes WHERE season < ?"
    rookie_outcomes = pd.DataFrame(conn.execute(rookie_query, (as_of_season_str,)))
    draft_query = "SELECT * FROM nba_draft WHERE draft_year < ? AND draft_year >= 2010"
    nba_draft = pd.DataFrame(conn.execute(draft_query, (as_of_season_int,)))

    print(rookie_outcomes, nba_draft)

conn = connect()
compute_draft_tier(conn, '2024-25', 2024)