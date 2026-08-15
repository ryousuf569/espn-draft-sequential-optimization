from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "nba.sqlite"
SCHEMA_PATH = DATA_DIR / "schema.sql"
RAW_DIR = DATA_DIR / "raw"

# nba_api is an unofficial wrapper around a public endpoint and it throttles
NBA_SLEEP = 2.5
NBA_TIMEOUT = 90
NBA_RETRIES = 3

# FantasyPros robots.txt asks for Crawl-delay: 5
ADP_SLEEP = 5.0

# a season is labelled by the year it starts, so "2010-11" began in 2010.
# 2026-27 is the upcoming season: nba_api already serves its rosters and draft
# class, but no stats or game logs exist until opening night. Fetches that only
# make sense for a played season take stats_seasons(); roster and draft fetches
# take CURRENT_SEASON and work before a ball is tipped.
FIRST_STATS_YEAR = 2010
LAST_STATS_YEAR = 2026

# the season being drafted for: rosters yes, stats not yet
CURRENT_START_YEAR = 2026
CURRENT_SEASON = f"{CURRENT_START_YEAR}-{str(CURRENT_START_YEAR + 1)[-2:]}"

# ADP goes back less far than stats, so the two ranges differ on purpose.
# FantasyPros NBA ADP is only real from 2014: year=2013 returns HTTP 500 and
# year=2012 silently serves the CURRENT season's board. fetch_adp.py checks the
# page title to catch that, but there is no reason to request known-bad years.
FIRST_ADP_YEAR = 2014
LAST_ADP_YEAR = 2025


# 2019 -> "2019-20"
def season_str(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def stats_seasons():
    return [season_str(y) for y in range(FIRST_STATS_YEAR, LAST_STATS_YEAR + 1)]


# every season that has actually been played, so stats and game-log fetches do
# not waste requests on 2026-27 and store a season of empty rows
def played_seasons():
    return [season_str(y) for y in range(FIRST_STATS_YEAR, CURRENT_START_YEAR)]


# the NBA draft classes to pull. Rookie priors are fit on realized outcomes, so
# this starts a year before FIRST_STATS_YEAR: a player drafted in 2009 has his
# rookie season in 2009-10, which is outside the stats range and is skipped by
# the backfill anyway -- the extra year costs one request and avoids an
# off-by-one when FIRST_STATS_YEAR moves.
FIRST_DRAFT_YEAR = 2009
LAST_DRAFT_YEAR = CURRENT_START_YEAR


def draft_years():
    return list(range(FIRST_DRAFT_YEAR, LAST_DRAFT_YEAR + 1))


# ADP is fetched by start year, since that is what FantasyPros takes
def adp_seasons():
    return list(range(FIRST_ADP_YEAR, LAST_ADP_YEAR + 1))
