from nba_api.stats.endpoints import (
    commonallplayers,
    commonplayerinfo,
    commonteamroster,
    drafthistory,
    leaguedashplayerstats,
    playergamelogs,
)
from nba_api.stats.static import teams
import argparse
import time

import sqlite_helpers
from config import (
    CURRENT_SEASON,
    NBA_RETRIES,
    NBA_SLEEP,
    NBA_TIMEOUT,
    draft_years,
    played_seasons,
    season_str,
)


# every nba_api request goes through here so nothing skips the rate limit
def call(endpoint, **kwargs):
    for attempt in range(NBA_RETRIES):
        # sleep before the call, and back off further on each retry
        time.sleep(NBA_SLEEP * (attempt + 1))
        try:
            return endpoint(timeout=NBA_TIMEOUT, **kwargs).get_data_frames()
        except Exception as e:
            if attempt == NBA_RETRIES - 1:
                raise
            print(f"Request failed: {e}. Retrying...")


# nba_api leaves gaps as NaN, sqlite wants None
def num(value, cast=int):
    if value is None or value != value:
        return None
    return cast(value)


def fetch_players(conn):
    df = call(commonallplayers.CommonAllPlayers, is_only_current_season=0)[0]

    rows = [
        {
            "player_id": int(r.PERSON_ID),
            "name": r.DISPLAY_FIRST_LAST,
            "position": None,
            "birth_date": None,
        }
        for r in df.itertuples()
    ]

    # INSERT OR IGNORE so we never wipe a birth_date fetch_bios already filled in
    conn.executemany(
        "INSERT OR IGNORE INTO players (player_id, name, position, birth_date) "
        "VALUES (:player_id, :name, :position, :birth_date)",
        rows,
    )
    conn.commit()
    print(f"players: {len(rows)} seen, {sqlite_helpers.count(conn, 'players')} in db")


def fetch_season_stats(conn, seasons):
    for season in seasons:
        df = call(
            leaguedashplayerstats.LeagueDashPlayerStats,
            season=season,
            per_mode_detailed="Totals",
        )[0]

        rows = []
        for r in df.itertuples():
            gp = num(r.GP)
            total_min = num(r.MIN, float)
            rows.append(
                {
                    "player_id": int(r.PLAYER_ID),
                    "season": season,
                    "gp": gp,
                    # MIN comes back as a season total, the schema wants per game
                    "mpg": round(total_min / gp, 2) if gp else None,
                    "pts": num(r.PTS),
                    "reb": num(r.REB),
                    "ast": num(r.AST),
                    "stl": num(r.STL),
                    "blk": num(r.BLK),
                    "tov": num(r.TOV),
                    "fg3m": num(r.FG3M),
                    "fgm": num(r.FGM),
                    "fga": num(r.FGA),
                    "ftm": num(r.FTM),
                    "fta": num(r.FTA),
                }
            )

        sqlite_helpers.upsert(conn, "season_stats", rows)
        print(f"season_stats {season}: {len(rows)}")


def fetch_game_logs(conn, seasons):
    for season in seasons:
        # one bulk request per season, not one per player
        df = call(
            playergamelogs.PlayerGameLogs,
            season_nullable=season,
            season_type_nullable="Regular Season",
        )[0]

        rows = [
            {
                "player_id": int(r.PLAYER_ID),
                "game_id": r.GAME_ID,
                "season": season,
                "date": r.GAME_DATE[:10],  # trim the T00:00:00
                "min": num(r.MIN, float),
                "pts": num(r.PTS),
                "reb": num(r.REB),
                "ast": num(r.AST),
                "stl": num(r.STL),
                "blk": num(r.BLK),
                "tov": num(r.TOV),
                "fg3m": num(r.FG3M),
                "fgm": num(r.FGM),
                "fga": num(r.FGA),
                "ftm": num(r.FTM),
                "fta": num(r.FTA),
            }
            for r in df.itertuples()
        ]

        sqlite_helpers.upsert(conn, "game_logs", rows)
        print(f"game_logs {season}: {len(rows)}")


def fetch_bios(conn):
    # the only per-player endpoint, so only ask about players still missing a
    # bio. Drafted players are included even with no season_stats rows: an
    # incoming rookie has none by definition, and age_at_draft is one of the two
    # signals the rookie prior has before he plays a game.
    missing = conn.execute(
        "SELECT player_id FROM players "
        "WHERE birth_date IS NULL "
        "  AND (player_id IN (SELECT DISTINCT player_id FROM season_stats) "
        "    OR player_id IN (SELECT player_id FROM nba_draft))"
    ).fetchall()
    print(f"bios: {len(missing)} to fetch (~{len(missing) * NBA_SLEEP / 60:.0f} min)")

    for i, (player_id,) in enumerate(missing, 1):
        try:
            df = call(commonplayerinfo.CommonPlayerInfo, player_id=player_id)[0]
        except Exception as e:
            print(f"    skip {player_id}: {e}")
            continue

        row = df.iloc[0]
        birth = row["BIRTHDATE"]
        conn.execute(
            "UPDATE players SET position = ?, birth_date = ? WHERE player_id = ?",
            (row["POSITION"] or None, birth[:10] if birth else None, player_id),
        )

        # commit periodically so a crash halfway does not lose everything
        if i % 25 == 0:
            conn.commit()
            print(f"    {i}/{len(missing)}")

    conn.commit()


def fetch_draft(conn, years):
    known = {r[0] for r in conn.execute("SELECT player_id FROM players")}
    rows = []

    for year in years:
        df = call(drafthistory.DraftHistory, season_year_nullable=str(year))[0]

        for r in df.itertuples():
            player_id = int(r.PERSON_ID)
            # draft history includes picks who never signed and so never appear
            # in commonallplayers; the foreign key would reject them
            if player_id not in known:
                continue
            rows.append(
                {
                    "player_id": player_id,
                    "draft_year": year,
                    "round_number": num(r.ROUND_NUMBER),
                    # picks that were forfeited come back as 0, not NULL
                    "overall_pick": num(r.OVERALL_PICK) or None,
                    "team": r.TEAM_ABBREVIATION or None,
                    "organization": r.ORGANIZATION or None,
                    "age_at_draft": None,  # filled by backfill_draft_ages
                    }
            )

        print(f"nba_draft {year}: {len(df)} picks")

    sqlite_helpers.upsert(conn, "nba_draft", rows)
    print(f"nba_draft: {len(rows)} linked to a player")


# age at draft is the second pre-debut signal after pick number, and it needs
# birth_date, which fetch_bios only fills for players with season_stats rows.
# Run after fetch_bios; recomputed each run so late-arriving bios get picked up.
def backfill_draft_ages(conn):
    updated = conn.execute(
        # NBA drafts are held in late June, so June 26 of the draft year is a
        # closer anchor than Jan 1 and keeps one-and-dones on the right side of
        # a birthday. Exact draft dates are not in the API.
        "UPDATE nba_draft SET age_at_draft = ("
        "  SELECT ROUND((JULIANDAY(nba_draft.draft_year || '-06-26') "
        "                - JULIANDAY(p.birth_date)) / 365.25, 2) "
        "  FROM players p WHERE p.player_id = nba_draft.player_id "
        "    AND p.birth_date IS NOT NULL"
        ") WHERE EXISTS ("
        "  SELECT 1 FROM players p WHERE p.player_id = nba_draft.player_id "
        "    AND p.birth_date IS NOT NULL"
        ")"
    ).rowcount
    conn.commit()
    print(f"nba_draft ages: {updated} filled")


def fetch_rosters(conn, seasons):
    known = {r[0] for r in conn.execute("SELECT player_id FROM players")}

    for season in seasons:
        rows = []
        # the only fetch that is per-team rather than per-season, so it is 30
        # requests; worth it because this is the sole source of 2026-27 rows
        for team in teams.get_teams():
            df = call(
                commonteamroster.CommonTeamRoster,
                team_id=team["id"],
                season=season,
            )[0]

            rows.extend(
                {
                    "player_id": int(r.PLAYER_ID),
                    "season": season,
                    "team": team["abbreviation"],
                    "position": r.POSITION or None,
                    "exp": str(r.EXP) if r.EXP else None,
                }
                for r in df.itertuples()
                if int(r.PLAYER_ID) in known  # respect the foreign key
            )

        sqlite_helpers.upsert(conn, "rosters", rows)
        print(f"rosters {season}: {len(rows)}")


# Realized rookie seasons, the training set for the rookie prior. A player's
# rookie season is the one starting in his draft year, so this is a join rather
# than a fetch -- no requests, and it re-derives cleanly whenever stats change.
def backfill_rookie_outcomes(conn):
    conn.execute("DELETE FROM rookie_outcomes")
    inserted = conn.execute(
        "INSERT INTO rookie_outcomes ("
        "  player_id, draft_year, season, overall_pick, age_at_draft, "
        "  gp, mpg, total_min, pts, reb, ast, stl, blk, tov, "
        "  fg3m, fgm, fga, ftm, fta) "
        "SELECT d.player_id, d.draft_year, s.season, d.overall_pick, "
        "       d.age_at_draft, s.gp, s.mpg, ROUND(s.mpg * s.gp, 1), "
        "       s.pts, s.reb, s.ast, s.stl, s.blk, s.tov, "
        "       s.fg3m, s.fgm, s.fga, s.ftm, s.fta "
        "FROM nba_draft d "
        "JOIN season_stats s ON s.player_id = d.player_id "
        # the season starting in the draft year is the rookie season; a player
        # who did not play that year has no row and is correctly absent
        " AND s.season = d.draft_year || '-' || SUBSTR(d.draft_year + 1, 3, 2) "
        "WHERE s.gp > 0"
    ).rowcount
    conn.commit()
    print(f"rookie_outcomes: {inserted}")


def fetch_status(conn):
    df = call(commonallplayers.CommonAllPlayers, is_only_current_season=1)[0]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    known = {r[0] for r in conn.execute("SELECT player_id FROM players")}

    rows = [
        {
            "player_id": int(r.PERSON_ID),
            "team": r.TEAM_ABBREVIATION or None,
            # ROSTERSTATUS comes back as "1"/"0", unreadable in a query
            "status": "active" if str(r.ROSTERSTATUS) == "1" else "inactive",
            "updated_at": now,
        }
        for r in df.itertuples()
        if int(r.PERSON_ID) in known  # respect the foreign key
    ]

    # current state only, no history, so replace the table wholesale
    conn.execute("DELETE FROM player_status")
    sqlite_helpers.upsert(conn, "player_status", rows)
    print(f"player_status: {len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", help="e.g. 2024-25 2025-26")
    ap.add_argument("--skip-logs", action="store_true", help="skip game logs")
    ap.add_argument("--skip-bios", action="store_true", help="skip birth dates")
    ap.add_argument("--skip-rosters", action="store_true", help="skip rosters (30 requests/season)")
    args = ap.parse_args()

    # stats only for seasons that have been played; rosters include the upcoming
    # one, which is the whole point of fetching them separately
    seasons = args.seasons or played_seasons()
    roster_seasons = args.seasons or [*played_seasons(), CURRENT_SEASON]

    conn = sqlite_helpers.connect()
    sqlite_helpers.init(conn)

    fetch_players(conn)
    fetch_season_stats(conn, seasons)
    if not args.skip_logs:
        fetch_game_logs(conn, seasons)
    fetch_draft(conn, draft_years())
    if not args.skip_bios:
        fetch_bios(conn)
    # after fetch_bios so it sees the birth dates that run just filled in
    backfill_draft_ages(conn)
    if not args.skip_rosters:
        fetch_rosters(conn, roster_seasons)
    backfill_rookie_outcomes(conn)
    fetch_status(conn)

    conn.close()
    print("done")


if __name__ == "__main__":
    main()
