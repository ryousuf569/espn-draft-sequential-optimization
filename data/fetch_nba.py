from nba_api.stats.endpoints import (
    commonallplayers,
    commonplayerinfo,
    leaguedashplayerstats,
    playergamelogs,
)
import argparse
import time

import sqlite_helpers
from config import NBA_RETRIES, NBA_SLEEP, NBA_TIMEOUT, stats_seasons


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
    # the only per-player endpoint, so only ask about players still missing a bio
    missing = conn.execute(
        "SELECT player_id FROM players "
        "WHERE birth_date IS NULL "
        "  AND player_id IN (SELECT DISTINCT player_id FROM season_stats)"
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
    args = ap.parse_args()

    seasons = args.seasons or stats_seasons()
    conn = sqlite_helpers.connect()
    sqlite_helpers.init(conn)

    fetch_players(conn)
    fetch_season_stats(conn, seasons)
    if not args.skip_logs:
        fetch_game_logs(conn, seasons)
    if not args.skip_bios:
        fetch_bios(conn)
    fetch_status(conn)

    conn.close()
    print("done")


if __name__ == "__main__":
    main()
