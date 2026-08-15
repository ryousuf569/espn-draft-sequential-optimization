import sys

import sqlite_helpers
from config import CURRENT_SEASON, DB_PATH

MIN_ADP_SEASONS = 8  # the backtest's hard floor
MIN_ADP_ROWS_PER_SEASON = 100
MIN_LINK_RATE = 0.95

# the rookie prior is fit on realized rookie seasons; below this the
# pick-number fit is too thin to beat a flat league-wide rookie average
MIN_ROOKIE_OUTCOMES = 400
MIN_ROOKIE_AGE_RATE = 0.90
MIN_CURRENT_ROSTER_TEAMS = 30


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not DB_PATH.exists():
        sys.exit(f"FAIL no database at {DB_PATH}")

    conn = sqlite_helpers.connect()
    failures = []

    def check(ok, message):
        print(f"{'PASS' if ok else 'FAIL'}  {message}")
        if not ok:
            failures.append(message)

    # a table added to the schema after a database was built is missing, not
    # empty, and querying it raises instead of failing a check. Report it as a
    # failure so the message says which fetch to rerun.
    present = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    for table in ("nba_draft", "rosters", "rookie_outcomes"):
        check(table in present, f"{table} exists (rerun fetch_nba.py if missing)")
    if failures:
        conn.close()
        sys.exit(f"\n{len(failures)} check(s) failed")

    for table in ("players", "season_stats", "adp"):
        check(sqlite_helpers.count(conn, table) > 0, f"{table} is not empty")

    # historical ADP is the binding risk for the backtest, so check its depth
    seasons = list(
        conn.execute("SELECT season, COUNT(*) FROM adp GROUP BY season ORDER BY season")
    )
    check(
        len(seasons) >= MIN_ADP_SEASONS,
        f"adp covers {len(seasons)} seasons (need >= {MIN_ADP_SEASONS})",
    )

    thin = [s for s, n in seasons if n < MIN_ADP_ROWS_PER_SEASON]
    check(not thin, f"every adp season has >= {MIN_ADP_ROWS_PER_SEASON} rows; thin: {thin}")

    # the silent-fallback tell: two seasons with identical boards means we stored
    # the same page twice under different labels
    dupes = conn.execute(
        "SELECT a.season, b.season FROM "
        "(SELECT season, GROUP_CONCAT(adp_name) AS board FROM "
        "  (SELECT season, adp_name FROM adp ORDER BY season, adp) GROUP BY season) a "
        "JOIN "
        "(SELECT season, GROUP_CONCAT(adp_name) AS board FROM "
        "  (SELECT season, adp_name FROM adp ORDER BY season, adp) GROUP BY season) b "
        "ON a.board = b.board AND a.season < b.season"
    ).fetchall()
    check(not dupes, f"no two adp seasons are identical; duplicates: {dupes}")

    # name linking, so joins from adp to stats actually resolve
    total = sqlite_helpers.count(conn, "adp")
    linked = conn.execute("SELECT COUNT(*) FROM adp WHERE player_id IS NOT NULL").fetchone()[0]
    rate = linked / total if total else 0
    check(rate >= MIN_LINK_RATE, f"adp link rate {rate:.1%} (need >= {MIN_LINK_RATE:.0%})")

    orphans = conn.execute(
        "SELECT COUNT(*) FROM season_stats s "
        "LEFT JOIN players p ON s.player_id = p.player_id WHERE p.player_id IS NULL"
    ).fetchone()[0]
    check(orphans == 0, f"no season_stats rows without a player ({orphans} orphans)")

    # the upcoming season is roster-only: it is the board being drafted, so an
    # empty or partial pull means every incoming rookie is silently missing
    roster_teams = conn.execute(
        "SELECT COUNT(DISTINCT team) FROM rosters WHERE season = ?", (CURRENT_SEASON,)
    ).fetchone()[0]
    check(
        roster_teams >= MIN_CURRENT_ROSTER_TEAMS,
        f"rosters covers {roster_teams} teams for {CURRENT_SEASON} "
        f"(need {MIN_CURRENT_ROSTER_TEAMS})",
    )

    incoming = conn.execute(
        "SELECT COUNT(*) FROM rosters WHERE season = ? AND exp = 'R'", (CURRENT_SEASON,)
    ).fetchone()[0]
    check(incoming > 0, f"{CURRENT_SEASON} rosters include {incoming} rookies")

    # training set for the rookie prior
    n_rookies = sqlite_helpers.count(conn, "rookie_outcomes")
    check(
        n_rookies >= MIN_ROOKIE_OUTCOMES,
        f"rookie_outcomes has {n_rookies} rows (need >= {MIN_ROOKIE_OUTCOMES})",
    )

    # age_at_draft is the second of two pre-debut signals; if bios did not land
    # it is silently NULL and the prior quietly degrades to pick-number only
    with_age = conn.execute(
        "SELECT COUNT(*) FROM rookie_outcomes WHERE age_at_draft IS NOT NULL"
    ).fetchone()[0]
    age_rate = with_age / n_rookies if n_rookies else 0
    check(
        age_rate >= MIN_ROOKIE_AGE_RATE,
        f"rookie_outcomes age_at_draft {age_rate:.1%} filled "
        f"(need >= {MIN_ROOKIE_AGE_RATE:.0%})",
    )

    # a rookie season must start in the draft year, so a mismatch means the
    # season-string join drifted and the prior is being fit on the wrong year
    misaligned = conn.execute(
        "SELECT COUNT(*) FROM rookie_outcomes "
        "WHERE CAST(SUBSTR(season, 1, 4) AS INTEGER) != draft_year"
    ).fetchone()[0]
    check(misaligned == 0, f"every rookie_outcomes season starts in its draft year ({misaligned} bad)")

    print(f"\nseasons: {', '.join(s for s, _ in seasons)}")
    for table in (
        "players",
        "season_stats",
        "game_logs",
        "adp",
        "draft_results",
        "nba_draft",
        "rosters",
        "rookie_outcomes",
    ):
        print(f"  {table:<16} {sqlite_helpers.count(conn, table):>8,}")

    conn.close()

    # non-zero exit so CI refuses to publish a bad database
    if failures:
        sys.exit(f"\n{len(failures)} check(s) failed")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
