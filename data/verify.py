import sys

import sqlite_helpers
from config import DB_PATH

MIN_ADP_SEASONS = 8  # the backtest's hard floor
MIN_ADP_ROWS_PER_SEASON = 100
MIN_LINK_RATE = 0.95


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

    print(f"\nseasons: {', '.join(s for s, _ in seasons)}")
    for table in ("players", "season_stats", "game_logs", "adp", "draft_results"):
        print(f"  {table:<14} {sqlite_helpers.count(conn, table):>8,}")

    conn.close()

    # non-zero exit so CI refuses to publish a bad database
    if failures:
        sys.exit(f"\n{len(failures)} check(s) failed")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
