# data/

Builds `nba.sqlite`: player stats, game logs, historical ADP, NBA draft history
and rosters.

The DB is **not committed**. CI publishes it as a release asset (`data-latest`),
because a nightly binary in git history would bloat the repo permanently.

```bash
pip install -r requirements.txt

python data/fetch_nba.py          # players, stats, game logs, draft, rosters
python data/fetch_adp.py          # historical ADP
python data/verify.py             # sanity checks; non-zero exit on failure
python data/draft_slots.py --season 2026-27
```

Grab the latest build instead of rebuilding:

```bash
gh release download data-latest --pattern 'nba.sqlite.gz' --dir data/
gunzip data/nba.sqlite.gz
```

## Files

| File | Role |
| --- | --- |
| `config.py` | Seasons, rate limits, paths. Everything tunable. |
| `schema.sql` | Table definitions. |
| `sqlite_helpers.py` | `connect` / `init` / `upsert` / `count`. |
| `fetch_nba.py` | nba_api: players, season stats, game logs, draft, rosters, injury status. |
| `fetch_adp.py` | FantasyPros historical ADP. |
| `draft_slots.py` | Per-player draft-slot distributions — all Model B consumes. |
| `verify.py` | Data quality gate. CI refuses to publish if it fails. |
| `test_draft_slots.py` | Tests for the autodraft noise filter. |

## Verified coverage

ADP was the binding risk, so it was checked before anything was built on it.
**12 seasons, 2014-15 through 2025-26**, 2,787 rows — against a requirement of 8.

| Table | Source | Notes |
| --- | --- | --- |
| `players` | `commonallplayers` | 5,205 rows, all-time. |
| `season_stats` | `leaguedashplayerstats` | One request per season. |
| `game_logs` | `playergamelogs` | Bulk per season, ~26k rows/season. |
| `adp` | FantasyPros | 12 seasons, 99.4% linked to `player_id`. |
| `player_status` | `commonallplayers` | Current only; replaced each run. |
| `nba_draft` | `drafthistory` | Draft classes 2009-2026, one request per year. |
| `rosters` | `commonteamroster` | 30 requests per season; 580 rows for 2026-27. |
| `rookie_outcomes` | *(derived)* | Join, no requests. Rebuilt each run. |
| `draft_results` | *(none yet)* | Empty; see below. |

## Traps

These are real behaviours found while building, not hypotheticals.

**FantasyPros fails soft on bad years.** `?year=2012` returns **HTTP 200 with the
current season's board** — 2013 returns a 500, but 2012 silently serves 2025-26.
Ingesting that would quietly corrupt a backtest with no error anywhere.
`fetch_adp.py` parses the season out of the page title and rejects any response
that disagrees with the request. `verify.py` independently asserts that no two
seasons share an identical board. Do not remove either check.

**Historical team columns are joined live.** A 2015-16 row lists Harden as CLE.
Team is never stored from ADP pages; only name and ADP are trustworthy.

**Column counts vary by season.** Some years carry a CBS column, some don't
(3 sources in 2014-21, 2 in 2024-25). The parser reads by header name, never by
index, and `n_observations` records how many sites backed each row.

**Accented names break naive matching.** nba_api writes `Nikola Jokić`,
FantasyPros writes `Nikola Jokic`. Without Unicode normalization the best
players on the board silently fail to link — this cost 18 rows including Jokić
and Dončić before it was fixed. Link rate is now 99.4%; the 5 unmatched are
players who never appeared in an NBA game.

**2026-27 has rosters but no stats.** nba_api already serves the 2026-27 rosters
and the 2026 draft class, with real `PERSON_ID`s, but no stats until opening
night. Stats and game logs use `played_seasons()`, which stops at the last played
season; rosters and draft use `CURRENT_SEASON`. Pointing a stats fetch at 2026-27
just writes a season of empty rows.

**Rookie bios need an exemption.** `fetch_bios` only asks about players who
already appear in `season_stats`, which excludes every incoming rookie. Without
the `nba_draft` clause in that query `age_at_draft` stays NULL for exactly the
players that need it, with no error anywhere. So `fetch_draft` runs before
`fetch_bios` and `backfill_draft_ages` after it.

**580 roster spots is not 450.** Preseason rosters include camp, two-way and
Exhibit-10 players who get cut, so `rosters` holds ~19 per team rather than the
~15 that stick. 80 are flagged `EXP = 'R'` against 60 drafted; the rest went
undrafted. `CommonAllPlayers` says 79, so neither count is authoritative.

**Forfeited picks come back as 0.** `OVERALL_PICK` is `0` rather than NULL for a
forfeited pick, so `fetch_draft` maps it to NULL. A literal 0 would sort ahead of
the first pick. Draft history also lists players who never signed and never
appear in `commonallplayers`, so those are skipped.

**Autodraft artifacts.** Abandoned drafts contaminate raw pick data: a team on
autopilot takes the top of the queue at its turn every round, piling mass onto
multiples of the team count. `autodraft_suspect()` flags a distribution with a
secondary spike parked on a round boundary. It deliberately compares each spike
against the rest of the distribution rather than the global peak — in badly
contaminated data the artifact *is* the tallest bar, so a peak-anchored check
compares the spike against itself and misses exactly the case it exists to
catch. This flags for review; it does not silently drop rows.

## Rate limiting

nba_api is an unofficial wrapper around a public endpoint and it throttles.
Every request goes through `call()` in `fetch_nba.py`, which sleeps 2.5s and
backs off further on retry. Nothing calls nba_api directly.

Season stats and game logs are fetched **in bulk per season**, not per player —
one request returns ~26k game logs. The only per-player endpoint is
`commonplayerinfo` for birth dates, which is why `fetch_bios` asks only about
players whose bio is still missing, making reruns nearly free.

Rosters are the one fetch that is per team, so a season costs 30 requests, about
90 seconds at the current sleep. `--skip-rosters` skips it when you only want
stats.

FantasyPros `robots.txt` sets `Crawl-delay: 5`; `ADP_SLEEP` honours it. ADP is
historical and immutable, so it needs a full pull only once.

## `rookie_outcomes` is derived

A join, not a fetch: `nba_draft` against the `season_stats` row for the season
starting in the player's draft year, filtered to `gp > 0`. It costs no requests
and is dropped and rebuilt each run, so it re-derives whenever stats change. A
drafted player who never played has no row.

`total_min` is stored rather than recomputed from `mpg * gp`, since `mpg` is
rounded at ingest. `overall_pick` is NULL for undrafted players and stored raw,
not bucketed. `verify.py` checks that every row's season starts in its draft
year, which catches the season-string join drifting.

## `draft_results` is empty

The schema includes it and `draft_slots.py` prefers it whenever rows exist, but
nothing populates it. No source publishes per-pick NBA draft logs at scale:
Sleeper has no NBA ADP endpoint and its draft IDs aren't enumerable, Yahoo's
`draft_analysis` is OAuth-gated and current-season only, and Underdog has no
public API. Per the plan, published aggregate ADP is used instead of live mock
scraping.

Until real drafts are loaded, `draft_slots.py` derives each distribution from
ADP: a normal centred on `adp`, with `adp_sd` for spread where sites disagree,
falling back to a spread that grows with ADP since late picks are far noisier.
The interface is identical, so dropping real drafts in later changes nothing
downstream.
