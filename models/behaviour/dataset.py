# Builds the survival table: one row per (draft_id, player_id) with a duration and
# a 0/1 event flag, which is the shape KaplanMeierFitter and CoxPHFitter both take.

import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (ADP_TIERS, DRAFT_LENGTH_TOLERANCE, PLATFORM_ADP_SOURCE,
                    RANDOM_SEED, SEASONS, SYNTHETIC_DRAFTS_PER_SEASON,
                    TOTAL_PICKS, UNRANKED_TIER, connect)

# this package data.py, loaded by path -- plain `import data` finds the repo
# package instead, see the note in config.py
tables = config.load_tables()


# ADP rank -> tier label, the grouping the fallback KM curves are cut on
def tier_for_rank(platform_rank):
    if platform_rank is None or pd.isna(platform_rank):
        return UNRANKED_TIER

    for name, lo, hi in ADP_TIERS:
        if lo <= platform_rank <= hi:
            return name

    return UNRANKED_TIER


# Every player who could have been taken: the season's ADP board, plus anyone who
# appears in the draft itself, which catches off-board picks the board missed.
def get_draft_universe(conn, draft_id, season):
    board = tables.load_adp(conn, season=season, source=PLATFORM_ADP_SOURCE)
    ranked = board.loc[board["player_id"].notna(), "player_id"]

    picked = tables.load_draft_results(conn, season=season)
    picked = picked[(picked["draft_id"] == draft_id) & picked["player_id"].notna()]

    universe = pd.unique(pd.concat([ranked, picked["player_id"]], ignore_index=True))
    return [int(pid) for pid in universe]


# The censoring boundary: the last pick that actually happened, since an abandoned
# draft would otherwise be credited survival through picks never made.
def get_draft_length(conn, draft_id):
    row = conn.execute(
        "SELECT MAX(pick_number) FROM draft_results WHERE draft_id = ?",
        (draft_id,),
    ).fetchone()

    last_pick = row[0] if row and row[0] is not None else None

    if last_pick is None:
        return TOTAL_PICKS, False

    return int(last_pick), abs(int(last_pick) - TOTAL_PICKS) > DRAFT_LENGTH_TOLERANCE


# One row per (draft_id, player_id) from real drafts: duration is the pick he went
# at or the draft length if unpicked, event_observed is 1 drafted and 0 censored.
def build_survival_records(conn, season=None):
    seasons = [season] if season is not None else list(SEASONS)
    frames = []

    for s in seasons:
        picks = tables.load_draft_results(conn, season=s)
        picks = picks[picks["player_id"].notna()]

        for draft_id, drafted in picks.groupby("draft_id"):
            length, _ = get_draft_length(conn, draft_id)
            universe = get_draft_universe(conn, draft_id, s)

            taken = dict(zip(drafted["player_id"].astype(int),
                             drafted["pick_number"].astype(int)))

            frames.append(pd.DataFrame({
                "draft_id": draft_id,
                "season": s,
                "player_id": universe,
                # censored at the boundary, not at his pick: the draft ended before his turn
                "duration": [min(taken.get(pid, length), length) for pid in universe],
                "event_observed": [1 if pid in taken else 0 for pid in universe],
            }))

    if not frames:
        return pd.DataFrame(columns=["draft_id", "season", "player_id",
                                     "duration", "event_observed"])

    return pd.concat(frames, ignore_index=True)


# --- synthetic drafts -------------------------------------------------------

# draft_results is EMPTY and stays that way -- no source publishes per-pick NBA
# logs at scale (data/README.md), so build_survival_records has nothing to read.


# Sample one draft: jitter each ADP by its own disagreement spread and sort, so the
# induced ranks are a permutation and nobody can be taken twice.
def sample_draft(board, rng, length=TOTAL_PICKS):
    adp = board["adp"].to_numpy(dtype=float)

    # same fallback as draft_slots.from_adp: sd 0 would make the draft deterministic,
    # and late picks are far noisier than early ones
    sd = board["adp_sd"].to_numpy(dtype=float)
    sd = np.where(np.isnan(sd) | (sd <= 0), np.maximum(1.5, adp * 0.20), sd)

    order = np.argsort(adp + rng.normal(0.0, sd))
    picks = order[:length]

    duration = np.full(len(board), length, dtype=int)
    event = np.zeros(len(board), dtype=int)

    duration[picks] = np.arange(1, len(picks) + 1)
    event[picks] = 1

    return duration, event


# hash() is salted per process, so it cannot seed anything reproducible
def season_seed(seed, season):
    return (int(seed) * 1000 + int(str(season)[:4])) % (2 ** 32)


# The same long-format table from sampled drafts, used whenever draft_results is
# empty -- which right now is always.
def build_synthetic_records(conn, season=None, n_drafts=SYNTHETIC_DRAFTS_PER_SEASON,
                            seed=RANDOM_SEED):
    seasons = [season] if season is not None else list(SEASONS)
    frames = []

    for s in seasons:
        board = tables.load_adp(conn, season=s, source=PLATFORM_ADP_SOURCE)
        board = board[board["player_id"].notna()].reset_index(drop=True)

        if board.empty:
            continue

        # seeded per season, so adding a season does not reshuffle the ones already built
        rng = np.random.default_rng(season_seed(seed, s))
        length = min(TOTAL_PICKS, len(board))

        for i in range(n_drafts):
            duration, event = sample_draft(board, rng, length)
            frames.append(pd.DataFrame({
                "draft_id": f"syn-{s}-{i:03d}",
                "season": s,
                "player_id": board["player_id"].astype(int).to_numpy(),
                "duration": duration,
                "event_observed": event,
            }))

    if not frames:
        return pd.DataFrame(columns=["draft_id", "season", "player_id",
                                     "duration", "event_observed"])

    return pd.concat(frames, ignore_index=True)


# The survival table every other module asks for. Real drafts win wherever they
# exist; is_synthetic travels with the rows so no report can lose track of which.
def build_dataset(conn, season=None, n_synthetic=SYNTHETIC_DRAFTS_PER_SEASON):
    real = build_survival_records(conn, season)
    real_seasons = set(real["season"].unique()) if not real.empty else set()

    seasons = [season] if season is not None else list(SEASONS)
    missing = [s for s in seasons if s not in real_seasons]

    frames = []
    if not real.empty:
        real = real.copy()
        real["is_synthetic"] = 0
        frames.append(real)

    for s in missing:
        part = build_synthetic_records(conn, s, n_synthetic)
        if not part.empty:
            part["is_synthetic"] = 1
            frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["draft_id", "season", "player_id", "duration",
                                     "event_observed", "is_synthetic"])

    out = pd.concat(frames, ignore_index=True)
    return attach_adp_tier(out, conn)


# platform_rank and its tier, joined per season -- both fitters group on the tier
# and Cox uses the rank itself as a covariate
def attach_adp_tier(survival_df, conn):
    board = tables.load_adp(conn, source=PLATFORM_ADP_SOURCE)
    board = board[board["player_id"].notna()].copy()
    board["player_id"] = board["player_id"].astype(int)

    # rank within the season board, so rank 1 is the consensus first pick
    board["platform_rank"] = board.groupby("season")["adp"].rank(method="first")

    merged = survival_df.merge(
        board[["player_id", "season", "adp", "platform_rank"]],
        on=["player_id", "season"], how="left")

    merged["adp_tier"] = merged["platform_rank"].map(tier_for_rank)
    return merged


# the table has to hold up as survival data before anything is fit on it
def verify(df, conn):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    check("table is not empty", not df.empty, f"{len(df)} rows")
    check("one row per draft and player",
          not df.duplicated(["draft_id", "player_id"]).any())
    check("event is 0 or 1", bool(df["event_observed"].isin((0, 1)).all()))
    check("durations are positive", bool((df["duration"] >= 1).all()),
          f"min {df['duration'].min()}")

    # a drafted player's duration is his pick, so it cannot exceed the draft
    lengths = df.groupby("draft_id")["duration"].max()
    check("durations stay inside the draft", bool((lengths <= TOTAL_PICKS).all()),
          f"longest {int(lengths.max())}")

    # censoring is the whole point: with no censored rows every curve falls to zero
    censored = int((df["event_observed"] == 0).sum())
    check("censored rows exist", censored > 0,
          f"{censored} of {len(df)} ({censored / len(df):.1%})")

    # each draft's event count is its length, or the risk set is being double counted
    per_draft = df[df["event_observed"] == 1].groupby("draft_id").size()
    check("events per draft equal the draft length",
          bool((per_draft == lengths.reindex(per_draft.index)).all()),
          f"{int(per_draft.iloc[0])} picks")

    # earlier ADP has to survive less, or the table is scrambled
    drafted = df[df["event_observed"] == 1]
    corr = drafted["platform_rank"].corr(drafted["duration"])
    check("better ADP goes earlier", corr > 0.5, f"corr {corr:.3f}")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    conn = connect()
    df = build_dataset(conn)

    n_real = int((df["is_synthetic"] == 0).sum())
    print(f"{len(df)} rows, {df['draft_id'].nunique()} drafts, "
          f"{df['season'].nunique()} seasons, {n_real} from real drafts\n")
    verify(df, conn)

    print("\nby season")
    print(df.groupby("season").agg(
        drafts=("draft_id", "nunique"),
        rows=("player_id", "size"),
        players=("player_id", "nunique"),
        drafted_rate=("event_observed", "mean"),
        synthetic=("is_synthetic", "max"),
    ).round(3).to_string())

    print("\nby adp tier")
    print(df.groupby("adp_tier").agg(
        rows=("player_id", "size"),
        mean_duration=("duration", "mean"),
        drafted_rate=("event_observed", "mean"),
    ).round(3).to_string())
