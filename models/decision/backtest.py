# §8.2: replay historical drafts, your agent on the model and opponents on that
# season's ADP, and score final rosters on realized production.

import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (ARTIFACT_DIR, BACKTEST_SEASONS, N_CANDIDATES, N_SEEDS, N_SIMS,
                    RANDOM_SEED, connect)
from board import Board
from policies import (POLICIES, PROJECTION_POLICIES, QUESTIONS, SEQUENCING_POLICIES,
                      get_policy)
from roster import RosterState, roster_size
from rollout import build_hazard_matrix, picks_for_slot, team_at_pick

OUT_CSV = ARTIFACT_DIR / "backtest.csv"
LADDER_CSV = ARTIFACT_DIR / "ablation_ladder.csv"


# Opponents draft that season's ADP. Their realism is not the point; holding them
# fixed is, because it is what makes two policies comparable at all.
def opponent_order(board, rng, jitter=6.0):
    noise = rng.normal(0.0, jitter, len(board))
    return np.argsort(board.ranks + noise)


# One replayed draft: your slot uses the policy, every other takes the best player
# left in its jittered ADP order that its roster can hold.
def replay_draft(board, policy_name, my_slot, n_teams, total_picks, rng,
                 behaviour=None, hazard=None, n_sims=N_SIMS,
                 n_candidates=N_CANDIDATES, seed=RANDOM_SEED):
    policy = get_policy(policy_name)
    available = np.ones(len(board), dtype=bool)

    my_state = RosterState()
    opponents = [RosterState() for _ in range(n_teams)]
    my_picks = picks_for_slot(my_slot, n_teams, total_picks)
    my_roster = []

    order = opponent_order(board, rng)

    for pick in range(1, total_picks + 1):
        slot = team_at_pick(pick, n_teams)

        if slot == my_slot:
            remaining = [p for p in my_picks if p >= pick]
            idx = policy(board, available, my_state, behaviour=behaviour,
                         my_picks=remaining, opponent_states=opponents,
                         n_teams=n_teams, total_picks=total_picks, hazard=hazard,
                         n_sims=n_sims, n_candidates=n_candidates, seed=seed)

            if idx is None or idx < 0:
                continue

            my_state.add(int(board.positions[idx]))
            my_roster.append(int(idx))
            available[idx] = False
            continue

        idx = next_in_order(order, available, board, opponents[slot])
        if idx < 0:
            continue

        opponents[slot].add(int(board.positions[idx]))
        available[idx] = False

    return my_roster, my_state


# the first player in this ADP order still on the board who fits
def next_in_order(order, available, board, state):
    for idx in order:
        if not available[idx]:
            continue
        if state.can_add(int(board.positions[idx])):
            return int(idx)
    return -1


# What those players actually produced. Starters carry the score, but the bench is
# what kept the roster legal, so it is reported rather than dropped.
def score_roster(board, roster_indices, slots=None):
    if not roster_indices:
        return {"total": 0.0, "starters": 0.0, "bench": 0.0, "n": 0}

    realized = board.realized[roster_indices]
    positions = board.positions[roster_indices]

    slots = slots or config.ROSTER_SLOTS
    from roster import position_index
    capacity = {position_index(p): n for p, n in slots.items()}

    # scored by filling starting slots with the best players at each position
    used = np.zeros(len(roster_indices), dtype=bool)
    starters = 0.0

    for pos_idx, n_slots in capacity.items():
        at_pos = [i for i in range(len(roster_indices)) if positions[i] == pos_idx]
        at_pos.sort(key=lambda i: realized[i], reverse=True)
        for i in at_pos[:n_slots]:
            starters += realized[i]
            used[i] = True

    return {
        "total": float(realized.sum()),
        "starters": float(starters),
        "bench": float(realized[~used].sum()),
        "n": len(roster_indices),
    }


# Every (policy, slot, seed) for one season and league size -- the sweep §8.2 asks
# for, so the output is a distribution.
def backtest_season(conn, season, n_teams, policies=None, n_seeds=N_SEEDS,
                    n_sims=N_SIMS, n_candidates=N_CANDIDATES, rounds=None,
                    behaviour=None, verbose=False):
    policies = policies or list(config.POLICY_NAMES)
    rounds = rounds or roster_size()
    total_picks = n_teams * rounds

    board = Board(conn, season, n_teams, use_projections=True)

    # Model B and its hazard matrix are built once per season: identical across every
    # policy, slot and seed, so refitting would multiply cost for no change.
    needs_model = any(p in SEQUENCING_POLICIES for p in policies)
    hazard = None

    if needs_model and behaviour is None:
        behaviour_query = config.load_behaviour_module("query")
        behaviour = behaviour_query.Behaviour(conn, season)

    if needs_model:
        hazard = build_hazard_matrix(behaviour, board, total_picks)

    has_projections = board.projected is not None
    rows = []

    for policy_name in policies:
        # Rung 4 needs Model A's projections. Saying so beats running it on board-implied
        # value, which would report rung 3 twice under two names.
        if policy_name in PROJECTION_POLICIES and not has_projections:
            if verbose:
                print(f"  {policy_name}: skipped, no projections for {season}")
            continue

        for my_slot in range(n_teams):
            for seed_i in range(n_seeds):
                rng = np.random.default_rng(
                    (RANDOM_SEED, hash_season(season), n_teams, my_slot, seed_i))

                roster, state = replay_draft(
                    board, policy_name, my_slot, n_teams, total_picks, rng,
                    behaviour=behaviour, hazard=hazard, n_sims=n_sims,
                    n_candidates=n_candidates, seed=RANDOM_SEED + seed_i)

                score = score_roster(board, roster)
                rows.append({
                    "season": season,
                    "n_teams": n_teams,
                    "policy": policy_name,
                    "draft_slot": my_slot + 1,
                    "seed": seed_i,
                    "roster_value": score["total"],
                    "starter_value": score["starters"],
                    "bench_value": score["bench"],
                    "roster_size": score["n"],
                    "legal": state.filled() == rounds,
                })

        if verbose:
            done = [r for r in rows if r["policy"] == policy_name]
            mean = np.mean([r["starter_value"] for r in done])
            print(f"  {policy_name:24} {len(done):5} drafts, "
                  f"mean starter value {mean:8.2f}")

    return pd.DataFrame(rows)


# a stable per-season integer, since hash() is salted per process
def hash_season(season):
    return int(str(season)[:4])


# Paired on the seed, so the draft's own luck cancels -- which is why this reports
# a win rate rather than a difference of two means.
def compare_to_baseline(df, baseline="rung0_adp", metric="starter_value"):
    keys = ["season", "n_teams", "draft_slot", "seed"]
    base = df[df["policy"] == baseline].set_index(keys)[metric]

    rows = []
    for policy, group in df.groupby("policy"):
        if policy == baseline:
            continue

        paired = group.set_index(keys)[metric]
        common = paired.index.intersection(base.index)
        if len(common) == 0:
            continue

        diff = (paired.loc[common] - base.loc[common]).to_numpy()

        rows.append({
            "policy": policy,
            "baseline": baseline,
            "n": len(diff),
            "mean_edge": float(diff.mean()),
            "median_edge": float(np.median(diff)),
            "win_rate": float((diff > 0).mean()),
            "p05": float(np.percentile(diff, 5)),
            "p95": float(np.percentile(diff, 95)),
            # a policy that wins on average but loses from a slot is a different object, so the
            # worst slot is reported next to the mean
            "worst_slot_edge": float(
                (paired.loc[common] - base.loc[common])
                .groupby("draft_slot").mean().min()),
            "slots_won": int(
                ((paired.loc[common] - base.loc[common])
                 .groupby("draft_slot").mean() > 0).sum()),
        })

    return pd.DataFrame(rows).sort_values("mean_edge", ascending=False)


# the ladder as the writeup reports it: each rung against the one below
def ablation_ladder(df, metric="starter_value"):
    rungs = [p for p in config.POLICY_NAMES if p in set(df["policy"])]
    keys = ["season", "n_teams", "draft_slot", "seed"]

    rows = []
    for i, policy in enumerate(rungs):
        scores = df[df["policy"] == policy].set_index(keys)[metric]
        row = {
            "rung": i,
            "policy": policy,
            "question": QUESTIONS.get(policy, ""),
            "n": len(scores),
            "mean": float(scores.mean()),
            "sd": float(scores.std()),
        }

        if i > 0:
            below = df[df["policy"] == rungs[i - 1]].set_index(keys)[metric]
            common = scores.index.intersection(below.index)
            diff = (scores.loc[common] - below.loc[common]).to_numpy()
            row["edge_over_below"] = float(diff.mean())
            row["win_rate_vs_below"] = float((diff > 0).mean())
            row["p05_vs_below"] = float(np.percentile(diff, 5))
            row["p95_vs_below"] = float(np.percentile(diff, 95))

        rows.append(row)

    return pd.DataFrame(rows)


# per-slot means, the distribution §8.2 insists on over a single average
def by_slot(df, metric="starter_value"):
    return (df.pivot_table(index="draft_slot", columns="policy", values=metric,
                           aggfunc="mean")
            .reindex(columns=[p for p in config.POLICY_NAMES
                              if p in set(df["policy"])]))


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25"])
    ap.add_argument("--league-sizes", nargs="*", type=int,
                    default=[config.N_TEAMS])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    conn = connect()
    frames = []

    print(f"backtest: {len(args.seasons)} seasons x {len(args.league_sizes)} league "
          f"sizes x {args.seeds} seeds, {args.sims} sims per rollout pick")

    t0 = time.time()
    for season in args.seasons:
        for n_teams in args.league_sizes:
            print(f"\n{season}, {n_teams} teams")
            part = backtest_season(conn, season, n_teams, policies=args.policies,
                                   n_seeds=args.seeds, n_sims=args.sims,
                                   n_candidates=args.candidates, verbose=True)
            frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n{len(df)} drafts in {time.time() - t0:.0f}s\n")

    print("\nthe ablation ladder, scored on realized starter value")
    ladder = ablation_ladder(df)
    print(ladder.round(3).to_string(index=False))

    print("\nagainst the incumbent (rung 0), paired on season/slot/seed")
    print(compare_to_baseline(df).round(4).to_string(index=False))

    print("\nmean starter value by draft slot")
    print(by_slot(df).round(2).to_string())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    ladder.to_csv(LADDER_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\nwrote {LADDER_CSV}")

    if args.plots:
        import plots
        paths = plots.plot_all(df)
        for p in paths:
            print(f"wrote {p}")
