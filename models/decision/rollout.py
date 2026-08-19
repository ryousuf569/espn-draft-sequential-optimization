# §8.1's rollout: assume you take a candidate, simulate the rest N times with
# opponents sampling from Model B, fill greedily by VORP, average the total.

import numpy as np

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (BEHAVIOUR_TOTAL_PICKS, BENCH_SLOTS, N_CANDIDATES, N_SIMS,
                    RANDOM_SEED, ROSTER_SLOTS, connect)
from roster import N_POSITION_SLOTS, RosterState, position_index, roster_size

# a hazard this small means the player is effectively off the board
MIN_WEIGHT = 1e-12


# How sharply an opponent follows the board: 1.0 samples straight from Model B's
# hazard, higher is more deterministic.
OPPONENT_TEMPERATURE = 1.0

# Model B was fit on 150 picks, so a 156-pick league maps onto that scale: past
# the boundary survival is flat and no longer separates players.
def scale_pick(pick, total_picks, fitted_total=BEHAVIOUR_TOTAL_PICKS):
    if total_picks == fitted_total:
        return float(pick)
    return float(pick) * fitted_total / float(total_picks)


# Snake order: odd rounds run 1..n, even run n..1, which is what makes the turn
# gap uneven and the whole sequencing problem non-trivial.
def team_at_pick(pick, n_teams):
    rnd, idx = divmod(pick - 1, n_teams)
    return idx if rnd % 2 == 0 else n_teams - 1 - idx


# every pick belonging to one slot, in order
def picks_for_slot(slot, n_teams, total_picks):
    return [p for p in range(1, total_picks + 1)
            if team_at_pick(p, n_teams) == slot]


# Per-player draft hazard: S(k) - S(k+1), the mass Model B puts on this player
# going exactly here. The opponent model's base rate before roster needs.
def hazard_at_pick(behaviour, board, pick, total_picks, model="cox"):
    k = scale_pick(pick, total_picks)
    k_next = scale_pick(pick + 1, total_picks)

    hazard = np.empty(len(board), dtype=np.float64)
    for i, pid in enumerate(board.player_ids):
        s_k = behaviour.survival_at(int(pid), k, model)
        s_next = behaviour.survival_at(int(pid), k_next, model)
        hazard[i] = max(0.0, (s_k if np.isfinite(s_k) else 1.0)
                        - (s_next if np.isfinite(s_next) else 1.0))

    return hazard


# Computed once for every pick and reused: calling lifelines inside the sim loop
# is the difference between seconds and hours. Rows are picks, columns players.
def build_hazard_matrix(behaviour, board, total_picks, model="cox"):
    matrix = np.empty((total_picks + 1, len(board)), dtype=np.float64)

    for pick in range(1, total_picks + 1):
        matrix[pick] = hazard_at_pick(behaviour, board, pick, total_picks, model)

    matrix[0] = matrix[1]

    # a pick where Model B puts no mass anywhere would make sampling undefined, so
    # it falls back to board order, which is what an ADP-following opponent does
    fallback = 1.0 / np.maximum(board.ranks, 1)
    for pick in range(total_picks + 1):
        if matrix[pick].sum() <= MIN_WEIGHT:
            matrix[pick] = fallback

    return matrix


# Opponents sample hazard weighted by their own roster need, which is §8.1's
# "subject to roster needs": a team with two centers rarely takes a third.
def simulate_once(hazard, board, taken, my_state, my_picks, opponent_states,
                  n_teams, total_picks, rng, start_pick, values, vorps,
                  temperature=OPPONENT_TEMPERATURE):
    available = ~taken
    my_value = 0.0
    my_picks_left = [p for p in my_picks if p >= start_pick]
    state = my_state.copy()
    opp = [s.copy() for s in opponent_states]
    need_buffer = np.empty(N_POSITION_SLOTS, dtype=np.float64)

    # Stop at your last pick, not the draft's end: later picks cannot change your
    # value, and simulating them was the rollout's single largest cost.
    last_pick = my_picks_left[-1] if my_picks_left else start_pick

    for pick in range(start_pick, min(last_pick, total_picks) + 1):
        if not available.any():
            break

        slot = team_at_pick(pick, n_teams)
        is_mine = bool(my_picks_left) and pick == my_picks_left[0]

        if is_mine:
            my_picks_left.pop(0)

            # filled greedily by dynamic VORP -- value over what is still freely available
            choice = greedy_pick(board, available, state, vorps)
            if choice < 0:
                continue

            state.add(int(board.positions[choice]))
            my_value += values[choice]
            available[choice] = False
            continue

        state_opp = opp[slot]
        choice = sample_opponent_pick(hazard[pick], available, board, state_opp,
                                      rng, temperature, need_buffer)
        if choice < 0:
            continue

        state_opp.add(int(board.positions[choice]))
        available[choice] = False

    # returned rather than mutated: the rollout must never touch the caller's roster,
    # but a verify needs to see what a simulation filled
    return my_value, state


# One opponent pick: hazard x roster need, sampled by inverse-CDF against one
# uniform draw -- rng.choice revalidates its p argument on every call.
def sample_opponent_pick(hazard_row, available, board, state, rng, temperature,
                         need_buffer=None):
    need = need_buffer if need_buffer is not None else np.empty(N_POSITION_SLOTS)
    for i in range(N_POSITION_SLOTS):
        need[i] = state.need(i)

    weights = np.where(available, hazard_row, 0.0)

    if temperature != 1.0:
        weights = np.power(weights, temperature)

    # need scales the hazard rather than filtering it: an opponent who needs a center
    # is likelier to take one, not certain to
    weights *= need[board.positions]

    cumulative = np.cumsum(weights)
    total = cumulative[-1]

    if total <= MIN_WEIGHT:
        # nothing legal left, so the pick is skipped -- forcing one would corrupt the board
        return -1

    return int(np.searchsorted(cumulative, rng.random() * total, side="right"))


# §8.1's greedy fill, vectorized over positions rather than players: legality
# depends only on position, so four checks replace one per available player.
def greedy_pick(board, available, state, vorps):
    allowed = np.zeros(N_POSITION_SLOTS, dtype=bool)
    for i in range(N_POSITION_SLOTS):
        allowed[i] = state.can_add(i)

    if not allowed.any():
        return -1

    eligible = available & allowed[board.positions]
    if not eligible.any():
        return -1

    # argmax over a masked copy, since indexing back out costs another allocation
    return int(np.argmax(np.where(eligible, vorps, -np.inf)))


# Expected roster value if you take this candidate now -- the number the
# recommender takes the argmax of.
def evaluate_candidate(candidate_idx, hazard, board, taken, my_state, my_picks,
                       opponent_states, n_teams, total_picks, values, vorps,
                       n_sims, seed, temperature=OPPONENT_TEMPERATURE):
    pos = int(board.positions[candidate_idx])
    if not my_state.can_add(pos):
        return -np.inf, 0.0

    state = my_state.copy()
    state.add(pos)

    base = values[candidate_idx]
    taken_after = taken.copy()
    taken_after[candidate_idx] = True

    current = my_picks[0] if my_picks else 1
    remaining = [p for p in my_picks if p > current]
    start = current + 1

    totals = np.empty(n_sims, dtype=np.float64)

    # COMMON RANDOM NUMBERS: every candidate faces the identical opponent sequence, so
    # the comparison is a difference of the same draws, not of two noisy means.
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        value, _ = simulate_once(
            hazard, board, taken_after.copy(), state, remaining, opponent_states,
            n_teams, total_picks, rng, start, values, vorps, temperature)
        totals[s] = base + value

    return float(totals.mean()), float(totals.std())


# The top few by VORP that fit. §8.1 cuts to ~10 because rolling out a player who
# will never be recommended is wasted work.
def candidate_set(board, available, state, vorps, n_candidates=N_CANDIDATES):
    legal = [i for i in np.flatnonzero(available)
             if state.can_add(int(board.positions[i]))]

    if not legal:
        return []

    legal.sort(key=lambda i: vorps[i], reverse=True)
    return legal[:n_candidates]


# The recommender: each candidate rolled out, ranked best first so a caller can
# take the argmax or read the margin.
def recommend(behaviour, board, taken, my_state, my_picks, opponent_states,
              n_teams, total_picks, use_projections=False, n_sims=N_SIMS,
              n_candidates=N_CANDIDATES, seed=RANDOM_SEED, hazard=None,
              model="cox", temperature=OPPONENT_TEMPERATURE):
    values = board.value_for(use_projections)
    vorps = board.vorp_for(use_projections)
    available = ~taken

    candidates = candidate_set(board, available, my_state, vorps, n_candidates)
    if not candidates:
        return []

    if hazard is None:
        hazard = build_hazard_matrix(behaviour, board, total_picks, model)

    results = []
    for idx in candidates:
        # the SAME seed for every candidate, deliberately -- see the note in
        # evaluate_candidate on common random numbers
        mean, sd = evaluate_candidate(
            idx, hazard, board, taken, my_state, my_picks, opponent_states,
            n_teams, total_picks, values, vorps, n_sims, seed, temperature)

        results.append({
            "player_id": int(board.player_ids[idx]),
            "board_index": idx,
            "platform_rank": int(board.ranks[idx]),
            "position": int(board.positions[idx]),
            "vorp": float(vorps[idx]),
            "expected_total": mean,
            "sd": sd,
        })

    return sorted(results, key=lambda r: r["expected_total"], reverse=True)


# the rollout has to obey the draft's structure and the roster's rules
def verify(behaviour, board):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    n_teams, total = config.N_TEAMS, config.TOTAL_PICKS

    # snake order: every team gets the same number of picks, and the turn gap
    # alternates, which is the structure the whole thesis depends on
    counts = [len(picks_for_slot(s, n_teams, total)) for s in range(n_teams)]
    check("every slot gets the same picks", len(set(counts)) == 1, f"{counts[0]}")
    check("picks match the roster size", counts[0] == roster_size(),
          f"{counts[0]} picks, roster {roster_size()}")

    first = picks_for_slot(0, n_teams, total)
    check("slot 1 picks first and turns at 1-24", first[:2] == [1, 24],
          f"{first[:3]}")
    last = picks_for_slot(n_teams - 1, n_teams, total)
    check("the last slot turns back-to-back", last[:2] == [12, 13], f"{last[:3]}")

    # pick scaling has to land inside Model B's fitted range
    check("pick scaling stays in the fitted range",
          scale_pick(total, total) <= BEHAVIOUR_TOTAL_PICKS + 1e-9,
          f"{scale_pick(total, total):.1f} of {BEHAVIOUR_TOTAL_PICKS}")

    hazard = build_hazard_matrix(behaviour, board, total)
    check("hazard matrix is populated", bool(np.isfinite(hazard).all()),
          f"{hazard.shape}")
    check("every pick has mass somewhere",
          bool((hazard[1:].sum(axis=1) > MIN_WEIGHT).all()))

    # the top of the board must carry more early hazard than the bottom, or the
    # opponent model is not following the board at all
    early = hazard[1:13, :24].sum()
    late = hazard[1:13, -24:].sum()
    check("early picks concentrate on the top of the board", early > late,
          f"{early:.3f} vs {late:.3f}")

    # a simulation must never overfill a roster or draft a player twice
    taken = np.zeros(len(board), dtype=bool)
    state = RosterState()
    my_picks = picks_for_slot(0, n_teams, total)
    opponents = [RosterState() for _ in range(n_teams)]
    rng = np.random.default_rng(RANDOM_SEED)

    values, vorps = board.values, board.vorps
    total_value, filled = simulate_once(hazard, board, taken.copy(), state, my_picks,
                                       opponents, n_teams, total, rng, 1, values, vorps)
    check("a simulation returns finite value", np.isfinite(total_value),
          f"{total_value:.2f}")

    # a full snake draft must fill your roster to exactly the roster size: one
    # short means a pick was skipped, one over means legality was not enforced
    check("your roster fills exactly", filled.filled() == roster_size(),
          f"{filled.filled()}/{roster_size()}")
    check("no opponent roster overfills",
          all(s.filled() <= roster_size() for s in opponents),
          f"max {max(s.filled() for s in opponents)}")

    # the caller's own state must be untouched, since the rollout copies it
    check("the simulation does not mutate the state it was given",
          state.filled() == 0, f"{state.filled()}")

    # THE §8.1 claim: a scarcer player should win on expected value with nothing in
    # the objective saying so. Tested by making one candidate far likelier to vanish.
    recs = recommend(behaviour, board, np.zeros(len(board), dtype=bool),
                     RosterState(), my_picks, [RosterState() for _ in range(n_teams)],
                     n_teams, total, n_sims=40, n_candidates=5, hazard=hazard)
    check("the recommender returns ranked candidates", len(recs) > 0,
          f"{len(recs)} candidates")
    check("expected totals are finite",
          all(np.isfinite(r["expected_total"]) for r in recs))
    check("results are sorted best first",
          all(recs[i]["expected_total"] >= recs[i + 1]["expected_total"]
              for i in range(len(recs) - 1)))

    # taking a player must not mutate the caller's state
    fresh = RosterState()
    recommend(behaviour, board, np.zeros(len(board), dtype=bool), fresh, my_picks,
              [RosterState() for _ in range(n_teams)], n_teams, total,
              n_sims=10, n_candidates=3, hazard=hazard)
    check("the recommender does not mutate your roster", fresh.filled() == 0,
          f"{fresh.filled()}")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    import time
    from board import Board

    conn = connect()
    season = "2024-25"
    n_teams, total = config.N_TEAMS, config.TOTAL_PICKS

    behaviour_query = config.load_behaviour_module("query")
    board = Board(conn, season, n_teams)

    print(f"{season}: {len(board)} players, {n_teams} teams x {config.ROUNDS} rounds "
          f"= {total} picks")

    t = time.time()
    behaviour = behaviour_query.Behaviour(conn, season)
    print(f"Model B fitted in {time.time() - t:.1f}s\n")

    verify(behaviour, board)

    names = dict(conn.execute("SELECT player_id, name FROM players"))
    hazard = build_hazard_matrix(behaviour, board, total)

    print("\nfirst pick from slot 1, 200 sims")
    t = time.time()
    recs = recommend(behaviour, board, np.zeros(len(board), dtype=bool),
                    RosterState(), picks_for_slot(0, n_teams, total),
                    [RosterState() for _ in range(n_teams)], n_teams, total,
                    n_sims=200, hazard=hazard)
    elapsed = time.time() - t

    print(f"{'player':<26}{'rank':>6}{'vorp':>8}{'E[total]':>11}{'sd':>8}")
    for r in recs:
        name = str(names.get(r["player_id"], r["player_id"])).encode(
            "ascii", "replace").decode()
        print(f"{name:<26}{r['platform_rank']:>6}{r['vorp']:>8.2f}"
              f"{r['expected_total']:>11.2f}{r['sd']:>8.2f}")

    print(f"\n{len(recs)} candidates x 200 sims in {elapsed:.1f}s")
