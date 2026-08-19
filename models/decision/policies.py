# §8.3's ladder: five policies behind one interface, so the harness swaps them and
# changes nothing else. Each returns a board index to draft.

import numpy as np

# config first so the repo root lands on sys.path before any data.* import
import config
from config import N_CANDIDATES, N_SIMS, RANDOM_SEED
from roster import N_POSITION_SLOTS
from rollout import greedy_pick, recommend


# highest-ranked player who fits, which is what following a default list does
def pick_by_rank(board, available, state, **kwargs):
    allowed = np.zeros(N_POSITION_SLOTS, dtype=bool)
    for i in range(N_POSITION_SLOTS):
        allowed[i] = state.can_add(i)

    eligible = available & allowed[board.positions]
    if not eligible.any():
        return -1

    # ranks ascend, so the best available is the smallest rank
    return int(np.argmin(np.where(eligible, board.ranks, np.iinfo(np.int32).max)))


# highest raw value that fits -- best available, no positional adjustment
def pick_by_value(board, available, state, values=None, **kwargs):
    values = board.values if values is None else values
    return greedy_pick(board, available, state, values)


# highest VORP that fits: the rollout's own fallback, with no rollout
def pick_by_vorp(board, available, state, vorps=None, **kwargs):
    vorps = board.vorps if vorps is None else vorps
    return greedy_pick(board, available, state, vorps)


# Rung 0: the incumbent, and the thing to beat -- what most of a real league does
def rung0_adp(board, available, state, **kwargs):
    return pick_by_rank(board, available, state)


# Rung 1: same board, ordered by the value it implies rather than rank position
def rung1_best_available(board, available, state, **kwargs):
    return pick_by_value(board, available, state, values=board.value_for(False))


# Rung 2: value minus replacement, so a scarce center outranks a replaceable guard
def rung2_vorp_greedy(board, available, state, **kwargs):
    return pick_by_vorp(board, available, state, vorps=board.vorp_for(False))


# Rung 3, the paper: identical inputs to rung 2, but it takes the argmax of
# expected final value instead of value right now.
def rung3_vorp_sequencing(board, available, state, behaviour=None, my_picks=None,
                          opponent_states=None, n_teams=None, total_picks=None,
                          hazard=None, n_sims=N_SIMS, n_candidates=N_CANDIDATES,
                          seed=RANDOM_SEED, **kwargs):
    if behaviour is None or my_picks is None:
        return pick_by_vorp(board, available, state, vorps=board.vorp_for(False))

    recs = recommend(behaviour, board, ~available, state, my_picks, opponent_states,
                     n_teams, total_picks, use_projections=False, n_sims=n_sims,
                     n_candidates=n_candidates, seed=seed, hazard=hazard)

    if not recs:
        return -1

    return int(recs[0]["board_index"])


# Rung 4: rung 3's sequencing, drafting on Model A's projections instead
def rung4_own_projections(board, available, state, behaviour=None, my_picks=None,
                          opponent_states=None, n_teams=None, total_picks=None,
                          hazard=None, n_sims=N_SIMS, n_candidates=N_CANDIDATES,
                          seed=RANDOM_SEED, **kwargs):
    if behaviour is None or my_picks is None:
        return pick_by_vorp(board, available, state, vorps=board.vorp_for(True))

    recs = recommend(behaviour, board, ~available, state, my_picks, opponent_states,
                     n_teams, total_picks, use_projections=True, n_sims=n_sims,
                     n_candidates=n_candidates, seed=seed, hazard=hazard)

    if not recs:
        return -1

    return int(recs[0]["board_index"])


POLICIES = {
    "rung0_adp": rung0_adp,
    "rung1_best_available": rung1_best_available,
    "rung2_vorp_greedy": rung2_vorp_greedy,
    "rung3_vorp_sequencing": rung3_vorp_sequencing,
    "rung4_own_projections": rung4_own_projections,
}

# which rungs need Model B, so the harness only builds the hazard matrix when used
SEQUENCING_POLICIES = ("rung3_vorp_sequencing", "rung4_own_projections")

# which rung reads Model A, so a season without projections is reported unavailable
PROJECTION_POLICIES = ("rung4_own_projections",)

# what each rung asks, for the ablation table in the writeup
QUESTIONS = {
    "rung0_adp": "the incumbent -- the thing to beat",
    "rung1_best_available": "does value-based drafting beat rank-following?",
    "rung2_vorp_greedy": "does positional scarcity matter?",
    "rung3_vorp_sequencing": "does sequence-awareness matter, projections fixed?",
    "rung4_own_projections": "what was the projection model worth?",
}


def get_policy(name):
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name}, expected one of {sorted(POLICIES)}")
    return POLICIES[name]


# the ladder has to be a ladder: each rung must differ from the one below it
def verify(board):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    from roster import RosterState

    available = np.ones(len(board), dtype=bool)
    check("every rung is registered",
          set(POLICIES) == set(config.POLICY_NAMES), f"{len(POLICIES)} rungs")

    # on an empty board each greedy rung must return something legal
    for name in config.BASELINE_POLICIES:
        idx = POLICIES[name](board, available, RosterState())
        legal = idx >= 0 and idx < len(board)
        check(f"{name} picks a legal player", legal, f"index {idx}")

    # rung 0 must take the consensus first pick, by definition
    idx0 = rung0_adp(board, available, RosterState())
    check("rung 0 takes board rank 1", int(board.ranks[idx0]) == 1,
          f"rank {int(board.ranks[idx0])}")

    # rung 1 must take the highest implied value, which need not be rank 1
    idx1 = rung1_best_available(board, available, RosterState())
    check("rung 1 takes the highest value",
          np.isclose(board.values[idx1], board.values.max()),
          f"value {board.values[idx1]:.2f}")

    # rung 2 must take the highest VORP, and this is the rung-1-vs-2 distinction
    idx2 = rung2_vorp_greedy(board, available, RosterState())
    check("rung 2 takes the highest vorp",
          np.isclose(board.vorps[idx2], board.vorps.max()),
          f"vorp {board.vorps[idx2]:.2f}")

    # if rungs 1 and 2 always agreed, scarcity would be doing nothing and the ladder
    # would carry a redundant rung
    agree = 0
    for pos_filled in range(4):
        state = RosterState()
        for _ in range(pos_filled):
            state.add(0)
        if (rung1_best_available(board, available, state)
                == rung2_vorp_greedy(board, available, state)):
            agree += 1
    check("value and vorp disagree somewhere", agree < 4, f"{agree}/4 agreed")

    # rungs 3 and 4 degrade to their greedy equivalent without Model B rather than
    # crash, so a fold with no fitted model still produces a roster
    idx3 = rung3_vorp_sequencing(board, available, RosterState())
    check("rung 3 falls back to vorp with no model", idx3 == idx2,
          f"{idx3} vs {idx2}")

    # a full roster must return -1 from every rung, not raise
    full = RosterState()
    for pos in (0,) * 4 + (1,) * 4 + (2,) * 2 + (0,) * 3:
        full.add(pos)
    for name in config.BASELINE_POLICIES:
        check(f"{name} returns -1 on a full roster",
              POLICIES[name](board, available, full) == -1)

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    from board import Board
    from config import connect

    conn = connect()
    board = Board(conn, "2024-25", config.N_TEAMS)

    print(f"the ablation ladder, {board.season}\n")
    for name in config.POLICY_NAMES:
        print(f"  {name:24} {QUESTIONS[name]}")
    print()

    verify(board)

    # what the greedy rungs actually take first, which is the clearest single view
    # of how the rungs differ
    from roster import RosterState
    names = dict(conn.execute("SELECT player_id, name FROM players"))
    available = np.ones(len(board), dtype=bool)

    print("\nfirst pick by rung, from an empty board")
    print(f"{'rung':<24}{'player':<26}{'rank':>6}{'value':>8}{'vorp':>8}")
    for name in config.BASELINE_POLICIES:
        idx = POLICIES[name](board, available, RosterState())
        player = str(names.get(int(board.player_ids[idx]),
                              board.player_ids[idx])).encode("ascii", "replace").decode()
        print(f"{name:<24}{player:<26}{int(board.ranks[idx]):>6}"
              f"{board.values[idx]:>8.2f}{board.vorps[idx]:>8.2f}")
