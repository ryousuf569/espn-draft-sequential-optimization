# The five rungs behind one interface. Every policy must respect roster legality
# and return a draftable board index, or the ladder compares illegal rosters.

import numpy as np
import pytest

from conftest import SEASON, load_package

_pkg = load_package("decision", ["board", "policies", "roster"])
config = _pkg["config"]
Board = _pkg["board"].Board
_policies = _pkg["policies"]
POLICIES = _policies.POLICIES
PROJECTION_POLICIES = _policies.PROJECTION_POLICIES
QUESTIONS = _policies.QUESTIONS
SEQUENCING_POLICIES = _policies.SEQUENCING_POLICIES
get_policy = _policies.get_policy
pick_by_rank = _policies.pick_by_rank
pick_by_value = _policies.pick_by_value
pick_by_vorp = _policies.pick_by_vorp
RosterState = _pkg["roster"].RosterState
position_index = _pkg["roster"].position_index
N_POSITION_SLOTS = _pkg["roster"].N_POSITION_SLOTS

GREEDY = ("rung0_adp", "rung1_best_available", "rung2_vorp_greedy")


def test_the_ladder_has_five_rungs():
    assert len(POLICIES) == 5


def test_every_rung_has_a_question():
    assert set(QUESTIONS) == set(POLICIES)


def test_get_policy_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        get_policy("rung9_wishful")


def test_only_the_top_rungs_need_the_survival_model():
    assert set(SEQUENCING_POLICIES) <= set(POLICIES)
    assert "rung2_vorp_greedy" not in SEQUENCING_POLICIES


# rung 4 is the only rung that reads Model A, which is what makes it a measurement
def test_only_rung_4_reads_the_projections():
    assert PROJECTION_POLICIES == ("rung4_own_projections",)


@pytest.fixture(scope="module")
def board(request):
    conn = request.getfixturevalue("conn")
    return Board(conn, SEASON, n_teams=12)


@pytest.fixture
def available(board):
    return np.ones(len(board), dtype=bool)


@pytest.mark.needs_db
def test_rank_following_takes_the_top_of_the_board(board, available):
    assert pick_by_rank(board, available, RosterState()) == 0


@pytest.mark.needs_db
def test_value_and_vorp_maximise_what_they_are_given(board, available):
    state = RosterState()

    by_value = pick_by_value(board, available, state, values=board.values)
    by_vorp = pick_by_vorp(board, available, state, vorps=board.vorps)

    assert board.values[by_value] == board.values.max()
    assert board.vorps[by_vorp] == board.vorps.max()


@pytest.mark.needs_db
def test_a_taken_player_is_never_returned(board):
    available = np.ones(len(board), dtype=bool)
    available[0] = False

    assert pick_by_rank(board, available, RosterState()) != 0


# the legality constraint lives inside the policy, not in a later filter
@pytest.mark.needs_db
@pytest.mark.parametrize("name", GREEDY)
def test_a_policy_never_returns_an_illegal_pick(board, available, name):
    policy = get_policy(name)

    # a roster with every guard slot and the whole bench gone cannot take a guard
    state = RosterState()
    for _ in range(4):
        state.add(position_index("G"))
    for _ in range(config.BENCH_SLOTS):
        state.add(position_index("G"))

    idx = policy(board, available, state)

    assert idx >= 0
    assert board.positions[idx] != position_index("G")


@pytest.mark.needs_db
@pytest.mark.parametrize("name", GREEDY)
def test_a_full_roster_returns_no_pick(board, available, name):
    policy = get_policy(name)

    state = RosterState()
    for pos in ("G",) * 4 + ("F",) * 4 + ("C",) * 2 + ("G",) * config.BENCH_SLOTS:
        state.add(position_index(pos))

    assert policy(board, available, state) == -1


@pytest.mark.needs_db
@pytest.mark.parametrize("name", GREEDY)
def test_an_empty_board_returns_no_pick(board, name):
    policy = get_policy(name)
    none_left = np.zeros(len(board), dtype=bool)

    assert policy(board, none_left, RosterState()) == -1


# On an empty board every rung agrees on the consensus first pick, which is the
# right answer. They diverge once the board thins and scarcity starts to bind.
@pytest.mark.needs_db
def test_the_rungs_agree_on_the_first_pick(board, available):
    picks = {name: get_policy(name)(board, available, RosterState()) for name in GREEDY}
    assert set(picks.values()) == {0}


# a ladder whose rungs never diverge measures nothing
@pytest.mark.needs_db
def test_the_rungs_diverge_once_the_board_thins(board):
    available = np.ones(len(board), dtype=bool)
    available[:24] = False

    state = RosterState()
    state.add(position_index("G"))
    state.add(position_index("G"))

    picks = {name: get_policy(name)(board, available, state) for name in GREEDY}
    assert len(set(picks.values())) > 1
