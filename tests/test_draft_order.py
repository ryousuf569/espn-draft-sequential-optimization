# Snake order and pick scaling. The uneven turn gap is the entire reason
# sequencing is a question, so an off-by-one here would quietly change the paper.

import pytest

from conftest import load_package

_rollout = load_package("decision", ["rollout"])["rollout"]
picks_for_slot = _rollout.picks_for_slot
scale_pick = _rollout.scale_pick
team_at_pick = _rollout.team_at_pick


def test_first_round_runs_forward():
    assert [team_at_pick(p, 10) for p in range(1, 11)] == list(range(10))


def test_second_round_runs_backward():
    assert [team_at_pick(p, 10) for p in range(11, 21)] == list(reversed(range(10)))


def test_third_round_runs_forward_again():
    assert [team_at_pick(p, 10) for p in range(21, 31)] == list(range(10))


# the turn ends of a snake pick back-to-back, which is the whole asymmetry
def test_the_turn_picks_back_to_back():
    assert team_at_pick(10, 10) == team_at_pick(11, 10) == 9
    assert team_at_pick(20, 10) == team_at_pick(21, 10) == 0


@pytest.mark.parametrize("n_teams", [8, 10, 12])
def test_every_pick_belongs_to_exactly_one_team(n_teams):
    total = n_teams * 13
    owned = [team_at_pick(p, n_teams) for p in range(1, total + 1)]

    assert set(owned) == set(range(n_teams))
    # a snake is balanced, so every team owns the same number of picks
    assert all(owned.count(t) == 13 for t in range(n_teams))


@pytest.mark.parametrize("slot", range(10))
def test_picks_for_slot_agrees_with_team_at_pick(slot):
    picks = picks_for_slot(slot, 10, 130)

    assert len(picks) == 13
    assert picks == sorted(picks)
    assert all(team_at_pick(p, 10) == slot for p in picks)


def test_slots_partition_the_board():
    all_picks = sorted(p for slot in range(10)
                       for p in picks_for_slot(slot, 10, 130))
    assert all_picks == list(range(1, 131))


# slot 0 waits the longest between its first two turns, slot n-1 the shortest.
# This gap is what a sequencing policy is supposed to exploit.
def test_the_first_slot_has_the_longest_wait():
    first = picks_for_slot(0, 10, 130)
    last = picks_for_slot(9, 10, 130)

    assert first[1] - first[0] == 19
    assert last[1] - last[0] == 1


def test_scale_pick_is_identity_at_the_fitted_size():
    assert scale_pick(37, 150, fitted_total=150) == 37.0


# Model B was fit on 150 picks, so a 156-pick league maps onto that scale
def test_scale_pick_maps_a_larger_league_down():
    assert scale_pick(156, 156, fitted_total=150) == pytest.approx(150.0)
    assert scale_pick(78, 156, fitted_total=150) == pytest.approx(75.0)


def test_scale_pick_is_monotone():
    scaled = [scale_pick(p, 156, fitted_total=150) for p in range(1, 157)]
    assert scaled == sorted(scaled)
