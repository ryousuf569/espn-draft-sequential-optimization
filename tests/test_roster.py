# Roster legality. A rollout that drafts 12 or 14 players scores a roster no real
# league would allow, so these are the assertions the harness depends on.

import pytest

from conftest import load_package

_pkg = load_package("decision", ["roster"])
roster = _pkg["roster"]

N_POSITION_SLOTS = roster.N_POSITION_SLOTS
UNKNOWN_INDEX = roster.UNKNOWN_INDEX
RosterState = roster.RosterState
position_index = roster.position_index
roster_size = roster.roster_size
slot_capacity = roster.slot_capacity

BENCH = roster.BENCH_SLOTS


def test_roster_size_is_slots_plus_bench():
    assert roster_size() == 4 + 4 + 2 + BENCH


def test_slot_capacity_matches_the_configured_slots():
    capacity, bench = slot_capacity()
    assert capacity[position_index("G")] == 4
    assert capacity[position_index("F")] == 4
    assert capacity[position_index("C")] == 2
    assert bench == BENCH


# UNK is bench-only rather than dropped, so it has no starting capacity
def test_unknown_position_has_no_starting_slot():
    capacity, _ = slot_capacity()
    assert capacity[UNKNOWN_INDEX] == 0


def test_unknown_positions_map_to_the_unknown_bucket():
    assert position_index("UNK") == UNKNOWN_INDEX
    assert position_index("not-a-position") == UNKNOWN_INDEX


# the bench is slack, so a starting slot must never consume it while it is open
def test_starting_slots_fill_before_the_bench():
    state = RosterState()
    for _ in range(4):
        state.add(position_index("G"))

    assert state.bench_used == 0
    assert state.counts[position_index("G")] == 4


def test_a_fifth_guard_takes_a_bench_seat():
    state = RosterState()
    for _ in range(5):
        state.add(position_index("G"))

    assert state.bench_used == 1
    assert state.counts[position_index("G")] == 4


def test_filled_counts_every_seat_including_the_bench():
    state = RosterState()
    for _ in range(5):
        state.add(position_index("G"))

    assert state.filled() == 5


# this is the invariant that keeps a draft from returning the wrong roster length
def test_a_full_roster_rejects_every_position():
    state = RosterState()
    for pos in ("G",) * 4 + ("F",) * 4 + ("C",) * 2 + ("G",) * BENCH:
        state.add(position_index(pos))

    assert state.is_full()
    assert state.filled() == state.size
    assert not any(state.can_add(i) for i in range(N_POSITION_SLOTS))
    assert state.open_positions() == []


def test_adding_to_a_full_roster_raises():
    state = RosterState()
    for pos in ("G",) * 4 + ("F",) * 4 + ("C",) * 2 + ("G",) * BENCH:
        state.add(position_index(pos))

    with pytest.raises(ValueError):
        state.add(position_index("G"))


def test_unknown_positions_exhaust_the_bench_and_stop():
    state = RosterState()
    assert state.can_add(UNKNOWN_INDEX)

    for _ in range(BENCH):
        state.add(UNKNOWN_INDEX)

    assert not state.can_add(UNKNOWN_INDEX)
    # a real position still has its starting slot
    assert state.can_add(position_index("G"))


# one simulation mutating another would correlate every rollout draw
def test_copies_do_not_alias():
    original = RosterState()
    original.add(position_index("G"))

    clone = original.copy()
    clone.add(position_index("G"))

    assert original.counts[position_index("G")] == 1
    assert clone.counts[position_index("G")] == 2
    assert original.filled() == 1 and clone.filled() == 2


def test_need_falls_as_a_position_fills():
    state = RosterState()
    before = state.need(position_index("C"))
    state.add(position_index("C"))

    assert state.need(position_index("C")) < before


def test_need_is_zero_when_a_position_cannot_be_added():
    state = RosterState()
    for pos in ("G",) * 4 + ("F",) * 4 + ("C",) * 2 + ("G",) * BENCH:
        state.add(position_index(pos))

    assert state.need(position_index("G")) == 0.0


# bench-only means wanted but not needed, which is what keeps the opponent model
# from chasing a position it has already filled
def test_bench_only_need_is_below_a_starting_slot():
    state = RosterState()
    for _ in range(2):
        state.add(position_index("C"))

    assert 0.0 < state.need(position_index("C")) < 1.0


def test_open_positions_tracks_can_add():
    state = RosterState()
    for _ in range(4):
        state.add(position_index("G"))

    expected = [i for i in range(N_POSITION_SLOTS) if state.can_add(i)]
    assert state.open_positions() == expected
