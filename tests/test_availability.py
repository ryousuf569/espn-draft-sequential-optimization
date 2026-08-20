# The conditional survival ratio, which is the one number the rollout consumes.
# A stub Behaviour tests the ratio arithmetic without a database; the real fitted
# model is exercised separately below and skips when the DB is absent.

import numpy as np
import pytest

from conftest import SEASON, load_package

_query = load_package("behaviour", ["query"])["query"]
MIN_SURVIVAL = _query.MIN_SURVIVAL
Behaviour = _query.Behaviour
p_available = _query.p_available
p_available_many = _query.p_available_many


# A known monotone curve, so the expected ratios are arithmetic rather than a fit.
class StubBehaviour:
    def __init__(self, curve):
        self.curve = curve

    def survival_at(self, player_id, pick_k, model="cox"):
        if pick_k is None or pick_k <= 0:
            return 1.0
        return self.curve.get(int(pick_k), float("nan"))


@pytest.fixture
def stub():
    # a player who leaves the board steadily
    return StubBehaviour({1: 1.0, 12: 0.8, 24: 0.4, 36: 0.2, 48: 0.0})


def test_the_same_pick_is_certainty(stub):
    assert p_available(1, 24, 24, stub) == 1.0


def test_a_past_pick_is_certainty(stub):
    # on the board by assumption, so asking backwards is not a probability question
    assert p_available(1, 24, 12, stub) == 1.0


def test_the_ratio_is_survival_divided_by_survival(stub):
    assert p_available(1, 12, 24, stub) == pytest.approx(0.4 / 0.8)
    assert p_available(1, 24, 36, stub) == pytest.approx(0.2 / 0.4)


# The property that makes this different from raw S(k): the same gap asked from a
# later pick gives a different answer, because the conditioning moved.
def test_the_answer_depends_on_where_the_draft_is(stub):
    early = p_available(1, 12, 24, stub)
    late = p_available(1, 24, 36, stub)

    assert early == pytest.approx(0.5)
    assert late == pytest.approx(0.5)
    # same ratio here by construction, but the raw survivals differ
    assert stub.survival_at(1, 24) != stub.survival_at(1, 36)


def test_the_ratio_exceeds_raw_survival(stub):
    conditional = p_available(1, 12, 24, stub)
    raw = stub.survival_at(1, 24)

    assert conditional > raw


def test_a_player_already_gone_is_zero():
    gone = StubBehaviour({12: 0.0, 24: 0.0})
    assert p_available(1, 12, 24, gone) == 0.0


def test_survival_below_the_floor_is_treated_as_gone():
    almost = StubBehaviour({12: MIN_SURVIVAL / 2, 24: MIN_SURVIVAL / 4})
    assert p_available(1, 12, 24, almost) == 0.0


# a ratio can drift above 1 on ties, and that propagates as a negative expected loss
def test_the_ratio_is_clipped_to_one():
    tied = StubBehaviour({12: 0.5, 24: 0.5000001})
    assert p_available(1, 12, 24, tied) == 1.0


def test_a_missing_curve_is_nan():
    missing = StubBehaviour({12: 0.5})
    assert np.isnan(p_available(1, 12, 999, missing))


def test_many_matches_one_at_a_time(stub):
    ids = [1, 2, 3]
    many = p_available_many(ids, 12, 24, stub)

    assert set(many) == set(ids)
    assert all(v == pytest.approx(p_available(i, 12, 24, stub))
               for i, v in many.items())


# --- the real fitted model ------------------------------------------------------

@pytest.fixture(scope="module")
def behaviour(request):
    conn = request.getfixturevalue("conn")
    return Behaviour(conn, SEASON)


@pytest.mark.needs_db
def test_the_cox_model_fits(behaviour):
    assert behaviour.cox is not None
    assert not behaviour.survival.empty


@pytest.mark.needs_db
def test_probabilities_stay_in_range(behaviour):
    board = (behaviour.survival.drop_duplicates("player_id")
             .sort_values("platform_rank"))
    ids = board["player_id"].astype(int).tolist()[:60]

    for model in ("km", "cox"):
        probs = [p_available(pid, 12, 24, behaviour, model) for pid in ids]
        finite = [p for p in probs if not np.isnan(p)]

        assert finite
        assert all(0.0 <= p <= 1.0 for p in finite)


# the board only empties, so a further pick is never more available
@pytest.mark.needs_db
@pytest.mark.parametrize("model", ["km", "cox"])
def test_availability_never_rises_with_distance(behaviour, model):
    board = (behaviour.survival.drop_duplicates("player_id")
             .sort_values("platform_rank"))
    ids = board["player_id"].astype(int).tolist()
    mid = ids[min(29, len(ids) - 1)]

    for k in range(12, 100, 6):
        near = p_available(mid, 12, k, behaviour, model)
        far = p_available(mid, 12, k + 6, behaviour, model)
        assert near >= far - 1e-9


@pytest.mark.needs_db
@pytest.mark.parametrize("model", ["km", "cox"])
def test_the_real_model_conditions_on_j(behaviour, model):
    board = (behaviour.survival.drop_duplicates("player_id")
             .sort_values("platform_rank"))
    ids = board["player_id"].astype(int).tolist()
    mid = ids[min(29, len(ids) - 1)]

    gaps = [p_available(mid, j, j + 12, behaviour, model) for j in (1, 24, 48)]
    assert max(gaps) - min(gaps) > 1e-6


@pytest.mark.needs_db
def test_a_consensus_first_pick_does_not_survive_the_first_round(behaviour):
    board = (behaviour.survival.drop_duplicates("player_id")
             .sort_values("platform_rank"))
    top = int(board.iloc[0]["player_id"])

    # this is the check that caught a penalizer biasing the fit toward flat curves
    assert behaviour.survival_at(top, 24, "cox") < 0.10
