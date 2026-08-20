# The survival table and the Kaplan-Meier curves. Everything the Cox model and the
# rollout stand on: if the table is scrambled or the curves are not monotone, every
# availability number downstream is wrong in a way nothing else would catch.

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from conftest import SEASON, load_package

_pkg = load_package("behaviour", ["dataset", "kaplan_meier", "features"])
_config = _pkg["config"]
_dataset = _pkg["dataset"]
_km = _pkg["kaplan_meier"]
_features = _pkg["features"]

TOTAL_PICKS = _config.TOTAL_PICKS


@pytest.fixture(scope="module")
def survival(request):
    conn = request.getfixturevalue("conn")
    return _dataset.build_dataset(conn, SEASON)


# fit_all groups on position, so the table needs the same preparation
# query.Behaviour gives it
@pytest.fixture(scope="module")
def fitted(survival, request):
    conn = request.getfixturevalue("conn")
    return _km.fit_all(_features.attach_position(survival, SEASON, conn))


# --- the survival table ---------------------------------------------------------

@pytest.mark.needs_db
def test_the_table_is_populated(survival):
    assert not survival.empty


@pytest.mark.needs_db
def test_one_row_per_draft_and_player(survival):
    assert not survival.duplicated(["draft_id", "player_id"]).any()


@pytest.mark.needs_db
def test_the_event_flag_is_binary(survival):
    assert survival["event_observed"].isin((0, 1)).all()


@pytest.mark.needs_db
def test_durations_are_positive(survival):
    assert (survival["duration"] >= 1).all()


# a drafted player's duration is his pick, so it cannot exceed the draft
@pytest.mark.needs_db
def test_durations_stay_inside_the_draft(survival):
    longest = survival.groupby("draft_id")["duration"].max()
    assert (longest <= TOTAL_PICKS).all()


# censoring is the whole point: with no censored rows every curve falls to zero
@pytest.mark.needs_db
def test_censored_rows_exist(survival):
    censored = int((survival["event_observed"] == 0).sum())

    assert censored > 0
    assert 0.1 < censored / len(survival) < 0.9


# each draft's event count is its length, or the risk set is double counted
@pytest.mark.needs_db
def test_events_per_draft_equal_the_draft_length(survival):
    lengths = survival.groupby("draft_id")["duration"].max()
    events = survival[survival["event_observed"] == 1].groupby("draft_id").size()

    assert (events == lengths.reindex(events.index)).all()


# earlier ADP has to survive less, or the join is misaligned
@pytest.mark.needs_db
def test_a_better_board_rank_goes_earlier(survival):
    drafted = survival[survival["event_observed"] == 1]
    assert drafted["platform_rank"].corr(drafted["duration"]) > 0.5


# --- the Kaplan-Meier curves ----------------------------------------------------

@pytest.mark.needs_db
def test_curves_were_fit(fitted):
    assert len(fitted["player_curves"]) + len(fitted["group_curves"]) > 0


@pytest.mark.needs_db
def test_survival_is_a_probability(fitted):
    curve = (list(fitted["player_curves"].values())
             or list(fitted["group_curves"].values()))[0]
    values = [_km.curve_survival(curve, k) for k in range(1, TOTAL_PICKS + 1)]

    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.needs_db
def test_survival_never_increases(fitted):
    curve = (list(fitted["player_curves"].values())
             or list(fitted["group_curves"].values()))[0]
    values = [_km.curve_survival(curve, k) for k in range(1, TOTAL_PICKS + 1)]

    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


@pytest.mark.needs_db
def test_survival_starts_at_one(fitted):
    curve = (list(fitted["player_curves"].values())
             or list(fitted["group_curves"].values()))[0]

    assert np.isclose(_km.curve_survival(curve, 0), 1.0)


# the elite tier has to empty faster than the late tier, or the group cut does nothing
@pytest.mark.needs_db
def test_the_elite_tier_goes_before_the_late_tier(fitted):
    groups = fitted["group_curves"]
    elite = [c for k, c in groups.items() if "elite" in k]
    late = [c for k, c in groups.items() if "late" in k]

    if not (elite and late):
        pytest.skip("no elite/late group split in this fit")

    assert _km.curve_survival(elite[0], 24) < _km.curve_survival(late[0], 24)


# every player must resolve to something, or the rollout treats him as always there
@pytest.mark.needs_db
def test_every_player_resolves_to_a_curve(fitted):
    unresolved = [
        pid for pid in list(fitted["player_groups"])[:500]
        if _km.resolve_curve(pid, fitted["player_curves"], fitted["group_curves"],
                             fitted["player_groups"])[0] is None
    ]

    assert unresolved == []


# this is the check that caught a penalizer biasing the fit toward flat curves
@pytest.mark.needs_db
def test_the_consensus_first_pick_does_not_survive_the_draft(fitted, survival):
    top = survival.loc[survival["platform_rank"] == 1, "player_id"]
    if top.empty:
        pytest.skip("no rank-1 player on this board")

    end = _km.km_survival_prob(int(top.iloc[0]), TOTAL_PICKS,
                               fitted["player_curves"], fitted["group_curves"],
                               fitted["player_groups"])
    assert end < 0.05
