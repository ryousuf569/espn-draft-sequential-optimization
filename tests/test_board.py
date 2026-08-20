# The board and the diagnosis. The paper's central claim is an ordering of rank
# correlations, so it is recomputed here rather than only asserted in prose.

import numpy as np
import pytest

from conftest import SEASON, load_package

_pkg = load_package("decision", ["board"])
config = _pkg["config"]
Board = _pkg["board"].Board
_spearman = _pkg["board"]._spearman


def test_spearman_is_one_on_a_monotone_pair():
    a = np.arange(50, dtype=float)
    b = a ** 3
    assert _spearman(a, b) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed():
    a = np.arange(50, dtype=float)
    assert _spearman(a, -a) == pytest.approx(-1.0)


# Pearson understates a convex curve, which is why the diagnosis uses ranks
def test_spearman_beats_pearson_on_a_convex_curve():
    a = np.arange(2, 60, dtype=float)
    b = np.exp(a / 8.0)

    pearson = float(np.corrcoef(a, b)[0, 1])
    assert _spearman(a, b) > pearson


@pytest.fixture(scope="module")
def board(request):
    conn = request.getfixturevalue("conn")
    return Board(conn, SEASON, n_teams=12, use_projections=True)


@pytest.mark.needs_db
def test_the_board_is_populated(board):
    assert len(board) > 100


@pytest.mark.needs_db
def test_ranks_are_unique_and_dense(board):
    assert np.array_equal(np.sort(board.ranks), np.arange(1, len(board) + 1))


@pytest.mark.needs_db
def test_the_board_is_sorted_by_rank(board):
    assert list(board.ranks) == sorted(board.ranks)


@pytest.mark.needs_db
def test_every_value_is_finite(board):
    assert np.isfinite(board.values).all()
    assert np.isfinite(board.vorps).all()
    assert np.isfinite(board.realized).all()


# a player who never played earns the floor, not NaN: dropping him would flatter
# whoever drafted him
@pytest.mark.needs_db
def test_players_who_never_played_get_the_floor(board):
    assert not np.isnan(board.realized).any()


@pytest.mark.needs_db
def test_index_of_round_trips(board):
    for i in (0, len(board) // 2, len(board) - 1):
        assert board.index_of[int(board.player_ids[i])] == i


# rungs 0-3 must see identical inputs, so switching the flag off changes nothing
@pytest.mark.needs_db
def test_baseline_rungs_share_inputs(board):
    assert np.array_equal(board.value_for(False), board.values)
    assert np.array_equal(board.vorp_for(False), board.vorps)


@pytest.mark.needs_db
def test_projections_actually_change_the_values(board):
    if board.projected is None:
        pytest.skip("no projections for this season")

    # rung 4 minus rung 3 is only a measurement if the inputs differ
    assert not np.array_equal(board.value_for(True), board.values)


# --- the diagnosis --------------------------------------------------------------

# The finding: ADP rank predicts realized value better than anything fitted here.
# It is a pooled result, not a per-season one -- board value wins in 8 of 24
# season x league-size combinations, so the claim is tested where it is made.
@pytest.fixture(scope="module")
def rho_by_season(request):
    conn = request.getfixturevalue("conn")
    rows = []

    for season in config.BACKTEST_SEASONS:
        board = Board(conn, season, n_teams=12)
        rows.append({
            "season": season,
            "adp": _spearman(-board.ranks.astype(float), board.realized),
            "value": _spearman(board.values, board.realized),
            "vorp": _spearman(board.vorps, board.realized),
        })

    return rows


@pytest.mark.needs_db
def test_adp_rank_outpredicts_every_fitted_estimate_on_average(rho_by_season):
    adp = np.mean([r["adp"] for r in rho_by_season])
    value = np.mean([r["value"] for r in rho_by_season])
    vorp = np.mean([r["vorp"] for r in rho_by_season])

    assert adp > value
    assert adp > vorp


# the margin is small, which is exactly why the policies lose by so little
@pytest.mark.needs_db
def test_the_ranking_margin_is_narrow(rho_by_season):
    adp = np.mean([r["adp"] for r in rho_by_season])
    value = np.mean([r["value"] for r in rho_by_season])

    assert 0.0 < adp - value < 0.10


# a single season does not settle it, and the suite should not pretend otherwise
@pytest.mark.needs_db
def test_adp_does_not_win_every_season(rho_by_season):
    wins = sum(1 for r in rho_by_season if r["adp"] > r["value"])

    assert wins > len(rho_by_season) / 2
    assert wins < len(rho_by_season)


# every estimate is still informative, so a broken join would show up as noise
@pytest.mark.needs_db
def test_every_estimate_is_positively_correlated_with_realized_value(board):
    assert _spearman(-board.ranks.astype(float), board.realized) > 0.3
    assert _spearman(board.values, board.realized) > 0.3
    assert _spearman(board.vorps, board.realized) > 0.3
