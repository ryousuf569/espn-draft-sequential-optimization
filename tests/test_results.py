# The committed result artifacts. The writeup quotes these numbers, so a rerun that
# silently changes them should fail here rather than in a PDF nobody rebuilds.

import pytest

pd = pytest.importorskip("pandas")

from conftest import ROOT

ARTIFACTS = ROOT / "artifacts"
LADDER = ARTIFACTS / "ablation_ladder.csv"
BACKTEST = ARTIFACTS / "backtest.csv"

RUNGS = ("rung0_adp", "rung1_best_available", "rung2_vorp_greedy",
         "rung3_vorp_sequencing", "rung4_own_projections")


@pytest.fixture(scope="module")
def ladder():
    if not LADDER.exists():
        pytest.skip("no ablation ladder artifact")
    return pd.read_csv(LADDER)


@pytest.fixture(scope="module")
def backtest():
    if not BACKTEST.exists():
        pytest.skip("no backtest artifact")
    return pd.read_csv(BACKTEST)


def test_the_ladder_has_all_five_rungs(ladder):
    assert list(ladder["policy"]) == list(RUNGS)


def test_every_rung_is_paired_on_the_same_drafts(ladder):
    # rungs 0-3 share inputs, so an unequal n would mean the pairing broke
    assert ladder[ladder["rung"] < 4]["n"].nunique() == 1


def test_rung_4_runs_on_one_season_only(ladder):
    # it needs projections, which exist for the target season only
    assert ladder.loc[ladder["rung"] == 4, "n"].iloc[0] < ladder["n"].max()


# the headline: following ADP beats every policy built on a fitted value estimate
def test_adp_is_the_best_rung(ladder):
    best = ladder.loc[ladder["mean"].idxmax(), "policy"]
    assert best == "rung0_adp"


def test_every_learned_rung_loses_to_the_one_below(ladder):
    edges = ladder[ladder["rung"] > 0]["edge_over_below"]
    assert (edges < 0).all()


# rung 3 vs rung 2 is the comparison the design exists for
def test_sequencing_does_not_beat_greedy(ladder):
    rung3 = ladder[ladder["policy"] == "rung3_vorp_sequencing"].iloc[0]

    assert rung3["edge_over_below"] < 0
    assert rung3["win_rate_vs_below"] < 0.5


# the interval that told me the earlier +2.35 was noise
def test_the_sequencing_interval_straddles_zero(ladder):
    rung3 = ladder[ladder["policy"] == "rung3_vorp_sequencing"].iloc[0]

    assert rung3["p05_vs_below"] < 0 < rung3["p95_vs_below"]


# rung 4 is the one rung whose interval excludes zero, and it is negative
def test_the_projection_rung_loses_decisively(ladder):
    rung4 = ladder[ladder["policy"] == "rung4_own_projections"].iloc[0]

    assert rung4["p95_vs_below"] < 0
    assert rung4["win_rate_vs_below"] < 0.05


def test_the_backtest_covers_every_policy(backtest):
    assert set(RUNGS) <= set(backtest["policy"].unique())


def test_every_draft_returns_a_legal_roster(backtest):
    # 12 or 14 players would mean the legality check leaked
    assert backtest["legal"].all()
    assert (backtest["roster_size"] == 13).all()


# realized value is signed, so starters + bench is the identity, not an inequality
def test_starters_and_bench_sum_to_the_roster(backtest):
    total = backtest["starter_value"] + backtest["bench_value"]
    assert (total - backtest["roster_value"]).abs().max() < 1e-6


def test_the_sweep_is_paired_across_policies(backtest):
    keys = ["season", "n_teams", "draft_slot", "seed"]
    counts = backtest.groupby("policy")[keys].size()

    baseline = counts[list(RUNGS[:4])]
    assert baseline.nunique() == 1


def test_the_sweep_covers_every_slot(backtest):
    # the artifact writes slots 1-indexed, so a 10-team league runs 1..10
    for n_teams, group in backtest.groupby("n_teams"):
        assert set(group["draft_slot"].unique()) == set(range(1, n_teams + 1))
