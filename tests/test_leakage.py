# Leak checks for Model A's features. A draft is a point in time, so a feature that
# reads the season it is meant to predict reports a better number for itself and the
# whole backtest becomes meaningless. These were the training gate before they were
# tests, which is why they are the strictest assertions in the suite.

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from conftest import load_package

_pkg = load_package("projection", ["train_rates", "feature_rates", "feature_minutes"])
_train_rates = _pkg["train_rates"]
_feature_rates = _pkg["feature_rates"]
_feature_minutes = _pkg["feature_minutes"]

CATEGORIES = _train_rates.CATEGORIES
RAW_STAT_COLS = _train_rates.RAW_STAT_COLS
feature_columns = _train_rates.feature_columns
is_safe_feature = _train_rates.is_safe_feature

# features are built as-of this season, so every row must predate it
AS_OF = "2026-27"


@pytest.fixture(scope="module")
def rate_features(request):
    conn = request.getfixturevalue("conn")
    return _feature_rates.build_rate_features(conn, AS_OF)


@pytest.fixture(scope="module")
def minutes_features(request):
    conn = request.getfixturevalue("conn")
    return _feature_minutes.build_minutes_features(conn, AS_OF)


# --- the name tests -------------------------------------------------------------

def test_raw_totals_are_not_safe_features():
    # a raw total is the target's own numerator
    assert not any(is_safe_feature(c) for c in RAW_STAT_COLS)


def test_lagged_and_age_features_are_safe():
    assert is_safe_feature("pts_rate_lag1")
    assert is_safe_feature("age")


@pytest.mark.needs_db
def test_features_were_selected(rate_features):
    assert len(feature_columns(rate_features)) > 0


@pytest.mark.needs_db
def test_no_raw_current_season_totals_are_used(rate_features):
    leaked = [c for c in feature_columns(rate_features) if c in RAW_STAT_COLS]
    assert leaked == []


@pytest.mark.needs_db
def test_no_rate_target_is_used_as_a_feature(rate_features):
    targets = {c + "_rate" for c in CATEGORIES}
    assert [c for c in feature_columns(rate_features) if c in targets] == []


@pytest.mark.needs_db
def test_every_feature_is_lagged_or_age_like(rate_features):
    unsafe = [c for c in feature_columns(rate_features) if not is_safe_feature(c)]
    assert unsafe == []


# --- the empirical test ---------------------------------------------------------

# Catches a leak no name test can see: permute the current season's stats and
# rebuild. A feature that moves was reading the season it is supposed to predict.
@pytest.mark.needs_db
def test_features_ignore_the_current_seasons_stats(rate_features):
    scrambled = rate_features.copy()
    rng = np.random.default_rng(0)

    for col in RAW_STAT_COLS:
        if col in scrambled.columns:
            scrambled[col] = rng.permutation(scrambled[col].to_numpy())

    moved = [c for c in feature_columns(rate_features)
             if c in scrambled.columns
             and not scrambled[c].equals(rate_features[c])]

    assert moved == []


# --- the lag arithmetic ---------------------------------------------------------

@pytest.mark.needs_db
def test_rate_lag1_equals_a_manual_shift(rate_features):
    grouped = rate_features.sort_values(["player_id", "season"]).groupby("player_id")
    manual = grouped["pts_rate"].shift(1)

    assert np.isclose(rate_features["pts_rate_lag1"], manual, equal_nan=True).all()


# career_min[t+1] - career_min[t] should be exactly the minutes played at t
@pytest.mark.needs_db
def test_career_minutes_exclude_the_current_season(rate_features):
    grouped = rate_features.sort_values(["player_id", "season"]).groupby("player_id")
    drift = grouped.apply(
        lambda d: (d["career_min"].shift(-1) - d["career_min"] - d["total_min"]).abs().max(),
        include_groups=False)

    assert np.nanmax(drift) < 1e-6


# a feature matching its target too well would mean the season leaked
@pytest.mark.needs_db
def test_the_lag_correlates_but_is_not_the_target(rate_features):
    corr = rate_features["pts_rate"].corr(rate_features["pts_rate_lag1"])
    assert 0.3 < corr < 0.95


@pytest.mark.needs_db
def test_minutes_lag1_equals_a_manual_shift(minutes_features):
    grouped = minutes_features.sort_values(["player_id", "season"]).groupby("player_id")

    assert np.isclose(minutes_features["mpg_lag1"], grouped["mpg"].shift(1),
                      equal_nan=True).all()


# --- the cutoff -----------------------------------------------------------------

@pytest.mark.needs_db
@pytest.mark.parametrize("frame", ["rate_features", "minutes_features"])
def test_every_season_predates_the_cutoff(request, frame):
    df = request.getfixturevalue(frame)
    assert (df["season"] < AS_OF).all()


@pytest.mark.needs_db
@pytest.mark.parametrize("frame", ["rate_features", "minutes_features"])
def test_one_row_per_player_season(request, frame):
    df = request.getfixturevalue(frame)
    assert not df.duplicated(["player_id", "season"]).any()


# today's injury list must not reach a backtest, and the cutoff is not the
# current season, so the column has to be empty here
@pytest.mark.needs_db
def test_no_status_leak_into_a_backtest(minutes_features):
    if AS_OF == _feature_minutes.CURRENT_SEASON:
        pytest.skip("as-of is the current season, where status is legitimate")

    assert minutes_features["is_inactive"].isna().all()


# --- rookies have no history to lag ---------------------------------------------

@pytest.mark.needs_db
def test_rookie_rate_lags_are_nan(rate_features):
    rookies = ~rate_features["has_history"]

    assert int(rookies.sum()) == rate_features["player_id"].nunique()
    assert rate_features.loc[rookies, "pts_rate_lag1"].isna().all()
    assert rate_features.loc[~rookies, "pts_rate_lag1"].notna().all()


@pytest.mark.needs_db
def test_rookie_minutes_lags_are_nan(minutes_features):
    rookies = minutes_features["is_rookie"] == 1

    assert int(minutes_features["is_rookie"].sum()) == minutes_features["player_id"].nunique()
    assert minutes_features.loc[rookies, "mpg_lag1"].isna().all()


# --- ranges ---------------------------------------------------------------------

@pytest.mark.needs_db
@pytest.mark.parametrize("frame", ["rate_features", "minutes_features"])
def test_ages_are_plausible(request, frame):
    df = request.getfixturevalue(frame)
    assert df["age"].between(17, 46).all()


@pytest.mark.needs_db
def test_percentages_are_in_range(rate_features):
    pct = rate_features[["fg_pct_rate", "ft_pct_rate"]].to_numpy().ravel()
    finite = pct[~np.isnan(pct)]

    assert ((finite >= 0) & (finite <= 1)).all()


@pytest.mark.needs_db
def test_no_negative_rates(rate_features):
    rates = rate_features[[c + "_rate" for c in CATEGORIES]].to_numpy().ravel()

    assert (rates[~np.isnan(rates)] >= 0).all()


@pytest.mark.needs_db
def test_every_target_varies(rate_features):
    # a target with no spread means the model has nothing to learn
    for cat in CATEGORIES:
        col = cat + "_rate"
        if col in rate_features.columns:
            assert float(rate_features[col].std()) > 0


@pytest.mark.needs_db
def test_the_vacated_minutes_share_is_a_share(minutes_features):
    vacated = minutes_features["team_min_vacated_pct"].dropna()

    assert ((vacated >= 0) & (vacated <= 1)).all()


@pytest.mark.needs_db
def test_team_turnover_is_mostly_populated(minutes_features):
    assert minutes_features["team_departures"].notna().mean() > 0.85
