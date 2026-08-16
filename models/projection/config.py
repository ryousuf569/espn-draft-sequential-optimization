"""Constants for the projection model. Imported by the feature builders, the
shrinkage fit and the scoring code, so it is the one place these are written.
"""

import sys
from pathlib import Path

# running a script by path only puts that script's directory on sys.path, so the
# repo root goes on explicitly and `data.*` imports work however this is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# re-exported so model code has one config to import, but owned by data/config.py:
# resolving them here would point at models/projection/, where there is no db
from data.config import DATA_DIR, DB_PATH, season_str  # noqa: E402

# the 9 ESPN roto categories. Every per-category array indexes on this order.
CATEGORIES = ("pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "fg_pct", "ft_pct")

# tov hurts, so the sign flips once at the z-score step and nowhere else
NEGATIVE_CATEGORIES = ("tov",)

# the ratios are derived, and shrink on attempts rather than on games
RATIO_CATEGORIES = {
    "fg_pct": ("fgm", "fga"),
    "ft_pct": ("ftm", "fta"),
}

# the other 7, stored as season totals and projected as per-minute rates
COUNTING_CATEGORIES = tuple(c for c in CATEGORIES if c not in RATIO_CATEGORIES)

# Empirical-Bayes weight on a player's own rate: w = n / (n + k), blended as
# w * observed + (1 - w) * prior. n is the category's exposure, so k is in
# minutes for a counting stat and attempts for a ratio.
#
# PLACEHOLDER, eyeballed to get the pipeline running. Step 7 fits these on
# out-of-sample error and overwrites the dict.
SHRINKAGE_K = {
    "pts": 500.0,
    "reb": 500.0,
    "ast": 500.0,
    "stl": 500.0,
    "blk": 500.0,
    "fg3m": 500.0,
    "tov": 500.0,
    "fg_pct": 300.0,
    "ft_pct": 150.0,
}

# so a category missing from the dict falls back to a heavy prior, not a KeyError
DEFAULT_SHRINKAGE_K = 500.0

# Pick buckets for the rookie prior: a rookie has no NBA rate of his own, so the
# prior comes from what players taken near the same slot actually did. Bounds
# are inclusive on overall_pick.
DRAFT_TIERS = (
    ("lottery", 1, 14),
    ("late_first", 15, 30),
    ("second_round", 31, 60),
)

# separate because undrafted is overall_pick IS NULL, not a pick range
UNDRAFTED_TIER = "undrafted"

# every tier label in order, for grouping and for the prior table's columns
TIER_NAMES = tuple(name for name, _, _ in DRAFT_TIERS) + (UNDRAFTED_TIER,)

# Training seasons, oldest first. The model is fit on season-to-season
# transitions and the backtest walks forward, so index i-1 has to be the season
# before index i. Ends at 2025-26, the last season with a full stat line.
FIRST_SEASON_YEAR = 2010
LAST_SEASON_YEAR = 2025

SEASONS = tuple(season_str(y) for y in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1))

# what every projection is aimed at, and the one season never used as a label
TARGET_SEASON = season_str(LAST_SEASON_YEAR + 1)

# too noisy to learn from. Players under these can still be projected.
MIN_GP = 20
MIN_MPG = 10.0
