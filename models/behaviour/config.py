# Constants for the draft-behaviour model. Imported by the survival-table build,
# both fitters and the query layer, so it is the one place these are written.

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# a script run by path only sees its own directory, so the repo root goes on too
sys.path.insert(0, str(_ROOT))


# This package has a data.py, so `import data.config` finds it and deadlocks --
# claim the real package's name first so every data.* import resolves to it.
def _claim_data_package():
    existing = sys.modules.get("data")
    if existing is not None and getattr(existing, "__path__", None):
        return

    spec = importlib.util.spec_from_loader("data", loader=None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(_ROOT / "data")]
    sys.modules["data"] = module


_claim_data_package()

# owned by data/config.py, re-exported so model code has one config to import
from data.config import DATA_DIR, DB_PATH, season_str  # noqa: E402
from data.sqlite_helpers import connect  # noqa: E402


# The flip side of claiming "data" above: this package's own data.py needs a way
# in, so it is loaded by path. Every module here does config.load_tables().
def load_tables():
    name = "behaviour_data"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / "data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

ARTIFACT_DIR = _ROOT / "artifacts"

# A standard ESPN league. TOTAL_PICKS is the censoring boundary, and only the
# fallback: get_draft_length() prefers each draft's own last pick.
N_TEAMS = 10
ROUNDS_PER_DRAFT = 15
TOTAL_PICKS = N_TEAMS * ROUNDS_PER_DRAFT

# how far a draft may fall short of TOTAL_PICKS before it counts as truncated
DRAFT_LENGTH_TOLERANCE = 2 * N_TEAMS

# "Platform rank" against real data: adp has exactly one source, fantasypros_avg,
# so this is the blended consensus, not one platform. §6 narrows -- see README.
PLATFORM_ADP_SOURCE = "fantasypros_avg"

# below this many drafts a player gets no curve of his own and routes to his group
MIN_DRAFTS_FOR_PLAYER_KM = 25

# ADP buckets for the fallback curves, inclusive on platform_rank. Cut on ADP
# because pooling every guard would share one curve across the whole board.
ADP_TIERS = (
    ("elite", 1, 12),
    ("early", 13, 36),
    ("mid", 37, 84),
    ("late", 85, 150),
)

# separate because it is "ranked below the board", not a rank range
UNRANKED_TIER = "unranked"

TIER_NAMES = tuple(name for name, _, _ in ADP_TIERS) + (UNRANKED_TIER,)

# the §6 covariates. position is one-hot encoded with a reference level dropped
COX_COVARIATES = (
    "platform_rank",
    "vorp_rank_diff",
    "position",
    "injury_flag",
    "rookie_flag",
)

# the level held out of the one-hot encoding, so the design matrix is full rank
POSITION_REFERENCE = "G"

# Walk-forward folds, never random: training on 2025 to predict 2019 is a leak.
# ADP has nothing real before 2014-15, so the folds start there too.
FIRST_SEASON_YEAR = 2014
LAST_SEASON_YEAR = 2025

SEASONS = tuple(season_str(y) for y in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1))

# the last two seasons are held out; everything before is available to train
VAL_SEASONS = (season_str(2024), season_str(2025))
TRAIN_SEASONS = tuple(s for s in SEASONS if s not in VAL_SEASONS)

# the season being drafted for, which has ADP but no draft to learn from
TARGET_SEASON = season_str(2026)

# Synthetic drafts stand in for real ones: draft_results is empty and no source
# fills it. A model fit on these recovers the ADP distribution it came from.
SYNTHETIC_DRAFTS_PER_SEASON = 60

RANDOM_SEED = 17


# Model A's modules do `from config import ...` meaning THEIR config, so the name
# is swapped for the import. Reusing add_vorp() keeps the two models from drifting.
def load_projection_module(name):
    key = f"projection_{name}"
    if key in sys.modules:
        return sys.modules[key]

    proj_dir = _ROOT / "models" / "projection"
    saved = {k: sys.modules.get(k) for k in ("config", name)}
    sys.path.insert(0, str(proj_dir))

    try:
        for module_name in ("config", name):
            spec = importlib.util.spec_from_file_location(
                module_name, proj_dir / f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if module_name == name:
                sys.modules[key] = module
        return sys.modules[key]
    finally:
        sys.path.remove(str(proj_dir))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

