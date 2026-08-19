# Constants for the decision layer. Imported by the rollout, the policies and the
# backtest harness, so it is the one place these are written.

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# a script run by path only sees its own directory, so the repo root goes on too
sys.path.insert(0, str(_ROOT))


# models/behaviour/config.py already solves the data.py name clash, so loading
# it here reuses that fix and gives this package one import for both models.
def load_behaviour_module(name):
    key = f"behaviour_{name}"
    if key in sys.modules:
        return sys.modules[key]

    beh_dir = _ROOT / "models" / "behaviour"
    saved = {k: sys.modules.get(k) for k in ("config", name)}
    sys.path.insert(0, str(beh_dir))

    try:
        for module_name in ("config", name):
            if module_name != name and f"behaviour_{module_name}" in sys.modules:
                sys.modules[module_name] = sys.modules[f"behaviour_{module_name}"]
                continue
            spec = importlib.util.spec_from_file_location(
                module_name, beh_dir / f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            sys.modules[f"behaviour_{module_name}"] = module
        return sys.modules[key]
    finally:
        sys.path.remove(str(beh_dir))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# the same trick for models/projection/, whose modules also do `from config import`
def load_projection_module(name):
    return load_behaviour_module("config").load_projection_module(name)


_behaviour_config = load_behaviour_module("config")

DB_PATH = _behaviour_config.DB_PATH
ARTIFACT_DIR = _behaviour_config.ARTIFACT_DIR
connect = _behaviour_config.connect
season_str = _behaviour_config.season_str

# §8's headline league: 12 teams x 13 rounds. rollout.scale_pick handles the
# mismatch with Model B's 150-pick fit.
N_TEAMS = 12
ROUNDS = 13
TOTAL_PICKS = N_TEAMS * ROUNDS

# §8.2 sweeps league sizes, so this is a parameter. 8 makes the board deepest,
# where sequencing should have the least to exploit.
LEAGUE_SIZES = (8, 10, 12)

# Starting slots are what make a position scarce. Same shape as Model A's
# vorp.DEFAULT_SLOTS so replacement level means the same thing in both.
ROSTER_SLOTS = {"G": 4, "F": 4, "C": 2}

# Bench seats take any position, so they are the slack in the legality check.
BENCH_SLOTS = 3

# every position the slots recognise, plus the unknown bucket Model A emits
POSITIONS = ("G", "F", "C")
UNKNOWN_POSITION = "UNK"

# §8.1 budgets ~1000 sims. 250 is enough here because every candidate is scored
# against common random numbers, and the argmax is already stable by ~80.
N_SIMS = 250
N_CANDIDATES = 10

# §8.1's fallback when the clock is tight, named so the cut is a decision
N_CANDIDATES_FAST = 8

# §8.3's ladder. Rung 3 is the paper: it holds projections fixed against rung 2
# so the only thing that changes is sequencing.
POLICY_NAMES = (
    "rung0_adp",
    "rung1_best_available",
    "rung2_vorp_greedy",
    "rung3_vorp_sequencing",
    "rung4_own_projections",
)

# Rungs 0-3 run on the same inputs, so any gap between them is the policy alone.
# Only rung 4 swaps in Model A's projections.
BASELINE_POLICIES = POLICY_NAMES[:4]

# the seasons with both an ADP board to draft from and realized stats to score on
BACKTEST_SEASONS = _behaviour_config.SEASONS

# every draft slot is swept exhaustively; seeds vary the opponents' sampling
N_SEEDS = 24

RANDOM_SEED = 17

# Model B was fit on a 150-pick draft, so a 156-pick league maps its picks onto
# that scale -- past the boundary the curve is flat and says nothing.
BEHAVIOUR_TOTAL_PICKS = _behaviour_config.TOTAL_PICKS
