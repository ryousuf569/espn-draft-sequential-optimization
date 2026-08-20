# The only file the Monte Carlo rollout imports. Everything else here is how the
# numbers are made; this is how they are asked for.

import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
import config
from config import TARGET_SEASON, TOTAL_PICKS, connect
from cox import design_for_prediction, fit_cox_model
from dataset import build_dataset
from features import attach_position, build_cox_features
from kaplan_meier import (curve_survival, fit_all, km_survival_prob, resolve_curve)

# survival below this is zero: dividing by it turns float noise into a probability
MIN_SURVIVAL = 1e-9


# Load once, query many times: the rollout runs thousands of simulations per pick,
# so nothing here may refit or re-read the DB per call.
class Behaviour:
    def __init__(self, conn, season=TARGET_SEASON, fit_cox=True):
        self.season = season

        survival = build_dataset(conn, season)
        self.survival = attach_position(survival, season, conn)

        # KM first, and it works alone -- which is why Cox is optional here
        self.km = fit_all(self.survival)

        self.cox = None
        self.cox_rows = None

        if fit_cox:
            feats = build_cox_features(survival, season, conn)
            self.cox = fit_cox_model(feats)

            # one row per player, so a Cox query is a lookup and not a feature-table rebuild
            self.cox_rows = feats.drop_duplicates("player_id").set_index("player_id")

            # every curve computed once: the rollout asks about the same few hundred players
            # over and over, so this trades a few MB for not predicting per query
            self.cox_curves = self._precompute_cox_curves()

    # S(k | x) for every player, as a DataFrame indexed by pick
    def _precompute_cox_curves(self):
        design = design_for_prediction(self.cox, self.cox_rows)
        curves = self.cox.predict_survival_function(design)
        curves.columns = list(self.cox_rows.index)
        return curves

    # S(k) from the precomputed Cox curve, as a step-function lookup at or below k --
    # never an interpolation between steps
    def cox_survival(self, player_id, pick_k):
        pid = int(player_id)

        if self.cox is None or pid not in self.cox_curves.columns:
            return float("nan")

        if pick_k is None or pd.isna(pick_k):
            return float("nan")

        k = float(pick_k)
        if k <= 0:
            return 1.0

        index = self.cox_curves.index.to_numpy()
        position = np.searchsorted(index, k, side="right") - 1

        if position < 0:
            return 1.0

        return float(np.clip(self.cox_curves.iloc[position][pid], 0.0, 1.0))

    # S(k) from the KM curve for this player, personal or group
    def km_survival(self, player_id, pick_k):
        return km_survival_prob(player_id, pick_k, self.km["player_curves"],
                                self.km["group_curves"], self.km["player_groups"])

    def survival_at(self, player_id, pick_k, model="cox"):
        if model == "km":
            return self.km_survival(player_id, pick_k)

        value = self.cox_survival(player_id, pick_k)

        # Cox has nothing for him, so fall through to KM rather than return NaN to a rollout
        if pd.isna(value):
            return self.km_survival(player_id, pick_k)

        return value

    # which curve answered, which the calibration report needs to group on
    def source_for(self, player_id):
        if self.cox is not None and int(player_id) in self.cox_curves.columns:
            return "cox"
        return resolve_curve(player_id, self.km["player_curves"],
                            self.km["group_curves"], self.km["player_groups"])[1]


# P(still on the board at k | on the board at j). The only function the rollout calls.
def p_available(player_id, pick_j, pick_k, behaviour, model="cox"):
    if pick_k <= pick_j:
        # the same pick or one already past: he is on the board by assumption
        return 1.0

    s_j = behaviour.survival_at(player_id, pick_j, model)
    s_k = behaviour.survival_at(player_id, pick_k, model)

    if pd.isna(s_j) or pd.isna(s_k):
        return float("nan")

    # already gone at j, so 0.0 -- the useful answer when deciding whether to wait
    if s_j <= MIN_SURVIVAL:
        return 0.0

    # a ratio can drift above 1 on ties, and that propagates as a negative expected loss
    return float(np.clip(s_k / s_j, 0.0, 1.0))


# the same question for a whole board at once, which is how the rollout uses it
def p_available_many(player_ids, pick_j, pick_k, behaviour, model="cox"):
    return {int(pid): p_available(pid, pick_j, pick_k, behaviour, model)
            for pid in player_ids}


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    conn = connect()
    season = "2025-26"

    behaviour = Behaviour(conn, season)

    print(f"{season}: {behaviour.survival['player_id'].nunique()} players, "
          f"{len(behaviour.km['player_curves'])} player curves, "
          f"cox {'fit' if behaviour.cox is not None else 'skipped'}\n")

    names = dict(conn.execute("SELECT player_id, name FROM players"))
    board = (behaviour.survival.drop_duplicates("player_id")
             .sort_values("platform_rank").head(30))

    print("\nP(available at k | on the board at pick 12)")
    print(f"{'player':<26}{'rank':>6}{'src':>7}" +
          "".join(f"{k:>9}" for k in (13, 18, 24, 36)))

    for row in board.head(12).itertuples():
        pid = int(row.player_id)
        cells = "".join(f"{p_available(pid, 12, k, behaviour, 'cox'):>9.3f}"
                        for k in (13, 18, 24, 36))
        name = str(names.get(pid, pid)).encode("ascii", "replace").decode()
        print(f"{name:<26}{int(row.platform_rank):>6}"
              f"{behaviour.source_for(pid):>7}{cells}")

    print("\nthe same player, asked from different picks (km)")
    mid = int(board.iloc[len(board) // 2]["player_id"])
    name = str(names.get(mid, mid)).encode("ascii", "replace").decode()
    print(f"  {name}, gap of 12 picks")
    for j in (1, 12, 24, 48, 96):
        print(f"    from pick {j:>3}  ->  "
              f"{p_available(mid, j, j + 12, behaviour, 'cox'):.4f}")
