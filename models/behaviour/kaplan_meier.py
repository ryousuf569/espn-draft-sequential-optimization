# Kaplan-Meier survival curves: P(player is still on the board at pick k).

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# config first so the repo root lands on sys.path before any data.* import
import config
from config import MIN_DRAFTS_FOR_PLAYER_KM, TOTAL_PICKS, connect
from dataset import build_dataset
from features import attach_position

GROUP_COLS = ("position", "adp_tier")


# one curve over a set of (duration, event) rows -- the whole KM fit is this
def fit_curve(rows, label=None):
    kmf = KaplanMeierFitter()
    kmf.fit(rows["duration"], event_observed=rows["event_observed"], label=str(label))
    return kmf


# A personal curve for anyone with enough drafts behind him. Enough is about the
# risk set: a curve on three drafts reports 0.0 wherever he happened not to last.
def fit_player_km(survival_df, min_drafts=MIN_DRAFTS_FOR_PLAYER_KM):
    curves = {}

    for player_id, rows in survival_df.groupby("player_id"):
        if len(rows) < min_drafts:
            continue
        curves[int(player_id)] = fit_curve(rows, label=int(player_id))

    return curves


# The fallback everyone else routes to, rookies included. Cut on position for runs
# and on tier because board rank is the strongest thing there is about when he goes.
def fit_group_km(survival_df, group_cols=GROUP_COLS):
    group_cols = list(group_cols)
    curves = {}

    for key, rows in survival_df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        curves[key] = fit_curve(rows, label="-".join(str(k) for k in key))

    return curves


# Read S(k) off a fitted curve. lifelines indexes on observed event times, so
# predict() does a step-function lookup rather than interpolating between steps.
def curve_survival(kmf, pick_k):
    if pick_k is None or pd.isna(pick_k):
        return float("nan")

    k = float(pick_k)

    # before the first pick nobody is gone, so survival is exactly 1
    if k <= 0:
        return 1.0

    return float(np.clip(kmf.predict(k), 0.0, 1.0))


# Player curve, else his group, else his tier, else 1.0 -- each step widens the pool.
# Returning a fallback rather than raising keeps a KeyError out of the rollout.
def resolve_curve(player_id, player_curves, group_curves, player_groups,
                  group_cols=GROUP_COLS):
    pid = int(player_id)

    if pid in player_curves:
        return player_curves[pid], "player"

    key = player_groups.get(pid)
    if key is not None and key in group_curves:
        return group_curves[key], "group"

    # the cell is missing, so fall back within the tier: position is the weaker of the
    # two and the first thing to give up
    if key is not None and len(key) == len(group_cols):
        tier = key[group_cols.index("adp_tier")]
        same_tier = [c for k, c in group_curves.items() if tier in k]
        if same_tier:
            return same_tier[0], "tier"

    return None, "none"


# The public query. An unknown player returns 1.0 rather than an error: one nobody
# has seen drafted is, as far as the data goes, always available.
def km_survival_prob(player_id, pick_k, player_curves, group_curves, player_groups):
    kmf, _ = resolve_curve(player_id, player_curves, group_curves, player_groups)

    if kmf is None:
        return 1.0

    return curve_survival(kmf, pick_k)


# player_id -> (position, adp_tier), built once rather than re-derived per query
def build_player_groups(survival_df, group_cols=GROUP_COLS):
    group_cols = list(group_cols)
    per_player = survival_df.drop_duplicates("player_id").set_index("player_id")

    return {int(pid): tuple(row[c] for c in group_cols)
            for pid, row in per_player[group_cols].iterrows()}


# everything the query layer needs, fit in one pass
def fit_all(survival_df, min_drafts=MIN_DRAFTS_FOR_PLAYER_KM, group_cols=GROUP_COLS):
    return {
        "player_curves": fit_player_km(survival_df, min_drafts),
        "group_curves": fit_group_km(survival_df, group_cols),
        "player_groups": build_player_groups(survival_df, group_cols),
        "group_cols": tuple(group_cols),
    }


# the survival table needs position on it before the group cut can be made
def prepare(conn, season):
    survival = build_dataset(conn, season)
    return attach_position(survival, season, conn)


# a survival curve that is not monotone, or leaves [0, 1], is not a survival curve
def verify(fitted, survival_df):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    player_curves = fitted["player_curves"]
    group_curves = fitted["group_curves"]
    groups = fitted["player_groups"]

    check("curves were fit", len(player_curves) + len(group_curves) > 0,
          f"{len(player_curves)} player, {len(group_curves)} group")

    picks = list(range(1, TOTAL_PICKS + 1))
    sample = list(player_curves.values())[:1] or list(group_curves.values())[:1]
    curve = sample[0]
    values = [curve_survival(curve, k) for k in picks]

    check("survival is in [0, 1]", all(0.0 <= v <= 1.0 for v in values),
          f"{min(values):.3f}-{max(values):.3f}")
    check("survival never increases",
          all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1)))
    check("survival starts at 1", np.isclose(curve_survival(curve, 0), 1.0))

    # the elite tier has to empty faster than the late tier, or the group cut does nothing
    elite = [c for k, c in group_curves.items() if "elite" in k]
    late = [c for k, c in group_curves.items() if "late" in k]
    if elite and late:
        at_24 = (curve_survival(elite[0], 24), curve_survival(late[0], 24))
        check("elite goes before late", at_24[0] < at_24[1],
              f"{at_24[0]:.3f} vs {at_24[1]:.3f}")

    # every player must resolve to something, or the rollout treats him as always there
    unresolved = [pid for pid in list(groups)[:500]
                  if resolve_curve(pid, player_curves, group_curves, groups)[0] is None]
    check("every player resolves to a curve", not unresolved,
          f"{len(unresolved)} unresolved")

    # a top-ranked player must be gone by the end of the draft
    top = survival_df.loc[survival_df["platform_rank"] == 1, "player_id"]
    if not top.empty:
        pid = int(top.iloc[0])
        end = km_survival_prob(pid, TOTAL_PICKS, player_curves, group_curves, groups)
        check("the consensus first pick does not survive the draft", end < 0.05,
              f"S({TOTAL_PICKS}) = {end:.4f}")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    conn = connect()
    season = "2025-26"

    survival = prepare(conn, season)
    fitted = fit_all(survival)

    print(f"{season}: {len(survival)} rows, {survival['player_id'].nunique()} players, "
          f"{len(fitted['player_curves'])} player curves, "
          f"{len(fitted['group_curves'])} group curves\n")
    verify(fitted, survival)

    print("\ngroup curves, survival by pick")
    header = f"{'group':<22}" + "".join(f"{k:>9}" for k in (12, 24, 48, 96, 150))
    print(header)
    for key in sorted(fitted["group_curves"], key=lambda k: str(k)):
        curve = fitted["group_curves"][key]
        row = "".join(f"{curve_survival(curve, k):>9.3f}" for k in (12, 24, 48, 96, 150))
        print(f"{'-'.join(str(k) for k in key):<22}{row}")

    print("\ntop of the board")
    names = dict(conn.execute("SELECT player_id, name FROM players"))
    board = (survival.drop_duplicates("player_id")
             .sort_values("platform_rank").head(10))

    print(f"{'player':<26}{'rank':>6}" + "".join(f"{k:>9}" for k in (12, 24, 48)))
    for row in board.itertuples():
        pid = int(row.player_id)
        probs = "".join(
            f"{km_survival_prob(pid, k, fitted['player_curves'], fitted['group_curves'], fitted['player_groups']):>9.3f}"
            for k in (12, 24, 48))
        name = str(names.get(pid, pid)).encode("ascii", "replace").decode()
        print(f"{name:<26}{int(row.platform_rank):>6}{probs}")
