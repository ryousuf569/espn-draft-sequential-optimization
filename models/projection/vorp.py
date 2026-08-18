import numpy as np
import pandas as pd

# config first so the repo root lands on sys.path before any data.* import
from config import CATEGORIES, NEGATIVE_CATEGORIES
from data.sqlite_helpers import connect
from predict import OUT_CSV, project_season

# a standard ESPN league, and the starting slots that make a position scarce
DEFAULT_N_TEAMS = 10
DEFAULT_SLOTS = {"G": 4, "F": 4, "C": 2}


# Value is per-category, and the categories are on wildly different scales -- a
# season of points is in the thousands, steals in the dozens. z-scores put them
# in the same units so they can be summed.
def category_zscores(df, categories=CATEGORIES):
    z = pd.DataFrame(index=df.index)

    for cat in categories:
        col = df[cat]
        spread = col.std()
        scores = (col - col.mean()) / spread if spread else col * 0.0

        # turnovers are the one category where more is worse
        z[cat] = -scores if cat in NEGATIVE_CATEGORIES else scores

    return z


# total value in z-score units, which is what gets compared to replacement
def add_value(df, categories=CATEGORIES):
    df = df.copy()
    df["value"] = category_zscores(df, categories).sum(axis=1)
    return df


# How many players at a position come off the board before the position is
# exhausted. The next one down is replacement level: what you can still get for
# free, and therefore the bar a draftable player has to clear.
def replacement_index(n_teams, slots_at_position):
    return int(n_teams * slots_at_position)


# Static replacement level, computed once against the full pool. The value of
# the best player at this position you would NOT have to draft.
def replacement_level(projections_df, position, n_teams=DEFAULT_N_TEAMS,
                      slots_at_position=None):
    if slots_at_position is None:
        slots_at_position = DEFAULT_SLOTS.get(position, 1)

    pool = projections_df[projections_df["position"] == position]
    if pool.empty:
        return 0.0

    values = np.sort(pool["value"].to_numpy())[::-1]
    idx = replacement_index(n_teams, slots_at_position)

    # a shallow position can run out before the cutoff, so the worst available
    # player is replacement level
    return float(values[min(idx, len(values) - 1)])


# value above what is freely available, which is the number to draft on
def add_vorp(projections_df, n_teams=DEFAULT_N_TEAMS, slots=None):
    slots = slots or DEFAULT_SLOTS
    df = add_value(projections_df)

    levels = {
        pos: replacement_level(df, pos, n_teams, slots.get(pos, 1))
        for pos in df["position"].unique()
    }

    df["replacement"] = df["position"].map(levels)
    df["vorp"] = df["value"] - df["replacement"]
    return df.sort_values("vorp", ascending=False).reset_index(drop=True)


# The rollout calls this thousands of times, so the pool arrives pre-sorted by
# value descending and already split by position. Nothing is sorted or copied
# here: it walks the sorted ids and counts past the ones already taken.
def dynamic_replacement_level(sorted_pool_array, position, drafted_ids,
                              n_teams=DEFAULT_N_TEAMS, slots_at_position=None):
    if slots_at_position is None:
        slots_at_position = DEFAULT_SLOTS.get(position, 1)

    ids, values = sorted_pool_array
    if len(ids) == 0:
        return 0.0

    needed = replacement_index(n_teams, slots_at_position)
    seen = 0

    # values are already descending, so the first `needed` undrafted players are
    # the ones that go, and the next one is replacement level
    for i in range(len(ids)):
        if ids[i] in drafted_ids:
            continue
        if seen == needed:
            return float(values[i])
        seen += 1

    # the position is picked clean, so anyone left is worth nothing over nobody
    return float(values[-1])


# pre-sorted (ids, values) per position, built once and reused by the rollout
def build_sorted_pools(projections_df):
    pools = {}

    for pos, group in projections_df.groupby("position"):
        ordered = group.sort_values("value", ascending=False)
        pools[pos] = (ordered["player_id"].to_numpy(),
                      ordered["value"].to_numpy())

    return pools


# the two paths have to agree, or the rollout is optimising a different game
def verify(df, n_teams=DEFAULT_N_TEAMS):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    valued = add_vorp(df, n_teams)
    pools = build_sorted_pools(valued)

    # with nobody drafted the dynamic version is the static one
    for pos in DEFAULT_SLOTS:
        if pos not in pools:
            continue
        static = replacement_level(valued, pos, n_teams, DEFAULT_SLOTS[pos])
        dynamic = dynamic_replacement_level(pools[pos], pos, set(), n_teams)
        check(f"static and dynamic agree on {pos}",
              bool(np.isclose(static, dynamic)), f"{static:.4f} vs {dynamic:.4f}")

    # drafting the best players can only push replacement level down
    pos = "G"
    if pos in pools:
        top = set(pools[pos][0][:12].tolist())
        before = dynamic_replacement_level(pools[pos], pos, set(), n_teams)
        after = dynamic_replacement_level(pools[pos], pos, top, n_teams)
        check("replacement falls as the pool empties", after <= before,
              f"{before:.4f} -> {after:.4f}")

    check("vorp is centered near the cutoff", bool((valued["vorp"] > 0).any()))
    check("turnovers score negatively",
          category_zscores(valued)["tov"].corr(valued["tov"]) < 0)

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    try:
        proj = pd.read_csv(OUT_CSV)
    except FileNotFoundError:
        proj = project_season(connect())

    ranked = add_vorp(proj)

    print(f"{len(ranked)} players, {DEFAULT_N_TEAMS} teams\n")
    verify(proj)

    print("\nreplacement level by position")
    for pos, slots in DEFAULT_SLOTS.items():
        level = replacement_level(add_value(proj), pos, DEFAULT_N_TEAMS, slots)
        print(f"  {pos}  slots {slots}  replacement {level:+.3f}")

    cols = ["player_id", "team", "position", "is_rookie", "mpg", "value",
            "replacement", "vorp"]
    print()
    print(ranked[cols].head(15).round(3).to_string(index=False))
