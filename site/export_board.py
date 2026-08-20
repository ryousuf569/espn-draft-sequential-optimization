# Offline export. The only file in site/ that touches the database or imports
# lifelines -- the deployed app reads board.json and nothing else.
#
# Writes survival[player_id][pick] from the fitted Cox model as int16 scaled by
# 1000, which is what makes a 256x150 matrix a ~175 KB commit instead of a 1.5 MB
# one. The app divides two of those lookups; see the conditional note below.

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOUR_DIR = ROOT / "models" / "behaviour"
OUT_PATH = Path(__file__).resolve().parent / "board.json"

# The behaviour package is imported by path, the way its own modules do it: those
# do `import config` meaning THEIR config, so that directory has to lead sys.path.
sys.path.insert(0, str(BEHAVIOUR_DIR))

import config  # noqa: E402  (claims the data package; must precede data.* imports)
from config import PLATFORM_ADP_SOURCE, connect  # noqa: E402
from cox import design_for_prediction, fit_cox_model  # noqa: E402
from dataset import build_dataset  # noqa: E402
from features import build_cox_features  # noqa: E402

# 2025-26, not config.TARGET_SEASON: the adp table stops there, and a board with
# no ADP has no ranking to sort by and no platform_rank to condition the fit on.
SEASON = "2025-26"

# survival is quantised to 1/1000 -- three digits is finer than the model's own
# resolution and the app renders a whole percent
SCALE = 1000

# picks 1..150, the horizon the Cox curves are defined over (10 teams x 15 rounds)
LAST_PICK = 150

# rosters.position is the season's actual assignment and players.position is the
# career one, so the roster wins where it exists -- same precedence as features.py
POSITION_FALLBACK = {
    "Guard": "G", "Forward": "F", "Center": "C",
    "Guard-Forward": "G-F", "Forward-Guard": "F-G",
    "Forward-Center": "F-C", "Center-Forward": "C-F",
}


# name, team and position for the exported ids. This is the whole player identity
# the app shows -- no headshot url, because the app requests no images.
def load_identity(conn, player_ids, season=SEASON):
    ids = [int(p) for p in player_ids]
    marks = ",".join("?" * len(ids))

    players = pd.read_sql_query(
        f"SELECT player_id, name, position FROM players WHERE player_id IN ({marks})",
        conn, params=ids)
    players["position"] = players["position"].map(POSITION_FALLBACK)

    roster = pd.read_sql_query(
        "SELECT player_id, team, position AS roster_position FROM rosters "
        f"WHERE season = ? AND player_id IN ({marks})",
        conn, params=[season, *ids]).drop_duplicates("player_id")

    out = players.merge(roster, on="player_id", how="outer")
    out["position"] = out["roster_position"].combine_first(out["position"])
    return out.drop(columns="roster_position").set_index("player_id")


# adp_name covers a player the players table has under a different spelling, and
# it is also the only name for the handful of ids that never joined a roster
def load_adp_names(conn, season=SEASON):
    adp = pd.read_sql_query(
        "SELECT player_id, adp_name FROM adp WHERE season = ? AND source = ? "
        "AND player_id IS NOT NULL", conn, params=(season, PLATFORM_ADP_SOURCE))
    adp["player_id"] = adp["player_id"].astype(int)
    return adp.drop_duplicates("player_id").set_index("player_id")["adp_name"]


# S(k) for every player over picks 1..LAST_PICK, as a step lookup at or below k.
# reindex+ffill rather than interpolate: survival between two steps is the earlier
# step's value, and averaging them would invent a curve the model never fit.
def survival_matrix(cph, player_rows, last_pick=LAST_PICK):
    curves = cph.predict_survival_function(design_for_prediction(cph, player_rows))
    curves.columns = list(player_rows.index)

    picks = np.arange(1, last_pick + 1, dtype=float)
    grid = curves.reindex(curves.index.union(picks)).ffill().reindex(picks)

    # a pick before the first fitted step is survival 1.0, not a gap
    return grid.fillna(1.0).clip(0.0, 1.0)


def build_board(conn, season=SEASON, last_pick=LAST_PICK):
    survival = build_dataset(conn, season)
    feats = build_cox_features(survival, season, conn)

    cph = fit_cox_model(feats)
    rows = feats.drop_duplicates("player_id").set_index("player_id")

    grid = survival_matrix(cph, rows, last_pick)

    identity = load_identity(conn, rows.index, season)
    adp_names = load_adp_names(conn, season)

    # ascending platform_rank IS the ADP board, and 1..N contiguous because the app
    # sorts and displays this directly rather than re-ranking a float
    ranked = rows["platform_rank"].rank(method="first").astype(int)

    players, matrix = [], {}
    for pid in ranked.sort_values().index:
        pid = int(pid)
        info = identity.loc[pid] if pid in identity.index else None

        # A missing merge key arrives as NaN, which is a truthy float -- so `or`
        # is not a usable fallback here. 18 of these are free agents with no
        # 2025-26 roster row, and the app does .lower() on whatever it is given.
        def text(field, default):
            value = info.get(field) if info is not None else None
            return value.strip() if isinstance(value, str) and value.strip() else default

        name = text("name", None) or str(adp_names.get(pid, "")).strip()
        if not name or name == "nan":
            continue  # no name to show; a bare id is not a draft board row

        players.append({
            "player_id": pid,
            "name": name,
            "position": text("position", "UNK"),
            "team": text("team", "FA"),
            "adp_rank": int(ranked.loc[pid]),
        })

        column = grid[pid].to_numpy()
        matrix[str(pid)] = [int(v) for v in np.rint(column * SCALE).astype(np.int16)]

    return {
        "season": season,
        "scale": SCALE,
        "first_pick": 1,
        "last_pick": last_pick,
        "concordance": round(float(cph.concordance_index_), 4),
        "players": players,
        "survival": matrix,
    }


if __name__ == "__main__":
    conn = connect()
    board = build_board(conn)

    OUT_PATH.write_text(json.dumps(board, separators=(",", ":")), encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"{board['season']}: {len(board['players'])} players x "
          f"{board['last_pick']} picks, concordance {board['concordance']}")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}  {size_kb:.0f} KB")

    for row in board["players"][:5]:
        curve = board["survival"][str(row["player_id"])]
        # a Windows console is cp1252 and half these names are not
        name = row["name"][:24].encode("ascii", "replace").decode()
        print(f"  {row['adp_rank']:>3}  {name:<24} {row['team']:<4} "
              f"{row['position']:<4} S(12)={curve[11] / SCALE:.3f} "
              f"S(24)={curve[23] / SCALE:.3f}")
