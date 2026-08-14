from collections import defaultdict
import argparse
import math
import sys

import sqlite_helpers

# fallback spread when a source publishes no per-site disagreement,
# so a pick-10 player varies by ~2 slots and a pick-100 player by ~20
SD_FLOOR = 1.5
SD_SCALE = 0.20


# empirical histogram from real drafts: {player_id: {pick: probability}}
def from_draft_results(conn, season):
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for player_id, pick in conn.execute(
        "SELECT player_id, pick_number FROM draft_results "
        "WHERE season = ? AND player_id IS NOT NULL",
        (season,),
    ):
        counts[player_id][pick] += 1
        totals[player_id] += 1

    return {
        pid: {pick: n / totals[pid] for pick, n in picks.items()}
        for pid, picks in counts.items()
    }


# fallback when we have no real drafts: normal centred on ADP, truncated
def from_adp(conn, season, max_pick):
    dists = {}

    for player_id, adp, adp_sd in conn.execute(
        "SELECT player_id, adp, adp_sd FROM adp "
        "WHERE season = ? AND player_id IS NOT NULL",
        (season,),
    ):
        sd = adp_sd if adp_sd and adp_sd > 0 else max(SD_FLOOR, adp * SD_SCALE)
        weights = {
            pick: math.exp(-0.5 * ((pick - adp) / sd) ** 2)
            for pick in range(1, max_pick + 1)
        }
        total = sum(weights.values())
        if total:
            # drop negligible tails so the dict stays small
            dists[player_id] = {
                p: w / total for p, w in weights.items() if w / total > 1e-6
            }

    return dists


# P(player is still on the board when this pick comes up)
def survival(dist, pick):
    return sum(p for slot, p in dist.items() if slot >= pick)


# flag autodraft artifacts: a team on autopilot takes the top of the queue at its
# turn every round, piling mass onto multiples of the team count
def autodraft_suspect(dist, teams, min_mass=0.15):
    if len(dist) < 3:
        return None

    for pick, mass in dist.items():
        if pick % teams not in (0, 1) or mass < min_mass:
            continue

        # compare against the REST of the distribution, not the global peak. in
        # badly contaminated data the artifact is the tallest bar, so anchoring
        # on the peak compares the spike to itself and misses the whole point.
        rest = {s: m for s, m in dist.items() if s != pick}
        rest_mass = sum(rest.values())
        if not rest_mass:
            continue

        centre = sum(s * m for s, m in rest.items()) / rest_mass
        # sits inside the main body, that is just where he goes
        if abs(pick - centre) < teams:
            continue

        # a spike stands alone, a genuine cluster has support beside it
        neighbours = [dist.get(pick - 1, 0.0), dist.get(pick + 1, 0.0)]
        if mass > 2 * max(neighbours, default=0.0):
            return pick

    return None


def build(conn, season, max_pick, teams=12):
    dists = from_draft_results(conn, season)

    if dists:
        print(f"{season}: {len(dists)} players from real drafts")

        flagged = {}
        for pid, dist in dists.items():
            pick = autodraft_suspect(dist, teams)
            if pick is not None:
                flagged[pid] = pick

        # flag for review, never silently drop rows
        if flagged:
            names = dict(conn.execute("SELECT player_id, name FROM players"))
            print(f"  autodraft suspects ({len(flagged)}), inspect before trusting:")
            for pid, pick in list(flagged.items())[:10]:
                print(f"    {names.get(pid, pid)}: spike at pick {pick}")

        return dists

    dists = from_adp(conn, season, max_pick)
    print(f"{season}: {len(dists)} players from ADP (no drafts for this season)")
    return dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g. 2025-26")
    ap.add_argument("--max-pick", type=int, default=180, help="12 teams x 15 rounds")
    args = ap.parse_args()

    # names carry accents (Jokić, Dončić) that the Windows console cannot print
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    conn = sqlite_helpers.connect()
    dists = build(conn, args.season, args.max_pick)

    # print the top of the board as a sanity check
    top = sorted(dists.items(), key=lambda kv: sum(s * p for s, p in kv[1].items()))[:10]
    names = dict(conn.execute("SELECT player_id, name FROM players"))

    print(f"\n{'player':<26} {'E[pick]':>8} {'P(avail@12)':>12} {'P(avail@24)':>12}")
    for pid, dist in top:
        exp = sum(slot * p for slot, p in dist.items())
        print(
            f"{names.get(pid, pid):<26} {exp:>8.1f} "
            f"{survival(dist, 12):>12.2f} {survival(dist, 24):>12.2f}"
        )

    conn.close()


if __name__ == "__main__":
    main()
