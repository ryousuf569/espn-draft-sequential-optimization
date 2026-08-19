# Figures for §8.2 and §8.3. Every one shows spread, since a bar chart of five
# averages hides exactly what §8.2 says to look for.

import numpy as np
import pandas as pd

import matplotlib
# Agg so this runs headless and never tries to open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# config first so the repo root lands on sys.path before any data.* import
import config
from config import ARTIFACT_DIR, POLICY_NAMES
from backtest import ablation_ladder, by_slot, compare_to_baseline

PLOT_DIR = ARTIFACT_DIR / "plots"

# one colour per rung, fixed across every figure so a rung stays recognisable
RUNG_COLOURS = {
    "rung0_adp": "#8c8c8c",
    "rung1_best_available": "#4c72b0",
    "rung2_vorp_greedy": "#dd8452",
    "rung3_vorp_sequencing": "#c44e52",
    "rung4_own_projections": "#55a868",
}

# rung 3 is the paper, so it is drawn heaviest
EMPHASIS = "rung3_vorp_sequencing"

SHORT_NAMES = {
    "rung0_adp": "0: ADP",
    "rung1_best_available": "1: best avail",
    "rung2_vorp_greedy": "2: VORP",
    "rung3_vorp_sequencing": "3: +sequencing",
    "rung4_own_projections": "4: full system",
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def _policies_in(df):
    return [p for p in POLICY_NAMES if p in set(df["policy"])]


def _colour(policy):
    return RUNG_COLOURS.get(policy, "#333333")


def _label(policy):
    return SHORT_NAMES.get(policy, policy)


# The ladder as a picture: mean per rung with the spread behind it, and the
# rung-over-rung edge below.
def plot_ladder(df, metric="starter_value", path=None):
    _style()
    policies = _policies_in(df)
    ladder = ablation_ladder(df, metric)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), height_ratios=[3, 2], sharex=True)

    # a violin shows whether a rung wins everywhere or only on average
    data = [df.loc[df["policy"] == p, metric].to_numpy() for p in policies]
    parts = top.violinplot(data, positions=range(len(policies)), widths=0.7,
                           showmeans=True, showextrema=False)

    for body, policy in zip(parts["bodies"], policies):
        body.set_facecolor(_colour(policy))
        body.set_alpha(0.75 if policy == EMPHASIS else 0.45)
        body.set_edgecolor(_colour(policy))
        body.set_linewidth(1.6 if policy == EMPHASIS else 0.8)

    parts["cmeans"].set_color("#222222")
    parts["cmeans"].set_linewidth(1.2)

    for i, policy in enumerate(policies):
        mean = ladder.loc[ladder["policy"] == policy, "mean"].iloc[0]
        top.annotate(f"{mean:.1f}", (i, mean), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color="#222222")

    top.set_ylabel(f"realized {metric.replace('_', ' ')}")
    top.set_title("The ablation ladder, scored on realized end-of-season production\n"
                  "rung 3 isolates sequencing from projection quality",
                  loc="left")

    # rung-over-rung edge, which is what each §8.3 question actually asks
    edges = ladder["edge_over_below"].to_numpy()
    lows = ladder["p05_vs_below"].to_numpy()
    highs = ladder["p95_vs_below"].to_numpy()

    for i, policy in enumerate(policies):
        if i == 0 or not np.isfinite(edges[i]):
            continue

        colour = _colour(policy)
        bottom.bar(i, edges[i], width=0.55, color=colour,
                   alpha=0.85 if policy == EMPHASIS else 0.55,
                   edgecolor=colour, linewidth=1.4 if policy == EMPHASIS else 0.8)
        # the 5th-95th band, so a positive mean straddling zero is visibly not a result
        bottom.plot([i, i], [lows[i], highs[i]], color="#222222", linewidth=1.0)
        bottom.plot([i - 0.1, i + 0.1], [lows[i]] * 2, color="#222222", linewidth=1.0)
        bottom.plot([i - 0.1, i + 0.1], [highs[i]] * 2, color="#222222", linewidth=1.0)

    bottom.axhline(0, color="#222222", linewidth=0.9)
    bottom.set_ylabel("edge over the rung below")
    bottom.set_xticks(range(len(policies)))
    bottom.set_xticklabels([_label(p) for p in policies], rotation=15, ha="right")
    bottom.set_title("Paired difference against the rung below, with the 5th-95th "
                     "percentile band", loc="left", fontsize=9)

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "ablation_ladder.png")


# §8.2's central warning as a figure: winning 2% on average while losing from the
# 1st slot is a different object from winning uniformly.
def plot_by_slot(df, metric="starter_value", baseline="rung0_adp", path=None):
    _style()
    policies = [p for p in _policies_in(df) if p != baseline]
    keys = ["season", "n_teams", "draft_slot", "seed"]

    base = df[df["policy"] == baseline].set_index(keys)[metric]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for policy in policies:
        paired = df[df["policy"] == policy].set_index(keys)[metric]
        common = paired.index.intersection(base.index)
        edge = (paired.loc[common] - base.loc[common]).groupby("draft_slot")

        means = edge.mean()
        # standard error per slot, so an uncertain slot reads as uncertain
        errs = edge.std() / np.sqrt(edge.count())

        ax.errorbar(means.index, means.to_numpy(), yerr=errs.to_numpy(),
                    marker="o", markersize=4.5, capsize=2.5,
                    linewidth=2.0 if policy == EMPHASIS else 1.1,
                    color=_colour(policy), label=_label(policy),
                    alpha=1.0 if policy == EMPHASIS else 0.8)

    ax.axhline(0, color="#222222", linewidth=0.9, zorder=0)
    ax.set_xlabel("draft slot")
    ax.set_ylabel(f"edge over {SHORT_NAMES.get(baseline, baseline)}")
    ax.set_title("Edge by draft slot, paired on season/slot/seed\n"
                 "a policy that only wins from some slots is not a policy that wins",
                 loc="left")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "edge_by_slot.png")


# Win rate, which §8.2 leads with instead of the mean: a mean edge can come from
# one lucky draft, a win rate cannot.
def plot_win_rates(df, metric="starter_value", baseline="rung0_adp", path=None):
    _style()
    edges = compare_to_baseline(df, baseline, metric)
    if edges.empty:
        return None

    edges = edges.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    colours = [_colour(p) for p in edges["policy"]]
    bars = ax.barh(range(len(edges)), edges["win_rate"], height=0.6,
                   color=colours, alpha=0.8)

    for bar, policy in zip(bars, edges["policy"]):
        if policy == EMPHASIS:
            bar.set_alpha(1.0)
            bar.set_edgecolor("#222222")
            bar.set_linewidth(1.2)

    # 0.5 is the line that matters: below it a policy loses more drafts than it wins
    ax.axvline(0.5, color="#222222", linewidth=1.0, linestyle="--")
    ax.annotate("coin flip", (0.5, len(edges) - 0.35), fontsize=8,
                color="#222222", ha="center",
                textcoords="offset points", xytext=(0, 6))

    for i, row in enumerate(edges.itertuples()):
        ax.annotate(f"{row.win_rate:.1%}  ({row.slots_won} slots won)",
                    (row.win_rate, i), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8)

    ax.set_yticks(range(len(edges)))
    ax.set_yticklabels([_label(p) for p in edges["policy"]])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel(f"share of drafts beating {SHORT_NAMES.get(baseline, baseline)}")
    ax.set_title("Win rate against the incumbent, paired draft by draft", loc="left")

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "win_rates.png")


# The one comparison that is the paper: rung 3 against rung 2, identical inputs,
# sequencing the only difference.
def plot_sequencing_effect(df, metric="starter_value", path=None):
    _style()
    keys = ["season", "n_teams", "draft_slot", "seed"]

    if not {"rung2_vorp_greedy", EMPHASIS} <= set(df["policy"]):
        return None

    base = df[df["policy"] == "rung2_vorp_greedy"].set_index(keys)[metric]
    seq = df[df["policy"] == EMPHASIS].set_index(keys)[metric]
    common = seq.index.intersection(base.index)
    diff = (seq.loc[common] - base.loc[common]).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.hist(diff, bins=40, color=_colour(EMPHASIS), alpha=0.75,
            edgecolor="#ffffff", linewidth=0.4)
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.axvline(diff.mean(), color="#c44e52", linewidth=1.6, linestyle="--",
               label=f"mean {diff.mean():+.2f}")

    win = (diff > 0).mean()
    ax.annotate(f"beats VORP-greedy in {win:.1%} of drafts\n"
                f"median {np.median(diff):+.2f}, "
                f"5th-95th [{np.percentile(diff, 5):+.1f}, "
                f"{np.percentile(diff, 95):+.1f}]",
                (0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.5)

    ax.set_xlabel("rung 3 minus rung 2, realized starter value (paired)")
    ax.set_ylabel("drafts")
    ax.set_title("Rung 3 is the paper: sequencing on identical inputs\n"
                 "if this straddles zero, draft-order effects are smaller than the "
                 "framing assumes", loc="left")
    ax.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "sequencing_effect.png")


# a policy that only works in one season found that season's noise
def plot_by_season(df, metric="starter_value", path=None):
    _style()
    policies = _policies_in(df)
    seasons = sorted(df["season"].unique())

    if len(seasons) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / len(policies)

    for i, policy in enumerate(policies):
        means = (df[df["policy"] == policy].groupby("season")[metric].mean()
                 .reindex(seasons))
        offset = (i - (len(policies) - 1) / 2) * width
        ax.bar(np.arange(len(seasons)) + offset, means.to_numpy(), width=width,
               color=_colour(policy), label=_label(policy),
               alpha=1.0 if policy == EMPHASIS else 0.75)

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, rotation=20, ha="right")
    ax.set_ylabel(f"mean realized {metric.replace('_', ' ')}")
    ax.set_title("By season: a policy that only works in one season found noise",
                 loc="left")
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "by_season.png")


# league size sweeps, since a deeper board leaves sequencing less to exploit
def plot_by_league_size(df, metric="starter_value", baseline="rung0_adp", path=None):
    _style()
    sizes = sorted(df["n_teams"].unique())
    if len(sizes) < 2:
        return None

    policies = [p for p in _policies_in(df) if p != baseline]
    keys = ["season", "n_teams", "draft_slot", "seed"]
    base = df[df["policy"] == baseline].set_index(keys)[metric]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for policy in policies:
        paired = df[df["policy"] == policy].set_index(keys)[metric]
        common = paired.index.intersection(base.index)
        edge = (paired.loc[common] - base.loc[common]).groupby("n_teams").mean()

        ax.plot(edge.index, edge.to_numpy(), marker="o", markersize=5,
                linewidth=2.0 if policy == EMPHASIS else 1.2,
                color=_colour(policy), label=_label(policy))

    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(sizes)
    ax.set_xlabel("teams in the league")
    ax.set_ylabel(f"edge over {SHORT_NAMES.get(baseline, baseline)}")
    ax.set_title("By league size: a deeper board leaves sequencing less to exploit",
                 loc="left")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return _save(fig, path or PLOT_DIR / "by_league_size.png")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# every figure the backtest supports, skipping ones its sweep cannot fill
def plot_all(df, metric="starter_value", out_dir=None):
    out_dir = out_dir or PLOT_DIR
    made = []

    for fn in (plot_ladder, plot_win_rates, plot_sequencing_effect,
               plot_by_slot, plot_by_season, plot_by_league_size):
        name = fn.__name__.replace("plot_", "") + ".png"
        result = fn(df, metric=metric, path=out_dir / name)
        if result is not None:
            made.append(result)

    return made


# a figure that silently drops a policy is worse than no figure
def verify(df, paths):
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    check("figures were written", len(paths) > 0, f"{len(paths)} files")
    check("every file exists and is non-trivial",
          all(p.exists() and p.stat().st_size > 5000 for p in paths),
          f"smallest {min(p.stat().st_size for p in paths) // 1024}kb")

    # the ladder has to show every rung that ran, or a reader thinks one was skipped
    policies = _policies_in(df)
    check("every rung in the data has a colour",
          all(p in RUNG_COLOURS for p in policies), f"{len(policies)} rungs")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    from backtest import OUT_CSV

    if not OUT_CSV.exists():
        raise SystemExit(f"no backtest output at {OUT_CSV} -- run backtest.py first")

    df = pd.read_csv(OUT_CSV)
    print(f"{len(df)} drafts, {df['policy'].nunique()} policies, "
          f"{df['season'].nunique()} seasons\n")

    paths = plot_all(df)
    verify(df, paths)

    for p in paths:
        print(f"  wrote {p}")
