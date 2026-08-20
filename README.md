# espn-draft-sequential-optimization

A sequential decision system for fantasy basketball drafts: project what players
will produce, model when each one comes off the board, and use the second to
decide how to spend the first.

Then measure it against the ranking it was built to beat, over 4,400 replayed
drafts.

**It lost.** Average draft position (ADP) predicts realized production better
than anything fitted here, and every policy built on a fitted estimate finished
behind the policy that just follows ADP. The writeup is
[`writeup/backtest.pdf`](writeup/backtest.pdf).

That is the actual result, so it is the headline. What follows is how it was
measured and why the answer is trustworthy.

## The result

Five policies, one harness, scored on **realized end-of-season production** —
never on projections, since a policy scored against its own model can win by
agreeing with itself.

| Rung | Policy | Mean | Edge over below | Win rate | 90% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | Follow ADP | 61.50 | — | — | — |
| 1 | Best available by value | 60.21 | −1.29 | 37.5% | [−14.0, +10.7] |
| 2 | VORP, greedy | 54.22 | −5.99 | 29.3% | [−26.2, +14.7] |
| 3 | VORP + sequencing | 54.07 | −0.14 | 45.7% | [−14.2, +14.6] |
| 4 | Own projections + both | 3.76 | −46.32 | 2.3% | [−73.7, −11.0] |

1,056 paired drafts per rung; 176 for rung 4, which needs projections and so runs
on one season. Every comparison is paired on (season, league size, slot, seed),
so the draft's own luck cancels.

Rungs 0–3 see **identical inputs**, so any gap between them is the decision rule
and nothing else. Only rung 4 swaps in the projection model, which is what makes
rung 4 minus rung 3 a separate measurement of what that model was worth.

Rung 3 is the comparison the design exists for: it holds projections fixed
against rung 2 and changes only whether the policy simulates the board forward.
The answer is −0.14 points and a 45.7% win rate. **Sequencing does not help.**

### Why

Each value estimate tested directly against realized production, Spearman ρ,
pooled across six seasons:

| Ranker | Full board | Top 60 |
| --- | --- | --- |
| ADP rank | **0.574** | **0.368** |
| Board-implied value | 0.555 | 0.360 |
| Value over replacement | 0.529 | 0.316 |
| Own projections | 0.331 | 0.270 |

ADP wins on the full board and again in the top 60, where drafts are decided. So
the policies are not losing because the decision layer is broken — they are
optimizing a worse objective than the ranking they are trying to beat, and no
amount of sequencing repairs a worse input.

This is a market-efficiency finding, and in hindsight it should have been the
prior. ADP is a consensus of experts updating daily on injury news and minutes
expectations. The projection model sees prior-season box scores.

### The result I nearly published

The first run used one season and six seeds: 132 paired drafts. Rung 3 beat
rung 2 by **+2.35** with a 55.3% win rate, consistent across per-combination
cuts. At eight times the sample the edge collapsed to −0.14.

The 90% interval on that first estimate was [−16.3, +25.3] — roughly ten times
wider than the effect. Nothing about the model changed between those two runs.
It is in the writeup because the interval was visible the whole time and a coin
flip still got described as a direction.

## What works

The survival model is genuinely good; it just does not move the decision.

| Metric | Value |
| --- | --- |
| Held-out concordance (Cox) | 0.935 |
| Expected calibration error | 0.029 |
| Brier score | 0.031 |

Calibration matters more than ranking here because the decision layer consumes
the probability, not the order. Validation is walk-forward: train on prior
seasons, test on the next.

The one model that clearly helped was games played. The projection code
multiplied every per-minute rate by a flat 82 games, assuming everyone plays a
full season; league average is about 46. A games-played model cut RMSE from
**38.3 to 18.5**.

## Layout

| Directory | Role |
| --- | --- |
| [`data/`](data/) | Builds `nba.sqlite`: stats, game logs, historical ADP, draft history, rosters. Has its own README and a long list of real traps. |
| [`models/projection/`](models/projection/) | Model A — XGBoost per category, rookie priors, games played, walk-forward validation. |
| [`models/behaviour/`](models/behaviour/) | Model B — Kaplan–Meier and Cox survival models over draft durations, calibration, and the availability query the rollout calls. |
| [`models/decision/`](models/decision/) | The decision layer — Monte Carlo rollout, the five policies, the backtest harness, plots. |
| [`tests/`](tests/) | 141 tests. The 70 pure-logic ones need no database, so CI runs them on every push. |
| [`writeup/`](writeup/) | The paper, with figures. |

## The idea, briefly

A snake draft rewards thinking ahead. You pick at 6, then not again until 19,
then 30. Thirteen players leave the board between your turns, so if you want two
of them the order matters: take the one who will still be there later, and you
lose him.

Formally, at pick *j* with a next turn at *k*, the value of waiting on player *i*
depends on the **conditional survival ratio**

```
P(available at k | available at j) = S_i(k) / S_i(j)
```

not on `S_i(k)` alone. The distinction matters more than it looks. A player with
an ADP of 5 who is somehow still on the board at pick 20 is a different
proposition from the same player before the draft started, and only the ratio
knows that. `models/behaviour/query.py` is the only file the rollout imports.

## Running it

```bash
pip install -r requirements.txt

# the DB is a release asset, never committed
gh release download data-latest --pattern 'nba.sqlite.gz' --dir data/
gunzip data/nba.sqlite.gz
```

Every module runs standalone as a worked example:

```bash
python models/behaviour/query.py       # availability tables for a real board
python models/decision/roster.py       # roster legality
python models/decision/backtest.py     # the sweep (hours, not minutes)
python models/decision/plots.py        # figures into artifacts/plots/
```

Tests:

```bash
pytest                      # 141 tests
pytest -m "not needs_db"    # the 70 that need no database
```

Splits are temporal everywhere. A draft is a point in time, and training on 2025
to predict 2019 would be a leak that reports a better number for itself.

Leak detection is a gate, not just a test: `train_rates.py` refuses to train if it
fires. Beyond the name checks, it permutes the current season's stats and rebuilds
the features — any feature whose value moves was reading the season it is supposed
to predict, which is a leak no name test can see. `tests/test_leakage.py` runs the
same checks in CI.

## Notes

**Injuries are not predicted, and the gap is left visible.** Rung 4's weakest
result traces largely to four players (VanVleet, Lillard, Irving, Haliburton)
projected for 61–66 games who played zero, all ruled out after the ADP board
published. No feature built from prior-season statistics can know that. A live
injury feed could, and the database has no historical one: `player_status` is a
single current snapshot, so reading today's status onto a 2018 row would leak the
future. So the games-played model corrects the population-level bias, which was
real and large, and does not correct individual unforeseeable injuries. Nothing
here pretends otherwise by fitting something that appears to.

**A scoring choice, disclosed.** The ladder scores starters only. VORP-greedy
drafts depth on purpose and strands 9.23 points on the bench against ADP's 5.05,
so on total roster value its −5.99 becomes −2.10 and its win rate rises from 29%
to 44%. Rung 3 stays negative under both metrics (−0.14 starters, −2.48 total),
so the sequencing conclusion holds either way — but part of rung 2's deficit was
a choice about scoring, not something the policy did.

**Two name-matching bugs, both found by their downstream symptom.** Accented
names (`Jokić` vs `Jokic`) silently failed to link, costing 18 rows. Later,
stripping generational suffixes collapsed "Jaren Jackson Jr." onto his father,
linking seven players to the wrong person across twelve seasons and producing
position-less board entries that still carried an inflated VORP — worse than
being unmatched. The fix matches the exact name first and falls back to the
stripped form only when nobody claims the exact one. `data/README.md` documents
both.

## What I would do next

Not more sequencing. The survival model works and the decision layer works; the
inputs are the problem, and the diagnosis says ADP already contains what the
projections estimate. Beating it needs signal ADP does not have, and a live
injury feed is the one gap with a number attached.

The part worth keeping is the harness. Four thousand paired off-policy drafts
with per-slot and per-seed distributions is the machinery that caught the +2.35,
and it would catch the same thing in any idea tried next.
