# espn-draft-sequential-optimization

Fantasy basketball draft assistant. It projects what players will produce, models
when each one comes off the board, and uses the second to decide how to spend the
first.

Then it gets measured against the ranking it was built to beat, over 4,400
replayed drafts.

It lost. Average draft position predicts realized production better than anything
I fit, and every policy built on a fitted estimate finished behind the policy that
just follows ADP.

Full writeup, with the methodology and the diagnosis:
[`writeup/backtest.pdf`](writeup/backtest.pdf).

## The result

Five policies, one harness, scored on realized end-of-season production rather
than on projections. Score on projections and a policy can win by agreeing with
its own model.

| Rung | Policy | Mean | Edge over below | Win rate | 90% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | Follow ADP | 61.50 | | | |
| 1 | Best available by value | 60.21 | −1.29 | 37.5% | [−14.0, +10.7] |
| 2 | VORP, greedy | 54.22 | −5.99 | 29.3% | [−26.2, +14.7] |
| 3 | VORP + sequencing | 54.07 | −0.14 | 45.7% | [−14.2, +14.6] |
| 4 | Own projections + both | 3.76 | −46.32 | 2.3% | [−73.7, −11.0] |

1,056 paired drafts per rung, 176 for rung 4, which needs projections and so runs
on one season. Every comparison is paired on (season, league size, slot, seed), so
the draft's own luck cancels.

Rungs 0 through 3 see identical inputs, so any gap between them is the decision
rule and nothing else. Only rung 4 swaps in the projection model, which is what
makes rung 4 minus rung 3 a measurement of what that model was worth.

Rung 3 is the comparison the design exists for. It holds projections fixed against
rung 2 and changes only whether the policy simulates the board forward. The answer
is −0.14 points and a 45.7% win rate, so sequencing does not help.

### Why

Each value estimate tested against realized production, Spearman ρ, averaged over
six seasons:

| Ranker | Full board | Top 60 |
| --- | --- | --- |
| ADP rank | 0.574 | 0.368 |
| Board-implied value | 0.555 | 0.360 |
| Value over replacement | 0.529 | 0.316 |
| Own projections | 0.331 | 0.270 |

ADP wins on the full board and again in the top 60, where drafts are decided. The
policies are not losing because the decision layer is broken. They are optimizing
a worse objective than the ranking they are trying to beat.

The margin is small and it is not uniform. Board-implied value actually wins in 8
of 24 season and league-size combinations, so the pooled ordering is what holds.
`tests/test_board.py` asserts it at that level rather than per season.

## What works

The survival model is accurate. It just does not move the decision.

| Metric | Value |
| --- | --- |
| Held-out concordance (Cox) | 0.935 |
| Expected calibration error | 0.029 |
| Brier score | 0.031 |

Calibration matters more than ranking here, because the decision layer consumes
the probability rather than the order. Validation is walk-forward.

The one model that clearly paid for itself was games played. The projection code
multiplied every per-minute rate by a flat 82 games, so it assumed everyone plays
a full season; league average is about 46. A games-played model cut RMSE from 38.3
to 18.5.

## Layout

| Directory | Role |
| --- | --- |
| [`data/`](data/) | Builds `nba.sqlite`: stats, game logs, historical ADP, draft history, rosters. Has its own README and a long list of traps found the hard way. |
| [`models/projection/`](models/projection/) | Model A. XGBoost per category, rookie priors, games played, walk-forward validation. |
| [`models/behaviour/`](models/behaviour/) | Model B. Kaplan–Meier and Cox survival models over draft durations, calibration, and the availability query the rollout calls. |
| [`models/decision/`](models/decision/) | Rollout, the five policies, the backtest harness, plots. |
| [`tests/`](tests/) | 141 tests. The 70 that need no database run in CI on every push. |
| [`writeup/`](writeup/) | The paper and its figures. |

The interface between Model B and the decision layer is one function.
`models/behaviour/query.py` answers P(available at pick k, given still on the
board at pick j) as the ratio S(k)/S(j), and that ratio is the only thing the
rollout imports.

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

## Leak detection is a gate, not a report

Splits are temporal everywhere. A draft is a point in time, and training on 2025
to predict 2019 would be a leak that reports a better number for itself.

`train_rates.py` refuses to train if its leak check fires. Past checking feature
names, it permutes the current season's statistics and rebuilds the features. Any
feature whose value moves was reading the season it is supposed to predict, which
is a leak no name test can catch. `tests/test_leakage.py` runs the same checks in
CI.

## Two caveats on the table above

Injuries are not predicted, and the gap is left visible rather than papered over.
Four players in rung 4 were projected for 61 to 66 games and played zero, all
ruled out after the ADP board published. `player_status` in the database is a
single current snapshot, so there is no historical injury feed to learn from, and
reading today's status onto a 2018 row would leak the future. The games-played
model fixes the population-level bias and cannot fix individual surprises.

The ladder scores starters only, which flatters ADP. VORP-greedy drafts depth on
purpose and strands 9.23 points on the bench against ADP's 5.05, so on total
roster value its −5.99 becomes −2.10. Rung 3 stays negative either way (−0.14
starters, −2.48 total), so the sequencing conclusion survives the choice. But part
of rung 2's deficit was my scoring, not its drafting.
