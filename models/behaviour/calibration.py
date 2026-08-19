# Calibration, per §7, before the decision layer is wired up: the rollout consumes
# the NUMBER, not the ranking, and concordance cannot see a wrong level.

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# config first so the repo root lands on sys.path before any data.* import
import config
from config import RANDOM_SEED, TOTAL_PICKS, connect

# equal-width bins, because "is it right when it says 0.9" is a question about a
# region of probability space, not about a quantile of this particular sample
DEFAULT_BINS = 10


# One row per bin. Empty bins are dropped rather than reported as 0 observed, which
# would read as a calibration failure where there is simply no evidence.
def reliability_table(predicted_probs, observed_outcomes, n_bins=DEFAULT_BINS):
    pred = np.asarray(predicted_probs, dtype=float)
    obs = np.asarray(observed_outcomes, dtype=float)

    keep = ~(np.isnan(pred) | np.isnan(obs))
    pred, obs = pred[keep], obs[keep]

    if len(pred) == 0:
        return pd.DataFrame(columns=["bin", "bin_low", "bin_high", "n",
                                     "mean_predicted", "observed_rate", "gap"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # right-closed so a prediction of exactly 1.0 lands in the top bin
    index = np.clip(np.digitize(pred, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue

        rows.append({
            "bin": b,
            "bin_low": edges[b],
            "bin_high": edges[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(pred[mask].mean()),
            "observed_rate": float(obs[mask].mean()),
            "gap": float(pred[mask].mean() - obs[mask].mean()),
        })

    return pd.DataFrame(rows)


# ECE: the reliability gaps weighted by bin count, and the one number to quote --
# unweighted, a bin of three observations would count as much as one of three thousand.
def expected_calibration_error(predicted_probs, observed_outcomes,
                              n_bins=DEFAULT_BINS):
    table = reliability_table(predicted_probs, observed_outcomes, n_bins)

    if table.empty:
        return float("nan")

    return float(np.average(table["gap"].abs(), weights=table["n"]))


# The largest single-bin gap, worth reporting because ECE averages the failure away:
# a model wrong only near 0.9 is exactly the one a decision acts on hardest.
def max_calibration_error(predicted_probs, observed_outcomes, n_bins=DEFAULT_BINS):
    table = reliability_table(predicted_probs, observed_outcomes, n_bins)

    if table.empty:
        return float("nan")

    return float(table["gap"].abs().max())


# Brier: a proper scoring rule, so unlike ECE it cannot be gamed by predicting the
# base rate for everyone -- that model is perfectly calibrated and useless.
def brier_score(predicted_probs, observed_outcomes):
    pred = np.asarray(predicted_probs, dtype=float)
    obs = np.asarray(observed_outcomes, dtype=float)

    keep = ~(np.isnan(pred) | np.isnan(obs))
    if not keep.any():
        return float("nan")

    return float(np.mean((pred[keep] - obs[keep]) ** 2))


# Post-hoc monotone correction. Isotonic only assumes a higher raw probability stays
# higher, fixing the level without touching the ordering; Platt would force a sigmoid.
def calibrate(raw_probs, observed_outcomes, method="isotonic"):
    if method != "isotonic":
        raise ValueError(f"only isotonic is supported, got {method}")

    pred = np.asarray(raw_probs, dtype=float)
    obs = np.asarray(observed_outcomes, dtype=float)

    keep = ~(np.isnan(pred) | np.isnan(obs))

    # out_of_bounds="clip" so a prediction outside the fitted range returns the nearest
    # fitted value rather than NaN into a rollout
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                    out_of_bounds="clip")
    calibrator.fit(pred[keep], obs[keep])
    return calibrator


# apply a fitted calibrator, keeping NaN as NaN rather than silently becoming 0
def apply_calibrator(calibrator, raw_probs):
    pred = np.asarray(raw_probs, dtype=float)
    out = np.full(len(pred), np.nan)

    keep = ~np.isnan(pred)
    if keep.any():
        out[keep] = np.clip(calibrator.predict(pred[keep]), 0.0, 1.0)

    return out


# every metric in one call, which is what evaluate_fold reports
def calibration_report(predicted_probs, observed_outcomes, n_bins=DEFAULT_BINS):
    return {
        "n": int(np.sum(~(np.isnan(np.asarray(predicted_probs, dtype=float))
                          | np.isnan(np.asarray(observed_outcomes, dtype=float))))),
        "ece": expected_calibration_error(predicted_probs, observed_outcomes, n_bins),
        "max_ce": max_calibration_error(predicted_probs, observed_outcomes, n_bins),
        "brier": brier_score(predicted_probs, observed_outcomes),
        "mean_predicted": float(np.nanmean(predicted_probs)),
        "base_rate": float(np.nanmean(observed_outcomes)),
    }


# Observed availability, the ground truth the predictions are scored against: for
# each (draft, player) still on the board at j, did he last to k?
def observed_availability(survival_df, pick_j, pick_k):
    # on the board at j means he was not taken before it
    at_risk = survival_df[(survival_df["duration"] > pick_j)
                          | ((survival_df["duration"] == pick_j)
                             & (survival_df["event_observed"] == 0))]

    # still there at k means he was not taken in (j, k]
    available = ((at_risk["duration"] > pick_k)
                 | (at_risk["event_observed"] == 0)).astype(int)

    return at_risk.assign(observed=available.to_numpy())


# the metrics have to behave on cases where the answer is known in advance
def verify():
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        ok = ok and bool(passed)

    rng = np.random.default_rng(RANDOM_SEED)

    # a perfectly calibrated model: outcomes drawn at exactly the stated rate
    truth = rng.uniform(0, 1, 20000)
    outcomes = rng.binomial(1, truth)
    check("a calibrated model scores near zero",
          expected_calibration_error(truth, outcomes) < 0.02,
          f"ece {expected_calibration_error(truth, outcomes):.4f}")

    # a model that is confidently wrong: predictions inverted
    check("an inverted model scores badly",
          expected_calibration_error(1.0 - truth, outcomes) > 0.3,
          f"ece {expected_calibration_error(1.0 - truth, outcomes):.4f}")

    # Brier has to prefer the honest model over the base-rate one, which is what makes
    # it worth reporting alongside ECE
    base = np.full_like(truth, outcomes.mean())
    check("brier prefers the informative model",
          brier_score(truth, outcomes) < brier_score(base, outcomes),
          f"{brier_score(truth, outcomes):.4f} vs {brier_score(base, outcomes):.4f}")

    # ECE cannot tell them apart, which is exactly why both are reported
    check("ece cannot tell them apart",
          expected_calibration_error(base, outcomes) < 0.02,
          f"base-rate ece {expected_calibration_error(base, outcomes):.4f}")

    # isotonic has to repair a known distortion, fit and applied out of sample
    split = len(truth) // 2
    skewed = np.clip(truth ** 2, 0, 1)
    calibrator = calibrate(skewed[:split], outcomes[:split])
    fixed = apply_calibrator(calibrator, skewed[split:])

    before = expected_calibration_error(skewed[split:], outcomes[split:])
    after = expected_calibration_error(fixed, outcomes[split:])
    check("isotonic repairs a skewed model out of sample", after < before,
          f"ece {before:.4f} -> {after:.4f}")

    # it must not reorder anything, or it fixed the level by breaking the ranking
    order_before = np.argsort(skewed[split:])
    order_after = np.argsort(fixed, kind="stable")
    check("isotonic preserves the ordering",
          bool(np.all(fixed[order_before] == np.sort(fixed))),
          f"{len(fixed)} points")

    # a prediction of exactly 0 or 1 has to land in a bin
    table = reliability_table([0.0, 1.0, 0.5], [0, 1, 1], n_bins=10)
    check("the extremes land in bins", int(table["n"].sum()) == 3,
          f"{int(table['n'].sum())} of 3")

    print("all good" if ok else "SOMETHING IS WRONG")
    return ok


if __name__ == "__main__":
    pd.set_option("display.width", 250)

    verify()

    # the real thing: score the fitted model against what the drafts actually did
    from query import Behaviour, p_available

    conn = connect()
    season = "2025-26"
    behaviour = Behaviour(conn, season)

    print(f"\n{season}, predictions against observed availability")
    print(f"{'window':>14}{'n':>8}{'pred':>8}{'obs':>8}{'ece':>8}{'max':>8}{'brier':>8}")

    for j, k in ((12, 24), (24, 36), (36, 60), (60, 96)):
        rows = observed_availability(behaviour.survival, j, k)
        if rows.empty:
            continue

        preds = np.array([p_available(pid, j, k, behaviour, "cox")
                          for pid in rows["player_id"]])
        report = calibration_report(preds, rows["observed"].to_numpy())

        print(f"{f'{j} -> {k}':>14}{report['n']:>8}{report['mean_predicted']:>8.3f}"
              f"{report['base_rate']:>8.3f}{report['ece']:>8.4f}"
              f"{report['max_ce']:>8.4f}{report['brier']:>8.4f}")

    print("\nreliability, pick 12 -> 24")
    rows = observed_availability(behaviour.survival, 12, 24)
    preds = np.array([p_available(pid, 12, 24, behaviour, "cox")
                      for pid in rows["player_id"]])
    print(reliability_table(preds, rows["observed"].to_numpy()).round(4).to_string(index=False))
