# Walk-forward validation, same temporal-only rule as Model A: a draft is a point in
# time, and random k-fold would train on the future and report a better number for it.

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

# config first so the repo root lands on sys.path before any data.* import
import config
from config import (ARTIFACT_DIR, RANDOM_SEED, SEASONS, TOTAL_PICKS, VAL_SEASONS,
                    connect)
from calibration import (apply_calibrator, calibrate, calibration_report,
                         observed_availability, reliability_table)
from cox import design_for_prediction, fit_cox_model
from dataset import build_dataset
from features import attach_position, build_cox_features, usable_covariates
from kaplan_meier import fit_all, km_survival_prob, resolve_curve

OUT_CSV = ARTIFACT_DIR / "behaviour_validation.csv"

# the windows a 10-team snake draft actually asks about: your next turn is 2*N_TEAMS
# away, so the 12 and 24 pick gaps are the ones the decision layer leans on
EVAL_WINDOWS = ((12, 24), (24, 36), (36, 60), (60, 96))


# Expanding-window folds, not sliding: there is no reason to throw away 2014 when
# predicting 2025, and ADP coverage is only 12 seasons deep to begin with.
def walk_forward_splits(seasons=SEASONS, val_seasons=VAL_SEASONS, min_train=3):
    seasons = list(seasons)
    splits = []

    for val in (val_seasons if val_seasons else seasons):
        if val not in seasons:
            continue

        train = [s for s in seasons if s < val]
        if len(train) < min_train:
            continue

        splits.append((train, val))

    return splits


# every fold's data, built once so a fold does not rebuild the dataset per metric
def build_fold(conn, train_seasons, val_season):
    train = pd.concat([build_dataset(conn, s) for s in train_seasons],
                      ignore_index=True)
    val = build_dataset(conn, val_season)

    return (attach_position(train, train_seasons[-1], conn),
            attach_position(val, val_season, conn))


# Concordance on the validation season. cph.concordance_index_ is free but in-sample,
# so it is reported alongside and never as the result.
def fold_concordance(cph, val_df):
    design = design_for_prediction(cph, val_df)
    hazard = cph.predict_partial_hazard(design).to_numpy()

    return float(concordance_index(val_df["duration"], -hazard,
                                   val_df["event_observed"]))


# KM has no covariates, so its ranking is E[pick] = sum of S(k), the standard
# identity for a non-negative discrete duration.
def km_expected_picks(fitted, player_ids):
    picks = np.arange(1, TOTAL_PICKS + 1, dtype=float)
    cache = {}
    expected = {}

    for pid in player_ids:
        curve, source = resolve_curve(int(pid), fitted["player_curves"],
                                      fitted["group_curves"],
                                      fitted["player_groups"])

        # a group curve is shared, so the curve object is the key and the cache dedupes
        key = id(curve) if curve is not None else None

        if key not in cache:
            if curve is None:
                cache[key] = float(len(picks))
            else:
                cache[key] = float(np.clip(curve.predict(picks), 0.0, 1.0).sum())

        expected[int(pid)] = cache[key]

    return expected


def km_concordance(fitted, val_df):
    expected = km_expected_picks(fitted, val_df["player_id"].unique())
    predicted = val_df["player_id"].astype(int).map(expected).to_numpy()

    return float(concordance_index(val_df["duration"], predicted,
                                   val_df["event_observed"]))


# raw p_available predictions and what actually happened, per window
def fold_predictions(model, val_df, windows=EVAL_WINDOWS):
    frames = []

    for j, k in windows:
        rows = observed_availability(val_df, j, k)
        if rows.empty:
            continue

        preds = model(rows["player_id"].to_numpy(), j, k)

        frames.append(pd.DataFrame({
            "pick_j": j,
            "pick_k": k,
            "player_id": rows["player_id"].to_numpy(),
            "predicted": preds,
            "observed": rows["observed"].to_numpy(),
        }))

    if not frames:
        return pd.DataFrame(columns=["pick_j", "pick_k", "player_id",
                                     "predicted", "observed"])

    return pd.concat(frames, ignore_index=True)


# S(k)/S(j) from a fitted Cox model, over a whole column of players at once
def cox_predictor(cph, val_df):
    rows = val_df.drop_duplicates("player_id").set_index("player_id")
    curves = cph.predict_survival_function(design_for_prediction(cph, rows))
    curves.columns = list(rows.index)

    index = curves.index.to_numpy()

    def at(pick):
        position = np.searchsorted(index, float(pick), side="right") - 1
        if position < 0:
            return pd.Series(1.0, index=curves.columns)
        return curves.iloc[position]

    def predict(player_ids, j, k):
        s_j, s_k = at(j), at(k)
        ids = pd.Index([int(p) for p in player_ids])

        num = s_k.reindex(ids).to_numpy(dtype=float)
        den = s_j.reindex(ids).to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 1e-9, num / den, 0.0)

        return np.clip(out, 0.0, 1.0)

    return predict


# The same from the KM curves, memoised per (curve, pick) for the same reason:
# the fold asks about the same shared group curves thousands of times.
def km_predictor(fitted):
    cache = {}

    def survival(pid, pick):
        curve, _ = resolve_curve(int(pid), fitted["player_curves"],
                                 fitted["group_curves"], fitted["player_groups"])
        if curve is None:
            return 1.0

        key = (id(curve), float(pick))
        if key not in cache:
            cache[key] = float(np.clip(curve.predict(float(pick)), 0.0, 1.0))

        return cache[key]

    def predict(player_ids, j, k):
        out = []
        for pid in player_ids:
            s_j = survival(pid, j)
            out.append(survival(pid, k) / s_j if s_j > 1e-9 else 0.0)

        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    return predict


# One fold, both models, discrimination and calibration.
def evaluate_fold(conn, train_seasons, val_season, windows=EVAL_WINDOWS):
    train, val = build_fold(conn, train_seasons, val_season)

    out = {"val_season": val_season, "train_seasons": len(train_seasons),
           "train_rows": len(train), "val_rows": len(val)}

    # KM, which is the bar Cox has to clear
    km_fitted = fit_all(train)
    km_preds = fold_predictions(km_predictor(km_fitted), val, windows)

    out["km_concordance_val"] = km_concordance(km_fitted, val)
    for key, value in calibration_report(km_preds["predicted"],
                                         km_preds["observed"]).items():
        out[f"km_{key}"] = value

    # --- Cox
    train_feats = build_cox_features(train, train_seasons[-1], conn)
    val_feats = build_cox_features(val, val_season, conn)

    # a fold with no usable covariates cannot be fit, and saying so beats reporting the
    # concordance of an intercept-only model
    kept, dropped = usable_covariates(train_feats)
    out["cox_covariates"] = len(kept)
    out["cox_dropped"] = ",".join(dropped)

    if len(kept) < 2:
        out["cox_concordance_val"] = float("nan")
        return out, km_preds, pd.DataFrame()

    cph = fit_cox_model(train_feats)
    out["cox_concordance_train"] = float(cph.concordance_index_)
    out["cox_concordance_val"] = fold_concordance(cph, val_feats)

    cox_preds = fold_predictions(cox_predictor(cph, val_feats), val_feats, windows)
    for key, value in calibration_report(cox_preds["predicted"],
                                         cox_preds["observed"]).items():
        out[f"cox_{key}"] = value

    # isotonic, fit and scored on disjoint halves of the validation fold
    half = len(cox_preds) // 2
    if half > 100:
        calibrator = calibrate(cox_preds["predicted"].to_numpy()[:half],
                               cox_preds["observed"].to_numpy()[:half])
        held = cox_preds.iloc[half:]
        fixed = apply_calibrator(calibrator, held["predicted"].to_numpy())

        out["cox_ece_raw_heldout"] = calibration_report(
            held["predicted"], held["observed"])["ece"]
        out["cox_ece_calibrated"] = calibration_report(
            fixed, held["observed"])["ece"]

    return out, km_preds, cox_preds


# every fold, in order
def walk_forward(conn, seasons=SEASONS, val_seasons=VAL_SEASONS,
                 windows=EVAL_WINDOWS):
    rows, predictions = [], []

    for train_seasons, val_season in walk_forward_splits(seasons, val_seasons):
        metrics, km_preds, cox_preds = evaluate_fold(conn, train_seasons,
                                                     val_season, windows)
        rows.append(metrics)

        for name, preds in (("km", km_preds), ("cox", cox_preds)):
            if preds.empty:
                continue
            part = preds.copy()
            part["model"] = name
            part["val_season"] = val_season
            predictions.append(part)

    folds = pd.DataFrame(rows)
    preds = (pd.concat(predictions, ignore_index=True) if predictions
             else pd.DataFrame())
    return folds, preds


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    np.random.seed(RANDOM_SEED)

    conn = connect()
    splits = walk_forward_splits()

    print("walk-forward folds")
    for train, val in splits:
        print(f"  train {train[0]}..{train[-1]} ({len(train)})  ->  val {val}")
    print()

    folds, preds = walk_forward(conn)

    cols = ["val_season", "train_seasons", "km_concordance_val",
            "cox_concordance_train", "cox_concordance_val",
            "km_ece", "cox_ece", "cox_ece_raw_heldout", "cox_ece_calibrated",
            "km_brier", "cox_brier"]
    print("\nfold metrics")
    print(folds[[c for c in cols if c in folds.columns]].round(4).to_string(index=False))

    if "cox_dropped" in folds.columns and folds["cox_dropped"].any():
        print(f"\ncovariates dropped as constant: {folds['cox_dropped'].iloc[0]}")
        print("  vorp_rank_diff: projections.csv covers the target season only")
        print("  injury_flag:    player_status has no history to read")

    print("\nreliability on the held-out seasons, cox")
    cox_preds = preds[preds["model"] == "cox"] if not preds.empty else pd.DataFrame()
    if not cox_preds.empty:
        print(reliability_table(cox_preds["predicted"],
                               cox_preds["observed"]).round(4).to_string(index=False))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    print("\nThese folds run on ADP-sampled drafts: draft_results is empty, so the")
    print("model is scored on recovering the distribution it was sampled from.")
    print("Read the numbers as the pipeline being correct, not as drafter behaviour.")
