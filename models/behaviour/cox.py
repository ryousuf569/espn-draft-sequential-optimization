# Cox proportional hazards. KM says when a player goes; Cox says what about him
# explains it, which is §6: the ratio of board weight to value weight.

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# config first so the repo root lands on sys.path before any data.* import
import config
from config import RANDOM_SEED, TOTAL_PICKS, connect
from dataset import build_dataset
from features import build_cox_features, cox_design, usable_covariates

# No ridge, deliberately: at 0.1 the platform_rank coefficient came out -0.026
# against -0.064, understating the §6 result 2.4x and flattening every curve.
PENALIZER = 0.0


# The whole fit. CoxPHFitter reads duration and event by name and treats every
# other column as a covariate, so cox_design has already stripped identifiers.
def fit_cox_model(cox_features_df, penalizer=PENALIZER):
    design = cox_design(cox_features_df)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(design, duration_col="duration", event_col="event_observed")
    return cph


# exp(B), which is what gets reported: above 1 comes off the board faster
def get_hazard_ratios(cph):
    out = pd.DataFrame({
        "hazard_ratio": cph.hazard_ratios_,
        "coef": cph.params_,
        "se": cph.standard_errors_,
        "p": cph.summary["p"],
    })

    # biggest effect first, so the reported table leads with what moves the draft
    out["abs_log_hr"] = out["coef"].abs()
    return out.sort_values("abs_log_hr", ascending=False).drop(columns="abs_log_hr")


# §6 as one number. Both are rank covariates on the same scale, which is why ranks
# are modelled rather than raw ADP and VORP -- those are in incomparable units.
def board_vs_value(cph):
    params = cph.params_
    board = float(params.get("platform_rank", float("nan")))
    value = float(params.get("vorp_rank_diff", float("nan")))

    return {
        "platform_rank_coef": board,
        "vorp_rank_diff_coef": value,
        "platform_rank_hr": float(np.exp(board)),
        "vorp_rank_diff_hr": float(np.exp(value)),
        # NaN when vorp_rank_diff was dropped as constant, i.e. every historical fold
        "board_weight_ratio": (abs(board) / abs(value)
                               if value and not np.isnan(value) else float("nan")),
    }


# Schoenfeld residuals, run regardless of the answer. Per §6 the assumption is KNOWN
# to be violated: a position's hazard spikes when a run starts and falls after.
def check_ph_assumption(cph, df, p_threshold=0.05, print_report=False):
    design = cox_design(df)

    if print_report:
        cph.check_assumptions(design, p_value_threshold=p_threshold, show_plots=False)

    # two transforms: a violation under only one is weaker evidence than under both
    result = proportional_hazard_test(cph, design, time_transform=["km", "rank"])

    out = result.summary.reset_index()
    out.columns = ["covariate", "time_transform", "test_statistic", "p", "neg_log2_p"]
    out["violates_ph"] = out["p"] < p_threshold

    return out.sort_values("p").reset_index(drop=True)


# the §6 sentence, as data: which covariates break proportional hazards
def ph_violations(ph_report):
    flagged = ph_report[ph_report["violates_ph"]]
    return sorted(flagged["covariate"].unique())


# S(k|x) for player rows, which is what the query layer reads j and k off of.
def design_for_prediction(cph, rows):
    design = cox_design(rows, drop_constant=False)
    wanted = list(cph.params_.index)

    # a level absent from these rows contributes 0, not a value to be imputed
    return design.reindex(columns=wanted, fill_value=0.0)


def predict_survival(cph, player_rows):
    return cph.predict_survival_function(design_for_prediction(cph, player_rows))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    np.random.seed(RANDOM_SEED)

    conn = connect()
    season = "2025-26"

    survival = build_dataset(conn, season)
    feats = build_cox_features(survival, season, conn)

    cph = fit_cox_model(feats)

    print(f"{season}: {len(feats)} rows, "
          f"{int(feats['event_observed'].sum())} events, "
          f"concordance {cph.concordance_index_:.4f}\n")

    print("\nhazard ratios")
    print(get_hazard_ratios(cph).round(5).to_string())

    print("\nboard against value")
    for key, value in board_vs_value(cph).items():
        print(f"  {key:24} {value:+.5f}" if not np.isnan(value)
              else f"  {key:24} n/a (covariate dropped as constant)")

    print("\nproportional hazards check")
    flagged = check_ph_assumption(cph, feats)
    if flagged.empty:
        print("  no covariate flagged")
    else:
        print(flagged.round(5).to_string(index=False))
        print("\n  Positional runs are a known violation (§6). Reported, not fixed in v1.")
