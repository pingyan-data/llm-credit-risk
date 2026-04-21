"""
06_explainability_policy.py
---------------------------
Three things:
  1. SHAP explainability — global + per-loan attribution for Model C
  2. Calibration curves — reliability diagrams for both models
  3. Decision policy — PD thresholds → Approve / Review / Decline

Outputs:
  outputs/shap_summary.csv         global mean |SHAP| per feature
  outputs/shap_sample.csv          SHAP values for 500 sampled loans
  outputs/calibration_curves.csv   for plotting
  outputs/policy_decisions.csv     loan-level decisions
  outputs/policy_summary.json      approval/decline/review rates
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path
warnings.filterwarnings("ignore")

X_C_PATH   = Path("data/processed/features_C.parquet")
X_A_PATH   = Path("data/processed/features_A.parquet")
Y_PATH     = Path("data/processed/labels.parquet")
PRED_PATH  = Path("outputs/oof_predictions.parquet")
OUT_DIR    = Path("outputs")

# Decision thresholds (calibrated PD)
THRESHOLD_APPROVE  = 0.10   # PD < 10% → Approve
THRESHOLD_DECLINE  = 0.25   # PD > 25% → Decline
# 10%–25% → Manual Review


def load_models():
    model_A = xgb.XGBClassifier()
    model_A.load_model(str(OUT_DIR / "model_A.json"))

    model_C = xgb.XGBClassifier()
    model_C.load_model(str(OUT_DIR / "model_C.json"))

    with open(OUT_DIR / "calibrator_A.pkl", "rb") as f:
        cal_A = pickle.load(f)
    with open(OUT_DIR / "calibrator_C.pkl", "rb") as f:
        cal_C = pickle.load(f)

    return model_A, model_C, cal_A, cal_C


def shap_analysis(model_C, X_C: pd.DataFrame, y: pd.Series, n_sample: int = 500):
    """Compute SHAP values using TreeExplainer."""
    print(f"Computing SHAP values (n={n_sample}) ...")

    sample = X_C.sample(min(n_sample, len(X_C)), random_state=42)

    explainer   = shap.TreeExplainer(model_C)
    shap_values = explainer.shap_values(sample)

    # Global importance: mean |SHAP|
    shap_df = pd.DataFrame(shap_values, columns=X_C.columns)
    global_imp = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    global_imp.columns = ["feature", "mean_abs_shap"]

    # Tag LLM vs structured features
    global_imp["type"] = global_imp["feature"].apply(
        lambda x: "LLM" if x.startswith("llm_") else "Structured"
    )

    print("\nTop 20 features by mean |SHAP|:")
    print(global_imp.head(20).to_string(index=False))

    llm_total     = global_imp[global_imp["type"] == "LLM"]["mean_abs_shap"].sum()
    struct_total  = global_imp[global_imp["type"] == "Structured"]["mean_abs_shap"].sum()
    grand_total   = llm_total + struct_total
    print(f"\nLLM SHAP share:        {llm_total/grand_total:.1%}")
    print(f"Structured SHAP share: {struct_total/grand_total:.1%}")

    # Per-loan SHAP (save sample)
    shap_sample = shap_df.copy()
    shap_sample["loan_idx"] = sample.index.values
    shap_sample["y"]        = y.loc[sample.index].values

    return global_imp, shap_sample


def calibration_curves(y, pred_df: pd.DataFrame):
    """Build calibration data for plotting (no matplotlib required)."""
    from sklearn.calibration import calibration_curve

    results = []
    for col_name, label in [
        ("prob_A_cal", "Model A — Calibrated"),
        ("prob_C_cal", "Model C — Calibrated"),
        ("prob_A_raw", "Model A — Raw"),
        ("prob_C_raw", "Model C — Raw"),
    ]:
        if col_name not in pred_df.columns:
            continue
        frac_pos, mean_pred = calibration_curve(
            y, pred_df[col_name], n_bins=10, strategy="quantile"
        )
        for fp, mp in zip(frac_pos, mean_pred):
            results.append({
                "model":          label,
                "mean_predicted": round(float(mp), 4),
                "fraction_pos":   round(float(fp), 4),
            })

    return pd.DataFrame(results)


def apply_policy(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Assign decisions based on calibrated PD thresholds."""

    def decision(pd_val):
        if pd_val < THRESHOLD_APPROVE:
            return "Approve"
        elif pd_val > THRESHOLD_DECLINE:
            return "Decline"
        else:
            return "Review"

    df = pred_df.copy()
    df["decision_A"] = df["prob_A_cal"].map(decision)
    df["decision_C"] = df["prob_C_cal"].map(decision)

    return df


def policy_summary(policy_df: pd.DataFrame) -> dict:
    """Compute approval/decline/review rates and default capture."""
    results = {}

    for model_col, dec_col in [("prob_A_cal", "decision_A"), ("prob_C_cal", "decision_C")]:
        label = "model_A" if "A" in dec_col else "model_C"
        total = len(policy_df)

        approve_mask  = policy_df[dec_col] == "Approve"
        decline_mask  = policy_df[dec_col] == "Decline"
        review_mask   = policy_df[dec_col] == "Review"

        # Among declined loans, what fraction were true defaults?
        decline_precision = policy_df.loc[decline_mask, "y"].mean() if decline_mask.any() else 0

        # Among approved loans, default rate
        approve_dr = policy_df.loc[approve_mask, "y"].mean() if approve_mask.any() else 0

        results[label] = {
            "approval_rate":     round(approve_mask.mean(), 3),
            "review_rate":       round(review_mask.mean(), 3),
            "decline_rate":      round(decline_mask.mean(), 3),
            "approved_default_rate":  round(float(approve_dr), 4),
            "declined_precision":     round(float(decline_precision), 4),
        }

    return results


def main():
    print("Loading data and models ...")
    X_C   = pd.read_parquet(X_C_PATH)
    X_A   = pd.read_parquet(X_A_PATH)
    y     = pd.read_parquet(Y_PATH).squeeze()
    preds = pd.read_parquet(PRED_PATH)
    preds["y"] = y.values

    model_A, model_C, cal_A, cal_C = load_models()

    # ── 1. SHAP ─────────────────────────────────────────────────────
    shap_global, shap_sample = shap_analysis(model_C, X_C, y)
    shap_global.to_csv(OUT_DIR / "shap_summary.csv", index=False)
    shap_sample.to_csv(OUT_DIR / "shap_sample.csv", index=False)
    print("✓ SHAP saved")

    # ── 2. Calibration curves ────────────────────────────────────────
    cal_df = calibration_curves(y, preds)
    cal_df.to_csv(OUT_DIR / "calibration_curves.csv", index=False)
    print("✓ Calibration curves saved")

    # ── 3. Decision policy ───────────────────────────────────────────
    print(f"\nApplying decision policy:")
    print(f"  Approve:  PD < {THRESHOLD_APPROVE:.0%}")
    print(f"  Review:   {THRESHOLD_APPROVE:.0%} ≤ PD ≤ {THRESHOLD_DECLINE:.0%}")
    print(f"  Decline:  PD > {THRESHOLD_DECLINE:.0%}")

    policy_df = apply_policy(preds)
    summary   = policy_summary(policy_df)

    print("\nPolicy summary:")
    for model, stats in summary.items():
        print(f"\n  {model}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.1%}" if isinstance(v, float) else f"    {k}: {v}")

    policy_df.to_csv(OUT_DIR / "policy_decisions.csv", index=False)
    with open(OUT_DIR / "policy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Policy outputs saved")

    # ── 4. LLM feature contribution summary ─────────────────────────
    llm_features = shap_global[shap_global["type"] == "LLM"].copy()
    print("\nLLM feature SHAP ranking:")
    print(llm_features.to_string(index=False))

    print("\n✓ All explainability outputs saved to outputs/")


if __name__ == "__main__":
    main()
