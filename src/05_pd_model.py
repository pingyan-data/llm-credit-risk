"""
05_pd_model.py
--------------
Train XGBoost PD models, calibrate with isotonic regression,
and run champion/challenger comparison.

Outputs:
  outputs/model_A.json          XGBoost model A (structured only)
  outputs/model_C.json          XGBoost model C (structured + LLM)
  outputs/calibrator_A.pkl      Isotonic calibrator A
  outputs/calibrator_C.pkl      Isotonic calibrator C
  outputs/evaluation_report.json  AUC, KS, Brier, Gini per model
  outputs/champion_challenger.csv  Per-decile comparison
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    average_precision_score,
)
import xgboost as xgb
warnings.filterwarnings("ignore")

X_A_PATH = Path("data/processed/features_A.parquet")
X_C_PATH = Path("data/processed/features_C.parquet")
Y_PATH   = Path("data/processed/labels.parquet")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS   = 5

# ── XGBoost hyperparameters ─────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma":            1.0,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "use_label_encoder": False,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}


# ── Metrics ─────────────────────────────────────────────────────────

def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov separation between score distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true, y_prob) -> float:
    return float(2 * roc_auc_score(y_true, y_prob) - 1)


def evaluate(y_true, y_prob, label: str) -> dict:
    auc    = roc_auc_score(y_true, y_prob)
    ks     = ks_statistic(y_true, y_prob)
    gini   = gini_coefficient(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\n── {label} ──")
    print(f"  AUC:       {auc:.4f}")
    print(f"  KS:        {ks:.4f}")
    print(f"  Gini:      {gini:.4f}")
    print(f"  Brier:     {brier:.4f}  (lower = better)")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    return {
        "model": label,
        "auc": round(auc, 4),
        "ks":  round(ks, 4),
        "gini": round(gini, 4),
        "brier": round(brier, 4),
        "pr_auc": round(pr_auc, 4),
    }


# ── Calibration ─────────────────────────────────────────────────────

def calibrate_isotonic(y_train, raw_probs_train, raw_probs_test):
    """Fit isotonic regression on train, apply to test."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_probs_train, y_train)
    return ir, ir.predict(raw_probs_test)


# ── Champion / Challenger decile table ──────────────────────────────

def decile_table(y_true, prob_A, prob_C) -> pd.DataFrame:
    """Compare both models at each score decile."""
    df = pd.DataFrame({
        "y":      y_true.values,
        "prob_A": prob_A,
        "prob_C": prob_C,
    })
    df["decile_A"] = pd.qcut(df["prob_A"], q=10, labels=False, duplicates="drop")
    df["decile_C"] = pd.qcut(df["prob_C"], q=10, labels=False, duplicates="drop")

    rows = []
    for d in range(10):
        sub_A = df[df["decile_A"] == d]
        sub_C = df[df["decile_C"] == d]
        rows.append({
            "decile":         d + 1,
            "count_A":        len(sub_A),
            "default_rate_A": round(sub_A["y"].mean(), 4),
            "avg_pd_A":       round(sub_A["prob_A"].mean(), 4),
            "count_C":        len(sub_C),
            "default_rate_C": round(sub_C["y"].mean(), 4),
            "avg_pd_C":       round(sub_C["prob_C"].mean(), 4),
        })

    return pd.DataFrame(rows)


# ── Main training loop ───────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> tuple:
    """
    Cross-validated XGBoost training with out-of-fold predictions.
    Returns (trained_model_on_full_data, oof_probs, test_probs_placeholder).
    """
    print(f"\nTraining {model_name} ({X.shape[1]} features) ...")

    skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof     = np.zeros(len(y))
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        fold_prob = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = fold_prob
        fold_aucs.append(roc_auc_score(y_val, fold_prob))

    print(f"  CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Retrain on full data
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, verbose=False)

    return final_model, oof


def main():
    print("Loading feature matrices ...")
    X_A = pd.read_parquet(X_A_PATH)
    X_C = pd.read_parquet(X_C_PATH)
    y   = pd.read_parquet(Y_PATH).squeeze()

    print(f"  X_A: {X_A.shape}, X_C: {X_C.shape}, y: {y.shape}")
    print(f"  Default rate: {y.mean():.1%}")

    # ── Train both models ───────────────────────────────────────────
    model_A, oof_A = train_model(X_A, y, "Model A — Structured only")
    model_C, oof_C = train_model(X_C, y, "Model C — Structured + LLM")

    # ── Calibration (on OOF predictions) ───────────────────────────
    print("\nCalibrating with Isotonic Regression ...")
    ir_A, cal_A = calibrate_isotonic(y, oof_A, oof_A)
    ir_C, cal_C = calibrate_isotonic(y, oof_C, oof_C)

    # ── Evaluate ────────────────────────────────────────────────────
    results = []
    results.append(evaluate(y, oof_A, "Model A — Raw XGBoost"))
    results.append(evaluate(y, cal_A, "Model A — Calibrated"))
    results.append(evaluate(y, oof_C, "Model C — Raw XGBoost"))
    results.append(evaluate(y, cal_C, "Model C — Calibrated"))

    # ── AUC lift ────────────────────────────────────────────────────
    auc_A = roc_auc_score(y, cal_A)
    auc_C = roc_auc_score(y, cal_C)
    lift  = auc_C - auc_A
    print(f"\n{'='*40}")
    print(f"AUC lift from LLM features: +{lift:.4f} ({lift*100:.2f} bps)")
    print(f"  Model A calibrated AUC: {auc_A:.4f}")
    print(f"  Model C calibrated AUC: {auc_C:.4f}")
    print(f"{'='*40}")

    # ── Feature importance for Model C ──────────────────────────────
    feat_names = list(X_C.columns)
    importances = model_C.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features — Model C:")
    print(fi_df.head(15).to_string(index=False))

    llm_importance = fi_df[fi_df["feature"].str.startswith("llm_")]["importance"].sum()
    total_importance = fi_df["importance"].sum()
    print(f"\nLLM features contribute {llm_importance/total_importance:.1%} of total feature importance")

    # ── Champion / Challenger table ──────────────────────────────────
    cc_table = decile_table(y, cal_A, cal_C)
    print("\nChampion/Challenger — Decile Table:")
    print(cc_table.to_string(index=False))

    # ── Save outputs ────────────────────────────────────────────────
    model_A.save_model(str(OUT_DIR / "model_A.json"))
    model_C.save_model(str(OUT_DIR / "model_C.json"))

    with open(OUT_DIR / "calibrator_A.pkl", "wb") as f:
        pickle.dump(ir_A, f)
    with open(OUT_DIR / "calibrator_C.pkl", "wb") as f:
        pickle.dump(ir_C, f)

    with open(OUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    cc_table.to_csv(OUT_DIR / "champion_challenger.csv", index=False)
    fi_df.to_csv(OUT_DIR / "feature_importance_C.csv", index=False)

    # Save OOF predictions for SHAP / calibration analysis
    pred_df = pd.DataFrame({
        "y":           y.values,
        "prob_A_raw":  oof_A,
        "prob_A_cal":  cal_A,
        "prob_C_raw":  oof_C,
        "prob_C_cal":  cal_C,
    })
    pred_df.to_parquet(OUT_DIR / "oof_predictions.parquet", index=False)

    print("\n✓ All outputs saved to outputs/")


if __name__ == "__main__":
    main()
