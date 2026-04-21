"""
07_ablation_study.py
---------------------
Answers: "Is the AUC lift from LLM features real, or just from more data?"

Three models compared with identical CV setup:
  Model A — structured features, N=5000 (baseline)
  Model B — structured features, N=10000 (2x data, no LLM)
  Model C — structured + LLM features, N=5000

If B ≈ A  → data volume alone doesn't help much
If C > A  → LLM features carry genuine signal
If C ≈ B  → LLM features just act like more data (less convincing)
If C > B  → LLM features add value beyond data volume (most convincing)

Run from project root:
    python src/07_ablation_study.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────
CLEAN_PATH = Path("data/processed/lc_clean.parquet")      # full LC dataset
FEAT_A     = Path("data/processed/features_A.parquet")    # structured only, N=5000
FEAT_C     = Path("data/processed/features_C.parquet")    # structured + LLM, N=5000
LABELS     = Path("data/processed/labels.parquet")        # labels for N=5000

RANDOM_STATE = 42
N_CV_FOLDS   = 5

XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,
    "gamma":             1.0,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "eval_metric":       "auc",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}


# ── Metrics ─────────────────────────────────────────────────────────

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_prob):
    auc   = roc_auc_score(y_true, y_prob)
    ks    = ks_stat(y_true, y_prob)
    gini  = 2 * auc - 1
    brier = brier_score_loss(y_true, y_prob)
    return {"auc": auc, "ks": ks, "gini": gini, "brier": brier}


def cross_val_oof(X, y, params):
    """Return out-of-fold predicted probabilities."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = prob
        fold_aucs.append(roc_auc_score(y.iloc[val_idx], prob))

    print(f"    CV AUC folds: {[round(a,4) for a in fold_aucs]}")
    print(f"    Mean: {np.mean(fold_aucs):.4f}  Std: {np.std(fold_aucs):.4f}")
    return oof, fold_aucs


def calibrate(y, oof):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof, y)
    return ir.predict(oof)


# ── Build Model B data (2x structured, no LLM) ─────────────────────

def build_model_b(feat_a_cols):
    """
    Sample 10000 rows from the full LC dataset using only structured features.
    This is 2x the size of Model A, but no LLM features.
    """
    print("  Building Model B dataset (10k rows, structured only) ...")
    df_full = pd.read_parquet(CLEAN_PATH)
    df_full = df_full.reset_index(drop=True)

    # Stratified sample of 10000
    n = 10_000
    d0 = df_full[df_full["default"] == 0].sample(min(n // 2, (df_full["default"]==0).sum()), random_state=RANDOM_STATE)
    d1 = df_full[df_full["default"] == 1].sample(min(n // 2, (df_full["default"]==1).sum()), random_state=RANDOM_STATE)
    df_b = pd.concat([d0, d1]).reset_index(drop=True)

    y_b = df_b["default"].copy()

    # ── Replicate feature engineering from 04 inline ────────────────
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder

    NUMERIC = [
        "loan_amnt", "term", "annual_inc", "dti", "fico_score",
        "inq_last_6mths", "delinq_2yrs", "open_acc", "pub_rec",
        "revol_bal", "revol_util", "total_acc", "emp_length",
    ]
    CATEGORICAL = ["home_ownership", "purpose", "verification_status", "addr_state"]

    num_avail = [c for c in NUMERIC if c in df_b.columns]
    imp = SimpleImputer(strategy="median")
    num_data = pd.DataFrame(imp.fit_transform(df_b[num_avail]), columns=num_avail)

    # Derived features
    num_data["loan_to_income"]   = num_data["loan_amnt"] / (num_data["annual_inc"] + 1)
    num_data["fico_dti_ratio"]   = num_data["fico_score"] / (num_data["dti"] + 1)
    num_data["credit_age_proxy"] = num_data["total_acc"] / (num_data["open_acc"] + 1)
    num_data["delinq_pub_sum"]   = num_data["delinq_2yrs"] + num_data["pub_rec"]
    num_data["log_annual_inc"]   = np.log1p(num_data["annual_inc"])
    num_data["log_revol_bal"]    = np.log1p(num_data["revol_bal"])

    cat_avail = [c for c in CATEGORICAL if c in df_b.columns]
    cat_raw   = df_b[cat_avail].fillna("MISSING")
    enc       = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_data  = pd.DataFrame(enc.fit_transform(cat_raw), columns=cat_avail)

    X_b = pd.concat([num_data, cat_data], axis=1)

    # Keep only columns that exist in Model A
    shared_cols = [c for c in feat_a_cols if c in X_b.columns]
    X_b = X_b[shared_cols]

    print(f"  Model B: {X_b.shape[0]:,} rows × {X_b.shape[1]} features")
    print(f"  Default rate: {y_b.mean():.1%}")
    return X_b, y_b


# ── Main ─────────────────────────────────────────────────────────────

def print_result(label, metrics, n_rows, n_features):
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  Rows: {n_rows:,}   Features: {n_features}")
    print(f"  AUC:   {metrics['auc']:.4f}")
    print(f"  KS:    {metrics['ks']:.4f}")
    print(f"  Gini:  {metrics['gini']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}  (lower = better)")


def main():
    print("=" * 60)
    print("  ABLATION STUDY")
    print("  Question: Data volume vs LLM feature value")
    print("=" * 60)

    # ── Load A and C ─────────────────────────────────────────────────
    print("\nLoading feature matrices ...")
    X_A = pd.read_parquet(FEAT_A)
    X_C = pd.read_parquet(FEAT_C)
    y   = pd.read_parquet(LABELS).squeeze()

    print(f"  Model A: {X_A.shape[0]:,} rows × {X_A.shape[1]} features")
    print(f"  Model C: {X_C.shape[0]:,} rows × {X_C.shape[1]} features")
    print(f"  Default rate: {y.mean():.1%}")

    # ── Build B ──────────────────────────────────────────────────────
    print("\nBuilding Model B (2x data, structured only) ...")
    X_B, y_B = build_model_b(list(X_A.columns))

    # ── Train all three ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Training Model A — structured only, N=5,000")
    oof_A, folds_A = cross_val_oof(X_A, y, XGB_PARAMS)
    cal_A = calibrate(y, oof_A)
    m_A   = evaluate(y, cal_A)

    print("\nTraining Model B — structured only, N=10,000")
    oof_B, folds_B = cross_val_oof(X_B, y_B, XGB_PARAMS)
    cal_B = calibrate(y_B, oof_B)
    m_B   = evaluate(y_B, cal_B)

    print("\nTraining Model C — structured + LLM features, N=5,000")
    oof_C, folds_C = cross_val_oof(X_C, y, XGB_PARAMS)
    cal_C = calibrate(y, oof_C)
    m_C   = evaluate(y, cal_C)

    # ── Results table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print_result("Model A  Structured only     (N=5,000)",  m_A, len(y),   X_A.shape[1])
    print_result("Model B  Structured only     (N=10,000)", m_B, len(y_B), X_B.shape[1])
    print_result("Model C  Structured + LLM    (N=5,000)",  m_C, len(y),   X_C.shape[1])

    # ── Lift analysis ────────────────────────────────────────────────
    lift_data_vol = m_B["auc"] - m_A["auc"]
    lift_llm      = m_C["auc"] - m_A["auc"]
    lift_llm_vs_b = m_C["auc"] - m_B["auc"]

    print("\n" + "=" * 60)
    print("  LIFT ANALYSIS")
    print(f"\n  A → B  (2x data, same features):  {lift_data_vol:+.4f}")
    print(f"  A → C  (LLM features, same N):    {lift_llm:+.4f}")
    print(f"  B → C  (LLM vs more data):        {lift_llm_vs_b:+.4f}")

    print("\n  INTERPRETATION:")
    if abs(lift_data_vol) < 0.005:
        print("  ✓ Data volume alone has minimal effect (A ≈ B)")
    else:
        print(f"  → More data moves AUC by {lift_data_vol:+.4f} (volume matters somewhat)")

    if lift_llm > lift_data_vol + 0.002:
        print("  ✓ LLM features add value BEYOND data volume (C > B)")
    elif lift_llm > 0.003:
        print("  ~ LLM features help, but effect is similar to more data")
    else:
        print("  ✗ LLM features add little incremental value in offline mode")
        print("    → Expected with offline rules (same source as structured data)")
        print("    → Real API-generated features would show genuine signal")

    print("\n  NOTE: Offline LLM features are derived from structured fields,")
    print("  so overlap is expected. Use real API for a clean experiment.")
    print("=" * 60)

    # ── Save summary ─────────────────────────────────────────────────
    summary = pd.DataFrame([
        {"model": "A — Structured (N=5k)",  "n_rows": len(y),   "n_features": X_A.shape[1], **m_A},
        {"model": "B — Structured (N=10k)", "n_rows": len(y_B), "n_features": X_B.shape[1], **m_B},
        {"model": "C — Struct+LLM (N=5k)",  "n_rows": len(y),   "n_features": X_C.shape[1], **m_C},
    ])
    summary = summary.round(4)
    out_path = Path("outputs/ablation_study.csv")
    out_path.parent.mkdir(exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
