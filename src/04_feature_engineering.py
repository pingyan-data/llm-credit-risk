"""
04_feature_engineering.py
--------------------------
Build two feature matrices:
  - X_A : structured features only          (Model A baseline)
  - X_C : structured + LLM-extracted feats  (Model C challenger)

Output: data/processed/features_A.parquet
        data/processed/features_C.parquet
        data/processed/labels.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

IN_PATH  = Path("data/processed/lc_with_llm_features.parquet")
OUT_A    = Path("data/processed/features_A.parquet")
OUT_C    = Path("data/processed/features_C.parquet")
OUT_Y    = Path("data/processed/labels.parquet")

# ── Feature definitions ─────────────────────────────────────────────

NUMERIC_COLS = [
    "loan_amnt",
    "term",
    "annual_inc",
    "dti",
    "fico_score",
    "inq_last_6mths",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "emp_length",
]

CATEGORICAL_COLS = [
    "home_ownership",
    "purpose",
    "verification_status",
    "addr_state",
]

LLM_COLS = [
    "llm_employment_stability",
    "llm_income_confidence",
    "llm_debt_stress_signal",
    "llm_repayment_intent_score",
    "llm_urgency_flag",
    "llm_financial_literacy_signal",
    "llm_purpose_clarity",
    "llm_future_orientation",
]

TARGET = "default"


def engineer_structured(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, impute, and encode structured features."""

    # ── Numeric: impute with median ─────────────────────────────────
    num_avail = [c for c in NUMERIC_COLS if c in df.columns]
    imp_num   = SimpleImputer(strategy="median")
    num_data  = pd.DataFrame(
        imp_num.fit_transform(df[num_avail]),
        columns=num_avail,
        index=df.index,
    )

    # ── Derived numeric features ────────────────────────────────────
    num_data["loan_to_income"]     = num_data["loan_amnt"] / (num_data["annual_inc"] + 1)
    num_data["fico_dti_ratio"]     = num_data["fico_score"] / (num_data["dti"] + 1)
    num_data["credit_age_proxy"]   = num_data["total_acc"] / (num_data["open_acc"] + 1)
    num_data["delinq_pub_sum"]     = num_data["delinq_2yrs"] + num_data["pub_rec"]
    num_data["log_annual_inc"]     = np.log1p(num_data["annual_inc"])
    num_data["log_revol_bal"]      = np.log1p(num_data["revol_bal"])

    # ── Categorical: ordinal encode ─────────────────────────────────
    cat_avail = [c for c in CATEGORICAL_COLS if c in df.columns]
    cat_raw   = df[cat_avail].fillna("MISSING")
    enc       = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_data  = pd.DataFrame(
        enc.fit_transform(cat_raw),
        columns=cat_avail,
        index=df.index,
    )

    structured = pd.concat([num_data, cat_data], axis=1)
    return structured


def main():
    print(f"Loading {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)
    print(f"  Shape: {df.shape}")

    y = df[TARGET].copy()

    print("Engineering structured features ...")
    X_struct = engineer_structured(df)
    print(f"  Structured feature matrix: {X_struct.shape}")

    # ── LLM features ────────────────────────────────────────────────
    llm_avail = [c for c in LLM_COLS if c in df.columns]
    if not llm_avail:
        raise ValueError("No LLM feature columns found. Run 03_llm_feature_extraction.py first.")

    X_llm = df[llm_avail].copy()
    # Impute any missing LLM features with 0.5 (neutral)
    X_llm = X_llm.fillna(0.5)
    print(f"  LLM feature matrix: {X_llm.shape}")

    # ── Assemble final matrices ─────────────────────────────────────
    X_A = X_struct.copy()
    X_C = pd.concat([X_struct, X_llm], axis=1)

    print(f"\nModel A features: {X_A.shape[1]} columns")
    print(f"Model C features: {X_C.shape[1]} columns  (+{X_llm.shape[1]} LLM)")
    print(f"Label distribution:\n{y.value_counts().to_string()}")

    # ── Save ────────────────────────────────────────────────────────
    X_A.to_parquet(OUT_A, index=False)
    X_C.to_parquet(OUT_C, index=False)
    y.to_frame().to_parquet(OUT_Y, index=False)

    print(f"\n✓ Saved features_A → {OUT_A}")
    print(f"✓ Saved features_C → {OUT_C}")
    print(f"✓ Saved labels     → {OUT_Y}")

    print("\nTop structured features (first 5):")
    print(X_A.head(3).round(3).iloc[:, :6].to_string())

    print("\nLLM feature preview:")
    print(X_llm.head(3).round(3).to_string())


if __name__ == "__main__":
    main()
