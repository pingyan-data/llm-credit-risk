"""
03_llm_feature_extraction_offline.py
--------------------------------------
OFFLINE VERSION — no API calls.

Derives the 8 LLM-style features directly from structured fields
using interpretable rules. Produces the same output schema as the
real LLM extraction step so the rest of the pipeline is unchanged.

Note on validity:
  These features overlap with structured inputs (same source),
  so the AUC lift in this offline run reflects redundancy, not
  genuine text understanding. Switch to the real API version
  for a meaningful champion/challenger result.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

IN_PATH  = Path("data/processed/lc_with_text.parquet")
OUT_PATH = Path("data/processed/lc_with_llm_features.parquet")

LLM_FEATURE_COLS = [
    "llm_employment_stability",
    "llm_income_confidence",
    "llm_debt_stress_signal",
    "llm_repayment_intent_score",
    "llm_urgency_flag",
    "llm_financial_literacy_signal",
    "llm_purpose_clarity",
    "llm_future_orientation",
]


def clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def extract_features_offline(row: pd.Series) -> dict:
    """
    Map structured fields → 8 pseudo-LLM features.
    Each feature is on [0, 1].
    """
    fico      = float(row.get("fico_score", 680))
    dti       = float(row.get("dti", 20))
    income    = float(row.get("annual_inc", 50000))
    emp       = float(row.get("emp_length", 3) or 0)
    delinq    = float(row.get("delinq_2yrs", 0))
    pub_rec   = float(row.get("pub_rec", 0))
    revol_util = float(row.get("revol_util", 50) or 50)
    inq       = float(row.get("inq_last_6mths", 0))
    purpose   = str(row.get("purpose", "other"))
    open_acc  = float(row.get("open_acc", 5))
    total_acc = float(row.get("total_acc", 10))
    loan_amnt = float(row.get("loan_amnt", 10000))

    # 1. employment_stability
    #    High emp_length + no delinquencies → stable
    emp_score = clamp(emp / 10.0)
    delinq_pen = clamp(delinq * 0.2)
    employment_stability = clamp(emp_score * 0.7 + (1 - delinq_pen) * 0.3)

    # 2. income_confidence
    #    High income relative to loan amount + low DTI → confident
    loan_to_income = loan_amnt / (income + 1)
    inc_score  = clamp(1 - loan_to_income * 2)
    dti_score  = clamp(1 - dti / 50.0)
    income_confidence = clamp(inc_score * 0.5 + dti_score * 0.5)

    # 3. debt_stress_signal
    #    High DTI + high revol_util + delinquencies → stressed
    dti_stress    = clamp(dti / 50.0)
    util_stress   = clamp(revol_util / 100.0)
    delinq_stress = clamp(min(delinq / 3.0, 1.0))
    debt_stress_signal = clamp(
        dti_stress * 0.4 + util_stress * 0.35 + delinq_stress * 0.25
    )

    # 4. repayment_intent_score
    #    Low delinquencies + good FICO + low pub_rec → strong intent
    fico_norm  = clamp((fico - 580) / (850 - 580))
    pub_pen    = clamp(pub_rec * 0.3)
    repayment_intent_score = clamp(
        fico_norm * 0.5 + (1 - delinq_pen) * 0.3 + (1 - pub_pen) * 0.2
    )

    # 5. urgency_flag
    #    High inq + high DTI + delinquencies → urgent/stressed
    inq_signal  = clamp(inq / 5.0)
    urgency_flag = clamp(
        inq_signal * 0.4 + dti_stress * 0.35 + delinq_pen * 0.25
    )

    # 6. financial_literacy_signal
    #    Many open accounts (managed well) + low revol_util + verified income
    acc_ratio  = clamp(open_acc / max(total_acc, 1))
    util_ok    = clamp(1 - revol_util / 100.0)
    financial_literacy_signal = clamp(acc_ratio * 0.4 + util_ok * 0.4 + fico_norm * 0.2)

    # 7. purpose_clarity
    #    Specific purposes score higher than "other"
    purpose_scores = {
        "debt_consolidation": 0.80,
        "home_improvement":   0.85,
        "medical":            0.90,
        "small_business":     0.75,
        "educational":        0.85,
        "car":                0.80,
        "moving":             0.75,
        "wedding":            0.70,
        "vacation":           0.60,
        "credit_card":        0.75,
        "major_purchase":     0.65,
        "house":              0.80,
        "renewable_energy":   0.85,
        "other":              0.40,
    }
    purpose_clarity = purpose_scores.get(purpose, 0.50)
    # Add slight noise for realism
    purpose_clarity = clamp(purpose_clarity + np.random.uniform(-0.05, 0.05))

    # 8. future_orientation
    #    Low delinq + good FICO + long emp → forward-looking
    future_orientation = clamp(
        fico_norm * 0.4 + emp_score * 0.35 + (1 - delinq_pen) * 0.25
    )

    return {
        "llm_employment_stability":    round(employment_stability, 4),
        "llm_income_confidence":       round(income_confidence, 4),
        "llm_debt_stress_signal":      round(debt_stress_signal, 4),
        "llm_repayment_intent_score":  round(repayment_intent_score, 4),
        "llm_urgency_flag":            round(urgency_flag, 4),
        "llm_financial_literacy_signal": round(financial_literacy_signal, 4),
        "llm_purpose_clarity":         round(purpose_clarity, 4),
        "llm_future_orientation":      round(future_orientation, 4),
    }


def main():
    np.random.seed(42)

    print(f"Loading {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)
    print(f"  {len(df):,} rows")

    print("Extracting features (offline rules) ...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        records.append(extract_features_offline(row))

    feat_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, feat_df], axis=1)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved → {OUT_PATH}")

    print("\nFeature preview:")
    print(df[["default"] + LLM_FEATURE_COLS].head(5).round(3).to_string())

    print("\nCorrelation with default label:")
    corr = df[LLM_FEATURE_COLS + ["default"]].corr()["default"].drop("default")
    print(corr.sort_values().round(3).to_string())

    print("\n⚠️  Note: offline features are derived from structured data.")
    print("   AUC lift reflects redundancy, not genuine LLM text understanding.")
    print("   Use 03_llm_feature_extraction.py with API key for real results.")


if __name__ == "__main__":
    main()
