"""
01_data_prep.py
---------------
Lending Club data loading, cleaning, and label construction.

Input:  data/raw/lending_club.csv   (download from Kaggle)
Output: data/processed/lc_clean.parquet

Key decisions:
  - Only keep loans with terminal status (Fully Paid / Charged Off)
  - Drop post-origination leakage columns
  - Binary label: default = 1 (Charged Off), paid = 0 (Fully Paid)
  - Keep a clean set of pre-application structured features
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH   = Path("data/raw/lending_club.csv")
OUT_PATH   = Path("data/processed/lc_clean.parquet")

# ------------------------------------------------------------------
# Columns available at application time (no post-origination leakage)
# ------------------------------------------------------------------
STRUCTURED_COLS = [
    "loan_amnt",          # requested loan amount
    "term",               # 36 or 60 months
    "annual_inc",         # self-reported annual income
    "dti",                # debt-to-income ratio
    "fico_range_low",     # FICO score lower bound
    "fico_range_high",    # FICO score upper bound
    "inq_last_6mths",     # credit enquiries in last 6 months
    "delinq_2yrs",        # delinquencies in past 2 years
    "open_acc",           # number of open credit lines
    "pub_rec",            # number of derogatory public records
    "revol_bal",          # total revolving credit balance
    "revol_util",         # revolving line utilisation rate
    "total_acc",          # total credit lines ever opened
    "emp_length",         # employment length (categorical)
    "home_ownership",     # RENT / OWN / MORTGAGE / OTHER
    "purpose",            # stated loan purpose
    "verification_status",# income verification status
    "addr_state",         # US state
]

# Text cols we will use to generate synthetic borrower explanations
TEXT_CONTEXT_COLS = [
    "emp_title",          # job title (free text) — context for generation
    "purpose",            # loan purpose category
    "annual_inc",
    "dti",
    "emp_length",
    "home_ownership",
    "loan_amnt",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "pub_rec",
]

TARGET_COL = "default"


def load_and_clean(path: Path, nrows: int = None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"  Raw shape: {df.shape}")

    # ── 1. Keep only terminal loans ──────────────────────────────────
    terminal = {"Fully Paid", "Charged Off"}
    df = df[df["loan_status"].isin(terminal)].copy()
    print(f"  After terminal filter: {df.shape[0]:,} rows")

    # ── 2. Binary target ─────────────────────────────────────────────
    df[TARGET_COL] = (df["loan_status"] == "Charged Off").astype(int)

    # ── 3. Select features ───────────────────────────────────────────
    keep_cols = list(set(STRUCTURED_COLS + TEXT_CONTEXT_COLS + [TARGET_COL]))
    available  = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # ── 4. Clean individual columns ──────────────────────────────────
    # term: "36 months" → 36
    if "term" in df.columns:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

    # emp_length: "10+ years" → 10, "< 1 year" → 0
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8,  "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(emp_map)

    # revol_util: strip "%" → float
    if "revol_util" in df.columns:
        df["revol_util"] = pd.to_numeric(
            df["revol_util"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )

    # FICO midpoint
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # ── 5. Drop extreme outliers ─────────────────────────────────────
    df = df[df["annual_inc"] < 1_000_000].copy()   # remove extreme income
    df = df[df["dti"] < 100].copy()                 # DTI sanity cap

    # ── 6. Drop rows missing critical fields ────────────────────────
    critical = ["annual_inc", "dti", "fico_score", "loan_amnt"]
    critical_avail = [c for c in critical if c in df.columns]
    df = df.dropna(subset=critical_avail)

    print(f"  Final clean shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Default rate: {df[TARGET_COL].mean():.1%}")

    return df


def main():
    df = load_and_clean(RAW_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved to {OUT_PATH}")
    print(df[STRUCTURED_COLS[:6] + [TARGET_COL]].head(3).to_string())


if __name__ == "__main__":
    main()
