"""
02_generate_borrower_text_offline.py
-------------------------------------
OFFLINE VERSION — no API calls.

Generates synthetic borrower text using rule-based templates
from structured fields. No cost, no internet required.

Use this to validate the full pipeline locally.
When ready to use real LLM generation, switch to 02_generate_borrower_text.py
"""

import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

IN_PATH  = Path("data/processed/lc_clean.parquet")
OUT_PATH = Path("data/processed/lc_with_text.parquet")

N_SAMPLES = 5_000
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

# ── Template components ─────────────────────────────────────────────

PURPOSE_PHRASES = {
    "debt_consolidation": [
        "consolidate my existing debts into one manageable payment",
        "simplify my finances by combining multiple debts",
        "reduce my monthly obligations by consolidating what I owe",
    ],
    "credit_card": [
        "pay off my high-interest credit card balances",
        "eliminate my credit card debt and save on interest",
        "clear my credit card bills which have been accumulating",
    ],
    "home_improvement": [
        "renovate my home and make some needed repairs",
        "fund improvements to my property",
        "cover the cost of home repairs I've been planning",
    ],
    "car": [
        "purchase a reliable vehicle for commuting to work",
        "buy a used car to replace my current one",
        "cover car repair costs that came up unexpectedly",
    ],
    "medical": [
        "cover unexpected medical expenses",
        "pay for medical treatment not covered by insurance",
        "handle medical bills that have been piling up",
    ],
    "small_business": [
        "invest in growing my small business",
        "fund equipment purchases for my business",
        "cover operating costs while my business expands",
    ],
    "major_purchase": [
        "finance a major purchase I have been planning",
        "cover the cost of a significant purchase",
        "fund an important purchase I need to make",
    ],
    "moving": [
        "cover relocation costs for a new job opportunity",
        "fund my move to a new city",
        "pay for moving expenses",
    ],
    "vacation": [
        "fund a family vacation we have been planning",
        "cover travel costs for an upcoming trip",
        "pay for a holiday my family deserves",
    ],
    "wedding": [
        "cover wedding expenses for our upcoming ceremony",
        "fund our wedding and related costs",
        "pay for wedding arrangements",
    ],
    "educational": [
        "fund further education and professional development",
        "cover tuition and course fees",
        "invest in my education to advance my career",
    ],
    "other": [
        "cover some personal expenses I have been managing",
        "handle various personal financial needs",
        "take care of some outstanding personal costs",
    ],
}

EMPLOYMENT_PHRASES = {
    "high": [   # emp_length >= 5
        "I have been with my current employer for {years} years and have stable income",
        "With {years} years at my job, I have a reliable and consistent income",
        "My employment of {years} years gives me a solid financial foundation",
    ],
    "medium": [  # 2 <= emp_length < 5
        "I have been employed for {years} years with a steady income",
        "With {years} years of work experience, my income is consistent",
        "I have maintained stable employment for {years} years",
    ],
    "low": [    # emp_length < 2
        "I recently started a new position and am building financial stability",
        "I am in the early stages of my career with growing income",
        "My employment situation is developing positively",
    ],
}

INCOME_PHRASES = {
    "high": [   # annual_inc >= 80000
        "earning ${income:,} annually",
        "with an annual income of ${income:,}",
        "my yearly earnings of ${income:,}",
    ],
    "medium": [  # 40000 <= annual_inc < 80000
        "earning ${income:,} per year",
        "with an income of ${income:,} annually",
        "on a salary of ${income:,} a year",
    ],
    "low": [    # annual_inc < 40000
        "with my current income of ${income:,}",
        "earning ${income:,} annually and managing my budget carefully",
        "on an income of ${income:,}",
    ],
}

DTI_PHRASES = {
    "low": [    # dti < 15
        "My debt obligations are minimal and I have significant capacity to repay",
        "I carry very little existing debt relative to my income",
        "My current debt load is low and well within my means",
    ],
    "medium": [  # 15 <= dti < 30
        "I manage my existing debt responsibly each month",
        "My debt-to-income ratio is reasonable and I keep up with all payments",
        "I have some existing obligations but handle them consistently",
    ],
    "high": [   # dti >= 30
        "I am working to reduce my debt burden and this loan will help",
        "I have existing obligations I am managing, and this will help consolidate",
        "I am committed to reducing my overall debt with a clear plan",
    ],
}

FICO_PHRASES = {
    "excellent": [  # fico >= 750
        "I have maintained an excellent credit history with no missed payments",
        "My credit record is strong with consistent on-time payments",
        "I pride myself on a clean credit history built over many years",
    ],
    "good": [       # 680 <= fico < 750
        "I have a good credit history and have always paid my bills on time",
        "My credit record shows responsible borrowing behaviour",
        "I have maintained a solid credit profile over the years",
    ],
    "fair": [       # 620 <= fico < 680
        "I am working to improve my credit and have been making progress",
        "My credit has had some challenges but I am committed to responsible repayment",
        "I have learned from past financial mistakes and am now more disciplined",
    ],
    "poor": [       # fico < 620
        "I acknowledge my credit history has some difficulties and I am addressing them",
        "I have faced financial challenges in the past but have a plan to move forward",
        "Despite some credit issues, I am committed to making every payment on time",
    ],
}

CLOSING_PHRASES = {
    "positive": [
        "I am confident in my ability to repay and appreciate your consideration.",
        "I have a clear repayment plan and am committed to fulfilling this obligation.",
        "Thank you for considering my application — I am a reliable borrower.",
    ],
    "neutral": [
        "I look forward to your decision on this application.",
        "Please let me know if you need any additional information.",
        "I appreciate you reviewing my application.",
    ],
    "urgent": [
        "I would appreciate a quick decision as this is time-sensitive.",
        "Any help would be greatly appreciated — this is important to me.",
        "I really need this to come through and will prioritise repayment.",
    ],
}


def classify_risk(row) -> str:
    """Classify overall borrower risk for template selection."""
    fico  = row.get("fico_score", 680)
    dti   = row.get("dti", 20)
    delinq = row.get("delinq_2yrs", 0)
    pub   = row.get("pub_rec", 0)

    score = 0
    if fico >= 720:  score += 2
    elif fico >= 660: score += 1
    if dti < 15:     score += 2
    elif dti < 25:   score += 1
    if delinq == 0:  score += 1
    if pub == 0:     score += 1

    if score >= 5:   return "low"
    elif score >= 3: return "medium"
    else:            return "high"


def generate_text(row: pd.Series) -> str:
    fico      = float(row.get("fico_score", 680))
    dti       = float(row.get("dti", 20))
    income    = int(row.get("annual_inc", 50000))
    emp       = float(row.get("emp_length", 3) or 0)
    purpose   = str(row.get("purpose", "other"))
    delinq    = int(row.get("delinq_2yrs", 0))
    risk      = classify_risk(row)

    # ── Select employment phrase ────────────────────────────────────
    if emp >= 5:
        emp_template = random.choice(EMPLOYMENT_PHRASES["high"])
    elif emp >= 2:
        emp_template = random.choice(EMPLOYMENT_PHRASES["medium"])
    else:
        emp_template = random.choice(EMPLOYMENT_PHRASES["low"])
    emp_phrase = emp_template.format(years=int(emp) if emp >= 1 else 1)

    # ── Select income phrase ────────────────────────────────────────
    if income >= 80_000:
        inc_phrase = random.choice(INCOME_PHRASES["high"]).format(income=income)
    elif income >= 40_000:
        inc_phrase = random.choice(INCOME_PHRASES["medium"]).format(income=income)
    else:
        inc_phrase = random.choice(INCOME_PHRASES["low"]).format(income=income)

    # ── Select DTI phrase ───────────────────────────────────────────
    if dti < 15:
        dti_phrase = random.choice(DTI_PHRASES["low"])
    elif dti < 30:
        dti_phrase = random.choice(DTI_PHRASES["medium"])
    else:
        dti_phrase = random.choice(DTI_PHRASES["high"])

    # ── Select FICO phrase ──────────────────────────────────────────
    if fico >= 750:
        fico_phrase = random.choice(FICO_PHRASES["excellent"])
    elif fico >= 680:
        fico_phrase = random.choice(FICO_PHRASES["good"])
    elif fico >= 620:
        fico_phrase = random.choice(FICO_PHRASES["fair"])
    else:
        fico_phrase = random.choice(FICO_PHRASES["poor"])

    # ── Select purpose phrase ───────────────────────────────────────
    purpose_options = PURPOSE_PHRASES.get(purpose, PURPOSE_PHRASES["other"])
    purpose_phrase  = random.choice(purpose_options)

    # ── Select closing phrase ───────────────────────────────────────
    if risk == "low":
        closing = random.choice(CLOSING_PHRASES["positive"])
    elif risk == "high" or delinq > 1:
        closing = random.choice(CLOSING_PHRASES["urgent"])
    else:
        closing = random.choice(CLOSING_PHRASES["neutral"])

    # ── Assemble text ───────────────────────────────────────────────
    text = (
        f"I am applying for this loan to {purpose_phrase}. "
        f"{emp_phrase}, {inc_phrase}. "
        f"{dti_phrase}. "
        f"{fico_phrase}. "
        f"{closing}"
    )

    return text


def main():
    print(f"Loading data from {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)

    # Stratified sample
    df = df.reset_index(drop=True)
    default_0 = df[df["default"] == 0].sample(
        min((df["default"] == 0).sum(), N_SAMPLES // 2), random_state=RANDOM_STATE
    )
    default_1 = df[df["default"] == 1].sample(
        min((df["default"] == 1).sum(), N_SAMPLES // 2), random_state=RANDOM_STATE
    )
    df = pd.concat([default_0, default_1]).reset_index(drop=True)

    print(f"Generating text for {len(df):,} rows (offline mode) ...")
    print(f"Default rate in sample: {df['default'].mean():.1%}")

    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        texts.append(generate_text(row))

    df["borrower_text"] = texts

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved {len(df):,} rows → {OUT_PATH}")

    print("\nSample generated texts:")
    for _, row in df.sample(3, random_state=42).iterrows():
        print(f"\n  [default={row['default']}, fico={row['fico_score']:.0f}, dti={row['dti']:.1f}]")
        print(f"  {row['borrower_text']}")


if __name__ == "__main__":
    main()
