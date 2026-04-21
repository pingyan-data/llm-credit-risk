"""
02_generate_borrower_text.py
-----------------------------
For each loan record, generate a realistic borrower explanation using Claude.

Design choices:
  - Prompt encodes the borrower profile (income, DTI, FICO, purpose, etc.)
  - Temperature=0.9 → natural variation across borrowers
  - Batch with rate-limit handling and resume from checkpoint
  - Output: data/processed/lc_with_text.parquet  (adds 'borrower_text' column)

Cost estimate:
  ~200 tokens per call (prompt+completion)
  At 5k rows: ~1M tokens → ~$0.60 with claude-haiku-4-5
  At 50k rows: ~10M tokens → ~$6 with claude-haiku-4-5
  Recommend starting with 2,000 rows for dev/testing.
"""

import os
import time
import json
import anthropic
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

IN_PATH  = Path("data/processed/lc_clean.parquet")
OUT_PATH = Path("data/processed/lc_with_text.parquet")
CKPT_PATH = Path("data/processed/text_checkpoint.jsonl")

# Tune these to control cost
N_SAMPLES   = 5_000   # rows to process (set to None for all)
BATCH_SIZE  = 20      # rows per batch before saving checkpoint
SLEEP_SECS  = 0.3     # pause between API calls (rate limit buffer)

MODEL = "claude-haiku-4-5-20251001"   # cheap + fast for bulk generation


PURPOSE_DESCRIPTIONS = {
    "debt_consolidation":  "consolidate existing debts into one manageable monthly payment",
    "credit_card":         "pay off high-interest credit card balances",
    "home_improvement":    "fund home renovation and improvement projects",
    "other":               "cover personal expenses",
    "major_purchase":      "finance a major purchase",
    "small_business":      "invest in their small business",
    "car":                 "purchase or repair a vehicle",
    "medical":             "cover unexpected medical expenses",
    "moving":              "cover relocation costs",
    "vacation":            "fund travel and vacation plans",
    "wedding":             "cover wedding expenses",
    "house":               "assist with housing-related costs",
    "educational":         "fund further education or training",
    "renewable_energy":    "invest in energy-efficient home improvements",
}


def build_prompt(row: pd.Series) -> str:
    fico   = int(row.get("fico_score", 680))
    income = int(row.get("annual_inc", 50000))
    dti    = round(float(row.get("dti", 15)), 1)
    loan   = int(row.get("loan_amnt", 10000))
    emp    = row.get("emp_length", 3)
    own    = row.get("home_ownership", "RENT").lower()
    purp   = str(row.get("purpose", "other"))
    delinq = int(row.get("delinq_2yrs", 0))
    pub    = int(row.get("pub_rec", 0))
    job    = str(row.get("emp_title", ""))

    purpose_desc = PURPOSE_DESCRIPTIONS.get(purp, "cover personal expenses")

    # Infer risk level to guide generation diversity
    if fico >= 740 and dti < 15 and delinq == 0:
        risk_hint = "This borrower has strong financials. Their explanation should sound confident, organised, and clear-headed."
    elif fico < 640 or dti > 30 or delinq > 1:
        risk_hint = "This borrower is under some financial pressure. Their explanation may show some urgency or stress, though not necessarily dishonest."
    else:
        risk_hint = "This borrower is typical, neither especially strong nor distressed. Their explanation should be straightforward."

    job_line = f"Job title: {job}. " if job and job not in ("nan", "") else ""

    prompt = f"""Generate a realistic loan application explanation (2–4 sentences) written in first person by a borrower applying for a personal loan on a P2P lending platform.

Borrower profile:
- Loan amount requested: ${loan:,}
- Purpose: {purpose_desc}
- Annual income: ${income:,}
- Debt-to-income ratio: {dti}%
- FICO credit score: {fico}
- Employment length: {emp} years
- Home ownership: {own}
- {job_line}Delinquencies in past 2 years: {delinq}
- Public records (bankruptcies, etc.): {pub}

{risk_hint}

Write only the borrower's explanation — no headers, no labels, no quotation marks. Sound human and natural, not corporate."""

    return prompt


def generate_text(client: anthropic.Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=0.9,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def load_checkpoint() -> dict:
    """Returns dict of {original_index: text} from checkpoint file."""
    results = {}
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            for line in f:
                obj = json.loads(line)
                results[obj["idx"]] = obj["text"]
    return results


def save_checkpoint(idx: int, text: str):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps({"idx": idx, "text": text}) + "\n")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable first.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading data from {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)

    if N_SAMPLES:
        df = df.reset_index(drop=True)
        default_0 = df[df["default"] == 0].sample(min((df["default"]==0).sum(), N_SAMPLES // 2), random_state=42)
        default_1 = df[df["default"] == 1].sample(min((df["default"]==1).sum(), N_SAMPLES // 2), random_state=42)
        df = pd.concat([default_0, default_1]).reset_index(drop=True)

    print(f"Processing {len(df):,} rows. Model: {MODEL}")
    print(f"Default rate in sample: {df['default'].mean():.1%}")

    # Resume from checkpoint
    checkpoint = load_checkpoint()
    print(f"Resuming: {len(checkpoint):,} rows already done.")

    texts = {}
    errors = 0

    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if idx in checkpoint:
            texts[idx] = checkpoint[idx]
            continue

        prompt = build_prompt(row)

        try:
            text = generate_text(client, prompt)
            texts[idx] = text
            save_checkpoint(idx, text)
        except Exception as e:
            print(f"\n  Error at idx {idx}: {e}")
            texts[idx] = ""
            errors += 1

        # Rate limit pause
        time.sleep(SLEEP_SECS)

        # Periodic save
        if (i + 1) % BATCH_SIZE == 0:
            print(f"  Batch saved at row {i+1}")

    df["borrower_text"] = df.index.map(texts)
    df = df[df["borrower_text"].notna() & (df["borrower_text"] != "")]

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved {len(df):,} rows with borrower_text → {OUT_PATH}")
    print(f"  Errors: {errors}")
    print("\nSample generated text:")
    print(df[["annual_inc", "dti", "fico_score", "default", "borrower_text"]].sample(3).to_string())


if __name__ == "__main__":
    main()
