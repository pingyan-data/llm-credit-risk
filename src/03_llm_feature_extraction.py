"""
03_llm_feature_extraction.py
-----------------------------
Extract structured risk features from borrower_text using Claude.

For each text, we extract 8 numerical features as JSON.
These become the "LLM features" added to Model C (structured + LLM).

Output: data/processed/lc_with_llm_features.parquet
         adds columns: llm_employment_stability, llm_income_confidence, ...
"""

import os
import re
import json
import time
import anthropic
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field, field_validator

IN_PATH  = Path("data/processed/lc_with_text.parquet")
OUT_PATH = Path("data/processed/lc_with_llm_features.parquet")
CKPT_PATH = Path("data/processed/llm_features_checkpoint.jsonl")

MODEL       = "claude-haiku-4-5-20251001"
SLEEP_SECS  = 0.3

# ── Pydantic schema for LLM output ─────────────────────────────────

class LLMRiskFeatures(BaseModel):
    """Validated output from LLM feature extraction."""

    employment_stability: float = Field(
        ..., ge=0.0, le=1.0,
        description="How stable/secure the borrower's employment appears (0=unemployed/unstable, 1=very stable)"
    )
    income_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence that income is sufficient and reliable for repayment (0=very low, 1=very high)"
    )
    debt_stress_signal: float = Field(
        ..., ge=0.0, le=1.0,
        description="Degree of financial stress or debt burden expressed (0=none, 1=severe stress)"
    )
    repayment_intent_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Strength of expressed repayment intention and responsibility (0=weak/absent, 1=very strong)"
    )
    urgency_flag: float = Field(
        ..., ge=0.0, le=1.0,
        description="Urgency or desperation in the language (0=none, 1=high urgency/pleading)"
    )
    financial_literacy_signal: float = Field(
        ..., ge=0.0, le=1.0,
        description="Evidence of financial awareness (budget planning, rate awareness, etc.) (0=none, 1=strong)"
    )
    purpose_clarity: float = Field(
        ..., ge=0.0, le=1.0,
        description="How clear and specific is the stated loan purpose (0=vague, 1=very specific with plan)"
    )
    future_orientation: float = Field(
        ..., ge=0.0, le=1.0,
        description="Focus on future planning vs. short-term/past problems (0=short-term/past, 1=long-term plan)"
    )

    @field_validator("*", mode="before")
    @classmethod
    def clamp(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5   # safe fallback


LLM_FEATURE_COLS = [f"llm_{f}" for f in LLMRiskFeatures.model_fields.keys()]


# ── Extraction prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a credit risk analyst specialising in extracting behavioural signals from loan application text.

Your task: read a borrower's self-written explanation and score 8 features on a 0.0–1.0 scale.

Rules:
- Base scores ONLY on what is expressed in the text — do not infer from what is absent
- Be consistent across different phrasings of similar ideas
- Output ONLY a valid JSON object with exactly these 8 keys — no explanation, no markdown, no extra keys:
  employment_stability, income_confidence, debt_stress_signal, repayment_intent_score,
  urgency_flag, financial_literacy_signal, purpose_clarity, future_orientation"""

def build_extraction_prompt(text: str) -> str:
    return f"""Borrower application text:
\"\"\"{text}\"\"\"

Extract the 8 risk features as a JSON object. All values must be floats between 0.0 and 1.0."""


def extract_features(client: anthropic.Anthropic, text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0.0,   # deterministic for feature extraction
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_extraction_prompt(text)}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    parsed = json.loads(raw)
    validated = LLMRiskFeatures(**parsed)
    return validated.model_dump()


def fallback_features() -> dict:
    """Return 0.5 for all features if extraction fails."""
    return {f: 0.5 for f in LLMRiskFeatures.model_fields.keys()}


def load_checkpoint() -> dict:
    results = {}
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            for line in f:
                obj = json.loads(line)
                results[obj["idx"]] = obj["features"]
    return results


def save_checkpoint(idx, features: dict):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps({"idx": idx, "features": features}) + "\n")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY first.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)
    print(f"  {len(df):,} rows to process")

    checkpoint = load_checkpoint()
    print(f"  Resuming: {len(checkpoint):,} already done")

    all_features = {}
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx in checkpoint:
            all_features[idx] = checkpoint[idx]
            continue

        text = str(row.get("borrower_text", ""))
        if not text or len(text) < 20:
            all_features[idx] = fallback_features()
            continue

        try:
            feats = extract_features(client, text)
            all_features[idx] = feats
            save_checkpoint(idx, feats)
        except Exception as e:
            print(f"\n  Error at {idx}: {e}")
            all_features[idx] = fallback_features()
            errors += 1

        time.sleep(SLEEP_SECS)

    # Attach features to dataframe
    feat_df = pd.DataFrame.from_dict(all_features, orient="index")
    feat_df.columns = [f"llm_{c}" for c in feat_df.columns]
    feat_df.index = feat_df.index.astype(df.index.dtype)

    df = df.join(feat_df)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ Saved → {OUT_PATH}")
    print(f"  Errors: {errors} / {len(df)}")

    print("\nSample LLM features:")
    print(df[["default"] + LLM_FEATURE_COLS].head(5).round(3).to_string())

    # Quick signal check: do LLM features correlate with default?
    print("\nCorrelation with default label:")
    corr = df[LLM_FEATURE_COLS + ["default"]].corr()["default"].drop("default")
    print(corr.sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
