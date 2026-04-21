# LLM Credit Risk Feature Extraction

A champion/challenger PD modelling experiment that quantifies whether LLM-extracted behavioural features from unstructured borrower text improve credit default prediction beyond structured financial variables alone.

**Stack:** Python · XGBoost · SHAP · Anthropic Claude · scikit-learn · MLflow-compatible

---

## Motivation

Traditional credit scorecards rely entirely on structured variables — FICO scores, DTI ratios, delinquency counts. But consumer loan applications contain rich unstructured signals that these features cannot capture: the urgency in a borrower's language, their financial literacy, the clarity of their stated purpose, their orientation toward future planning versus short-term pressure.

This project asks: **can an LLM extract predictive signals from application text that are genuinely incremental to structured features?**

---

## Pipeline

```
LendingClub structured data (2007–2018)
         │
         ▼
  01_data_prep.py            Clean, label, remove post-origination leakage
         │
         ▼
  02_generate_borrower_text  Generate realistic application narratives
         │                   per borrower profile (Claude API / offline rules)
         ▼
  03_llm_feature_extraction  Extract 8 structured risk signals from text
         │                   (Claude API with Pydantic schema validation)
         ▼
  04_feature_engineering     Build Model A and Model C feature matrices
         │
         ▼
  05_pd_model.py             XGBoost PD models, 5-fold CV,
         │                   isotonic calibration, champion/challenger table
         ▼
  06_explainability_policy   SHAP attribution, calibration curves,
                             decision policy (Approve / Review / Decline)
         │
         ▼
  07_ablation_study          Disentangle data volume effect vs LLM feature effect
```

---

## Models compared

| Model | Features | N | Purpose |
|-------|----------|---|---------|
| **Model A** | Structured only (FICO, DTI, income, etc.) | 5,000 | Baseline champion |
| **Model B** | Structured only | 10,000 | Volume control |
| **Model C** | Structured + 8 LLM-extracted features | 5,000 | LLM challenger |

---

## LLM-extracted features

For each borrower's application text, Claude extracts 8 behavioural signals (0.0–1.0 scale):

| Feature | What it captures |
|---------|-----------------|
| `llm_employment_stability` | Stability and tenure of employment as expressed in text |
| `llm_income_confidence` | Reliability of income for repayment |
| `llm_debt_stress_signal` | Expressed financial stress or burden |
| `llm_repayment_intent_score` | Strength of repayment commitment |
| `llm_urgency_flag` | Urgency or desperation in language |
| `llm_financial_literacy_signal` | Evidence of budgeting or financial awareness |
| `llm_purpose_clarity` | Specificity of stated loan purpose |
| `llm_future_orientation` | Focus on future planning vs. short-term pressure |

All features are validated against a Pydantic schema before entering the feature matrix.

---

## Key findings

### Ablation study — data volume vs LLM feature value

| Model | AUC | KS | Gini | Brier |
|-------|-----|----|------|-------|
| A — Structured (N=5k) | 0.6859 | 0.2636 | 0.3718 | 0.2223 |
| B — Structured (N=10k) | **0.6934** | **0.2808** | **0.3869** | **0.2208** |
| C — Struct + LLM (N=5k) | 0.6859 | 0.2736 | 0.3719 | 0.2225 |

**Interpretation:**

- A → B (+0.0075 AUC): doubling data volume produces a real improvement
- A → C (+0.000 AUC): offline rule-derived LLM features add no incremental value
- B > C on all metrics: more data outperforms offline LLM features

This is the expected result when LLM features are derived from the same structured source data using rules — there is no new information. It confirms that **genuine LLM value requires real text analysis**, not rules operating on structured fields.

### SHAP attribution — Model C

LLM features account for **28% of total SHAP importance** in Model C, with `llm_urgency_flag` ranking 3rd overall (mean |SHAP| = 0.189) — ahead of most structured features. This demonstrates that the model learns to use these features when they carry signal.

### Calibration

Both raw XGBoost models show systematic underestimation at high PD scores. Isotonic regression calibration corrects this to near-perfect alignment (predicted PD ≈ observed default rate across all deciles).

### Decision policy

| | Model A | Model C |
|---|---------|---------|
| Approval rate | 2.3% | 1.1% |
| Review rate | 2.0% | 3.5% |
| Decline rate | 95.7% | 95.4% |
| Approved default rate | **4.3%** | **0.0%** |

Model C applies a stricter threshold, routing more marginal cases to review rather than approval. Among loans it does approve, the observed default rate is 0% vs 4.3% for Model A.

---

## Roadmap

- [ ] Replace offline rules with real Claude API calls for text generation and extraction
- [ ] Re-run ablation study with genuine LLM features to measure true incremental AUC
- [ ] Add MLflow experiment tracking across model runs
- [ ] Build Streamlit dashboard for interactive champion/challenger exploration
- [ ] Add fairness audit (demographic parity, equalised odds across `addr_state`)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/pingyan-data/llm-credit-risk.git
cd llm-credit-risk
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# → data/raw/lending_club.csv

# 3a. Run offline (no API key needed)
bash run_pipeline_offline.sh

# 3b. Run with real LLM (requires Anthropic API key)
export ANTHROPIC_API_KEY=sk-ant-...
bash run_pipeline.sh
```

**Cost estimate for real LLM run (5,000 rows, claude-haiku):** ~$3–4

---

## Project structure

```
llm-credit-risk/
├── src/
│   ├── 01_data_prep.py                      Data loading and cleaning
│   ├── 02_generate_borrower_text.py         LLM text generation (API)
│   ├── 02_generate_borrower_text_offline.py LLM text generation (rules)
│   ├── 03_llm_feature_extraction.py         LLM feature extraction (API)
│   ├── 03_llm_feature_extraction_offline.py LLM feature extraction (rules)
│   ├── 04_feature_engineering.py            Feature matrix construction
│   ├── 05_pd_model.py                       XGBoost training + calibration
│   ├── 06_explainability_policy.py          SHAP + decision policy
│   └── 07_ablation_study.py                 Data volume vs LLM ablation
├── data/
│   ├── raw/                                 (gitignored — download from Kaggle)
│   └── processed/                           (gitignored — generated by pipeline)
├── outputs/                                 (gitignored — generated by pipeline)
├── notebooks/                               (in progress)
├── requirements.txt
├── run_pipeline.sh                          Full pipeline (API)
└── run_pipeline_offline.sh                  Full pipeline (offline)
```

---

## References

- Sanz-Guerrero & Arroyo (2024). *Credit Risk Meets Large Language Models.* arXiv:2401.16458
- Netzer, Lemaire & Herzenstein (2019). *When Words Sweat.* Journal of Marketing Research.
- Yu, Bai & Chen (2025). *GPT-LGBM: ChatGPT-based framework for credit scoring.*
