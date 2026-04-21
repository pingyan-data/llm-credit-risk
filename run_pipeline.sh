#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# Full pipeline from raw data to outputs.
# Run from project root: bash run_pipeline.sh
#
# Prerequisites:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   pip install -r requirements.txt
#   Place lending_club.csv in data/raw/

set -e

echo "======================================================"
echo "  LLM Credit Risk Feature Extraction Pipeline"
echo "======================================================"

echo ""
echo "Step 1: Data preparation ..."
python src/01_data_prep.py

echo ""
echo "Step 2: Generate synthetic borrower text (API calls) ..."
python src/02_generate_borrower_text.py

echo ""
echo "Step 3: LLM feature extraction (API calls) ..."
python src/03_llm_feature_extraction.py

echo ""
echo "Step 4: Feature engineering ..."
python src/04_feature_engineering.py

echo ""
echo "Step 5: PD model training + calibration ..."
python src/05_pd_model.py

echo ""
echo "Step 6: Explainability + decision policy ..."
python src/06_explainability_policy.py

echo ""
echo "======================================================"
echo "  Pipeline complete. Results in outputs/"
echo "======================================================"
