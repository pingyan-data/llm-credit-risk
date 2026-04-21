#!/usr/bin/env bash
# run_pipeline_offline.sh
# ------------------------
# Full pipeline — NO API calls required.
# Uses rule-based text generation and feature extraction.
#
# Run from project root:
#   bash run_pipeline_offline.sh

set -e

echo "======================================================"
echo "  LLM Credit Risk Pipeline — OFFLINE MODE"
echo "  (no API key needed)"
echo "======================================================"

echo ""
echo "Step 1: Data preparation ..."
python src/01_data_prep.py

echo ""
echo "Step 2: Generate borrower text (offline rules) ..."
python src/02_generate_borrower_text_offline.py

echo ""
echo "Step 3: Feature extraction (offline rules) ..."
python src/03_llm_feature_extraction_offline.py

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
echo "  Done. Results in outputs/"
echo "======================================================"
