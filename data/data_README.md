# Data

This directory is intentionally excluded from version control.

## How to get the data

1. Download Lending Club loan data from Kaggle:
   https://www.kaggle.com/datasets/wordsforthewise/lending-club

2. Place the file at:
   ```
   data/raw/lending_club.csv
   ```

3. Run the pipeline to generate processed files:
   ```bash
   python src/01_data_prep.py
   ```

Processed files in `data/processed/` are generated automatically by the pipeline.
