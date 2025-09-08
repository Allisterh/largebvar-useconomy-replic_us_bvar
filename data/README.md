# Data Directory

This directory contains datasets and data processing files for the Large Bayesian VAR model.

## Structure:
- Raw FRED data downloads (CSV format)
- Processed data with Crump et al. (2024) transformations
- Data summaries and metadata

## Files:
- `var_data_crump2024*.csv` - Main dataset with transformations
- `*_summary.csv` - Statistical summaries of each dataset

## Data Sources:
All data is downloaded from the Federal Reserve Economic Data (FRED) database using the FRED API.

## Note:
Large data files are excluded from git (see .gitignore). Run the data download step to regenerate:
```bash
python main.py --step 1
```