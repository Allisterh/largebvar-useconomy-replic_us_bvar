# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Vector Autoregression (VAR) modeling pipeline implementing the methodology from Crump et al. (2024) "A Large Bayesian VAR of the U.S. Economy". The codebase provides end-to-end macroeconomic analysis using FRED data with 31 key U.S. economic variables.

## Core Commands

### Run Complete Analysis Pipeline
```bash
python main.py
```
This executes all 7 steps: data download, VAR estimation, IRF analysis, variance decomposition, diagnostics, coefficient analysis, and forecasting.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Individual Steps
```bash
python main.py --step 1   # Download FRED data
python main.py --step 2   # Estimate VAR model
python main.py --step 3   # Compute impulse response functions
python main.py --step 4   # Variance decomposition analysis
python main.py --step 5   # Diagnostic tests
python main.py --step 6   # Coefficient analysis
python main.py --step 7   # Generate forecasts
```

### Common Command Options
```bash
python main.py --max-lags 4 --periods 16 --identification cholesky
python main.py --force-download --legacy-transforms
python main.py --output-dir results --plots-dir charts
```

### Testing
```bash
python tests/test_transformations.py
python test_data_loading.py
python validate_implementation.py
```

## Architecture Overview

### Pipeline Structure (main.py:49-504)
The `VARPipeline` class orchestrates the complete analysis workflow through 7 sequential steps. Each step can be run independently or as part of the complete pipeline via `run_complete_analysis()`.

### Data Layer (src/data_downloader.py)
- **FREDDataDownloader**: Handles FRED API interactions with automatic retry logic
- **Crump Transformations**: Implements exact Crump et al. (2024) data transformations
- **Seasonal Adjustment**: Custom X-13ARIMA-SEATS implementation for quarterly conversion
- **Frequency Conversion**: Smart handling of daily/monthly/quarterly data alignment

### Model Layer (src/var_model.py)
- **BayesianVAR**: Core VAR estimation with Minnesota priors
- **Model Selection**: Information criteria-based lag selection (AIC/BIC/HQ)
- **Forecasting**: Multi-step ahead forecasts with confidence intervals

### Analysis Modules
- **IRF Analysis** (src/irf_analysis.py): Cholesky and generalized identification schemes
- **Variance Decomposition** (src/variance_decomposition.py): FEVD and historical decomposition
- **Coefficient Analysis** (src/coefficient_analysis.py): Statistical inference with t-tests, p-values, confidence intervals
- **Diagnostics** (src/diagnostics.py): Residual tests, stability analysis, normality tests

### Configuration System
- **Variables** (config/variables.py): FRED series mappings with transformation specifications
- **Cholesky Ordering** (config/cholesky_ordering.py): Economic theory-based variable ordering for structural identification
- **Plot Utils** (config/plot_utils.py): Standardized visualization formatting

## Key Implementation Details

### Data Transformations
The pipeline implements exact Crump et al. (2024) specifications:
- **log_100**: log(x) * 100 for most real and price variables
- **raw**: levels for interest rates, unemployment, capacity utilization
- **Seasonal Adjustment**: X-13ARIMA-SEATS for proper quarterly conversion
- **Missing Data**: Linear interpolation with validation

### Identification Strategy
Uses theoretically-motivated Cholesky ordering (config/cholesky_ordering.py:13-66):
1. Real Activity (slow-moving): GDP, consumption, investment, industrial production
2. Labor Market (medium-slow): employment, unemployment, wages  
3. Income/Sentiment: disposable income, consumer sentiment
4. Prices (medium): GDP deflator, core PCE, CPI measures
5. Financial (fast-moving): fed funds rate, treasury yields, corporate spreads
6. Asset Prices (fastest): exchange rates, equity indices, commodities, volatility

### Output Structure
- **data/**: Raw and processed FRED data with summaries
- **output/**: Model results, coefficient tables, forecasts, analysis summaries
- **plots/**: All visualizations including IRF matrices, FEVD charts, diagnostic plots

### Error Handling & Logging
- Comprehensive logging to `var_analysis.log`
- Automatic backup creation before data updates
- Graceful handling of missing FRED series with alternative mappings
- Validation of data transformations and model estimation

## Development Patterns

### Adding New Variables
1. Update `config/variables.py` with FRED series ID and transformation
2. Add to appropriate frequency list (QUARTERLY_VARIABLES, MONTHLY_VARIABLES, DAILY_VARIABLES)
3. Update Cholesky ordering in `config/cholesky_ordering.py` based on economic theory
4. Test with `python test_data_loading.py`

### Extending Analysis
- Analysis modules follow common interface: accept VAR results, return structured output
- Plot functions use standardized formatting from `config/plot_utils.py`
- All modules support both save_plots and return results for programmatic use

### Configuration Management
- Environment variables in `.env` (FRED API key)
- Data specifications centralized in `config/variables.py`
- Economic assumptions explicit in `config/cholesky_ordering.py`