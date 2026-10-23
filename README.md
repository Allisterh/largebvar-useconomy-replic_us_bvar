# Large Bayesian VAR Model Implementation

This project implements a comprehensive Vector Autoregression (VAR) modeling pipeline inspired by the methodology from Crump et al. (2024) "A Large Bayesian VAR of the U.S. Economy". The implementation provides a complete framework for macroeconomic analysis using FRED data, with both a practical default profile and a paper‑faithful alternative profile.

## Features

- **Automated FRED Data Download**: Downloads 31 key macroeconomic variables from FRED API
- **VAR Model Estimation**: Bayesian VAR with Minnesota + sum‑of‑coefficients priors (Gibbs sampler)
- **Impulse Response Functions**: Computes IRFs with multiple identification schemes
- **Variance Decomposition**: Forecast error variance decomposition and historical decomposition
- **Coefficient Analysis**: Detailed statistical inference for all VAR coefficients with t-statistics, p-values, and confidence intervals
- **Comprehensive Diagnostics**: Residual tests, stability analysis, and model fit statistics
- **Professional Visualizations**: High-quality plots for all analyses including coefficient heatmaps
- **Posterior Uncertainty Utilities**: Posterior IRF/FEVD summaries (medians + 68/90% bands) available via API
- **Conditional Forecasts (Scenarios)**: Step 8 supports constraint‑based scenarios (e.g., financial turmoil, TCJA proxy)
- **Entropic Tilting (Anchoring)**: Step 9 reweights posterior predictive draws to match external targets
- **Complete Pipeline**: Single command execution of entire analysis workflow

## Project Structure

```
largebvar-useconomy/
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies  
├── CLAUDE.md              # Claude Code instructions
├── README.md              # This file
│
├── src/                   # Core source code
│   ├── data_downloader.py # FRED data download and processing
│   ├── var_model.py       # VAR model estimation
│   ├── irf_analysis.py    # Impulse response functions
│   ├── variance_decomposition.py # FEVD and historical decomposition
│   ├── coefficient_analysis.py   # Coefficient inference
│   ├── diagnostics.py     # Model diagnostics
│   ├── bayesian/         # Bayesian VAR implementation
│   │   ├── core_bayesian_var.py # Main Bayesian VAR class
│   │   ├── priors/       # Prior specifications
│   │   └── estimation/   # Gibbs sampling and conjugate posteriors
│   ├── computational/    # Phase 2 computational optimizations
│   │   ├── kronecker_ops.py     # Kronecker operations
│   │   ├── matrix_normal.py     # Matrix normal distributions
│   │   ├── sparse_methods.py    # Sparse VAR methods
│   │   └── memory_management.py # Memory optimization
│   └── dimension_reduction/     # Dimension reduction methods
│       ├── factor_models.py     # Dynamic factor models
│       ├── favar.py            # Factor-augmented VAR
│       └── variable_selection.py # Variable selection
│
├── config/                # Configuration files
│   ├── variables.py                  # Default variable definitions (includes DFF; growth‑rate transforms)
│   ├── variables_alternative.py      # Paper‑aligned variables (omits DFF; log‑level transforms)
│   ├── variables_deposit_study.py    # Deposit‑focused profile (adds deposits, FCI, EPU)
│   ├── cholesky_ordering.py          # Identification schemes and standard economic ordering
│   └── plot_utils.py                 # Plotting utilities
│
├── data/                  # Data storage (gitignored)
│   └── README.md         # Data directory documentation
│
├── output/               # Model results (gitignored)
│   └── README.md         # Output directory documentation
│
├── plots/                # Generated visualizations (gitignored)
│   └── README.md         # Plots directory documentation
│
├── docs/                 # Documentation
│   ├── research/         # Research papers and analysis
│   ├── implementation/   # Implementation guides
│   └── README.md         # Documentation index
│
├── scripts/              # Utility and test scripts
│   ├── test_*.py         # Test scripts
│   ├── example_*.py      # Example scripts
│   ├── replicate_paper.py # Paper replication script
│   └── README.md         # Scripts documentation
│
├── validation/           # Validation and testing
│   ├── validate_phase2_implementation.py
│   └── README.md         # Validation documentation
│
├── tools/                # Development tools
│   └── README.md         # Tools documentation
│
└── tests/                # Unit and integration tests
    ├── unit/             # Unit tests
    └── integration/      # Integration tests
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up FRED API key**:
   Create a local `.env` file for the Python pipeline (already ignored by Git):
   ```
   FRED_API_KEY=your_fred_api_key_here
   ```

   Never commit a real API key. For the R scripts, set `FRED_API_KEY` in
   your shell or an untracked `.Renviron` file before starting R.
   R reads `.Renviron` at startup; the R scripts do not load Python's `.env` file.

## Quick Start

### Run Complete Analysis Pipeline

```bash
python main.py
```

This runs the complete pipeline:
1. Download FRED data (31 variables; 1973:Q1–2019:Q1)
2. Estimate Bayesian VAR (lag selected; Gibbs sampler)
3. Compute IRFs (Cholesky or GIRF). Posterior bands are available via the Bayesian API
4. Perform FEVD and historical decomposition
5. Run diagnostic tests
6. Coefficient analysis (tables + plots)
7. Generate forecasts

### Customization Options

```bash
# Specify custom parameters
python main.py --max-lags 4 --periods 16 --forecast-periods 8

# Identification: generalized IRFs (order-invariant)
python main.py --identification generalized --periods 20

# Force data re-download
python main.py --force-download

# Run specific steps only (1–9)
python main.py --step 1                       # Download data only
python main.py --step 2 --gibbs-draws 1000    # Estimate BVAR (posterior draws)
python main.py --step 3 --periods 20          # IRFs (posterior bands if available)
python main.py --step 4 --periods 20          # FEVD + historical decomposition
python main.py --step 5                       # Diagnostics
python main.py --step 6                       # Coefficient analysis
python main.py --step 7 --forecast-periods 12 # Forecasts
python main.py --step 8 --constraints examples/scenarios/financial_turmoil_canonical.json --scenario-horizon 12
python main.py --step 9 --tilt-targets path/to/tilt_targets.json

# Dev run: small model, fewer draws, fewer lags (fast sanity test)
python main.py --dev-mode --max-lags 2 --gibbs-draws 200 --periods 8 --forecast-periods 4
```

### Command Line Options

- `--output-dir`: Output directory (default: 'output')
- `--data-dir`: Data directory (default: 'data')
- `--plots-dir`: Plots directory (default: 'plots')
- `--max-lags`: Maximum lags for VAR (default: 8)
- `--gibbs-draws`: Number of Gibbs sampler draws (default: 20000; see Fast Mode note)
- `--periods`: Periods for IRF/FEVD (default: 12)
- `--identification`: Identification scheme ('cholesky' or 'generalized')
- `--forecast-periods`: Forecast periods (default: 12)
- `--force-download`: Force data re-download
- `--step`: Run specific step only (1–9)
- `--scenario`: Scenario mode (currently 'custom' supported)
- `--constraints`: Path to JSON constraints for Step 8 (horizon→{var: value})
- `--scenario-horizon`: Forecast horizon for scenarios (default: 12)
- `--tilt-targets`: Path to JSON mean targets for Step 9 (entropic tilting)
 - `--variables-profile`: Choose variables set: `default`, `paper`, or `deposit`

See `examples/scenarios/README.md` for scenario details and examples.

### Variable Profiles (CLI and Python)

- `--variables-profile {default|paper|deposit}` selects the variables set:
  - **default** → `config/variables.py` (includes `DFF`; many `log_diff_100` transforms)
  - **paper** → `config/variables_alternative.py` (omits `DFF`; mostly `log_level_100`)
  - **deposit** → `config/variables_deposit_study.py` (adds deposits, FCI/uncertainty indices)

CLI examples:
```bash
# Paper baseline (Crump-style transforms by default)
python main.py --variables-profile paper --step 1

# Deposit study profile, estimate and IRFs
python main.py --variables-profile deposit --step 3 --max-lags 5 --periods 20

# Default profile, full run with caching
python main.py --variables-profile default

# Dev mode with deposit profile
python main.py --variables-profile deposit --dev-mode --max-lags 2 --gibbs-draws 200
```

Python examples:
```python
from main import VARPipeline
from config.variables_deposit_study import FRED_VARIABLES as DEPOSIT_VARS

pipe = VARPipeline()
pipe.run_complete_analysis(fred_variables=DEPOSIT_VARS)
```

Deposit additions include: total deposits (all banks), checkable and demand deposits, small‑denomination time deposits, small/large bank deposit splits; financial conditions (`KCFSI`, `ANFCI`), and policy uncertainty (`USEPUINDXM`).

## Variables Included

The model includes 31 key U.S. macroeconomic variables:

### Real Activity
- Real GDP and components (consumption, investment, trade, government)
- Employment and unemployment
- Industrial production and capacity utilization
- Housing starts

### Prices
- GDP deflator, PCE price indices
- CPI (headline and core)
- Compensation per hour

### Financial Variables
- Treasury yields (2, 5, 10 year)
- Corporate bond yields (Aaa, Baa)
- Stock prices (S&P 500)
- Exchange rates
- Commodity prices
- VIX volatility index

## Methodology

### VAR Model
- Bayesian Vector Autoregression with shrinkage
- Supports both paper (log‑level) and default (growth‑rate) transform profiles
- Automatic lag order selection using information criteria
- Handles missing data through interpolation

### Identification Schemes
1. **Cholesky Decomposition**: Recursive identification with industry-standard economic ordering
   - Uses theoretically-motivated variable ordering (Real Activity → Labor → Prices → Financial)
   - Automatic application of standard macroeconomic ordering conventions
2. **Generalized IRFs**: Pesaran-Shin approach for order-invariant responses

### Posterior Uncertainty (IRFs/FEVD)

The Bayesian backend exposes posterior summaries even if the default plotting uses point IRFs. You can obtain medians and 68/90% credible bands programmatically:

```python
# IRF summary
summary, intervals = var_model.results.get_irf_summary(horizon=20, identification='cholesky',
                                                       interval_levels=[0.68, 0.90])
# FEVD summary
fevd_summary, fevd_intervals = var_model.results.get_fevd_summary(horizon=20, identification='cholesky')
```

Variance decomposition in the pipeline already prefers posterior medians when available.

### Analysis Components
- **Impulse Response Functions**: Dynamic responses to structural shocks
- **Forecast Error Variance Decomposition**: Contribution of shocks to forecast variance
- **Historical Decomposition**: Decomposition of historical movements
- **Coefficient Analysis**: Statistical inference for all VAR coefficients including t-statistics, p-values, and confidence intervals
- **Diagnostic Tests**: Autocorrelation, normality, heteroskedasticity, stability

## Output Files

### Data Files
- `data/var_data_{profile}.csv` or `data/var_data_{profile}_crump2024.csv`: Downloaded and processed FRED data
  - Examples: `var_data_default_crump2024.csv`, `var_data_paper_crump2024.csv`, `var_data_deposit_crump2024.csv`
- `data/var_data_{profile}[_crump2024]_summary.csv`: Data summary statistics (auto‑generated)
- `data/var_data_{profile}[_crump2024]_stationarity.csv`: Stationarity diagnostics (ADF)

### Model Results
- `output/var_model_summary.txt`: VAR model estimation results
- `output/var_coefficients.xlsx`: Comprehensive coefficient tables with statistical inference
- `output/var_coefficients_all.csv`: Combined coefficient table (CSV format)
- `output/coefficients_[VARIABLE].csv`: Individual equation coefficient tables
- `output/forecasts.csv`: Point forecasts
- `output/forecast_lower_bounds.csv`: Forecast confidence intervals (lower)
- `output/forecast_upper_bounds.csv`: Forecast confidence intervals (upper)
- `output/analysis_summary.txt`: Complete analysis summary

### Analysis Results
- `plots/irf_results_[ident].csv`: IRF numerical results (point IRFs by default)
- `plots/fevd_results_[ident].csv`: Variance decomposition results
- `plots/historical_decomposition_[ident].csv`: Historical decomposition
- `plots/diagnostic_report.csv`: Diagnostic test results

### Visualizations
- `plots/all_irfs_[ident].png`: Complete IRF matrix
- `plots/irf_[VARIABLE]_[ident].png`: Individual IRF plots
- `plots/fevd_all_cholesky.png`: All variance decompositions
- `plots/fevd_[VARIABLE]_cholesky.png`: Individual FEVD plots
- `plots/hist_decomp_[VARIABLE]_cholesky.png`: Historical decomposition plots
- `plots/residual_diagnostics.png`: Residual diagnostic plots
- `plots/fitted_vs_actual.png`: Model fit visualization
- `output/coefficient_heatmap.png`: Coefficient magnitude and significance heatmaps
- `output/coefficient_significance.png`: Coefficient significance analysis plots

## Usage Examples

Data is cached per selected `--variables-profile`. Reruns reuse existing CSVs unless `--force-download` is provided.

### CLI Examples
```bash
# 1) End-to-end with 2 lags and 1000 draws
python main.py --max-lags 2 --gibbs-draws 1000 --periods 20 --forecast-periods 12

# 2) Posterior GIRFs only
python main.py --step 3 --identification generalized --periods 20

# 3) FEVD + historical decomposition (posterior medians)
python main.py --step 4 --periods 20

# 4) Conditional-forecast scenario (financial turmoil canonical)
python main.py --step 8 --constraints examples/scenarios/financial_turmoil_canonical.json --scenario-horizon 12

# 5) Entropic tilting to anchor long-run core PCE mean at h=12
python main.py --step 9 --tilt-targets examples/tilt_targets_core_pce_h12.json

# 6) Dev-mode fast run
python main.py --dev-mode --max-lags 2 --gibbs-draws 200 --periods 8 --forecast-periods 4

# 7) Use the paper baseline variables
python main.py --variables-profile paper --step 1

# 8) Deposit study profile (adds deposits/FCI/EPU)
python main.py --variables-profile deposit --step 2 --max-lags 5 --gibbs-draws 1000
```

### Python Examples
```python
from src.data_downloader import download_var_data
from src.var_model import estimate_var_model
from src.irf_analysis import analyze_irfs
from src.coefficient_analysis import analyze_var_coefficients

# Download data
data = download_var_data('data/var_data.csv')

# Estimate VAR
var_model = estimate_var_model(data, max_lags=4)

# Analyze IRFs
irf_analyzer = analyze_irfs(var_model.results, periods=20)

# Analyze coefficients
coeff_analyzer = analyze_var_coefficients(var_model.results)
```

### Coefficient Analysis Examples
```python
from src.coefficient_analysis import VARCoefficientAnalysis

# Initialize coefficient analyzer
coeff_analyzer = VARCoefficientAnalysis(var_model.results)

# Create detailed coefficient tables
coeff_analyzer.create_coefficient_tables()

# Generate summary statistics
summary = coeff_analyzer.generate_summary_statistics()
print(f"Significant coefficients: {summary['significant_at_5pct']}/{summary['total_coefficients']}")

# Export tables to Excel
coeff_analyzer.export_tables(output_dir='output', format='excel')

# Create visualization plots
fig1 = coeff_analyzer.plot_coefficient_heatmap()
fig2 = coeff_analyzer.plot_coefficient_significance()
```

### Custom Analysis
```python
# Use specific date range
var_model = estimate_var_model(data, start_date='2000-01-01', end_date='2020-12-31')

# Different identification
irf_analyzer = analyze_irfs(var_model.results, identification='generalized')

# Custom variable ordering for Cholesky
ordering = ['GDPC1', 'CPIAUCSL', 'UNRATE', 'DGS10']
irf_analyzer = analyze_irfs(var_model.results, ordering=ordering)

# Custom confidence intervals for coefficients
coeff_analyzer = VARCoefficientAnalysis(var_model.results)
coeff_analyzer._compute_confidence_intervals(alpha=0.01)  # 99% CI
```

## Coefficient Analysis Features

The coefficient analysis module provides comprehensive statistical inference for VAR model parameters:

### Statistical Inference
- **Coefficient Estimates**: All parameter values from the VAR estimation
- **Standard Errors**: Computed from the coefficient covariance matrix
- **t-Statistics**: Test statistics for coefficient significance
- **P-Values**: Two-tailed significance tests
- **Confidence Intervals**: Customizable confidence levels (default 95%)

### Output Tables
- **Individual Equations**: Separate tables for each dependent variable
- **Combined Tables**: All coefficients in a single comprehensive table
- **Excel Export**: Multi-sheet workbooks with formatted tables
- **CSV Export**: Individual and combined CSV files

### Visualizations
- **Coefficient Heatmaps**: Visual representation of coefficient magnitudes and significance
- **Significance Analysis**: Distribution of t-statistics and significance by lag order
- **Cross-Variable Effects**: Analysis of inter-variable relationships

### Summary Statistics
- Overall significance rates (5% and 1% levels)
- Equation-by-equation significance analysis
- Lag-order significance patterns
- Most significant coefficient identification

### Usage in Pipeline
The coefficient analysis is integrated as Step 6 in the complete pipeline:
```bash
# Run complete analysis (includes coefficient analysis)
python main.py

# Run only coefficient analysis step
python main.py --step 6

# Run standalone coefficient analysis example
python example_coefficient_analysis.py
```

## Technical Notes

### Data Transformations
- Supported profiles:
  - Default (`config/variables.py`): many series in growth rates (`log_diff_100`), rates/unemployment in levels
  - Paper (`config/variables_alternative.py`): most series in log‑levels ×100 (`log_level_100`); rates/unemployment in levels; `DFF` omitted
  - Deposit study (`config/variables_deposit_study.py`): paper‑aligned macro block plus deposit aggregates and FCI/EPU; deposits default to `log_level_100` (can be flipped to `log_diff_100` if desired)
- Seasonal adjustment is applied at native frequency (X‑13/STL), then aggregated to quarterly
- Sample window: 1973:Q1–2019:Q1 (configurable)
- Transform table (machine‑readable): `config/transform_table.csv`
- Missing data policy: logs per‑variable missingness; drops variables with ≥50% missing and rows with NaNs post‑processing

### Model Specifications
- VAR in levels (paper profile) or growth‑rate transforms (default profile); no explicit cointegration restrictions
- Bayesian estimation via Gibbs sampler (Minnesota + sum‑of‑coefficients dummy priors)
- Automatic lag order selection using information criteria
- Posterior IRFs/FEVD summaries available via Bayesian API; default IRF plots are point paths

### Computational Considerations
- Large models (23+ variables) may require significant computation time
- IRF bootstrap (if used) is time‑intensive; posterior summaries are preferred
- Memory requirements scale with number of variables and lags
- Fast Mode note: for datasets with >20 variables, the estimator internally caps draws to ~500 to keep runtimes manageable. Use smaller systems/dev‑mode for iteration, and increase draws (and disable fast caps in code) for final paper‑style runs.

## Limitations and Extensions

### Current Limitations
- Simplified Minnesota prior implementation
- No time-varying parameters
- No stochastic volatility
- Limited to quarterly frequency

### Potential Extensions
- Full Bayesian estimation with MCMC
- Time-varying parameter models
- Mixed-frequency data handling
- Structural identification schemes
- Sign restrictions
- Factor-augmented VARs

## References

1. Crump, R. K., Eusepi, S., Giannone, D., Qian, E., & Sbordone, A. (2024). "A Large Bayesian VAR of the U.S. Economy." *International Journal of Central Banking* (Forthcoming).

2. Litterman, R. B. (1980). "A Bayesian Procedure for Forecasting with Vector Autoregressions." Working Paper, Federal Reserve Bank of Minneapolis.

3. Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48.

4. Pesaran, M. H., & Shin, Y. (1998). "Generalized Impulse Response Analysis in Linear Multivariate Models." *Economics Letters*, 58(1), 17-29.

## Support

For questions or issues:
1. Check the log files (`var_analysis.log`) for detailed execution information
2. Review the diagnostic reports for model validation
3. Examine the data summary files for data quality issues

## License

This implementation is provided for educational and research purposes. Please cite the original Crump et al. (2024) paper when using this methodology.
