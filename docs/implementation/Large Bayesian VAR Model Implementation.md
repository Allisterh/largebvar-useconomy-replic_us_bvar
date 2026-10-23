# Large Bayesian VAR Model Implementation

This project implements a comprehensive Vector Autoregression (VAR) modeling pipeline inspired by the methodology from Crump et al. (2024) "A Large Bayesian VAR of the U.S. Economy". The implementation provides a complete framework for macroeconomic analysis using FRED data.

## Features

- **Automated FRED Data Download**: Downloads 31 key macroeconomic variables from FRED API
- **VAR Model Estimation**: Implements Bayesian VAR with simplified Minnesota priors
- **Impulse Response Functions**: Computes IRFs with multiple identification schemes
- **Variance Decomposition**: Forecast error variance decomposition and historical decomposition
- **Comprehensive Diagnostics**: Residual tests, stability analysis, and model fit statistics
- **Professional Visualizations**: High-quality plots for all analyses
- **Complete Pipeline**: Single command execution of entire analysis workflow

## Project Structure

```
var_project/
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
├── .env                   # FRED API key configuration
├── README.md              # This documentation
├── config/
│   └── variables.py       # Variable definitions and FRED mappings
├── src/
│   ├── data_downloader.py # FRED data download functionality
│   ├── var_model.py       # VAR model estimation
│   ├── irf_analysis.py    # Impulse response functions
│   ├── variance_decomposition.py # Variance decomposition analysis
│   └── diagnostics.py     # Diagnostic tests and plots
├── data/                  # Downloaded data files
├── plots/                 # Generated plots and visualizations
└── output/                # Analysis results and reports
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

This will execute the entire VAR analysis pipeline:
1. Download FRED data (31 variables)
2. Estimate VAR model with optimal lag selection
3. Compute impulse response functions
4. Perform variance decomposition analysis
5. Run diagnostic tests
6. Generate forecasts

### Customization Options

```bash
# Specify custom parameters
python main.py --max-lags 4 --periods 16 --forecast-periods 8

# Use generalized identification
python main.py --identification generalized

# Force data re-download
python main.py --force-download

# Run specific steps only
python main.py --step 1  # Download data only
python main.py --step 2  # Estimate VAR only
```

### Command Line Options

- `--output-dir`: Output directory (default: 'output')
- `--data-dir`: Data directory (default: 'data')
- `--plots-dir`: Plots directory (default: 'plots')
- `--max-lags`: Maximum lags for VAR (default: 8)
- `--periods`: Periods for IRF/FEVD (default: 20)
- `--identification`: Identification scheme ('cholesky' or 'generalized')
- `--forecast-periods`: Forecast periods (default: 12)
- `--force-download`: Force data re-download
- `--step`: Run specific step only (1-6)

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
- Implements Vector Autoregression in levels
- Automatic lag order selection using information criteria
- Simplified Minnesota prior for Bayesian shrinkage
- Handles missing data through interpolation

### Identification Schemes
1. **Cholesky Decomposition**: Recursive identification with variable ordering
2. **Generalized IRFs**: Pesaran-Shin approach for order-invariant responses

### Analysis Components
- **Impulse Response Functions**: Dynamic responses to structural shocks
- **Forecast Error Variance Decomposition**: Contribution of shocks to forecast variance
- **Historical Decomposition**: Decomposition of historical movements
- **Diagnostic Tests**: Autocorrelation, normality, heteroskedasticity, stability

## Output Files

### Data Files
- `data/var_data.csv`: Downloaded and processed FRED data
- `data/var_data_summary.csv`: Data summary statistics

### Model Results
- `output/var_model_summary.txt`: VAR model estimation results
- `output/forecasts.csv`: Point forecasts
- `output/forecast_lower_bounds.csv`: Forecast confidence intervals (lower)
- `output/forecast_upper_bounds.csv`: Forecast confidence intervals (upper)
- `output/analysis_summary.txt`: Complete analysis summary

### Analysis Results
- `plots/irf_results_cholesky.csv`: IRF numerical results
- `plots/fevd_results_cholesky.csv`: Variance decomposition results
- `plots/historical_decomposition_cholesky.csv`: Historical decomposition
- `plots/diagnostic_report.csv`: Diagnostic test results

### Visualizations
- `plots/all_irfs_cholesky.png`: Complete IRF matrix
- `plots/irf_[VARIABLE]_cholesky.png`: Individual IRF plots
- `plots/fevd_all_cholesky.png`: All variance decompositions
- `plots/fevd_[VARIABLE]_cholesky.png`: Individual FEVD plots
- `plots/hist_decomp_[VARIABLE]_cholesky.png`: Historical decomposition plots
- `plots/residual_diagnostics.png`: Residual diagnostic plots
- `plots/fitted_vs_actual.png`: Model fit visualization

## Usage Examples

### Basic Analysis
```python
from src.data_downloader import download_var_data
from src.var_model import estimate_var_model
from src.irf_analysis import analyze_irfs

# Download data
data = download_var_data('data/var_data.csv')

# Estimate VAR
var_model = estimate_var_model(data, max_lags=4)

# Analyze IRFs
irf_analyzer = analyze_irfs(var_model.results, periods=20)
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
```

## Technical Notes

### Data Transformations
- Most variables enter in log-levels (×100)
- Interest rates and unemployment enter in levels
- Quarterly frequency (higher frequency data averaged)
- Missing data handled via linear interpolation

### Model Specifications
- VAR in levels without cointegration restrictions
- Minnesota prior with automatic hyperparameter selection
- Information criteria for lag order selection
- Bootstrap confidence intervals for IRFs

### Computational Considerations
- Large models (23+ variables) may require significant computation time
- Bootstrap procedures can be time-intensive
- Memory requirements scale with number of variables and lags

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

