# Project Overview

This project provides a complete pipeline for conducting Vector Autoregression (VAR) analysis on U.S. macroeconomic data. It implements the methodology described in "A Large Bayesian VAR of the U.S. Economy" by Crump et al. (2024).

The core functionalities include:
- Downloading and transforming macroeconomic data from the FRED API.
- Estimating a Bayesian VAR model with automatic lag selection.
- Performing Impulse Response Function (IRF) analysis.
- Conducting Forecast Error Variance Decomposition (FEVD).
- Generating detailed coefficient analysis with statistical inference.
- Running a suite of diagnostic tests to validate the model.

The project is structured as a Python application with a clear separation of concerns, including data handling, modeling, analysis, and configuration.

## Key Technologies

- **Language:** Python 3
- **Core Libraries:** `statsmodels`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `fredapi`

# Building and Running

## Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up FRED API Key:**
    Create a `.env` file in the root directory and add your FRED API key:
    ```
    FRED_API_KEY=your_api_key_here
    ```

## Running the Analysis

The main entry point for the project is `main.py`.

-   **Run the complete pipeline:**
    This command will execute all steps: data download, model estimation, IRF analysis, variance decomposition, diagnostics, coefficient analysis, and forecasting.
    ```bash
    python main.py
    ```

-   **Run a specific step:**
    You can run a single step of the pipeline using the `--step` argument. For example, to only download the data:
    ```bash
    python main.py --step 1
    ```

-   **Customizing the analysis:**
    Several command-line arguments are available to customize the analysis:
    ```bash
    # Use different VAR parameters
    python main.py --max-lags 4 --periods 16

    # Use generalized impulse responses
    python main.py --identification generalized

    # Force a re-download of the data
    python main.py --force-download
    ```

## Running Examples

The `example_usage.py` script demonstrates how to use the different modules of the project programmatically.
```bash
python example_usage.py
```

# Development Conventions

## Code Structure

The project follows a standard Python project structure:
-   `main.py`: Main executable script for the pipeline.
-   `var_pipeline.py`: An alternative pipeline implementation.
-   `src/`: Contains the core logic for the application, with modules for each major functionality (e.g., `data_downloader.py`, `var_model.py`, `irf_analysis.py`).
-   `config/`: Configuration files, such as variable definitions (`variables.py`) and Cholesky ordering.
-   `data/`: Stores the downloaded and processed data.
-   `plots/`: Output directory for generated plots.
-   `output/`: Output directory for analysis results and reports.
-   `tests/`: Contains tests for the application.
-   `docs/`: Project documentation.

## Data Transformations

The project implements the data transformation methodology from Crump et al. (2024), as detailed in `docs/crump_transformation_methodology.md`. Most variables are transformed to 100 * log-levels, while interest rates and unemployment enter in levels.

## Coding Style

The code generally follows the PEP 8 style guide. It is well-documented with comments and docstrings. The use of type hints is present in the `main.py` script.

## Testing

The `tests/` directory contains scripts for validating the implementation, such as `test_transformations.py` and `validate_crump_data.py`.
