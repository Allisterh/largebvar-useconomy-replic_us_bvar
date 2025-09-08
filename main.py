#!/usr/bin/env python3
"""
Main Pipeline Script for Large Bayesian VAR Model
Replicates methodology from Crump et al. (2024)

This script provides a complete pipeline for:
1. Downloading FRED data
2. Estimating VAR model
3. Computing impulse response functions
4. Performing variance decomposition
5. Running diagnostic tests
6. Generating comprehensive plots and reports

Usage:
    python main.py [--config config.yaml] [--output-dir plots/] [--periods 20]
"""

import os
import sys
import argparse
import importlib
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

# Import our modules
from src.data_downloader import download_var_data, download_var_data_crump_transformations
from src.var_model import estimate_var_model
from src.irf_analysis import analyze_irfs
from src.variance_decomposition import analyze_variance_decomposition
from src.diagnostics import run_var_diagnostics
from src.coefficient_analysis import analyze_var_coefficients
from config.variables import FRED_VARIABLES, DEV_FRED_VARIABLES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('var_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VARPipeline:
    """
    Complete VAR modeling pipeline
    """
    
    def __init__(self, output_dir: str = 'output', data_dir: str = 'data', plots_dir: str = 'plots', profile_tag: str = 'default'):
        """
        Initialize VAR pipeline
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        data_dir : str
            Directory for data files
        plots_dir : str
            Directory for plots
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.plots_dir = plots_dir
        self.profile_tag = profile_tag or 'default'
        
        # Create directories
        for directory in [output_dir, data_dir, plots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.data = None
        self.var_model = None
        self.irf_analyzer = None
        self.var_decomp = None
        self.diagnostics = None
        self.coeff_analysis = None
        
        logger.info(f"Initialized VAR pipeline with output directory: {output_dir}")
    
    def step1_download_data(self, force_download: bool = False, use_crump_transformations: bool = True, fred_variables: dict = None) -> None:
        """
        Step 1: Download FRED data
        
        Parameters:
        -----------
        force_download : bool
            Whether to force re-download even if data exists
        use_crump_transformations : bool
            Whether to use Crump et al. (2024) transformation specifications
        """
        logger.info("=" * 60)
        logger.info("STEP 1: PREPARING FRED DATA")
        logger.info("=" * 60)

        # Use profile-tagged filenames to enable per-profile caching
        if use_crump_transformations:
            filename = f"var_data_{self.profile_tag}_crump2024.csv"
            data_path = os.path.join(self.data_dir, filename)
            # Backward-compatibility: handle older double-suffixed files
            legacy_double = data_path.replace('_crump2024.csv', '_crump2024_crump2024.csv')
            legacy_global = os.path.join(self.data_dir, 'var_data_crump2024_crump2024.csv')
            if not os.path.exists(data_path):
                if os.path.exists(legacy_double):
                    logger.info(f"Found legacy double-suffixed file. Using: {legacy_double}")
                    data_path = legacy_double
                elif os.path.exists(legacy_global):
                    logger.info(f"Found legacy double-suffixed file. Using: {legacy_global}")
                    data_path = legacy_global
            logger.info("Using Crump et al. (2024) transformation specifications")
        else:
            filename = f"var_data_{self.profile_tag}.csv"
            data_path = os.path.join(self.data_dir, filename)
            logger.info("Using legacy transformation specifications")

        # Check if data exists and is not forced to be re-downloaded
        if os.path.exists(data_path) and not force_download:
            logger.info(f"Data file found at {data_path}. Loading from disk.")
            import pandas as pd
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            if force_download:
                logger.info(f"Forcing re-download of data to {data_path}.")
            else:
                logger.info(f"Data file not found at {data_path}. Downloading from FRED.")
            
            if use_crump_transformations:
                # data_downloader will not append suffix if already present
                self.data = download_var_data_crump_transformations(data_path, fred_variables)
            else:
                self.data = download_var_data(data_path, fred_variables)
        
        logger.info(f"Data loaded: {self.data.shape[0]} observations, {self.data.shape[1]} variables")
        logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        # Save data summary
        summary_path = os.path.join(self.output_dir, 'data_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("VAR Data Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {self.data.shape}\n")
            f.write(f"Date range: {self.data.index.min()} to {self.data.index.max()}\n\n")
            f.write("Variables:\n")
            for i, var in enumerate(self.data.columns):
                f.write(f"{i+1:2d}. {var}\n")
            f.write(f"\nMissing values per variable:\n")
            f.write(str(self.data.isnull().sum()))
        
        logger.info(f"Data summary saved to: {summary_path}")
    
    def step2_estimate_var(self, max_lags: int = 8, start_date: str = None, end_date: str = None, n_draws: int = 500) -> None:
        """
        Step 2: Estimate VAR model
        
        Parameters:
        -----------
        max_lags : int
            Maximum lags to consider
        start_date : str, optional
            Start date for estimation
        end_date : str, optional
            End date for estimation
        n_draws : int, optional
            Number of Gibbs sampler draws
        """
        logger.info("=" * 60)
        logger.info("STEP 2: ESTIMATING VAR MODEL")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("Data must be downloaded first (run step1_download_data)")
        
        # Estimate VAR model
        self.var_model = estimate_var_model(
            self.data, 
            max_lags=max_lags, 
            start_date=start_date, 
            end_date=end_date,
            n_draws=n_draws
        )
        
        # Get model summary
        summary = self.var_model.get_model_summary()
        try:
            logger.info(f"Prepared data shape used for estimation: {self.var_model.prepared_data.shape}")
            logger.info(f"Prepared data date range: {self.var_model.prepared_data.index.min()} to {self.var_model.prepared_data.index.max()}")
        except Exception:
            pass
        
        logger.info("Bayesian VAR Model Estimation Results:")
        logger.info(f"  Variables: {summary['n_variables']}")
        logger.info(f"  Observations: {summary['n_observations']}")
        logger.info(f"  Lags: {summary['n_lags']}")
        logger.info(f"  Posterior draws: {summary.get('n_draws', 'N/A')}")
        logger.info(f"  Parameters: {summary.get('n_parameters', 'N/A')}")
        
        # Handle potential NaN values for Bayesian metrics
        ll = summary.get('log_likelihood', np.nan)
        aic = summary.get('aic', np.nan)
        bic = summary.get('bic', np.nan)
        
        if not np.isnan(ll):
            logger.info(f"  Log-likelihood (approx): {ll:.2f}")
        if not np.isnan(aic):
            logger.info(f"  AIC (approx): {aic:.2f}")
        if not np.isnan(bic):
            logger.info(f"  BIC (approx): {bic:.2f}")
        
        # Save model summary
        summary_path = os.path.join(self.output_dir, 'var_model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("VAR Model Summary\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                if key != 'variable_names':
                    f.write(f"{key}: {value}\n")
            f.write(f"\nVariables:\n")
            for i, var in enumerate(summary['variable_names']):
                f.write(f"{i+1:2d}. {var}\n")
        
        logger.info(f"Model summary saved to: {summary_path}")

        # Debug diagnostics for IRF flatness: report A_lag norms and sigma off-diagonals
        try:
            A_list = getattr(self.var_model.results, 'coefs', [])
            for idx, A in enumerate(A_list, start=1):
                max_abs = float(np.nanmax(np.abs(A)))
                off = A.copy()
                np.fill_diagonal(off, 0.0)
                max_off = float(np.nanmax(np.abs(off))) if off.size else 0.0
                logger.info(f"Lag {idx} A matrix: max|A|={max_abs:.6g}, max|offdiag(A)|={max_off:.6g}")
            Sigma = np.array(self.var_model.results.sigma_u)
            Sigma_off = Sigma.copy()
            np.fill_diagonal(Sigma_off, 0.0)
            logger.info(f"Sigma: max|offdiag|={float(np.nanmax(np.abs(Sigma_off))):.6g}, max|diag|={float(np.nanmax(np.abs(np.diag(Sigma)))):.6g}")
        except Exception as e:
            logger.warning(f"IRF debug diagnostics failed: {e}")
    
    def step3_impulse_responses(self, periods: int = 20, identification: str = 'cholesky') -> None:
        """
        Step 3: Compute impulse response functions
        
        Parameters:
        -----------
        periods : int
            Number of periods for IRF
        identification : str
            Identification scheme ('cholesky', 'generalized')
        """
        logger.info("=" * 60)
        logger.info("STEP 3: COMPUTING IMPULSE RESPONSE FUNCTIONS")
        logger.info("=" * 60)
        
        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")
        
        # Compute IRFs
        self.irf_analyzer = analyze_irfs(
            self.var_model.results,
            periods=periods,
            identification=identification,
            save_plots=True,
            output_dir=self.plots_dir
        )
        
        logger.info(f"IRF analysis completed with {identification} identification")
        logger.info(f"Plots and results saved to: {self.plots_dir}")
    
    def step4_variance_decomposition(self, periods: int = 20, identification: str = 'cholesky') -> None:
        """
        Step 4: Perform variance decomposition analysis
        
        Parameters:
        -----------
        periods : int
            Number of periods for decomposition
        identification : str
            Identification scheme
        """
        logger.info("=" * 60)
        logger.info("STEP 4: VARIANCE DECOMPOSITION ANALYSIS")
        logger.info("=" * 60)
        
        if self.irf_analyzer is None:
            raise ValueError("IRF analysis must be completed first (run step3_impulse_responses)")
        
        # Perform variance decomposition
        self.var_decomp = analyze_variance_decomposition(
            self.var_model.results,
            self.irf_analyzer,
            periods=periods,
            identification=identification,
            save_plots=True,
            output_dir=self.plots_dir
        )
        
        logger.info(f"Variance decomposition completed")
        logger.info(f"Plots and results saved to: {self.plots_dir}")
    
    def step5_diagnostics(self) -> None:
        """
        Step 5: Run diagnostic tests
        """
        logger.info("=" * 60)
        logger.info("STEP 5: DIAGNOSTIC TESTS")
        logger.info("=" * 60)
        
        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")
        
        # Run diagnostics
        self.diagnostics = run_var_diagnostics(
            self.var_model.results,
            save_plots=True,
            output_dir=self.plots_dir
        )
        
        logger.info("Diagnostic tests completed")
        logger.info(f"Diagnostic plots and report saved to: {self.plots_dir}")
    
    def step6_coefficient_analysis(self) -> None:
        """
        Step 6: Analyze VAR coefficients with statistical inference
        """
        logger.info("=" * 60)
        logger.info("STEP 6: COEFFICIENT ANALYSIS")
        logger.info("=" * 60)
        
        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")
        
        # Run coefficient analysis
        self.coeff_analysis = analyze_var_coefficients(
            self.var_model.results,
            save_plots=True,
            save_tables=True,
            output_dir=self.output_dir
        )
        
        logger.info("Coefficient analysis completed")
        logger.info(f"Coefficient tables and plots saved to: {self.output_dir}")
    
    def step7_generate_forecasts(self, forecast_periods: int = 12) -> None:
        """
        Step 7: Generate forecasts
        
        Parameters:
        -----------
        forecast_periods : int
            Number of periods to forecast
        """
        logger.info("=" * 60)
        logger.info("STEP 7: GENERATING FORECASTS")
        logger.info("=" * 60)
        
        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")
        
        # Generate forecasts
        forecasts, lower_bounds, upper_bounds = self.var_model.forecast(steps=forecast_periods)
        
        # Save forecasts
        forecast_path = os.path.join(self.output_dir, 'forecasts.csv')
        forecasts.to_csv(forecast_path)
        
        lower_path = os.path.join(self.output_dir, 'forecast_lower_bounds.csv')
        lower_bounds.to_csv(lower_path)
        
        upper_path = os.path.join(self.output_dir, 'forecast_upper_bounds.csv')
        upper_bounds.to_csv(upper_path)
        
        logger.info(f"Forecasts generated for {forecast_periods} periods")
        logger.info(f"Forecast files saved to: {self.output_dir}")
        
        # Create forecast summary
        summary_path = os.path.join(self.output_dir, 'forecast_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("VAR Forecast Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Forecast periods: {forecast_periods}\n")
            f.write(f"Forecast start: {forecasts.index[0]}\n")
            f.write(f"Forecast end: {forecasts.index[-1]}\n\n")
            f.write("Variables forecasted:\n")
            for i, var in enumerate(forecasts.columns):
                f.write(f"{i+1:2d}. {var}\n")
    
    def run_complete_analysis(self, max_lags: int = 8, periods: int = 20,
                            identification: str = 'cholesky', forecast_periods: int = 12,
                            force_download: bool = False, use_crump_transformations: bool = True, fred_variables: dict = None, gibbs_draws: int = 500) -> None:
        """
        Run complete VAR analysis pipeline
        
        Parameters:
        -----------
        max_lags : int
            Maximum lags for VAR model
        periods : int
            Periods for IRF and variance decomposition
        identification : str
            Identification scheme
        forecast_periods : int
            Periods to forecast
        force_download : bool
            Whether to force data re-download
        use_crump_transformations : bool
            Whether to use Crump et al. (2024) transformation specifications
        """
        start_time = datetime.now()
        
        logger.info("*" * 80)
        logger.info("STARTING COMPLETE VAR ANALYSIS PIPELINE")
        logger.info(f"Start time: {start_time}")
        logger.info("*" * 80)
        
        try:
            # Run all steps
            self.step1_download_data(force_download=force_download, use_crump_transformations=use_crump_transformations, fred_variables=fred_variables)
            self.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            self.step3_impulse_responses(periods=periods, identification=identification)
            self.step4_variance_decomposition(periods=periods, identification=identification)
            self.step5_diagnostics()
            self.step6_coefficient_analysis()
            self.step7_generate_forecasts(forecast_periods=forecast_periods)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("*" * 80)
            logger.info("VAR ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"End time: {end_time}")
            logger.info(f"Total duration: {duration}")
            logger.info("*" * 80)
            
            # Create final summary
            self._create_final_summary(start_time, end_time, duration)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _create_final_summary(self, start_time: datetime, end_time: datetime, duration) -> None:
        """Create final analysis summary"""
        summary_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("VAR Analysis Pipeline Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis completed: {end_time}\n")
            f.write(f"Total duration: {duration}\n\n")
            
            if self.data is not None:
                f.write(f"Data: {self.data.shape[0]} observations, {self.data.shape[1]} variables\n")
            
            if self.var_model is not None:
                summary = self.var_model.get_model_summary()
                f.write(f"Bayesian VAR Model: {summary['n_lags']} lags, {summary['n_variables']} variables\n")
                f.write(f"Posterior draws: {summary.get('n_draws', 'N/A')}\n")
                f.write(f"Parameters: {summary.get('n_parameters', 'N/A')}\n")
                
                # Handle potential NaN values
                ll = summary.get('log_likelihood', np.nan)
                aic = summary.get('aic', np.nan)
                bic = summary.get('bic', np.nan)
                
                if not np.isnan(ll):
                    f.write(f"Log-likelihood (approx): {ll:.2f}\n")
                if not np.isnan(aic) and not np.isnan(bic):
                    f.write(f"AIC (approx): {aic:.2f}, BIC (approx): {bic:.2f}\n")
            
            f.write(f"\nOutput files generated:\n")
            f.write(f"- Data directory: {self.data_dir}\n")
            f.write(f"- Plots directory: {self.plots_dir}\n")
            f.write(f"- Output directory: {self.output_dir}\n")
            
            # List key output files
            key_files = [
                'var_data.csv',
                'var_model_summary.txt',
                'var_coefficients.xlsx',
                'coefficient_heatmap.png',
                'forecasts.csv',
                'analysis_summary.txt'
            ]
            
            f.write(f"\nKey output files:\n")
            for file in key_files:
                if os.path.exists(os.path.join(self.output_dir, file)) or os.path.exists(os.path.join(self.data_dir, file)):
                    f.write(f"[x] {file}\n")
        
        logger.info(f"Final summary saved to: {summary_path}")

    def step8_conditional_scenario(self, constraints_path: Optional[str], horizon: int = 12) -> None:
        """
        Step 8: Run conditional forecast scenario using constraints.

        constraints_path: Path to JSON file specifying constraints per horizon.
            Example JSON:
            {
              "1": {"GDPC1": 0.5, "CPIAUCSL": 0.2},
              "2": {"DFF": 3.0}
            }
        """
        import json
        import pandas as pd

        logger.info("=" * 60)
        logger.info("STEP 8: CONDITIONAL FORECAST SCENARIO")
        logger.info("=" * 60)

        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")

        if constraints_path is None or not os.path.exists(constraints_path):
            raise ValueError("constraints file is required for scenario run and must exist")

        # Load constraints
        with open(constraints_path, 'r') as f:
            constraints_raw = json.load(f)

        # Map variable names to indices
        var_names = list(self.var_model.results.names)
        name_to_idx = {name: i for i, name in enumerate(var_names)}

        # Build constraints dict expected by conditional_forecast: {t: {idx: value}}
        constraints_idx = {}
        for t_str, vars_map in constraints_raw.items():
            try:
                t = int(t_str)
            except Exception:
                raise ValueError(f"Constraint horizon key must be int-like, got {t_str}")
            constraints_idx[t] = {}
            for vname, val in vars_map.items():
                if vname not in name_to_idx:
                    raise ValueError(f"Unknown variable in constraints: {vname}")
                constraints_idx[t][name_to_idx[vname]] = float(val)

        # Run conditional forecast draws
        draws = self.var_model.results.get_conditional_forecast(horizon=horizon, constraints=constraints_idx)
        # draws shape: (n_draws, horizon, n_vars)

        # Summarize median and 90% intervals across draws
        median = np.nanmedian(draws, axis=0)  # (horizon, n_vars)
        lower = np.nanpercentile(draws, 5, axis=0)
        upper = np.nanpercentile(draws, 95, axis=0)

        # Build index continuing the data index if possible
        try:
            last_ts = self.var_model.prepared_data.index[-1]
            freq = pd.infer_freq(self.var_model.prepared_data.index)
            if freq is not None:
                idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
            else:
                idx = pd.RangeIndex(start=1, stop=horizon + 1)
        except Exception:
            idx = pd.RangeIndex(start=1, stop=horizon + 1)

        median_df = pd.DataFrame(median, index=idx, columns=var_names)
        lower_df = pd.DataFrame(lower, index=idx, columns=var_names)
        upper_df = pd.DataFrame(upper, index=idx, columns=var_names)

        # Save to output directory
        os.makedirs(self.output_dir, exist_ok=True)
        median_df.to_csv(os.path.join(self.output_dir, 'scenario_forecast_median.csv'))
        lower_df.to_csv(os.path.join(self.output_dir, 'scenario_forecast_lower90.csv'))
        upper_df.to_csv(os.path.join(self.output_dir, 'scenario_forecast_upper90.csv'))

        logger.info(f"Scenario forecasts saved to {self.output_dir}")

    def step9_entropic_tilting(self, tilt_targets_path: Optional[str]) -> None:
        """
        Step 9: Entropic tilting of posterior predictive forecasts to match mean targets.

        tilt_targets_path JSON schema:
        {
          "constraints": [
             {"h": 12, "var": "PCEPILFE", "target": 200.0}
          ],
          "horizon": 12
        }
        """
        import json
        import pandas as pd
        from analysis.entropic_tilting import entropic_tilt_mean, summarize_tilted

        logger.info("=" * 60)
        logger.info("STEP 9: ENTROPIC TILTING")
        logger.info("=" * 60)

        if self.var_model is None:
            raise ValueError("VAR model must be estimated first (run step2_estimate_var)")

        if tilt_targets_path is None or not os.path.exists(tilt_targets_path):
            raise ValueError("tilt-targets file is required and must exist")

        with open(tilt_targets_path, 'r') as f:
            cfg = json.load(f)
        constraints = cfg.get('constraints', [])
        horizon = int(cfg.get('horizon', 12))
        if not constraints:
            raise ValueError("tilt-targets must include a non-empty 'constraints' list")

        # Use conditional forecast engine with empty constraints to get unconditional draws
        draws = self.var_model.results.get_conditional_forecast(horizon=horizon, constraints={})
        var_names = list(self.var_model.results.names)

        weights = entropic_tilt_mean(draws, var_names, constraints)
        summary = summarize_tilted(draws, weights)

        # Save weights and selected horizon summaries
        os.makedirs(self.output_dir, exist_ok=True)
        # Weights
        pd.DataFrame({'weight': weights}).to_csv(os.path.join(self.output_dir, 'tilted_weights.csv'))

        # Save overall summaries at all horizons
        for key, arr in summary.items():
            df = pd.DataFrame(arr.reshape(horizon, len(var_names)), columns=var_names)
            df.to_csv(os.path.join(self.output_dir, f'tilted_{key}.csv'))

        logger.info(f"Entropic tilting completed and saved to {self.output_dir}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='VAR Analysis Pipeline')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--plots-dir', default='plots', help='Plots directory')
    parser.add_argument('--max-lags', type=int, default=5, help='Maximum lags for VAR')
    parser.add_argument('--periods', type=int, default=12, help='Periods for IRF/FEVD')
    parser.add_argument('--identification', default='cholesky',
                       choices=['cholesky', 'generalized'], help='Identification scheme')
    parser.add_argument('--forecast-periods', type=int, default=12, help='Forecast periods')
    parser.add_argument('--force-download', action='store_true', help='Force data re-download')
    parser.add_argument('--legacy-transforms', action='store_true',
                       help='Use legacy transformations instead of Crump et al. (2024)')
    parser.add_argument('--step', type=int, choices=range(1, 10),
                       help='Run specific step only (1-9)')
    parser.add_argument('--dev-mode', action='store_true', help='Run in development mode with a smaller model')
    parser.add_argument('--gibbs-draws', type=int, default=5000, help='Number of Gibbs sampler draws')
    # Variable profiles
    parser.add_argument(
        '--variables-profile',
        default='default',
        choices=['default', 'paper', 'deposit'],
        help='Choose variables set: default (config/variables.py), paper (config/variables_alternative.py), deposit (config/variables_deposit_study.py)'
    )
    # Entropic tilting
    parser.add_argument('--tilt-targets', help='Path to JSON file with mean constraints for entropic tilting')
    # Scenario analysis (conditional forecasts)
    parser.add_argument('--scenario', choices=['custom'], help='Run a conditional-forecast scenario')
    parser.add_argument('--constraints', help='Path to JSON constraints mapping horizon -> {varName: value}')
    parser.add_argument('--scenario-horizon', type=int, default=12, help='Scenario forecast horizon')
    
    args = parser.parse_args()

    # Resolve variables profile
    module_map = {
        'default': 'config.variables',
        'paper': 'config.variables_alternative',
        'deposit': 'config.variables_deposit_study',
    }

    mod_path = module_map.get(args.variables_profile, 'config.variables')
    try:
        var_mod = importlib.import_module(mod_path)
        logger.info(f"Loaded variables profile '{args.variables_profile}' from {mod_path}")
    except Exception as e:
        logger.warning(f"Failed to load profile '{args.variables_profile}' ({e}); falling back to default")
        var_mod = importlib.import_module('config.variables')

    fred_variables = getattr(var_mod, 'FRED_VARIABLES', FRED_VARIABLES)
    max_lags = args.max_lags
    periods = args.periods
    gibbs_draws = args.gibbs_draws

    if args.dev_mode:
        logger.info("Running in development mode with a smaller model")
        fred_variables = getattr(var_mod, 'DEV_FRED_VARIABLES', DEV_FRED_VARIABLES)
        max_lags = 5
        periods = 12
        gibbs_draws = 20000 # Keep default for dev mode or adjust as needed

    # Log selected variable count and a few keys for transparency
    try:
        logger.info(f"Variables selected: {len(fred_variables)}; sample keys: {list(fred_variables.keys())[:5]}")
    except Exception:
        pass

    # Initialize pipeline
    pipeline = VARPipeline(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
        profile_tag=args.variables_profile
    )
    
    if args.step:
        # Run specific step
        logger.info(f"Running step {args.step} only")
        
        use_crump = not args.legacy_transforms
        
        if args.step == 1:
            pipeline.step1_download_data(force_download=args.force_download, use_crump_transformations=use_crump, fred_variables=fred_variables)
        elif args.step == 2:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)  # Load existing data
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
        elif args.step == 3:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step3_impulse_responses(periods=periods, identification=args.identification)
        elif args.step == 4:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step3_impulse_responses(periods=periods, identification=args.identification)
            pipeline.step4_variance_decomposition(periods=periods, identification=args.identification)
        elif args.step == 5:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step5_diagnostics()
        elif args.step == 6:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step6_coefficient_analysis()
        elif args.step == 7:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step7_generate_forecasts(forecast_periods=args.forecast_periods)
        elif args.step == 9:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step9_entropic_tilting(tilt_targets_path=args.tilt_targets)
        elif args.step == 8:
            pipeline.step1_download_data(force_download=False, use_crump_transformations=use_crump, fred_variables=fred_variables)
            pipeline.step2_estimate_var(max_lags=max_lags, n_draws=gibbs_draws)
            pipeline.step8_conditional_scenario(constraints_path=args.constraints, horizon=args.scenario_horizon)
    else:
        # Run complete analysis
        pipeline.run_complete_analysis(
            max_lags=max_lags,
            periods=periods,
            identification=args.identification,
            forecast_periods=args.forecast_periods,
            force_download=args.force_download,
            use_crump_transformations=not args.legacy_transforms,
            fred_variables=fred_variables,
            gibbs_draws=gibbs_draws
        )

        # Optional scenario run after full pipeline if requested
        if args.scenario:
            pipeline.step8_conditional_scenario(constraints_path=args.constraints, horizon=args.scenario_horizon)
        # Optional tilting run after full pipeline
        if args.tilt_targets:
            pipeline.step9_entropic_tilting(tilt_targets_path=args.tilt_targets)


if __name__ == "__main__":
    main()
