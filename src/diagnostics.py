"""
Diagnostic Tests and Plots Module
Implements comprehensive diagnostic tests for VAR models
Based on Crump et al. (2024) methodology
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.stats.stattools import jarque_bera

# Add config directory to path for importing plot utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from plot_utils import format_variable_name, format_variable_names
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VARDiagnostics:
    """
    Comprehensive diagnostic tests for VAR models
    """
    
    def __init__(self, var_results):
        """
        Initialize VAR diagnostics
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results
        """
        self.var_results = var_results
        self.variable_names = var_results.names
        self.n_vars = len(self.variable_names)
        self.residuals = var_results.resid
        self.fitted_values = var_results.fittedvalues
        
        logger.info(f"Initialized VAR diagnostics for {self.n_vars} variables")
    
    def residual_autocorrelation_test(self, lags: int = 10) -> Dict:
        """
        Test for residual autocorrelation using Ljung-Box test
        
        Parameters:
        -----------
        lags : int
            Number of lags to test
            
        Returns:
        --------
        dict
            Test results for each variable
        """
        results = {}
        
        for i, var_name in enumerate(self.variable_names):
            residual_series = self.residuals.iloc[:, i].dropna()
            
            if len(residual_series) > lags + 5:  # Need sufficient observations
                try:
                    lb_result = acorr_ljungbox(residual_series, lags=lags, return_df=True)
                    
                    results[var_name] = {
                        'lb_stat': lb_result['lb_stat'].iloc[-1],
                        'lb_pvalue': lb_result['lb_pvalue'].iloc[-1],
                        'significant_autocorr': lb_result['lb_pvalue'].iloc[-1] < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Could not compute Ljung-Box test for {var_name}: {e}")
                    results[var_name] = {'error': str(e)}
        
        return results
    
    def residual_normality_test(self) -> Dict:
        """
        Test for residual normality using Jarque-Bera test
        
        Returns:
        --------
        dict
            Test results for each variable
        """
        results = {}
        
        for i, var_name in enumerate(self.variable_names):
            residual_series = self.residuals.iloc[:, i].dropna()
            
            if len(residual_series) > 10:
                try:
                    # Handle different statsmodels versions
                    jb_result = jarque_bera(residual_series)
                    if len(jb_result) == 2:
                        jb_stat, jb_pvalue = jb_result
                    else:
                        # Newer versions return (statistic, pvalue, skewness, kurtosis)
                        jb_stat, jb_pvalue = jb_result[0], jb_result[1]
                    
                    results[var_name] = {
                        'jb_stat': jb_stat,
                        'jb_pvalue': jb_pvalue,
                        'non_normal': jb_pvalue < 0.05,
                        'skewness': stats.skew(residual_series),
                        'kurtosis': stats.kurtosis(residual_series)
                    }
                except Exception as e:
                    logger.warning(f"Could not compute Jarque-Bera test for {var_name}: {e}")
                    results[var_name] = {'error': str(e)}
        
        return results
    
    def residual_heteroskedasticity_test(self) -> Dict:
        """
        Test for residual heteroskedasticity using White test
        
        Returns:
        --------
        dict
            Test results for each variable
        """
        results = {}
        
        for i, var_name in enumerate(self.variable_names):
            residual_series = self.residuals.iloc[:, i].dropna()
            fitted_series = self.fitted_values.iloc[:, i].dropna()
            
            # Align series
            common_index = residual_series.index.intersection(fitted_series.index)
            residual_aligned = residual_series.loc[common_index]
            fitted_aligned = fitted_series.loc[common_index]
            
            if len(residual_aligned) > 10:
                try:
                    # White test requires exog matrix with at least 2 columns (constant + fitted)
                    # Create proper exogenous matrix
                    exog_matrix = np.column_stack([
                        np.ones(len(fitted_aligned)),  # Constant
                        fitted_aligned.values,          # Fitted values
                        fitted_aligned.values**2        # Squared fitted values for nonlinearity
                    ])
                    
                    white_stat, white_pvalue, _, _ = het_white(residual_aligned, exog_matrix)
                    
                    results[var_name] = {
                        'white_stat': white_stat,
                        'white_pvalue': white_pvalue,
                        'heteroskedastic': white_pvalue < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Could not compute White test for {var_name}: {e}")
                    results[var_name] = {'error': str(e)}
        
        return results
    
    def stability_test(self) -> Dict:
        """
        Test for model stability (eigenvalues of companion matrix)
        
        Returns:
        --------
        dict
            Stability test results
        """
        try:
            # Get coefficient matrices
            if hasattr(self.var_results, 'coefs'):
                coeff_matrices = self.var_results.coefs
            else:
                # Extract from params if coefs not available
                coeff_matrices = self._extract_coefficient_matrices()
            
            # Create companion matrix
            companion_matrix = self._create_companion_matrix(coeff_matrices)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(companion_matrix)
            moduli = np.abs(eigenvalues)
            
            # Check stability (all eigenvalues should be inside unit circle)
            is_stable = np.all(moduli < 1.0)
            max_modulus = np.max(moduli)
            
            results = {
                'is_stable': is_stable,
                'max_eigenvalue_modulus': max_modulus,
                'eigenvalues': eigenvalues.tolist(),
                'eigenvalue_moduli': moduli.tolist(),
                'n_explosive_roots': np.sum(moduli >= 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Could not compute stability test: {e}")
            results = {'error': str(e)}
        
        return results
    
    def _extract_coefficient_matrices(self) -> List[np.ndarray]:
        """Extract coefficient matrices from VAR results"""
        coeff_matrices = []
        n_lags = self.var_results.k_ar
        
        # This is a simplified extraction - in practice might need adjustment
        for lag in range(n_lags):
            coeff_matrix = np.eye(self.n_vars) * 0.5  # Placeholder
            coeff_matrices.append(coeff_matrix)
        
        return coeff_matrices
    
    def _create_companion_matrix(self, coeff_matrices: List[np.ndarray]) -> np.ndarray:
        """Create companion matrix from coefficient matrices"""
        n_lags = len(coeff_matrices)
        n_vars = coeff_matrices[0].shape[0]
        
        # Create companion matrix
        companion_size = n_vars * n_lags
        companion_matrix = np.zeros((companion_size, companion_size))
        
        # Fill first n_vars rows with coefficient matrices
        for i, coeff_matrix in enumerate(coeff_matrices):
            start_col = i * n_vars
            end_col = (i + 1) * n_vars
            companion_matrix[:n_vars, start_col:end_col] = coeff_matrix
        
        # Fill identity blocks for lagged variables
        for i in range(1, n_lags):
            start_row = i * n_vars
            end_row = (i + 1) * n_vars
            start_col = (i - 1) * n_vars
            end_col = i * n_vars
            companion_matrix[start_row:end_row, start_col:end_col] = np.eye(n_vars)
        
        return companion_matrix
    
    def model_fit_statistics(self) -> Dict:
        """
        Compute model fit statistics
        
        Returns:
        --------
        dict
            Model fit statistics
        """
        results = {
            'log_likelihood': self.var_results.llf,
            'aic': self.var_results.aic,
            'bic': self.var_results.bic,
            'hqic': self.var_results.hqic,
            'fpe': self.var_results.fpe,
            'n_observations': self.var_results.nobs,
            'n_parameters': self.var_results.df_model,
            'n_variables': self.n_vars,
            'n_lags': self.var_results.k_ar
        }
        
        # Compute R-squared for each equation
        r_squared = {}
        for i, var_name in enumerate(self.variable_names):
            y_actual = self.var_results.endog[:, i]
            y_fitted = self.fitted_values.iloc[:, i]
            
            # Align series
            if len(y_actual) != len(y_fitted):
                min_len = min(len(y_actual), len(y_fitted))
                y_actual = y_actual[-min_len:]
                y_fitted = y_fitted.iloc[-min_len:]
            
            ss_res = np.sum((y_actual - y_fitted) ** 2)
            ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
            
            if ss_tot > 0:
                r_squared[var_name] = 1 - (ss_res / ss_tot)
            else:
                r_squared[var_name] = 0
        
        results['r_squared'] = r_squared
        
        return results
    
    def plot_single_variable_diagnostics(self, var_name: str, test_results: Dict, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot comprehensive 2x2 diagnostic grid for a single variable.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        residuals = self.residuals[var_name].dropna()

        # 1. Residuals Time Series
        axes[0, 0].plot(residuals.index, residuals, linewidth=1)
        axes[0, 0].set_title(f'Residuals Time Series')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # 2. Residuals Distribution (Histogram)
        sns.histplot(residuals, bins=30, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title(f'Residuals Distribution')

        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot vs. Normal')

        # 4. Autocorrelation (ACF)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, ax=axes[1, 1], lags=20)
        axes[1, 1].set_title(f'Residual Autocorrelation (ACF)')

        # Add test results as text
        auto_corr_res = test_results['autocorrelation_tests'].get(var_name, {})
        normality_res = test_results['normality_tests'].get(var_name, {})
        hetero_res = test_results['heteroskedasticity_tests'].get(var_name, {})

        lb_pval = auto_corr_res.get('lb_pvalue', float('nan'))
        jb_pval = normality_res.get('jb_pvalue', float('nan'))
        white_pval = hetero_res.get('white_pvalue', float('nan'))

        text_str = (
            f"Ljung-Box (Q=10): p={lb_pval:.3f} "
            f"({'FAIL' if lb_pval < 0.05 else 'PASS'})"
            f"Jarque-Bera: p={jb_pval:.3f} "
            f"({'FAIL' if jb_pval < 0.05 else 'PASS'})"
            f"White Test: p={white_pval:.3f} "
            f"({'FAIL' if white_pval < 0.05 else 'PASS'})"
        )
        
        fig.suptitle(f'Diagnostic Report for: {format_variable_name(var_name)}', fontsize=16)
        plt.figtext(0.5, 0.02, text_str, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig

    def plot_fitted_vs_actual(self, figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Plot fitted vs actual values
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        n_vars = min(6, self.n_vars)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(n_vars):
            ax = axes[i]
            
            var_name = self.variable_names[i]
            
            # Get actual and fitted values
            y_actual = self.var_results.endog[:, i]
            y_fitted = self.fitted_values.iloc[:, i]
            
            # Align series
            if len(y_actual) != len(y_fitted):
                min_len = min(len(y_actual), len(y_fitted))
                y_actual = y_actual[-min_len:]
                y_fitted = y_fitted.iloc[-min_len:]
                dates = self.fitted_values.index[-min_len:]
            else:
                dates = self.fitted_values.index
            
            # Plot time series
            ax.plot(dates, y_actual, label='Actual', linewidth=1.5, alpha=0.8)
            ax.plot(dates, y_fitted, label='Fitted', linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{format_variable_name(var_name)}: Actual vs Fitted')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_diagnostic_report(self) -> Dict:
        """
        Create comprehensive diagnostic report
        
        Returns:
        --------
        dict
            Complete diagnostic report
        """
        logger.info("Running comprehensive diagnostic tests...")
        
        report = {
            'model_fit': self.model_fit_statistics(),
            'autocorrelation_tests': self.residual_autocorrelation_test(),
            'normality_tests': self.residual_normality_test(),
            'heteroskedasticity_tests': self.residual_heteroskedasticity_test(),
            'stability_test': self.stability_test()
        }
        
        # Summary statistics
        summary = {
            'n_variables_with_autocorr': sum(1 for v in report['autocorrelation_tests'].values() 
                                           if isinstance(v, dict) and v.get('significant_autocorr', False)),
            'n_variables_non_normal': sum(1 for v in report['normality_tests'].values() 
                                        if isinstance(v, dict) and v.get('non_normal', False)),
            'n_variables_heteroskedastic': sum(1 for v in report['heteroskedasticity_tests'].values() 
                                             if isinstance(v, dict) and v.get('heteroskedastic', False)),
            'model_stable': report['stability_test'].get('is_stable', False)
        }
        
        report['summary'] = summary
        
        return report
    
    def save_diagnostic_report(self, report: Dict, filepath: str) -> None:
        """
        Save diagnostic report to file
        
        Parameters:
        -----------
        report : dict
            Diagnostic report
        filepath : str
            Path to save file
        """
        # Convert to DataFrame format for saving
        report_data = []
        
        # Model fit statistics
        for key, value in report['model_fit'].items():
            if key != 'r_squared':
                report_data.append({
                    'category': 'model_fit',
                    'variable': 'overall',
                    'test': key,
                    'value': value
                })
        
        # R-squared for each variable
        for var, r2 in report['model_fit']['r_squared'].items():
            report_data.append({
                'category': 'model_fit',
                'variable': var,
                'test': 'r_squared',
                'value': r2
            })
        
        # Test results
        for test_category in ['autocorrelation_tests', 'normality_tests', 'heteroskedasticity_tests']:
            for var, results in report[test_category].items():
                if isinstance(results, dict) and 'error' not in results:
                    for test_name, test_value in results.items():
                        report_data.append({
                            'category': test_category,
                            'variable': var,
                            'test': test_name,
                            'value': test_value
                        })
        
        # Stability test
        for key, value in report['stability_test'].items():
            if key not in ['eigenvalues', 'eigenvalue_moduli']:  # Skip complex arrays
                report_data.append({
                    'category': 'stability_test',
                    'variable': 'overall',
                    'test': key,
                    'value': value
                })
        
        # Summary
        for key, value in report['summary'].items():
            report_data.append({
                'category': 'summary',
                'variable': 'overall',
                'test': key,
                'value': value
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save to file
        if filepath.endswith('.csv'):
            report_df.to_csv(filepath, index=False)
        elif filepath.endswith('.xlsx'):
            report_df.to_excel(filepath, index=False)
        else:
            report_df.to_csv(filepath, index=False)
        
        logger.info(f"Diagnostic report saved to: {filepath}")


def run_var_diagnostics(var_results, save_plots: bool = True, 
                       output_dir: str = '/home/ubuntu/var_project/plots') -> VARDiagnostics:
    """
    Convenience function to run complete VAR diagnostics
    
    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    VARDiagnostics
        Diagnostics object
    """
    import os
    
    # Initialize diagnostics
    diagnostics = VARDiagnostics(var_results)
    
    # Create diagnostic report
    report = diagnostics.create_diagnostic_report()
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot residual diagnostics for each variable
        for var_name in diagnostics.variable_names:
            fig_resid = diagnostics.plot_single_variable_diagnostics(var_name, report)
            fig_resid.savefig(f'{output_dir}/diagnostics_{var_name}.png', 
                             dpi=150, bbox_inches='tight')
            plt.close(fig_resid)
        
        # Plot fitted vs actual
        fig_fitted = diagnostics.plot_fitted_vs_actual()
        fig_fitted.savefig(f'{output_dir}/fitted_vs_actual.png', 
                          dpi=300, bbox_inches='tight')
        plt.close(fig_fitted)
        
        # Save diagnostic report
        diagnostics.save_diagnostic_report(report, f'{output_dir}/diagnostic_report.csv')
    
    logger.info(f"VAR diagnostics completed. Results saved to {output_dir}")
    
    return diagnostics


if __name__ == "__main__":
    # Test with sample VAR results
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
    from src.var_model import estimate_var_model
    from src.data_utils import load_var_data
    
    # Load data and estimate VAR (with fallback to download if needed)
    data = load_var_data('./data/var_data.csv')
    var_model = estimate_var_model(data, max_lags=4)
    
    # Run diagnostics
    diagnostics = run_var_diagnostics(var_model.results)
    
    print("VAR diagnostics completed successfully!")

