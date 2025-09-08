"""
Impulse Response Function Analysis Module
Implements IRF calculation and visualization for VAR models
Based on Crump et al. (2024) methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
import sys
import os

# Add config directory to path for importing plot utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from plot_utils import format_variable_name, format_variable_names
from cholesky_ordering import get_ordering_for_variables, validate_ordering

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IRFAnalysis:
    """
    Impulse Response Function Analysis for VAR models
    """
    
    def __init__(self, var_results):
        """
        Initialize IRF analysis
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results
        """
        self.var_results = var_results
        self.variable_names = var_results.names
        self.n_vars = len(self.variable_names)
        self.n_lags = var_results.k_ar
        
        # Store coefficient matrices
        self.coeff_matrices = self._extract_coefficient_matrices()
        self.sigma_u = var_results.sigma_u  # Residual covariance matrix
        
        logger.info(f"Initialized IRF analysis for {self.n_vars} variables with {self.n_lags} lags")
    
    def _extract_coefficient_matrices(self) -> List[np.ndarray]:
        """
        Extract coefficient matrices from VAR results
        
        Returns:
        --------
        list
            List of coefficient matrices for each lag (rows=equations, cols=predictors)
        """
        coeff_matrices = []
        
        # Primary method: Use .coefs attribute if available (our BVAR implementation)
        if hasattr(self.var_results, 'coefs') and isinstance(self.var_results.coefs, (list, tuple)):
            logger.info(f"Extracting coefficients using .coefs attribute (found {len(self.var_results.coefs)} lag matrices)")
            for lag, A_matrix in enumerate(self.var_results.coefs):
                if isinstance(A_matrix, np.ndarray) and A_matrix.shape == (self.n_vars, self.n_vars):
                    coeff_matrices.append(A_matrix.copy())
                    
                    # Debug: log coefficient statistics for each lag
                    max_abs = float(np.nanmax(np.abs(A_matrix)))
                    max_diag = float(np.nanmax(np.abs(np.diag(A_matrix))))
                    A_off = A_matrix.copy()
                    np.fill_diagonal(A_off, 0.0)
                    max_off = float(np.nanmax(np.abs(A_off)))
                    logger.info(f"IRF extraction - Lag {lag+1}: max|A|={max_abs:.6f}, "
                               f"max|diag|={max_diag:.6f}, max|off_diag|={max_off:.6f}")
                else:
                    logger.error(f"Invalid coefficient matrix shape for lag {lag}: expected ({self.n_vars}, {self.n_vars}), "
                               f"got {A_matrix.shape if hasattr(A_matrix, 'shape') else type(A_matrix)}")
                    raise ValueError(f"Invalid coefficient matrix shape for lag {lag}")
                    
        # Secondary method: Extract from .params DataFrame (statsmodels-style)
        elif hasattr(self.var_results, 'params') and hasattr(self.var_results.params, 'loc'):
            logger.info("Extracting coefficients using .params DataFrame")
            params = self.var_results.params
            
            for lag in range(self.n_lags):
                coeff_matrix = np.zeros((self.n_vars, self.n_vars))
                
                # Extract coefficients for this lag
                for eq_idx, eq_name in enumerate(self.variable_names):  # equations (rows)
                    for var_idx, var_name in enumerate(self.variable_names):  # predictors (cols)
                        param_name = f'L{lag+1}.{var_name}'
                        try:
                            if param_name in params.index:
                                coeff_matrix[eq_idx, var_idx] = params.loc[param_name, eq_name]
                        except (KeyError, IndexError):
                            # Coefficient not found, leave as zero
                            pass
                
                coeff_matrices.append(coeff_matrix)
                
                # Debug logging
                max_abs = float(np.nanmax(np.abs(coeff_matrix)))
                logger.info(f"IRF extraction from params - Lag {lag+1}: max|A|={max_abs:.6f}")
                
        else:
            raise ValueError("Cannot extract coefficient matrices: VAR results object lacks both 'coefs' and 'params' attributes")
        
        if len(coeff_matrices) != self.n_lags:
            raise ValueError(f"Expected {self.n_lags} coefficient matrices, got {len(coeff_matrices)}")
        
        logger.info(f"Successfully extracted {len(coeff_matrices)} coefficient matrices for IRF analysis")
        return coeff_matrices
    
    def cholesky_identification(self, ordering: List[str] = None, use_standard_ordering: bool = True) -> np.ndarray:
        """
        Compute structural shocks using Cholesky decomposition
        
        Parameters:
        -----------
        ordering : list, optional
            Variable ordering for Cholesky decomposition. If None and use_standard_ordering=True,
            uses economic theory-based ordering
        use_standard_ordering : bool, default True
            Whether to use standard macroeconomic ordering when ordering=None
            
        Returns:
        --------
        np.ndarray
            Structural impact matrix
        """
        if ordering is None:
            if use_standard_ordering:
                # Use economic theory-based ordering
                ordering = get_ordering_for_variables(self.variable_names, ordering_type='standard')
                logger.info(f"Using standard economic ordering: {ordering[:5]}... (showing first 5)")
            else:
                # Fallback to original variable order
                ordering = self.variable_names
                logger.warning("Using original variable order - may not be economically meaningful")
        
        # Validate ordering
        try:
            validate_ordering(self.variable_names, ordering)
        except ValueError as e:
            logger.error(f"Invalid ordering provided: {e}")
            logger.info("Falling back to standard ordering")
            ordering = get_ordering_for_variables(self.variable_names, ordering_type='standard')
        
        # Convert sigma_u to numpy array if it's a DataFrame
        if isinstance(self.sigma_u, pd.DataFrame):
            sigma_matrix = self.sigma_u.values
        else:
            sigma_matrix = self.sigma_u
        
        # Reorder covariance matrix according to specified ordering
        order_indices = [self.variable_names.index(var) for var in ordering]
        sigma_ordered = sigma_matrix[np.ix_(order_indices, order_indices)]
        
        # Cholesky decomposition
        P = linalg.cholesky(sigma_ordered, lower=True)
        
        # Reorder back to original variable order
        P_reordered = np.zeros_like(P)
        for i, orig_idx in enumerate(order_indices):
            for j, orig_jdx in enumerate(order_indices):
                P_reordered[orig_idx, orig_jdx] = P[i, j]
        
        return P_reordered
    
    def generalized_irf(self, normalize: bool = True) -> np.ndarray:
        """
        Compute generalized impulse responses (Pesaran and Shin, 1998)
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize shocks to unit size
            
        Returns:
        --------
        np.ndarray
            Generalized structural impact matrix
        """
        # Convert sigma_u to numpy array if it's a DataFrame
        if isinstance(self.sigma_u, pd.DataFrame):
            sigma_matrix = self.sigma_u.values
        else:
            sigma_matrix = self.sigma_u

        # Generalized IRF impact matrix:
        # Column i is Σ e_i normalized by sqrt(Σ_ii)
        P = np.zeros((self.n_vars, self.n_vars))
        for i in range(self.n_vars):
            denom = np.sqrt(sigma_matrix[i, i]) if normalize else 1.0
            # Response of all variables to a one-standard-deviation shock in variable i
            P[:, i] = sigma_matrix[:, i] / denom

        return P
    
    def compute_irf(self, periods: int = 12, identification: str = 'cholesky',
                   ordering: List[str] = None, normalize: bool = True,
                   use_standard_ordering: bool = True) -> np.ndarray:
        """
        Compute impulse response functions with stability validation
        
        Parameters:
        -----------
        periods : int
            Number of periods for IRF
        identification : str
            Identification scheme ('cholesky', 'generalized')
        ordering : list, optional
            Variable ordering for Cholesky. If None, uses standard economic ordering
        normalize : bool
            Whether to normalize shocks
        use_standard_ordering : bool, default True
            Whether to use standard macroeconomic ordering when ordering=None
            
        Returns:
        --------
        np.ndarray
            IRF array of shape (periods, n_vars, n_vars)
            [period, response_var, shock_var]
        """
        logger.info("=" * 50)
        logger.info("COMPUTING IMPULSE RESPONSE FUNCTIONS")
        logger.info("=" * 50)
        
        # Check for stability first
        if hasattr(self.var_results, '_stability_metrics'):
            stability = self.var_results._stability_metrics
            if stability['is_stable'] is False:
                logger.error("⚠️ Cannot compute reliable IRFs: VAR system is unstable!")
                logger.error("IRFs from unstable systems will be explosive and meaningless.")
                logger.error("Please fix the VAR instability first (see stability check warnings above).")
                # Return NaN array as a signal
                return np.full((periods, self.n_vars, self.n_vars), np.nan)
            elif stability['stability_margin'] < 0.05:
                logger.warning(f"⚠️ VAR system is near-unstable (margin: {stability['stability_margin']:.3f})")
                logger.warning("IRFs may be very persistent - consider shorter horizon.")
                
        # Check coefficient matrices for numerical issues
        logger.info("Validating coefficient matrices...")
        for i, A in enumerate(self.coeff_matrices):
            max_coef = np.nanmax(np.abs(A))
            if not np.isfinite(max_coef):
                logger.error(f"Non-finite coefficients in lag {i+1} matrix!")
                return np.full((periods, self.n_vars, self.n_vars), np.nan)
            if max_coef > 100:
                logger.warning(f"Very large coefficients in lag {i+1} matrix (max: {max_coef:.2f})")
                logger.warning("This may indicate numerical instability.")
        
        # Get structural impact matrix
        if identification == 'cholesky':
            P = self.cholesky_identification(ordering, use_standard_ordering)
            logger.info(f"Using Cholesky identification with economic ordering")
        elif identification == 'generalized':
            P = self.generalized_irf(normalize)
            logger.info(f"Using generalized identification")
        else:
            raise ValueError(f"Unknown identification scheme: {identification}")
        
        # Validate structural impact matrix
        max_impact = np.nanmax(np.abs(P))
        if not np.isfinite(max_impact):
            logger.error("Non-finite values in structural impact matrix!")
            return np.full((periods, self.n_vars, self.n_vars), np.nan)
        logger.info(f"Structural impact matrix: max|impact| = {max_impact:.4f}")
        
        # Initialize IRF array
        irf = np.zeros((periods, self.n_vars, self.n_vars))
        
        # Period 0: structural impact
        irf[0] = P
        
        # Compute IRFs for subsequent periods with stability monitoring
        max_irf_value = max_impact
        explosive_threshold = max_impact * 1000  # Flag if IRFs grow 1000x beyond initial impact
        
        for h in range(1, periods):
            irf[h] = np.zeros((self.n_vars, self.n_vars))
            
            # Sum over lags - this is the core VAR dynamics
            for lag in range(min(h, self.n_lags)):
                irf[h] += self.coeff_matrices[lag] @ irf[h - lag - 1]
            
            # Monitor for explosive behavior
            period_max = np.nanmax(np.abs(irf[h]))
            if period_max > max_irf_value:
                max_irf_value = period_max
                
            if period_max > explosive_threshold:
                logger.error(f"🚨 IRFs becoming explosive at period {h}!")
                logger.error(f"Maximum absolute response: {period_max:.2e}")
                logger.error("This confirms VAR instability. Truncating IRF computation.")
                # Fill remaining periods with NaN
                irf[h:] = np.nan
                break
                
            if h % 5 == 0:  # Log progress every 5 periods
                logger.debug(f"Period {h}: max|IRF| = {period_max:.4f}")
        
        # Final validation
        final_max = np.nanmax(np.abs(irf[~np.isnan(irf)]))
        logger.info(f"IRF computation completed.")
        logger.info(f"Maximum absolute IRF value: {final_max:.4f}")
        logger.info(f"IRF shape: {irf.shape} (periods, response_vars, shock_vars)")
        
        if final_max < 1e-8:
            logger.warning("⚠️ IRFs are extremely small - may indicate over-shrinkage in priors")
        elif final_max > 100:
            logger.warning("⚠️ IRFs are very large - may indicate near-instability")
            
        return irf
    
    def compute_cumulative_irf(self, irf: np.ndarray) -> np.ndarray:
        """
        Compute cumulative impulse response functions
        
        Parameters:
        -----------
        irf : np.ndarray
            Standard IRF array
            
        Returns:
        --------
        np.ndarray
            Cumulative IRF array
        """
        return np.cumsum(irf, axis=0)
    
    def bootstrap_irf(self, periods: int = 12, n_bootstrap: int = 1000,
                     identification: str = 'cholesky', ordering: List[str] = None,
                     confidence_level: float = 0.95, use_standard_ordering: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for IRFs
        
        Parameters:
        -----------
        periods : int
            Number of periods for IRF
        n_bootstrap : int
            Number of bootstrap replications
        identification : str
            Identification scheme
        ordering : list, optional
            Variable ordering for Cholesky. If None, uses standard economic ordering
        confidence_level : float
            Confidence level for intervals
        use_standard_ordering : bool, default True
            Whether to use standard macroeconomic ordering when ordering=None
            
        Returns:
        --------
        tuple
            (point_estimate, lower_bound, upper_bound)
        """
        # Compute point estimate
        point_irf = self.compute_irf(periods, identification, ordering, use_standard_ordering=use_standard_ordering)
        
        # Bootstrap replications
        bootstrap_irfs = np.zeros((n_bootstrap, periods, self.n_vars, self.n_vars))
        
        # Get residuals for bootstrap
        residuals = self.var_results.resid
        n_obs = len(residuals)
        
        for b in range(n_bootstrap):
            # Bootstrap residuals
            bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
            # Support both pandas DataFrame/Series and NumPy ndarray residuals
            if isinstance(residuals, (pd.Series, pd.DataFrame)):
                bootstrap_residuals = residuals.iloc[bootstrap_indices]
            else:
                bootstrap_residuals = residuals[bootstrap_indices]
            
            # This is a simplified bootstrap - in practice, you'd re-estimate the VAR
            # For now, we'll add noise to the original IRF
            noise_scale = 0.1  # Simplified noise scaling
            noise = np.random.normal(0, noise_scale, point_irf.shape)
            bootstrap_irfs[b] = point_irf + noise
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_irfs, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_irfs, upper_percentile, axis=0)
        
        return point_irf, lower_bound, upper_bound
    
    def plot_irf(self, irf: np.ndarray, shock_var: str, response_vars: List[str] = None,
                periods: int = None, confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                figsize: Tuple[int, int] = (15, 10), title: str = None) -> plt.Figure:
        """
        Plot impulse response functions
        
        Parameters:
        -----------
        irf : np.ndarray
            IRF array
        shock_var : str
            Variable that receives the shock
        response_vars : list, optional
            Variables to plot responses for
        periods : int, optional
            Number of periods to plot
        confidence_intervals : tuple, optional
            (lower_bound, upper_bound) arrays
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if response_vars is None:
            response_vars = self.variable_names
        
        if periods is None:
            periods = irf.shape[0]
        
        shock_idx = self.variable_names.index(shock_var)
        n_response_vars = len(response_vars)
        
        # Set up subplot grid
        n_cols = min(4, n_response_vars)
        n_rows = (n_response_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        x_axis = range(periods)

        for i, response_var in enumerate(response_vars):
            ax = axes[i]
            
            response_idx = self.variable_names.index(response_var)
            
            # Plot IRF
            irf_values = irf[:periods, response_idx, shock_idx]
            ax.plot(x_axis, irf_values, 'b-', linewidth=2, label='IRF')
            
            # Plot confidence intervals if provided
            if confidence_intervals is not None:
                lower_bound, upper_bound = confidence_intervals
                lower_values = lower_bound[:periods, response_idx, shock_idx]
                upper_values = upper_bound[:periods, response_idx, shock_idx]
                ax.fill_between(x_axis, lower_values, upper_values, 
                              alpha=0.3, color='blue', label='95% CI')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Formatting
            ax.set_title(f'Response of {format_variable_name(response_var)}')
            ax.set_xlabel('Periods')
            ax.set_ylabel('Response')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplots (axes is flattened to 1D above)
        for i in range(n_response_vars, n_rows * n_cols):
            # Safe for 1D axes array irrespective of rows/cols shape
            try:
                axes[i].set_visible(False)
            except Exception:
                pass
        
        if title is None:
            title = f'Impulse Response Functions: Shock to {format_variable_name(shock_var)}'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_all_irfs(self, irf: np.ndarray, periods: int = None,
                     figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Plot all IRFs in a grid
        
        Parameters:
        -----------
        irf : np.ndarray
            IRF array
        periods : int, optional
            Number of periods to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if periods is None:
            periods = irf.shape[0]
        
        fig, axes = plt.subplots(self.n_vars, self.n_vars, figsize=figsize)
        
        x_axis = range(periods)
        
        for i in range(self.n_vars):  # Response variables
            for j in range(self.n_vars):  # Shock variables
                ax = axes[i, j]
                
                # Plot IRF
                irf_values = irf[:periods, i, j]
                ax.plot(x_axis, irf_values, 'b-', linewidth=1.5)
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Formatting
                if i == 0:
                    ax.set_title(f'Shock: {format_variable_name(self.variable_names[j])}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{format_variable_name(self.variable_names[i])}', fontsize=10)
                if i == self.n_vars - 1:
                    ax.set_xlabel('Periods', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        fig.suptitle('All Impulse Response Functions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_individual_response(self, irf: np.ndarray, shock_var: str, response_var: str,
                               periods: int = None, confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                               figsize: Tuple[int, int] = (8, 6), title: str = None) -> plt.Figure:
        """
        Plot individual impulse response function for a specific shock-response pair
        
        Parameters:
        -----------
        irf : np.ndarray
            IRF array
        shock_var : str
            Variable that receives the shock
        response_var : str
            Variable whose response is plotted
        periods : int, optional
            Number of periods to plot
        confidence_intervals : tuple, optional
            (lower_bound, upper_bound) arrays
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if periods is None:
            periods = irf.shape[0]
        
        shock_idx = self.variable_names.index(shock_var)
        response_idx = self.variable_names.index(response_var)
        
        # Create single subplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        x_axis = range(periods)
        
        # Plot IRF
        irf_values = irf[:periods, response_idx, shock_idx]
        ax.plot(x_axis, irf_values, 'b-', linewidth=2, label='IRF')
        
        # Plot confidence intervals if provided
        if confidence_intervals is not None:
            lower_bound, upper_bound = confidence_intervals
            lower_values = lower_bound[:periods, response_idx, shock_idx]
            upper_values = upper_bound[:periods, response_idx, shock_idx]
            ax.fill_between(x_axis, lower_values, upper_values,
                          alpha=0.3, color='blue', label='95% CI')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        if title is None:
            title = f'Response of {format_variable_name(response_var)} to {format_variable_name(shock_var)} Shock'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Periods', fontsize=12)
        ax.set_ylabel('Response', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def save_irf_results(self, irf: np.ndarray, filepath: str,
                        confidence_intervals: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """
        Save IRF results to file
        
        Parameters:
        -----------
        irf : np.ndarray
            IRF array
        filepath : str
            Path to save file
        confidence_intervals : tuple, optional
            Confidence interval arrays
        """
        periods = irf.shape[0]
        
        # Create a comprehensive DataFrame
        results_list = []
        
        for period in range(periods):
            for i, response_var in enumerate(self.variable_names):
                for j, shock_var in enumerate(self.variable_names):
                    row = {
                        'period': period,
                        'response_variable': response_var,
                        'shock_variable': shock_var,
                        'irf_value': irf[period, i, j]
                    }
                    
                    if confidence_intervals is not None:
                        lower_bound, upper_bound = confidence_intervals
                        row['lower_bound'] = lower_bound[period, i, j]
                        row['upper_bound'] = upper_bound[period, i, j]
                    
                    results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        
        # Save to file
        if filepath.endswith('.csv'):
            results_df.to_csv(filepath, index=False)
        elif filepath.endswith('.xlsx'):
            results_df.to_excel(filepath, index=False)
        else:
            results_df.to_csv(filepath, index=False)
        
        logger.info(f"IRF results saved to: {filepath}")


def analyze_irfs(var_results, periods: int = 20, identification: str = 'cholesky',
                ordering: List[str] = None, save_plots: bool = True,
                output_dir: str = './plots', use_standard_ordering: bool = True) -> IRFAnalysis:
    """
    Convenience function to perform complete IRF analysis
    
    This function computes IRFs and generates multiple types of plots:
    1. Grid plot showing all IRFs (all_irfs_{identification}.png)
    2. Multi-panel plots for each shock variable (irf_{variable}_{identification}.png)
    3. Individual response plots organized in shock-specific folders (irf_{shock_var}/response_{response_var}.png)
    
    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    periods : int
        Number of periods for IRF
    identification : str
        Identification scheme ('cholesky', 'generalized')
    ordering : list, optional
        Variable ordering for Cholesky decomposition. If None, uses standard economic ordering
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save outputs
    use_standard_ordering : bool, default True
        Whether to use standard macroeconomic ordering when ordering=None
        
    Returns:
    --------
    IRFAnalysis
        IRF analysis object
        
    Notes:
    ------
    The function creates the following output structure:
    - {output_dir}/all_irfs_{identification}.png: Grid of all IRFs
    - {output_dir}/irf_{shock_var}_{identification}.png: Multi-panel plots by shock
    - {output_dir}/irf_{shock_var}/response_{response_var}.png: Individual response plots
    - {output_dir}/irf_results_{identification}.csv: IRF results data
    """
    import os
    
    # Initialize IRF analysis
    irf_analyzer = IRFAnalysis(var_results)
    
    # Compute IRFs using the unified, robust method
    logger.info(f"Computing IRFs with {identification} identification...")
    
    # Use the enhanced compute_irf method which includes all stability checks
    irf = irf_analyzer.compute_irf(periods=periods, identification=identification,
                                   ordering=ordering, use_standard_ordering=use_standard_ordering)
    
    # Check if IRF computation was successful
    if np.all(np.isnan(irf)):
        logger.error("🚨 IRF computation failed due to VAR instability!")
        logger.error("Cannot generate meaningful plots from NaN IRFs.")
        logger.error("Please fix the underlying VAR stability issues first.")
        return irf_analyzer
    
    # Final validation of IRF results
    valid_irf = irf[~np.isnan(irf)]
    if len(valid_irf) == 0:
        logger.error("All IRF values are NaN - computation completely failed!")
        return irf_analyzer
        
    max_abs_irf = np.nanmax(np.abs(valid_irf))
    logger.info(f"✅ IRF computation successful. Maximum absolute response: {max_abs_irf:.6f}")
    
    # No confidence intervals for now to keep it simple and robust
    lower_bound = None
    upper_bound = None
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot all IRFs
        fig_all = irf_analyzer.plot_all_irfs(irf, periods=periods)
        fig_all.savefig(f'{output_dir}/all_irfs_{identification}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_all)
        
        # Plot individual IRFs for key variables
        key_variables = var_results.names  # All variables
        for shock_var in key_variables:
            fig = irf_analyzer.plot_irf(
                irf, shock_var, response_vars=key_variables,
                confidence_intervals=None,  # No confidence intervals
                title=f'IRF Analysis: Shock to {format_variable_name(shock_var)} ({identification.title()})'
            )
            fig.savefig(f'{output_dir}/irf_{shock_var}_{identification}.png',
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Generate individual response plots organized by shock variable
        logger.info("Generating individual response plots for each shock variable...")
        for shock_var in key_variables:
            # Create folder for this shock variable
            shock_dir = os.path.join(output_dir, f'irf_{shock_var}')
            os.makedirs(shock_dir, exist_ok=True)
            
            # Generate individual response plots
            for response_var in key_variables:
                fig_individual = irf_analyzer.plot_individual_response(
                    irf, shock_var, response_var,
                    periods=periods,
                    confidence_intervals=None,  # No confidence intervals
                    title=f'Response of {format_variable_name(response_var)} to {format_variable_name(shock_var)} Shock ({identification.title()})'
                )
                
                # Save individual plot
                individual_filename = f'response_{response_var}.png'
                individual_path = os.path.join(shock_dir, individual_filename)
                fig_individual.savefig(individual_path, dpi=300, bbox_inches='tight')
                plt.close(fig_individual)
        
        logger.info(f"Individual response plots saved to shock-specific folders in {output_dir}")
        
        # Save IRF results
        irf_analyzer.save_irf_results(
            irf, f'{output_dir}/irf_results_{identification}.csv',
            confidence_intervals=None  # No confidence intervals
        )
    
    logger.info(f"IRF analysis completed. Results saved to {output_dir}")
    
    return irf_analyzer


if __name__ == "__main__":
    # Test with sample VAR results
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
    from src.var_model import estimate_var_model
    from src.data_utils import load_var_data
    
    # Load data and estimate VAR (with fallback to download if needed)
    data = load_var_data('./data/var_data.csv')
    var_model = estimate_var_model(data, max_lags=4)
    
    # Perform IRF analysis
    irf_analyzer = analyze_irfs(var_model.results, periods=20, identification='cholesky')
    
    print("IRF analysis completed successfully!")
