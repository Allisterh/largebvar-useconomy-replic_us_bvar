"""
Variance Decomposition Analysis Module
Implements forecast error variance decomposition and historical decomposition for VAR models
Based on Crump et al. (2024) methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
import sys
import os

# Add config directory to path for importing plot utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from plot_utils import format_variable_name, format_variable_names

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VarianceDecomposition:
    """
    Variance Decomposition Analysis for VAR models
    """
    
    def __init__(self, var_results, irf_analysis):
        """
        Initialize variance decomposition analysis
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results
        irf_analysis : IRFAnalysis
            IRF analysis object
        """
        self.var_results = var_results
        self.irf_analysis = irf_analysis
        self.variable_names = var_results.names
        self.n_vars = len(self.variable_names)
        
        logger.info(f"Initialized variance decomposition for {self.n_vars} variables")
    
    def forecast_error_variance_decomposition(self, periods: int = 20, 
                                            identification: str = 'cholesky',
                                            ordering: List[str] = None) -> np.ndarray:
        """
        Compute forecast error variance decomposition
        
        Parameters:
        -----------
        periods : int
            Number of periods for decomposition
        identification : str
            Identification scheme ('cholesky', 'generalized')
        ordering : list, optional
            Variable ordering for Cholesky
            
        Returns:
        --------
        np.ndarray
            FEVD array of shape (periods, n_vars, n_vars)
            [period, variable, shock_contribution]
        """
        # Compute IRFs
        irf = self.irf_analysis.compute_irf(periods=periods, 
                                          identification=identification, 
                                          ordering=ordering)
        
        # Initialize FEVD array
        fevd = np.zeros((periods, self.n_vars, self.n_vars))
        
        # Compute variance decomposition for each period and variable
        for h in range(periods):
            for i in range(self.n_vars):  # Variable whose variance we're decomposing
                
                # Total variance at horizon h
                total_variance = 0
                shock_variances = np.zeros(self.n_vars)
                
                # Sum over all previous periods (0 to h)
                for s in range(h + 1):
                    for j in range(self.n_vars):  # Shock variable
                        shock_variances[j] += irf[s, i, j] ** 2
                
                total_variance = np.sum(shock_variances)
                
                # Compute proportions
                if total_variance > 0:
                    fevd[h, i, :] = shock_variances / total_variance
                else:
                    fevd[h, i, :] = 0
        
        return fevd
    
    def historical_decomposition(self, start_date: str = None, end_date: str = None,
                               identification: str = 'cholesky',
                               ordering: List[str] = None) -> pd.DataFrame:
        """
        Compute historical decomposition of variables
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for decomposition
        end_date : str, optional
            End date for decomposition
        identification : str
            Identification scheme
        ordering : list, optional
            Variable ordering
            
        Returns:
        --------
        pd.DataFrame
            Historical decomposition results
        """
        # Get structural shocks
        if identification == 'cholesky':
            P = self.irf_analysis.cholesky_identification(ordering)
        elif identification == 'generalized':
            P = self.irf_analysis.generalized_irf()
        else:
            raise ValueError(f"Unknown identification scheme: {identification}")
        
        # Get residuals and ensure a DataFrame with a usable index
        residuals = self.var_results.resid
        if isinstance(residuals, (pd.Series, pd.DataFrame)):
            residuals_df = pd.DataFrame(residuals, columns=self.variable_names)
        else:
            residuals_df = pd.DataFrame(residuals, columns=self.variable_names)

        # Apply date filtering if we have a DatetimeIndex
        if (start_date or end_date) and isinstance(residuals_df.index, pd.DatetimeIndex):
            residuals_df = residuals_df.loc[start_date:end_date]
        
        # Compute structural shocks and keep as DataFrame for .iloc compatibility
        shocks_array = residuals_df.values @ np.linalg.inv(P).T
        structural_shocks = pd.DataFrame(shocks_array, index=residuals_df.index, columns=self.variable_names)
        
        # Get coefficient matrices
        coeff_matrices = self.irf_analysis.coeff_matrices
        n_lags = len(coeff_matrices)
        
        # Initialize decomposition
        n_obs = len(residuals_df)
        decomposition = np.zeros((n_obs, self.n_vars, self.n_vars))
        
        # Compute historical decomposition
        for t in range(n_obs):
            for i in range(self.n_vars):  # Variable
                for j in range(self.n_vars):  # Shock
                    
                    # Sum contributions from all lags
                    contribution = 0
                    for lag in range(min(t + 1, n_lags)):
                        if t - lag >= 0:
                            # IRF coefficient at this lag
                            irf_coeff = P[i, j]  # Impact effect
                            
                            # Add lagged effects
                            for s in range(1, lag + 1):
                                if s <= len(coeff_matrices):
                                    temp_effect = 0
                                    for k in range(self.n_vars):
                                        temp_effect += coeff_matrices[s-1][i, k] * P[k, j]
                                    irf_coeff = temp_effect
                            
                            contribution += irf_coeff * structural_shocks.iloc[t - lag, j]
                    
                    decomposition[t, i, j] = contribution
        
        # Convert to DataFrame
        decomp_data = []
        for t in range(n_obs):
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    decomp_data.append({
                        'date': residuals_df.index[t],
                        'variable': self.variable_names[i],
                        'shock': self.variable_names[j],
                        'contribution': decomposition[t, i, j]
                    })
        
        decomp_df = pd.DataFrame(decomp_data)
        
        return decomp_df
    
    def plot_fevd(self, fevd: np.ndarray, variable: str, periods: int = None,
                 figsize: Tuple[int, int] = (12, 8), stacked: bool = True) -> plt.Figure:
        """
        Plot forecast error variance decomposition
        
        Parameters:
        -----------
        fevd : np.ndarray
            FEVD array
        variable : str
            Variable to plot decomposition for
        periods : int, optional
            Number of periods to plot
        figsize : tuple
            Figure size
        stacked : bool
            Whether to create stacked area plot
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if periods is None:
            periods = fevd.shape[0]
        
        var_idx = self.variable_names.index(variable)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_axis = range(periods)
        
        if stacked:
            # Stacked area plot
            bottom = np.zeros(periods)
            colors = plt.cm.Set3(np.linspace(0, 1, self.n_vars))
            
            for j, shock_var in enumerate(self.variable_names):
                values = fevd[:periods, var_idx, j] * 100  # Convert to percentage
                ax.fill_between(x_axis, bottom, bottom + values,
                              label=format_variable_name(shock_var), alpha=0.8, color=colors[j])
                bottom += values
        else:
            # Line plot
            for j, shock_var in enumerate(self.variable_names):
                values = fevd[:periods, var_idx, j] * 100
                ax.plot(x_axis, values, label=format_variable_name(shock_var), linewidth=2)
        
        ax.set_xlabel('Periods')
        ax.set_ylabel('Percentage of Variance')
        ax.set_title(f'Forecast Error Variance Decomposition: {format_variable_name(variable)}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if stacked:
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        return fig
    
    def plot_all_fevd(self, fevd: np.ndarray, periods: int = None,
                     figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Plot FEVD for all variables
        
        Parameters:
        -----------
        fevd : np.ndarray
            FEVD array
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
            periods = fevd.shape[0]
        
        n_cols = min(4, self.n_vars)
        n_rows = (self.n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        x_axis = range(periods)
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_vars))

        for i, variable in enumerate(self.variable_names):
            ax = axes[i]
            
            # Stacked area plot
            bottom = np.zeros(periods)
            
            for j, shock_var in enumerate(self.variable_names):
                values = fevd[:periods, i, j] * 100
                ax.fill_between(x_axis, bottom, bottom + values,
                              label=format_variable_name(shock_var) if i == 0 else "",
                              alpha=0.8, color=colors[j])
                bottom += values
            
            ax.set_title(f'{format_variable_name(variable)}', fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(self.n_vars, n_rows * n_cols):
            axes[i].set_visible(False)
        
        # Add legend
        if self.n_vars > 0:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle('Forecast Error Variance Decomposition - All Variables', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_historical_decomposition(self, hist_decomp: pd.DataFrame, variable: str,
                                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot historical decomposition for a variable
        
        Parameters:
        -----------
        hist_decomp : pd.DataFrame
            Historical decomposition results
        variable : str
            Variable to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        # Filter data for the specific variable
        var_data = hist_decomp[hist_decomp['variable'] == variable].copy()
        
        # Pivot to get shocks as columns
        pivot_data = var_data.pivot(index='date', columns='shock', values='contribution')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked area plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_data.columns)))
        
        # Separate positive and negative contributions
        positive_data = pivot_data.where(pivot_data >= 0, 0)
        negative_data = pivot_data.where(pivot_data < 0, 0)
        
        # Plot positive contributions
        ax.stackplot(pivot_data.index, *[positive_data[col] for col in positive_data.columns],
                    labels=positive_data.columns, alpha=0.8, colors=colors)
        
        # Plot negative contributions
        ax.stackplot(pivot_data.index, *[negative_data[col] for col in negative_data.columns],
                    alpha=0.8, colors=colors)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Contribution')
        ax.set_title(f'Historical Decomposition: {format_variable_name(variable)}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def save_fevd_results(self, fevd: np.ndarray, filepath: str) -> None:
        """
        Save FEVD results to file
        
        Parameters:
        -----------
        fevd : np.ndarray
            FEVD array
        filepath : str
            Path to save file
        """
        periods = fevd.shape[0]
        
        # Create DataFrame
        results_list = []
        
        for period in range(periods):
            for i, variable in enumerate(self.variable_names):
                for j, shock_var in enumerate(self.variable_names):
                    results_list.append({
                        'period': period,
                        'variable': variable,
                        'shock': shock_var,
                        'variance_share': fevd[period, i, j]
                    })
        
        results_df = pd.DataFrame(results_list)
        
        # Save to file
        if filepath.endswith('.csv'):
            results_df.to_csv(filepath, index=False)
        elif filepath.endswith('.xlsx'):
            results_df.to_excel(filepath, index=False)
        else:
            results_df.to_csv(filepath, index=False)
        
        logger.info(f"FEVD results saved to: {filepath}")


def analyze_variance_decomposition(var_results, irf_analysis, periods: int = 20,
                                 identification: str = 'cholesky', ordering: List[str] = None,
                                 save_plots: bool = True, 
                                 output_dir: str = './plots') -> VarianceDecomposition:
    """
    Convenience function to perform complete variance decomposition analysis
    
    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    irf_analysis : IRFAnalysis
        IRF analysis object
    periods : int
        Number of periods for decomposition
    identification : str
        Identification scheme
    ordering : list, optional
        Variable ordering
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    VarianceDecomposition
        Variance decomposition analysis object
    """
    import os
    
    # Initialize variance decomposition
    var_decomp = VarianceDecomposition(var_results, irf_analysis)
    
    # Compute FEVD (prefer posterior summary if available)
    logger.info(f"Computing forecast error variance decomposition...")
    fevd = None
    if hasattr(var_results, 'get_fevd_summary'):
        try:
            summary, intervals = var_results.get_fevd_summary(
                horizon=periods, identification=identification, interval_levels=[0.68, 0.90]
            )
            # summary['median'] shape: (n_response, n_impulse, horizon)
            median = summary['median']
            # Convert to (periods, variable, shock)
            fevd = np.transpose(median, (2, 0, 1))
            logger.info("Using posterior FEVD median from Bayesian draws.")
        except Exception as e:
            logger.warning(f"Posterior FEVD summary unavailable ({e}). Falling back to deterministic FEVD.")

    if fevd is None:
        fevd = var_decomp.forecast_error_variance_decomposition(
            periods=periods, identification=identification, ordering=ordering
        )
    
    # Compute historical decomposition
    logger.info("Computing historical decomposition...")
    hist_decomp = var_decomp.historical_decomposition(
        identification=identification, ordering=ordering
    )
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot all FEVD
        fig_all = var_decomp.plot_all_fevd(fevd, periods=periods)
        fig_all.savefig(f'{output_dir}/fevd_all_{identification}.png', 
                       dpi=300, bbox_inches='tight')
        plt.close(fig_all)
        
        # Plot individual FEVD for key variables
        key_variables = var_results.names # All variables
        for variable in key_variables:
            fig = var_decomp.plot_fevd(fevd, variable, periods=periods)
            fig.savefig(f'{output_dir}/fevd_{variable}_{identification}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Plot historical decomposition for key variables
        for variable in key_variables:
            fig = var_decomp.plot_historical_decomposition(hist_decomp, variable)
            fig.savefig(f'{output_dir}/hist_decomp_{variable}_{identification}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Save results
        var_decomp.save_fevd_results(fevd, f'{output_dir}/fevd_results_{identification}.csv')
        hist_decomp.to_csv(f'{output_dir}/historical_decomposition_{identification}.csv', index=False)
    
    logger.info(f"Variance decomposition analysis completed. Results saved to {output_dir}")
    
    return var_decomp


if __name__ == "__main__":
    # Test with sample VAR results
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
    from src.var_model import estimate_var_model
    from src.irf_analysis import IRFAnalysis
    from src.data_utils import load_var_data
    
    # Load data and estimate VAR (with fallback to download if needed)
    data = load_var_data('./data/var_data.csv')
    var_model = estimate_var_model(data, max_lags=4)
    
    # Create IRF analysis
    irf_analyzer = IRFAnalysis(var_model.results)
    
    # Perform variance decomposition analysis
    var_decomp = analyze_variance_decomposition(
        var_model.results, irf_analyzer, periods=20, identification='cholesky'
    )
    
    print("Variance decomposition analysis completed successfully!")
