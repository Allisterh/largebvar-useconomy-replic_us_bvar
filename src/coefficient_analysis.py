"""
VAR Coefficient Analysis Module
Provides detailed statistical analysis of VAR model coefficients
Includes t-statistics, p-values, confidence intervals, and significance tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

class VARCoefficientAnalysis:
    """
    Comprehensive coefficient analysis for VAR models
    Provides statistical inference, tables, and visualizations
    """
    
    def __init__(self, var_results):
        """
        Initialize coefficient analysis
        
        Parameters:
        -----------
        var_results : VARResults
            Fitted VAR model results from statsmodels
        """
        self.var_results = var_results
        self.variable_names = var_results.names
        self.n_vars = len(self.variable_names)
        self.n_lags = var_results.k_ar
        self.n_obs = var_results.nobs
        
        # Extract key components
        self.coefficients = None
        self.std_errors = None
        self.t_statistics = None
        self.p_values = None
        self.conf_intervals = None
        
        # Computed tables
        self.coefficient_tables = {}
        self.summary_stats = {}
        
        logger.info(f"Initialized coefficient analysis for {self.n_vars} variables, {self.n_lags} lags")
        
        # Compute all statistics
        self._extract_coefficients()
        self._compute_statistics()
    
    def _extract_coefficients(self) -> None:
        """Extract coefficient matrices and covariance from VAR results"""
        try:
            # Get coefficient parameters - VAR results has shape (n_params, n_equations)
            # We need to stack/flatten this properly for analysis
            params_df = self.var_results.params
            
            # Stack the parameters into a long format for analysis
            # Each column represents an equation, each row a parameter
            stacked_params = []
            param_names = []
            equation_names = []
            
            for eq_name in params_df.columns:
                for param_name in params_df.index:
                    stacked_params.append(params_df.loc[param_name, eq_name])
                    param_names.append(f"{eq_name}.{param_name}")
                    equation_names.append(eq_name)
            
            # Create a properly indexed Series for all coefficients
            self.coefficients = pd.Series(stacked_params, index=param_names)
            self.equation_mapping = pd.Series(equation_names, index=param_names)
            
            # Get coefficient covariance matrix
            self.coeff_cov = self.var_results.cov_params()
            
            # Store original parameter structure for table creation
            self.params_df = params_df
            
            logger.info("Coefficients extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting coefficients: {e}")
            raise
    
    def _compute_statistics(self) -> None:
        """Compute standard errors, t-statistics, p-values, and confidence intervals"""
        try:
            # Compute standard errors from covariance matrix diagonal
            std_errors_array = np.sqrt(np.diag(self.coeff_cov))
            
            # Create Series with same index as coefficients
            self.std_errors = pd.Series(std_errors_array, index=self.coefficients.index)
            
            # Compute t-statistics (element-wise division)
            self.t_statistics = self.coefficients / self.std_errors
            
            # Compute p-values (two-tailed test)
            # Degrees of freedom = n_obs - n_parameters per equation
            n_params_per_eq = len(self.params_df.index)
            df = self.n_obs - n_params_per_eq
            
            p_values_array = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), df))
            self.p_values = pd.Series(p_values_array, index=self.coefficients.index)
            
            # Compute confidence intervals (default 95%)
            self._compute_confidence_intervals()
            
            logger.info("Statistical inference computed successfully")
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            raise
    
    def _compute_confidence_intervals(self, alpha: float = 0.05) -> None:
        """
        Compute confidence intervals for coefficients
        
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05 for 95% CI)
        """
        n_params_per_eq = len(self.params_df.index)
        df = self.n_obs - n_params_per_eq
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * self.std_errors
        
        self.conf_intervals = pd.DataFrame({
            'lower': self.coefficients - margin_error,
            'upper': self.coefficients + margin_error
        }, index=self.coefficients.index)
    
    def create_coefficient_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Create formatted coefficient tables for each equation
        
        Returns:
        --------
        dict
            Dictionary with equation names as keys and coefficient DataFrames as values
        """
        self.coefficient_tables = {}
        
        # Create tables directly from the original params DataFrame structure
        for eq_name in self.variable_names:
            table_data = []
            
            # Get parameters for this equation from the original DataFrame
            for param_name in self.params_df.index:
                # Create full parameter key for lookup in our stacked series
                full_param_key = f"{eq_name}.{param_name}"
                
                coeff_val = self.coefficients.loc[full_param_key]
                std_err = self.std_errors.loc[full_param_key]
                t_stat = self.t_statistics.loc[full_param_key]
                p_val = self.p_values.loc[full_param_key]
                ci_lower = self.conf_intervals.loc[full_param_key, 'lower']
                ci_upper = self.conf_intervals.loc[full_param_key, 'upper']
                
                # Determine significance
                significant = p_val < 0.05
                
                table_data.append({
                    'Parameter': param_name,
                    'Coefficient': coeff_val,
                    'Std.Error': std_err,
                    't-statistic': t_stat,
                    'p-value': p_val,
                    'CI_Lower_95%': ci_lower,
                    'CI_Upper_95%': ci_upper,
                    'Significant': significant
                })
            
            # Create DataFrame
            eq_table = pd.DataFrame(table_data)
            eq_table = eq_table.round(6)
            
            self.coefficient_tables[eq_name] = eq_table
        
        logger.info(f"Created coefficient tables for {len(self.coefficient_tables)} equations")
        return self.coefficient_tables
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate summary statistics about coefficient significance
        
        Returns:
        --------
        dict
            Summary statistics
        """
        if not self.coefficient_tables:
            self.create_coefficient_tables()
        
        summary = {
            'total_coefficients': len(self.coefficients),
            'significant_at_5pct': np.sum(self.p_values < 0.05),
            'significant_at_1pct': np.sum(self.p_values < 0.01),
            'by_equation': {}
        }
        
        # Statistics by equation
        for eq_name, table in self.coefficient_tables.items():
            eq_stats = {
                'total_params': len(table),
                'significant_5pct': np.sum(table['Significant']),
                'significant_1pct': np.sum(table['p-value'] < 0.01),
                'max_t_stat': table['t-statistic'].abs().max(),
                'min_p_value': table['p-value'].min()
            }
            summary['by_equation'][eq_name] = eq_stats
        
        # Overall significance rates
        summary['significance_rate_5pct'] = summary['significant_at_5pct'] / summary['total_coefficients']
        summary['significance_rate_1pct'] = summary['significant_at_1pct'] / summary['total_coefficients']
        
        self.summary_stats = summary
        return summary
    
    def export_tables(self, output_dir: str = 'output', format: str = 'csv') -> None:
        """
        Export coefficient tables to files
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        format : str
            Export format ('csv', 'excel', 'both')
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.coefficient_tables:
            self.create_coefficient_tables()
        
        if format in ['csv', 'both']:
            # Export individual tables
            for eq_name, table in self.coefficient_tables.items():
                filename = os.path.join(output_dir, f'coefficients_{eq_name}.csv')
                table.to_csv(filename, index=False)
                logger.info(f"Exported {eq_name} coefficients to {filename}")
            
            # Export combined table
            combined_table = self._create_combined_table()
            combined_filename = os.path.join(output_dir, 'var_coefficients_all.csv')
            combined_table.to_csv(combined_filename, index=False)
            logger.info(f"Exported combined coefficients to {combined_filename}")
        
        if format in ['excel', 'both']:
            # Export to Excel with multiple sheets (if openpyxl is available)
            try:
                excel_filename = os.path.join(output_dir, 'var_coefficients.xlsx')
                with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                    for eq_name, table in self.coefficient_tables.items():
                        table.to_excel(writer, sheet_name=eq_name, index=False)
                    
                    # Add summary sheet
                    summary_df = self._create_summary_dataframe()
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Exported all coefficients to {excel_filename}")
            except ImportError:
                logger.warning("openpyxl not available, skipping Excel export. Install with: pip install openpyxl")
            except Exception as e:
                logger.warning(f"Could not export to Excel: {e}")
    
    def _create_combined_table(self) -> pd.DataFrame:
        """Create combined table with all coefficients"""
        combined_data = []
        
        for eq_name, table in self.coefficient_tables.items():
            table_copy = table.copy()
            table_copy['Equation'] = eq_name
            combined_data.append(table_copy)
        
        return pd.concat(combined_data, ignore_index=True)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        if not self.summary_stats:
            self.generate_summary_statistics()
        
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Metric': 'Total Coefficients',
            'Value': self.summary_stats['total_coefficients']
        })
        summary_data.append({
            'Metric': 'Significant at 5%',
            'Value': self.summary_stats['significant_at_5pct']
        })
        summary_data.append({
            'Metric': 'Significant at 1%',
            'Value': self.summary_stats['significant_at_1pct']
        })
        summary_data.append({
            'Metric': 'Significance Rate (5%)',
            'Value': f"{self.summary_stats['significance_rate_5pct']:.3f}"
        })
        summary_data.append({
            'Metric': 'Significance Rate (1%)',
            'Value': f"{self.summary_stats['significance_rate_1pct']:.3f}"
        })
        
        return pd.DataFrame(summary_data)
    
    def plot_coefficient_heatmap(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot coefficient heatmap showing magnitude and significance
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if not self.coefficient_tables:
            self.create_coefficient_tables()
        
        # Create coefficient matrix for heatmap
        coeff_matrix = np.zeros((self.n_vars, self.n_vars * self.n_lags + 1))
        significance_matrix = np.zeros((self.n_vars, self.n_vars * self.n_lags + 1))
        
        # Parameter labels
        param_labels = ['const']
        for lag in range(1, self.n_lags + 1):
            for var in self.variable_names:
                param_labels.append(f"{var}_L{lag}")
        
        # Fill matrices
        for eq_idx, (eq_name, table) in enumerate(self.coefficient_tables.items()):
            for param_idx, row in table.iterrows():
                if param_idx < coeff_matrix.shape[1]:
                    coeff_matrix[eq_idx, param_idx] = row['Coefficient']
                    significance_matrix[eq_idx, param_idx] = 1 if row['Significant'] else 0
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Coefficient magnitude heatmap
        sns.heatmap(coeff_matrix,
                   xticklabels=param_labels,
                   yticklabels=format_variable_names(self.variable_names),
                   cmap='RdBu_r',
                   center=0,
                   ax=ax1,
                   cbar_kws={'label': 'Coefficient Value'})
        ax1.set_title('VAR Coefficient Magnitudes')
        ax1.set_xlabel('')
        
        # Significance heatmap
        sns.heatmap(significance_matrix,
                   xticklabels=param_labels,
                   yticklabels=format_variable_names(self.variable_names),
                   cmap='Reds',
                   vmin=0, vmax=1,
                   ax=ax2,
                   cbar_kws={'label': 'Significant (p<0.05)'})
        ax2.set_title('Coefficient Significance (p < 0.05)')
        ax2.set_xlabel('Parameters')
        
        plt.tight_layout()
        return fig
    
    def plot_coefficient_significance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot coefficient significance by equation and lag
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        if not self.coefficient_tables:
            self.create_coefficient_tables()
        
        # Prepare data for plotting
        plot_data = []
        
        for eq_name, table in self.coefficient_tables.items():
            for _, row in table.iterrows():
                param_name = row['Parameter']
                
                # Extract lag information
                if 'L' in param_name and param_name != 'const':
                    try:
                        if '.L' in param_name:
                            parts = param_name.split('.L')
                            var_name = parts[0]
                            lag = int(parts[1])
                        elif '_L' in param_name:
                            parts = param_name.split('_L')
                            var_name = parts[0]
                            lag = int(parts[1])
                        else:
                            # Handle cases like 'L1.GDPC1'
                            if param_name.startswith('L'):
                                lag_part = param_name[1:].split('.')[0]
                                lag = int(lag_part)
                                var_name = param_name.split('.', 1)[1] if '.' in param_name else param_name
                            else:
                                lag = 0
                                var_name = param_name
                    except (ValueError, IndexError):
                        lag = 0
                        var_name = param_name
                else:
                    lag = 0
                    var_name = 'const' if param_name == 'const' else param_name
                
                plot_data.append({
                    'equation': eq_name,
                    'parameter': param_name,
                    'variable': var_name,
                    'lag': lag,
                    'coefficient': row['Coefficient'],
                    't_statistic': abs(row['t-statistic']),
                    'p_value': row['p-value'],
                    'significant': row['Significant']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Significance by lag
        lag_significance = plot_df.groupby('lag')['significant'].agg(['sum', 'count']).reset_index()
        lag_significance['rate'] = lag_significance['sum'] / lag_significance['count']
        
        ax1.bar(lag_significance['lag'], lag_significance['rate'], alpha=0.7)
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Significance Rate')
        ax1.set_title('Coefficient Significance by Lag')
        ax1.set_ylim(0, 1)
        
        # Plot 2: t-statistics distribution
        significant_t = plot_df[plot_df['significant']]['t_statistic']
        non_significant_t = plot_df[~plot_df['significant']]['t_statistic']
        
        ax2.hist(non_significant_t, bins=20, alpha=0.7, label='Non-significant', color='lightcoral')
        ax2.hist(significant_t, bins=20, alpha=0.7, label='Significant', color='skyblue')
        ax2.axvline(x=1.96, color='red', linestyle='--', label='5% Critical Value')
        ax2.set_xlabel('|t-statistic|')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of t-statistics')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def print_summary_report(self) -> None:
        """Print comprehensive summary report"""
        if not self.summary_stats:
            self.generate_summary_statistics()
        
        print("=" * 80)
        print("VAR COEFFICIENT ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Model: {self.n_vars} variables, {self.n_lags} lags, {self.n_obs} observations")
        print(f"Total coefficients: {self.summary_stats['total_coefficients']}")
        print(f"Significant at 5%: {self.summary_stats['significant_at_5pct']} ({self.summary_stats['significance_rate_5pct']:.1%})")
        print(f"Significant at 1%: {self.summary_stats['significant_at_1pct']} ({self.summary_stats['significance_rate_1pct']:.1%})")
        
        print("\nBy Equation:")
        print("-" * 60)
        for eq_name, stats in self.summary_stats['by_equation'].items():
            print(f"{eq_name:12s}: {stats['significant_5pct']:3d}/{stats['total_params']:3d} significant " +
                  f"({stats['significant_5pct']/stats['total_params']:.1%}), " +
                  f"max |t|={stats['max_t_stat']:.2f}")
        
        print("\nTop 5 Most Significant Coefficients:")
        print("-" * 60)
        # Get all coefficients sorted by p-value
        all_coeffs = []
        for eq_name, table in self.coefficient_tables.items():
            for _, row in table.iterrows():
                all_coeffs.append({
                    'equation': eq_name,
                    'parameter': row['Parameter'],
                    'coefficient': row['Coefficient'],
                    't_statistic': row['t-statistic'],
                    'p_value': row['p-value']
                })
        
        all_coeffs_df = pd.DataFrame(all_coeffs)
        top_coeffs = all_coeffs_df.nsmallest(5, 'p_value')
        
        for _, row in top_coeffs.iterrows():
            print(f"{row['equation']:12s}.{row['parameter']:15s}: " +
                  f"coeff={row['coefficient']:8.4f}, t={row['t_statistic']:6.2f}, " +
                  f"p={row['p_value']:.6f}")


def analyze_var_coefficients(var_results, save_plots: bool = True, 
                           save_tables: bool = True, output_dir: str = 'output') -> VARCoefficientAnalysis:
    """
    Convenience function to run complete coefficient analysis
    
    Parameters:
    -----------
    var_results : VARResults
        Fitted VAR model results
    save_plots : bool
        Whether to save plots
    save_tables : bool
        Whether to save coefficient tables
    output_dir : str
        Output directory
        
    Returns:
    --------
    VARCoefficientAnalysis
        Coefficient analysis object
    """
    # Initialize analysis
    coeff_analyzer = VARCoefficientAnalysis(var_results)
    
    # Create coefficient tables
    coeff_analyzer.create_coefficient_tables()
    
    # Generate summary statistics
    coeff_analyzer.generate_summary_statistics()
    
    # Print summary report
    coeff_analyzer.print_summary_report()
    
    # Save tables if requested
    if save_tables:
        coeff_analyzer.export_tables(output_dir=output_dir, format='both')
    
    # Save plots if requested
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Coefficient heatmap
        fig1 = coeff_analyzer.plot_coefficient_heatmap()
        fig1.savefig(os.path.join(output_dir, 'coefficient_heatmap.png'), 
                     dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # Significance plots
        fig2 = coeff_analyzer.plot_coefficient_significance()
        fig2.savefig(os.path.join(output_dir, 'coefficient_significance.png'),
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        logger.info(f"Coefficient plots saved to {output_dir}")
    
    return coeff_analyzer


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load existing VAR results for testing
    try:
        import pandas as pd
        from var_model import estimate_var_model
        
        from src.data_utils import load_var_data
        
        # Load data (with fallback to download if needed)
        data = load_var_data('./data/var_data.csv')
        
        # Estimate VAR model
        var_model = estimate_var_model(data, max_lags=4)
        
        # Run coefficient analysis
        coeff_analysis = analyze_var_coefficients(var_model.results, output_dir='../output')
        
        print("Coefficient analysis completed successfully!")
        
    except Exception as e:
        print(f"Could not run example: {e}")
        print("This is normal if data files don't exist yet.")