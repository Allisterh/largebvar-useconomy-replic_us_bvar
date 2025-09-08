"""
FRED Data Downloader for Large Bayesian VAR Model
Based on Crump et al. (2024) methodology
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import os
from dotenv import load_dotenv
import warnings
from typing import Dict, List, Optional, Tuple
import logging
from statsmodels.tsa.stattools import adfuller
from src.seasonal_adjustment import SeasonalAdjuster

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDDataDownloader:
    """
    Downloads and processes FRED data for VAR modeling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED API connection
        
        Parameters:
        -----------
        api_key : str, optional
            FRED API key. If None, will try to load from environment
        """
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
        
        if api_key is None:
            raise ValueError("FRED API key not found. Please set FRED_API_KEY environment variable.")
        
        self.fred = Fred(api_key=api_key)
        self.data = {}
        self.transformed_data = {}
        self.seasonal_adjuster = SeasonalAdjuster()
        
    def download_series(self, series_id: str, start_date: str = '1960-01-01', 
                       end_date: str = '2024-12-31') -> pd.Series:
        """
        Download a single FRED series
        
        Parameters:
        -----------
        series_id : str
            FRED series identifier
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.Series
            Downloaded time series data
        """
        try:
            series = self.fred.get_series(series_id, start=start_date, end=end_date)
            logger.info(f"Downloaded {series_id}: {len(series)} observations")
            return series
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {str(e)}")
            return pd.Series(dtype=float)
    
    def transform_series(self, series: pd.Series, config: Dict) -> pd.Series:
        """
        Apply Crump et al. (2024) transformations to time series
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        config : dict
            Variable configuration with 'transform', 'adjustment_status', and 'frequency'
            
        Returns:
        --------
        pd.Series
            Transformed series according to Crump et al. specification
        """
        transform_type = config.get('transform', 'raw')
        adjustment_status = config.get('adjustment_status', 'raw')
        frequency = config.get('frequency', 'M')
        series_name = config.get('name', 'unknown')
        
        # Step 1: Apply transformation
        if transform_type in ('raw', 'level', 'rate_level'):
            return series
        elif transform_type in ('log_level_100', 'log'):
            # Log levels scaled by 100: 100*log(X_t)
            # Guard against non-positive values
            s = series.copy()
            s = s.where(s > 0)
            return 100.0 * np.log(s)
        elif transform_type in ('log_diff_100', 'log_100', 'log_diff'):
            # Growth rates: 100*log(X_t/X_{t-1})
            s = series.copy()
            s = s.where(s > 0)
            log_diff = 100.0 * np.log(s / s.shift(1))
            # Optional annualization if explicitly desired (commented out by default)
            # if adjustment_status in ('sa', 'saar') and frequency == 'Q':
            #     log_diff = log_diff * 4
            return log_diff
        elif transform_type == 'diff':
            return series.diff()
        else:
            logger.warning(f"Unknown transformation type: {transform_type}")
            return series

    def prepare_series(self, series: pd.Series, config: Dict) -> pd.Series:
        """
        Prepare a series by applying seasonal adjustment at native frequency then converting to quarterly.
        """
        series_name = config.get('name', 'unknown')
        adjustment_status = config.get('adjustment_status', 'raw')
        freq = config.get('frequency', 'M')

        processed = series

        # Step 1: Apply seasonal adjustment at native frequency if NSA
        if adjustment_status == 'nsa':
            logger.info(f"Applying seasonal adjustment to {series_name} at native frequency")
            processed = self.seasonal_adjuster.adjust_series(processed, series_name)

        # Step 2: Convert to quarterly frequency
        if freq in ['M', 'D']:
            processed = self.convert_to_quarterly(processed)

        return processed

    def convert_to_quarterly(self, series: pd.Series, method: str = 'mean') -> pd.Series:
        """
        Convert higher frequency data to quarterly
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        method : str
            Aggregation method ('mean', 'last', 'sum')
            
        Returns:
        --------
        pd.Series
            Quarterly time series
        """
        if method == 'mean':
            return series.resample('Q').mean()
        elif method == 'last':
            return series.resample('Q').last()
        elif method == 'sum':
            return series.resample('Q').sum()
        else:
            logger.warning(f"Unknown aggregation method: {method}")
            return series.resample('Q').mean()
    
    def download_all_variables(self, variables_config: Dict, 
                             start_date: str = '1960-01-01',
                             end_date: str = '2024-12-31') -> Dict[str, pd.Series]:
        """
        Download all variables from configuration
        
        Parameters:
        -----------
        variables_config : dict
            Dictionary with variable configurations
        start_date : str
            Start date for data download
        end_date : str
            End date for data download
            
        Returns:
        --------
        dict
            Dictionary of downloaded series
        """
        prepared_data = {}
        # Import alternative series mapping lazily to avoid circular imports
        try:
            import sys as _sys
            _sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
            from config.variables import ALTERNATIVE_SERIES as _ALT
        except Exception:
            _ALT = {}
        
        for series_id, config in variables_config.items():
            logger.info(f"Downloading and preparing {series_id}: {config['name']}")
            
            # Download the series (with fallback if needed)
            series = self.download_series(series_id, start_date, end_date)
            used_alt = False
            if series.empty and series_id in _ALT:
                alt_id = _ALT[series_id]
                logger.warning(f"Primary series {series_id} empty. Trying alternative {alt_id}.")
                series = self.download_series(alt_id, start_date, end_date)
                used_alt = not series.empty
            
            if not series.empty:
                # Prepare series (to quarterly, seasonal adjustment)
                prepared_series = self.prepare_series(series, config)
                prepared_data[series_id] = prepared_series
                if used_alt:
                    logger.info(f"Used alternative series for {series_id}")
            else:
                logger.warning(f"No data downloaded for {series_id}")

        transformed_data = {}
        for series_id, config in variables_config.items():
            if series_id in prepared_data:
                logger.info(f"Transforming {series_id}")
                # Apply final transformation
                transformed_series = self.transform_series(prepared_data[series_id], config)
                transformed_data[series_id] = transformed_series

        return transformed_data
    
    def create_dataframe(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Create a DataFrame from dictionary of series
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of time series
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all series
        """
        df = pd.DataFrame(data_dict)
        
        # Sort by date
        df = df.sort_index()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        logger.info(f"Created DataFrame with shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing data in the DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        method : str
            Method for handling missing data ('interpolate', 'forward_fill', 'drop')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing data handled
        """
        if method == 'interpolate':
            df_filled = df.interpolate(method='linear')
        elif method == 'forward_fill':
            df_filled = df.fillna(method='ffill')
        elif method == 'drop':
            df_filled = df.dropna()
        else:
            logger.warning(f"Unknown missing data method: {method}")
            df_filled = df.interpolate(method='linear')
        
        # Report missing data statistics
        missing_before = df.isnull().sum().sum()
        missing_after = df_filled.isnull().sum().sum()
        
        logger.info(f"Missing values before: {missing_before}")
        logger.info(f"Missing values after: {missing_after}")
        
        return df_filled
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to file
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filepath : str
            Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save based on file extension
        if filepath.endswith('.csv'):
            df.to_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df.to_excel(filepath)
        elif filepath.endswith('.pkl'):
            df.to_pickle(filepath)
        else:
            # Default to CSV
            df.to_csv(filepath)
        
        logger.info(f"Data saved to: {filepath}")
    
    def get_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        summary = pd.DataFrame({
            'count': df.count(),
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'missing_pct': (df.isnull().sum() / len(df)) * 100
        })
        
        return summary

    def test_stationarity(self, series: pd.Series, variable_name: str = "Unknown", 
                         alpha: float = 0.05) -> Dict:
        """
        Test stationarity using Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        variable_name : str
            Name of the variable for logging
        alpha : float
            Significance level for the test
            
        Returns:
        --------
        dict
            Dictionary with test results
        """
        # Remove NaN values for the test
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            logger.warning(f"Insufficient data for stationarity test: {variable_name}")
            return {
                'variable': variable_name,
                'is_stationary': None,
                'adf_statistic': None,
                'p_value': None,
                'critical_values': None,
                'warning': 'Insufficient data'
            }
        
        try:
            # Perform ADF test
            result = adfuller(series_clean, autolag='AIC')
            
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Determine if series is stationary
            is_stationary = p_value < alpha
            
            # Log results
            status = "STATIONARY" if is_stationary else "NON-STATIONARY"
            logger.info(f"Stationarity test for {variable_name}: {status} (p-value: {p_value:.4f})")
            
            if not is_stationary:
                logger.warning(f"⚠️ {variable_name} is non-stationary! This may cause explosive IRFs.")
                logger.warning(f"   Consider using log differences instead of levels.")
            
            return {
                'variable': variable_name,
                'is_stationary': is_stationary,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'warning': None
            }
            
        except Exception as e:
            logger.error(f"Stationarity test failed for {variable_name}: {str(e)}")
            return {
                'variable': variable_name,
                'is_stationary': None,
                'adf_statistic': None,
                'p_value': None,
                'critical_values': None,
                'warning': f'Test failed: {str(e)}'
            }

    def validate_data_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate stationarity of all variables in DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all variables
            
        Returns:
        --------
        pd.DataFrame
            Summary of stationarity tests
        """
        logger.info("=" * 60)
        logger.info("STATIONARITY VALIDATION")
        logger.info("=" * 60)
        
        results = []
        non_stationary_count = 0
        
        for column in df.columns:
            result = self.test_stationarity(df[column], column)
            results.append(result)
            
            if result['is_stationary'] is False:
                non_stationary_count += 1
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Log overall summary
        total_vars = len(df.columns)
        stationary_count = sum(1 for r in results if r['is_stationary'] is True)
        
        logger.info(f"Stationarity Summary:")
        logger.info(f"  Total variables: {total_vars}")
        logger.info(f"  Stationary: {stationary_count}")
        logger.info(f"  Non-stationary: {non_stationary_count}")
        
        if non_stationary_count > 0:
            logger.warning(f"⚠️ {non_stationary_count} non-stationary variables detected!")
            logger.warning("This will likely cause VAR instability and explosive IRFs.")
            logger.warning("Recommendation: Ensure proper differencing/transformation of non-stationary variables.")
            
            # List non-stationary variables
            non_stationary_vars = [r['variable'] for r in results if r['is_stationary'] is False]
            logger.warning(f"Non-stationary variables: {non_stationary_vars}")
        else:
            logger.info("✅ All variables appear stationary - good for stable VAR estimation!")
        
        return summary_df


def download_var_data(save_path: str = None, fred_variables: Optional[Dict] = None) -> pd.DataFrame:
    """
    Main function to download all VAR data
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the data
    fred_variables : dict, optional
        Dictionary of FRED variables to download. If None, uses default.
        
    Returns:
    --------
    pd.DataFrame
        Complete dataset for VAR modeling
    """
    # Import variables configuration
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
    from config.variables import FRED_VARIABLES, DATA_START, DATA_END

    if fred_variables is None:
        fred_variables = FRED_VARIABLES
    
    # Initialize downloader
    downloader = FREDDataDownloader()
    
    # Download all variables
    logger.info("Starting data download...")
    data_dict = downloader.download_all_variables(
        fred_variables, 
        start_date=DATA_START, 
        end_date=DATA_END
    )
    
    # Create DataFrame
    df = downloader.create_dataframe(data_dict)
    
    # Handle missing data
    df_clean = downloader.handle_missing_data(df, method='interpolate')
    
    # Generate summary
    summary = downloader.get_data_summary(df_clean)
    logger.info("Data Summary:")
    logger.info(f"\n{summary}")
    
    # Validate stationarity
    stationarity_summary = downloader.validate_data_stationarity(df_clean)
    
    # Save data if path provided
    if save_path:
        downloader.save_data(df_clean, save_path)
        
        # Also save summaries
        summary_path = save_path.replace('.csv', '_summary.csv')
        downloader.save_data(summary, summary_path)
        
        stationarity_path = save_path.replace('.csv', '_stationarity.csv')
        downloader.save_data(stationarity_summary, stationarity_path)
    
    return df_clean

def download_var_data_crump_transformations(save_path: str = None, fred_variables: Optional[Dict] = None) -> pd.DataFrame:
    """
    Download VAR data using Crump et al. (2024) transformation specifications
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the data (will add '_crump2024' suffix automatically)
    fred_variables : dict, optional
        Dictionary of FRED variables to download. If None, uses default.
        
    Returns:
    --------
    pd.DataFrame
        Complete dataset for VAR modeling with Crump et al. transformations
    """
    # Import variables configuration
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
    from config.variables import FRED_VARIABLES, DATA_START, DATA_END

    if fred_variables is None:
        fred_variables = FRED_VARIABLES
    
    # Initialize downloader
    downloader = FREDDataDownloader()
    
    # Download all variables with new transformation system
    logger.info("Starting data download with Crump et al. (2024) transformations...")
    logger.info("Transform system includes:")
    logger.info("- log_100: 100 × log(X_t/X_{t-1}) transformations")
    logger.info("- Seasonal adjustment using X-13ARIMA-SEATS with STL fallback")
    logger.info("- Annualization factors for SA variables")
    
    data_dict = downloader.download_all_variables(
        fred_variables,
        start_date=DATA_START,
        end_date=DATA_END
    )
    
    # Create DataFrame
    df = downloader.create_dataframe(data_dict)
    
    # Handle missing data
    df_clean = downloader.handle_missing_data(df, method='interpolate')
    
    # Generate summary
    summary = downloader.get_data_summary(df_clean)
    logger.info("Crump et al. (2024) Transformed Data Summary:")
    logger.info(f"\n{summary}")
    
    # Validate stationarity with Crump transformations
    stationarity_summary = downloader.validate_data_stationarity(df_clean)
    
    # Save data if path provided
    if save_path:
        # Add crump2024 suffix to indicate transformation methodology.
        # Avoid double-appending if the caller already provided a name with the suffix.
        import os as _os
        base_name = _os.path.basename(save_path)
        if '_crump2024' in base_name:
            crump_save_path = save_path
        elif save_path.endswith('.csv'):
            crump_save_path = save_path.replace('.csv', '_crump2024.csv')
        else:
            crump_save_path = save_path + '_crump2024.csv'

        downloader.save_data(df_clean, crump_save_path)
        logger.info(f"Crump et al. (2024) transformed data saved to: {crump_save_path}")
        
        # Also save summaries
        summary_path = crump_save_path.replace('.csv', '_summary.csv')
        downloader.save_data(summary, summary_path)
        logger.info(f"Data summary saved to: {summary_path}")
        
        stationarity_path = crump_save_path.replace('.csv', '_stationarity.csv')
        downloader.save_data(stationarity_summary, stationarity_path)
        logger.info(f"Stationarity summary saved to: {stationarity_path}")
    
    return df_clean


if __name__ == "__main__":
    # Download data
    data = download_var_data('/home/ubuntu/var_project/data/var_data.csv')
    print(f"Downloaded data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
