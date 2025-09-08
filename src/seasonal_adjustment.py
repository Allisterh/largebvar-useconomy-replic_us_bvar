"""
Seasonal Adjustment Module for VAR Data Processing
Based on Crump et al. (2024) methodology

Provides X-13ARIMA-SEATS seasonal adjustment with STL fallback
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import warnings

# Set up logging
logger = logging.getLogger(__name__)

class SeasonalAdjuster:
    """
    Handles seasonal adjustment using X-13ARIMA-SEATS with STL fallback
    """
    
    def __init__(self, x13_path: Optional[str] = None):
        """
        Initialize seasonal adjuster
        
        Parameters:
        -----------
        x13_path : str, optional
            Path to X-13ARIMA-SEATS executable
        """
        self.x13_path = x13_path
        self.x13_available = self._check_x13_availability()
        
        if not self.x13_available:
            logger.warning("X-13ARIMA-SEATS not available. Will use STL fallback for seasonal adjustment.")
    
    def _check_x13_availability(self) -> bool:
        """Check if X-13ARIMA-SEATS is available"""
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            # Try a simple test to see if X-13 is properly configured
            test_data = pd.Series([1, 2, 3, 4] * 12, 
                                index=pd.date_range('2020-01-01', periods=48, freq='M'))
            x13_arima_analysis(test_data, x12path=self.x13_path)
            return True
        except Exception as e:
            logger.debug(f"X-13 availability check failed: {e}")
            return False
    
    def adjust_series_x13(self, data: pd.Series, series_name: str = "series") -> pd.Series:
        """
        Apply X-13ARIMA-SEATS seasonal adjustment
        
        Parameters:
        -----------
        data : pd.Series
            Time series to adjust
        series_name : str
            Name of the series for logging
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            
            # Clean data - remove NaN values for X-13
            clean_data = data.dropna()
            
            if len(clean_data) < 36:  # X-13 requires at least 3 years of data
                logger.warning(f"Insufficient data for X-13 adjustment of {series_name}. Using STL fallback.")
                return self.adjust_series_stl(data, series_name)
            
            # Run X-13ARIMA-SEATS
            result = x13_arima_analysis(
                clean_data,
                x12path=self.x13_path,
                trading=True,      # Include trading day effects
                outlier=True,      # Automatic outlier detection
                print_stdout=False, # Suppress X-13 output
                print_stderr=False
            )
            
            # Extract seasonally adjusted series
            seasadj = result.seasadj
            
            # Align with original index
            adjusted_series = pd.Series(index=data.index, dtype=float)
            adjusted_series.loc[seasadj.index] = seasadj
            
            # Forward fill any missing values at the end
            adjusted_series = adjusted_series.fillna(method='ffill')
            
            logger.info(f"Successfully applied X-13 seasonal adjustment to {series_name}")
            return adjusted_series
            
        except Exception as e:
            logger.warning(f"X-13 seasonal adjustment failed for {series_name}: {e}. Using STL fallback.")
            return self.adjust_series_stl(data, series_name)
    
    def adjust_series_stl(self, data: pd.Series, series_name: str = "series") -> pd.Series:
        """
        Apply STL seasonal adjustment as fallback
        
        Parameters:
        -----------
        data : pd.Series
            Time series to adjust
        series_name : str
            Name of the series for logging
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        try:
            from statsmodels.tsa.seasonal import STL
            
            # Clean data
            clean_data = data.dropna()
            
            if len(clean_data) < 24:  # STL requires at least 2 years
                logger.warning(f"Insufficient data for STL adjustment of {series_name}. Returning original series.")
                return data
            
            # Determine seasonal period based on frequency
            if hasattr(data.index, 'freq'):
                if data.index.freq == 'M' or 'M' in str(data.index.freq):
                    seasonal = 12
                elif data.index.freq == 'Q' or 'Q' in str(data.index.freq):
                    seasonal = 4
                else:
                    seasonal = 12  # Default to monthly
            else:
                # Infer from data frequency
                time_diff = (clean_data.index[1] - clean_data.index[0]).days
                if time_diff <= 32:  # Monthly or less
                    seasonal = 12
                elif time_diff <= 100:  # Quarterly
                    seasonal = 4
                else:
                    seasonal = 12
            
            # Apply STL decomposition
            stl = STL(clean_data, seasonal=seasonal, robust=True)
            decomposition = stl.fit()
            
            # Seasonally adjusted = trend + residual
            seasadj = decomposition.trend + decomposition.resid
            
            # Align with original index
            adjusted_series = pd.Series(index=data.index, dtype=float)
            adjusted_series.loc[seasadj.index] = seasadj
            
            # Forward fill any missing values
            adjusted_series = adjusted_series.fillna(method='ffill')
            
            logger.info(f"Successfully applied STL seasonal adjustment to {series_name}")
            return adjusted_series
            
        except Exception as e:
            logger.error(f"STL seasonal adjustment failed for {series_name}: {e}. Returning original series.")
            return data
    
    def adjust_series(self, data: pd.Series, series_name: str = "series", 
                     method: str = "auto") -> pd.Series:
        """
        Apply seasonal adjustment using preferred method
        
        Parameters:
        -----------
        data : pd.Series
            Time series to adjust
        series_name : str
            Name of the series for logging
        method : str
            Adjustment method ('x13', 'stl', 'auto')
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        if method == "x13" or (method == "auto" and self.x13_available):
            return self.adjust_series_x13(data, series_name)
        elif method == "stl" or method == "auto":
            return self.adjust_series_stl(data, series_name)
        else:
            logger.warning(f"Unknown adjustment method: {method}. Using STL fallback.")
            return self.adjust_series_stl(data, series_name)
    
    def batch_adjust(self, data_dict: Dict[str, pd.Series], 
                    adjustment_specs: Dict[str, str]) -> Dict[str, pd.Series]:
        """
        Apply seasonal adjustment to multiple series
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of series name -> pandas Series
        adjustment_specs : dict
            Dictionary of series name -> adjustment status
            ('saar', 'sa', 'nsa', 'raw')
            
        Returns:
        --------
        dict
            Dictionary of adjusted series
        """
        adjusted_data = {}
        
        for series_name, series_data in data_dict.items():
            adjustment_status = adjustment_specs.get(series_name, 'raw')
            
            if adjustment_status == 'nsa':
                # Need seasonal adjustment
                adjusted_data[series_name] = self.adjust_series(series_data, series_name)
                logger.info(f"Applied seasonal adjustment to {series_name} (was NSA)")
            elif adjustment_status in ['saar', 'sa', 'raw']:
                # Already adjusted or doesn't need adjustment
                adjusted_data[series_name] = series_data
                logger.debug(f"No seasonal adjustment needed for {series_name} (status: {adjustment_status})")
            else:
                logger.warning(f"Unknown adjustment status '{adjustment_status}' for {series_name}. No adjustment applied.")
                adjusted_data[series_name] = series_data
        
        return adjusted_data


def apply_seasonal_adjustment(data: pd.Series, series_name: str = "series", 
                            method: str = "auto") -> pd.Series:
    """
    Convenience function for single series seasonal adjustment
    
    Parameters:
    -----------
    data : pd.Series
        Time series to adjust
    series_name : str
        Name of the series
    method : str
        Adjustment method ('x13', 'stl', 'auto')
        
    Returns:
    --------
    pd.Series
        Seasonally adjusted series
    """
    adjuster = SeasonalAdjuster()
    return adjuster.adjust_series(data, series_name, method)


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)
    
    # Create test data with seasonal pattern
    dates = pd.date_range('2010-01-01', '2023-12-01', freq='M')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 2, len(dates))
    test_data = pd.Series(trend + seasonal + noise, index=dates)
    
    # Test seasonal adjustment
    adjuster = SeasonalAdjuster()
    adjusted = adjuster.adjust_series(test_data, "test_series")
    
    print(f"Original data range: {test_data.min():.2f} to {test_data.max():.2f}")
    print(f"Adjusted data range: {adjusted.min():.2f} to {adjusted.max():.2f}")
    print(f"Seasonal adjustment completed successfully!")