"""
Enhanced data utilities for large-scale VAR processing.

This module provides memory-efficient data processing utilities for handling
datasets with 100+ variables, including batch processing, streaming operations,
and memory management.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Generator, List, Dict, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import gc
import psutil
import warnings
from pathlib import Path
import h5py

try:
    from .computational.memory_management import MemoryManager, optimize_chunk_size
    from .data_downloader import FREDDataDownloader
except ImportError:
    # Fallback for standalone execution
    logging.warning("Could not import computational modules - some features may not be available")
    MemoryManager = None
    FREDDataDownloader = None

# Set up logger
logger = logging.getLogger(__name__)


def load_var_data(data_path: str = './data/var_data.csv', 
                  force_download: bool = False) -> pd.DataFrame:
    """
    Load VAR data from file or download if not present
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
    force_download : bool
        If True, download data even if file exists
        
    Returns:
    --------
    pd.DataFrame
        VAR data with datetime index
    """
    # Normalize path to handle relative paths consistently
    data_path = os.path.abspath(data_path)
    
    # Check if file exists and we're not forcing download
    if os.path.exists(data_path) and not force_download:
        logger.info(f"Loading existing data from: {data_path}")
        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded data shape: {data.shape}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load existing data: {str(e)}")
            logger.info("Falling back to download...")
    
    # Download data if file doesn't exist or loading failed
    logger.info(f"Downloading data to: {data_path}")
    
    # Import download function
    from src.data_downloader import download_var_data
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Download and save data
    data = download_var_data(data_path)
    logger.info(f"Downloaded and saved data to: {data_path}")
    
    return data


def get_default_data_path() -> str:
    """
    Get the default path for VAR data file
    
    Returns:
    --------
    str
        Default data file path
    """
    # Get the directory of this file and construct path to data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, 'data', 'var_data.csv')


class LargeScaleDataProcessor:
    """
    Memory-efficient processor for large-scale VAR datasets.
    
    This class handles datasets with hundreds of variables by implementing
    batch processing, parallel downloads, and memory-efficient storage.
    """
    
    def __init__(self, max_memory_mb: int = 4000, n_workers: int = None,
                 cache_dir: str = 'data/cache'):
        """
        Initialize large-scale data processor.
        
        Parameters:
        -----------
        max_memory_mb : int
            Maximum memory to use in MB
        n_workers : int
            Number of parallel workers (None for auto)
        cache_dir : str
            Directory for caching intermediate results
        """
        self.max_memory_mb = max_memory_mb
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if MemoryManager:
            self.memory_manager = MemoryManager(max_memory_fraction=0.7)
        else:
            self.memory_manager = None
        
        self.downloader = None
        
    def setup_fred_connection(self, api_key: Optional[str] = None):
        """Setup FRED API connection."""
        if FREDDataDownloader:
            self.downloader = FREDDataDownloader(api_key)
        else:
            logger.error("FREDDataDownloader not available")
    
    def download_large_dataset(self, variable_dict: Dict[str, Dict], 
                             batch_size: int = 50,
                             start_date: str = '1960-01-01',
                             end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Download large dataset using batch processing and parallelization.
        
        Parameters:
        -----------
        variable_dict : Dict[str, Dict]
            Variable specifications with transformations
        batch_size : int
            Number of variables to process per batch
        start_date : str
            Start date for data
        end_date : str
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            Combined dataset
        """
        if self.downloader is None:
            raise ValueError("FRED connection not setup. Call setup_fred_connection first.")
        
        logger.info(f"Starting large-scale download: {len(variable_dict)} variables")
        
        # Split variables into batches
        variable_items = list(variable_dict.items())
        batches = [variable_items[i:i + batch_size] 
                  for i in range(0, len(variable_items), batch_size)]
        
        logger.info(f"Processing {len(batches)} batches with {batch_size} variables each")
        
        # Process batches in parallel
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit batch jobs
            future_to_batch = {}
            for i, batch in enumerate(batches):
                future = executor.submit(
                    self._process_batch, 
                    batch, i, start_date, end_date
                )
                future_to_batch[future] = i
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    batch_results.append((batch_idx, result))
                    logger.info(f"Batch {batch_idx + 1}/{len(batches)} completed")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    continue
        
        # Sort results by batch index and combine
        batch_results.sort(key=lambda x: x[0])
        
        if not batch_results:
            raise RuntimeError("All batches failed to process")
        
        # Combine batch results
        logger.info("Combining batch results...")
        combined_data = self._combine_batch_results([result[1] for result in batch_results])
        
        logger.info(f"Download completed: {combined_data.shape}")
        return combined_data
    
    def _process_batch(self, variable_batch: List[Tuple[str, Dict]], 
                      batch_idx: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Process a single batch of variables."""
        batch_data = []
        batch_cache_file = self.cache_dir / f"batch_{batch_idx}.h5"
        
        # Check if batch is already cached
        if batch_cache_file.exists():
            try:
                logger.info(f"Loading cached batch {batch_idx}")
                return pd.read_hdf(batch_cache_file, key='data')
            except Exception as e:
                logger.warning(f"Failed to load cache for batch {batch_idx}: {e}")
        
        # Process variables in batch
        for var_name, var_config in variable_batch:
            try:
                # Download with retry logic
                series_data = self._download_with_retry(
                    var_config['series_id'], start_date, end_date, max_retries=3
                )
                
                if series_data is not None and not series_data.empty:
                    # Apply transformations
                    transformed_data = self._apply_transformations(
                        series_data, var_config
                    )
                    
                    if transformed_data is not None:
                        transformed_data.name = var_name
                        batch_data.append(transformed_data)
                        
            except Exception as e:
                logger.warning(f"Failed to process {var_name}: {str(e)}")
                continue
        
        if not batch_data:
            logger.warning(f"No data successfully processed in batch {batch_idx}")
            return pd.DataFrame()
        
        # Combine batch data
        batch_df = pd.concat(batch_data, axis=1, sort=True)
        
        # Cache the batch result
        try:
            batch_df.to_hdf(batch_cache_file, key='data', mode='w', complib='zlib')
        except Exception as e:
            logger.warning(f"Failed to cache batch {batch_idx}: {e}")
        
        return batch_df
    
    def _download_with_retry(self, series_id: str, start_date: str, 
                           end_date: str, max_retries: int = 3) -> Optional[pd.Series]:
        """Download series with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.downloader.download_series(series_id, start_date, end_date)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download {series_id} after {max_retries} attempts: {e}")
                    return None
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {series_id}, retrying...")
                    continue
    
    def _apply_transformations(self, series: pd.Series, config: Dict) -> Optional[pd.Series]:
        """Apply transformations to series."""
        try:
            transform = config.get('transform', 'raw')
            
            if transform == 'log':
                # Handle non-positive values
                if (series <= 0).any():
                    series = series.replace(0, np.nan)
                    series = series[series > 0]
                return np.log(series) * 100
                
            elif transform == 'log_100':
                if (series <= 0).any():
                    series = series.replace(0, np.nan)
                    series = series[series > 0]
                return np.log(series) * 100
                
            elif transform == 'diff':
                return series.diff()
                
            elif transform == 'log_diff':
                if (series <= 0).any():
                    series = series.replace(0, np.nan)
                    series = series[series > 0]
                log_series = np.log(series)
                return log_series.diff() * 100
                
            elif transform == 'raw':
                return series
                
            else:
                logger.warning(f"Unknown transformation: {transform}")
                return series
                
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            return None
    
    def _combine_batch_results(self, batch_results: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine results from multiple batches."""
        # Filter out empty dataframes
        valid_results = [df for df in batch_results if not df.empty]
        
        if not valid_results:
            raise ValueError("No valid batch results to combine")
        
        # Use memory-efficient concatenation for large datasets
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Simple concatenation for now - could be enhanced with memory management
        logger.info("Combining results in memory")
        return pd.concat(valid_results, axis=1, sort=True)


class DataQualityChecker:
    """
    Data quality checker for large VAR datasets.
    
    Performs comprehensive quality checks on large datasets including
    missing data analysis, outlier detection, and basic statistics.
    """
    
    def __init__(self, max_missing_pct: float = 0.2, outlier_method: str = 'iqr'):
        """
        Initialize data quality checker.
        
        Parameters:
        -----------
        max_missing_pct : float
            Maximum allowed missing data percentage
        outlier_method : str
            Outlier detection method ('iqr', 'zscore')
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_method = outlier_method
        
    def comprehensive_quality_check(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality check on dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to check
            
        Returns:
        --------
        Dict[str, Any]
            Quality check results
        """
        logger.info(f"Starting quality check on dataset: {data.shape}")
        
        results = {
            'dataset_shape': data.shape,
            'date_range': (data.index.min(), data.index.max()),
            'missing_data': self._check_missing_data(data),
            'outliers': self._detect_outliers(data),
            'data_types': self._check_data_types(data),
            'summary_stats': self._compute_summary_stats(data),
            'recommendations': []
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        logger.info("Quality check completed")
        return results
    
    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check missing data patterns."""
        missing_counts = data.isnull().sum()
        missing_pcts = (missing_counts / len(data)) * 100
        
        problematic_vars = missing_pcts[missing_pcts > self.max_missing_pct * 100]
        
        return {
            'total_missing': missing_counts.sum(),
            'missing_by_variable': missing_counts.to_dict(),
            'missing_percentages': missing_pcts.to_dict(),
            'problematic_variables': problematic_vars.to_dict(),
            'variables_to_drop': problematic_vars.index.tolist()
        }
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in the dataset."""
        outlier_results = {}
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                series = data[col].dropna()
                
                if len(series) == 0:
                    continue
                
                if self.outlier_method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    
                elif self.outlier_method == 'zscore':
                    z_scores = np.abs((series - series.mean()) / series.std())
                    outliers = series[z_scores > 3]
                
                else:
                    outliers = pd.Series([])  # Placeholder
                
                outlier_results[col] = {
                    'n_outliers': len(outliers),
                    'outlier_pct': (len(outliers) / len(series)) * 100,
                    'outlier_indices': outliers.index.tolist()[:10]  # First 10 only
                }
        
        return outlier_results
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Check data types of all variables."""
        return data.dtypes.astype(str).to_dict()
    
    def _compute_summary_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Compute summary statistics for numeric variables."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        summary_stats = {}
        for col in numeric_data.columns:
            stats = numeric_data[col].describe()
            summary_stats[col] = stats.to_dict()
        
        return summary_stats
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality check results."""
        recommendations = []
        
        # Missing data recommendations
        missing_data = results['missing_data']
        if missing_data['problematic_variables']:
            recommendations.append(
                f"Consider dropping {len(missing_data['problematic_variables'])} variables "
                f"with >{self.max_missing_pct*100}% missing data"
            )
        
        # Outlier recommendations
        outliers = results['outliers']
        high_outlier_vars = [var for var, info in outliers.items() 
                           if info['outlier_pct'] > 5]
        if high_outlier_vars:
            recommendations.append(
                f"Consider investigating {len(high_outlier_vars)} variables "
                f"with >5% outliers: {high_outlier_vars[:3]}..."
            )
        
        # Dataset size recommendations
        n_vars = results['dataset_shape'][1]
        if n_vars > 200:
            recommendations.append(
                "Large dataset detected. Consider using dimension reduction techniques."
            )
        
        return recommendations


if __name__ == "__main__":
    # Test the data loading function
    logging.basicConfig(level=logging.INFO)
    
    data_path = get_default_data_path()
    data = load_var_data(data_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Test quality checker if data is available
    if not data.empty:
        quality_checker = DataQualityChecker()
        quality_results = quality_checker.comprehensive_quality_check(data)
        print(f"\nQuality check results:")
        print(f"Missing data: {quality_results['missing_data']['total_missing']} total")
        print(f"Recommendations: {len(quality_results['recommendations'])}")
        for rec in quality_results['recommendations']:
            print(f"  - {rec}")