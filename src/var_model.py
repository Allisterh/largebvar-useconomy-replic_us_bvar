"""
VAR Model Estimation Module
Implements Vector Autoregression with Bayesian features
Based on Crump et al. (2024) methodology
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .bayesian.core_bayesian_var import LargeBayesianVAR

# Add config directory to path for importing plot utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from plot_utils import format_variable_name, format_variable_names
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianVAR:
    """
    Wrapper class for the LargeBayesianVAR model to integrate with the existing pipeline.
    """
    
    def __init__(self, data: pd.DataFrame, max_lags: int = 4, fast_mode: bool = False):
        """
        Initialize Bayesian VAR model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data for VAR estimation
        max_lags : int
            Maximum number of lags to consider
        fast_mode : bool
            Enable fast mode for quicker estimation
        """
        self.data = data.copy()
        self.max_lags = max_lags
        self.fast_mode = fast_mode
        self.model = None
        self.results = None
        self.optimal_lags = None
        self.prepared_data = None
        
        logger.info(f"Initialized Bayesian VAR with {data.shape[1]} variables and {data.shape[0]} observations")
    
    def prepare_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Prepare data for VAR estimation
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for estimation sample
        end_date : str, optional
            End date for estimation sample
            
        Returns:
        --------
        pd.DataFrame
            Prepared data
        """
        # Select date range if specified
        if start_date or end_date:
            data = self.data.loc[start_date:end_date]
        else:
            data = self.data.copy()
        
        # Report missingness before filtering
        missing_pct = (data.isnull().sum() / len(data)).sort_values(ascending=False)
        if (missing_pct > 0).any():
            logger.info("Missingness by variable (pre-filter):\n" + missing_pct.to_string())

        # Remove columns with too many missing values (>50%) with explicit logging
        drop_cols = missing_pct[missing_pct >= 0.5].index.tolist()
        if drop_cols:
            logger.warning(f"Dropping columns with >=50% missing: {drop_cols}")
        data = data.drop(columns=drop_cols, errors='ignore')
        
        # Drop rows with any remaining missing values
        before_rows = len(data)
        data = data.dropna()
        after_rows = len(data)
        if after_rows < before_rows:
            logger.info(f"Dropped {before_rows - after_rows} rows due to remaining missing values after preprocessing")
        
        logger.info(f"Prepared data shape: {data.shape}")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        
        return data
    
    def select_lag_order(self, data: pd.DataFrame, max_lags: int = None) -> int:
        """
        Select optimal lag order using information criteria
        
        Parameters:
        -----------
        data : pd.DataFrame
            Prepared data
        max_lags : int, optional
            Maximum lags to consider
            
        Returns:
        --------
        int
            Optimal lag order
        """
        if max_lags is None:
            max_lags = self.max_lags
        
        # Ensure we have enough observations
        max_possible_lags = min(max_lags, len(data) // 4)
        
        model = VAR(data)
        lag_order_results = model.select_order(maxlags=max_possible_lags)
        
        # Use AIC as primary criterion, with BIC as backup
        optimal_lags = lag_order_results.aic
        if optimal_lags is None or optimal_lags == 0:
            optimal_lags = lag_order_results.bic
        if optimal_lags is None or optimal_lags == 0:
            optimal_lags = 2  # Default fallback
        
        logger.info(f"Lag order selection results:")
        logger.info(f"AIC: {lag_order_results.aic}")
        logger.info(f"BIC: {lag_order_results.bic}")
        logger.info(f"Selected optimal lags: {optimal_lags}")
        
        return optimal_lags

    def estimate(self, start_date: str = None, end_date: str = None, 
                lags: int = None, n_draws: int = 1000, n_burnin: int = 500) -> 'LargeBayesianVAR':
        """
        Estimate the BVAR model using the new Bayesian backend.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for estimation
        end_date : str, optional
            End date for estimation
        lags : int, optional
            Number of lags (if None, will be selected automatically)
        n_draws : int
            Total number of Gibbs draws.
        n_burnin : int
            Number of burn-in draws to discard.
            
        Returns:
        --------
        LargeBayesianVAR
            The fitted Bayesian VAR model instance.
        """
        # Prepare data
        prepared_data = self.prepare_data(start_date, end_date)
        self.prepared_data = prepared_data
        
        # Select lag order if not specified
        if lags is None:
            self.optimal_lags = self.select_lag_order(prepared_data)
        else:
            self.optimal_lags = lags
        
        # Initialize and fit the LargeBayesianVAR model
        self.model = LargeBayesianVAR(n_lags=self.optimal_lags, fast_mode=self.fast_mode)
        # Pass DataFrame to preserve index and column names for diagnostics/plots
        self.model.fit(prepared_data, n_draws=n_draws, n_burnin=n_burnin)
        
        # Add variable names for compatibility with IRF analysis
        self.model.names = prepared_data.columns.tolist()
        
        # For compatibility with the rest of the pipeline, we can store the fitted model
        self.results = self.model
        
        logger.info(f"Bayesian VAR model estimated with {self.optimal_lags} lags.")
        
        return self.results

    def forecast(self, steps: int = 12, alpha: float = 0.05):
        """
        Produce multi-step forecasts with approximate confidence intervals.

        Parameters:
        -----------
        steps : int
            Number of periods to forecast
        alpha : float
            Significance level for intervals (default 0.05 for 95% CI)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Forecasts, lower bounds, upper bounds
        """
        if self.results is None or self.model is None:
            raise ValueError("Model must be estimated before forecasting.")
        if self.prepared_data is None:
            raise ValueError("Prepared data not available for forecasting.")

        # Coefficient matrices (A_1..A_p)
        A_mats = self.model.coefs  # list of (n_vars, n_vars), length p
        p = len(A_mats)
        n = self.results.n_vars

        # Intercept vector (if present)
        try:
            intercept = self.model.params.loc['const'].values
        except Exception:
            intercept = np.zeros(n)

        # Residual covariance (posterior mean)
        Sigma = np.array(self.model.sigma_u)

        # Seed with the last p observations
        history = self.prepared_data.values
        if history.shape[0] < p:
            raise ValueError("Not enough observations to initialize forecasts with given lag order.")
        last_vals = [history[-lag] for lag in range(1, p + 1)]  # y_{T-1}, y_{T-2}, ..., y_{T-p}

        # Recursive point forecasts
        forecasts = []
        lag_buffer = last_vals.copy()  # list length p, lag_buffer[0] = y_{t-1}
        for h in range(steps):
            y_hat = intercept.copy()
            for L in range(p):
                y_hat += A_mats[L] @ lag_buffer[L]
            forecasts.append(y_hat)
            # Update lag buffer: new y becomes lag 1, shift others
            lag_buffer = [y_hat] + lag_buffer[:-1]

        forecasts = np.vstack(forecasts)  # (steps, n)

        # Compute MA coefficients Psi_i up to horizon-1 for forecast error variance
        Psi = [np.eye(n)]
        for i in range(1, steps):
            psi_i = np.zeros((n, n))
            for k in range(1, min(i, p) + 1):
                psi_i += Psi[i - k] @ A_mats[k - 1]
            Psi.append(psi_i)

        # Forecast error covariance at each horizon: S_h = sum_{i=0}^{h-1} Psi_i Sigma Psi_i'
        S_list = []
        for h in range(1, steps + 1):
            S_h = np.zeros((n, n))
            for i in range(h):
                S_h += Psi[i] @ Sigma @ Psi[i].T
            S_list.append(S_h)

        # Confidence intervals
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        lower = []
        upper = []
        for h in range(steps):
            std_h = np.sqrt(np.diag(S_list[h]))
            lower.append(forecasts[h] - z * std_h)
            upper.append(forecasts[h] + z * std_h)
        lower = np.vstack(lower)
        upper = np.vstack(upper)

        # Build DataFrames with reasonable index/columns
        cols = self.prepared_data.columns
        # Try to create a date index continuing the data index
        idx = None
        try:
            last_ts = self.prepared_data.index[-1]
            freq = pd.infer_freq(self.prepared_data.index)
            if freq is not None:
                idx = pd.date_range(start=last_ts, periods=steps + 1, freq=freq)[1:]
        except Exception:
            idx = None
        if idx is None:
            idx = pd.RangeIndex(start=1, stop=steps + 1, step=1)

        forecasts_df = pd.DataFrame(forecasts, index=idx, columns=cols)
        lower_df = pd.DataFrame(lower, index=idx, columns=cols)
        upper_df = pd.DataFrame(upper, index=idx, columns=cols)

        return forecasts_df, lower_df, upper_df

    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary from the Bayesian model.
        """
        if self.results is None:
            raise ValueError("Model must be estimated first")
        
        # Extract summary statistics from the posterior draws
        beta_mean = np.mean(self.results.beta_draws, axis=0)
        sigma_mean = np.mean(self.results.sigma_draws, axis=0)
        
        # Calculate marginal likelihood approximation (Bayesian version of log-likelihood)
        # Using posterior mean as point estimate
        try:
            Y, X = self.results.Y, self.results.X
            residuals = Y - X @ beta_mean.reshape(X.shape[1], -1)
            log_likelihood = -0.5 * (Y.shape[0] * Y.shape[1] * np.log(2 * np.pi) + 
                                   Y.shape[0] * np.log(np.linalg.det(sigma_mean)) + 
                                   np.trace(residuals.T @ residuals @ np.linalg.inv(sigma_mean)))
        except:
            log_likelihood = np.nan  # Fallback if calculation fails
        
        # Bayesian Information Criteria approximations
        n_params = self.results.beta_draws.shape[1] + self.results.n_vars * (self.results.n_vars + 1) // 2
        n_obs = self.results.Y.shape[0]
        
        # Use marginal likelihood approximation for AIC/BIC
        aic = -2 * log_likelihood + 2 * n_params if not np.isnan(log_likelihood) else np.nan
        bic = -2 * log_likelihood + np.log(n_obs) * n_params if not np.isnan(log_likelihood) else np.nan
        
        # Prefer prepared_data/names for accuracy
        var_names = getattr(self.results, 'names', None)
        if var_names is None:
            var_names = list(self.data.columns)
        summary = {
            'n_variables': self.results.n_vars,
            'n_observations': self.results.Y.shape[0],
            'n_lags': self.results.n_lags,
            'n_draws': self.results.beta_draws.shape[0],
            'n_parameters': n_params,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'variable_names': var_names
        }
        
        return summary

def estimate_var_model(data: pd.DataFrame, max_lags: int = 8, 
                      start_date: str = None, end_date: str = None, 
                      n_draws: int = 5000, n_burnin: int = 500) -> BayesianVAR:
    """
    Convenience function to estimate VAR model
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    max_lags : int
        Maximum lags to consider
    start_date : str, optional
        Start date for estimation
    end_date : str, optional
        End date for estimation
    n_draws : int
        Total number of Gibbs draws.
    n_burnin : int
        Number of burn-in draws to discard.
        
    Returns:
    --------
    BayesianVAR
        Estimated VAR model
    """
    # Enable fast mode for datasets with many variables
    fast_mode = data.shape[1] > 50
    
    # Adjust burn-in for fast mode to prevent errors
    if fast_mode:
        n_draws = min(n_draws, 500)
        n_burnin = min(n_burnin, n_draws // 3)
    
    var_model = BayesianVAR(data, max_lags=max_lags, fast_mode=fast_mode)
    var_model.estimate(start_date=start_date, end_date=end_date, n_draws=n_draws, n_burnin=n_burnin)
    
    return var_model
