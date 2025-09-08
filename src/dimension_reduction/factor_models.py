"""
Dynamic factor models for dimension reduction in large VAR systems.

This module implements factor model approaches for handling high-dimensional
data by extracting common factors that drive the dynamics of many variables.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import chi2
from typing import Tuple, Optional, Dict, List
import warnings


class DynamicFactorModel:
    """
    Dynamic factor model for dimension reduction in large datasets.
    
    The model assumes:
    X_t = Λ F_t + u_t
    F_t = A F_{t-1} + η_t
    
    where F_t are r common factors, Λ is the loading matrix,
    and u_t are idiosyncratic errors.
    """
    
    def __init__(self, n_factors: int, n_lags: int = 1, 
                 standardize: bool = True, method: str = 'pc'):
        """
        Initialize dynamic factor model.
        
        Parameters:
        -----------
        n_factors : int
            Number of common factors
        n_lags : int
            Number of lags in factor VAR
        standardize : bool
            Whether to standardize data before factor extraction
        method : str
            Factor extraction method ('pc', 'ml', 'sw')
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.standardize = standardize
        self.method = method
        
        # Model components
        self.factors = None
        self.loadings = None
        self.factor_var_coeffs = None
        self.factor_var_sigma = None
        self.idiosyncratic_vars = None
        self.scaler = StandardScaler() if standardize else None
        
        # Fit diagnostics
        self.explained_variance_ratio = None
        self.ic_criteria = {}
    
    def fit(self, X: np.ndarray, max_factors: Optional[int] = None) -> 'DynamicFactorModel':
        """
        Fit dynamic factor model to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix (T x N)
        max_factors : int, optional
            Maximum factors to consider for selection
            
        Returns:
        --------
        DynamicFactorModel
            Fitted model
        """
        T, N = X.shape
        
        # Standardize if requested
        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Select number of factors if not specified
        if max_factors is not None:
            self.n_factors = self._select_factors(X_scaled, max_factors)
        
        # Extract factors based on method
        if self.method == 'pc':
            self._extract_factors_pc(X_scaled)
        elif self.method == 'ml':
            self._extract_factors_ml(X_scaled)
        elif self.method == 'sw':
            self._extract_factors_sw(X_scaled)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Estimate factor VAR
        self._estimate_factor_var()
        
        # Compute diagnostics
        self._compute_diagnostics(X_scaled)
        
        return self
    
    def _extract_factors_pc(self, X: np.ndarray):
        """Extract factors using principal components."""
        pca = PCA(n_components=self.n_factors)
        self.factors = pca.fit_transform(X)
        self.loadings = pca.components_.T * np.sqrt(X.shape[0] - 1)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
    
    def _extract_factors_ml(self, X: np.ndarray):
        """Extract factors using maximum likelihood."""
        fa = FactorAnalysis(n_components=self.n_factors, max_iter=1000)
        self.factors = fa.fit_transform(X)
        self.loadings = fa.components_.T
        self.idiosyncratic_vars = fa.noise_variance_
    
    def _extract_factors_sw(self, X: np.ndarray):
        """Extract factors using Stock & Watson (2002) method."""
        # This implements the two-step approach
        # Step 1: Initial factor estimation via PC
        self._extract_factors_pc(X)
        
        # Step 2: Re-estimate loadings via regression
        for i in range(X.shape[1]):
            # Regress each series on factors
            beta_i = np.linalg.lstsq(self.factors, X[:, i], rcond=None)[0]
            self.loadings[i, :] = beta_i
        
        # Re-extract factors with updated loadings
        self.factors = X @ self.loadings @ np.linalg.inv(self.loadings.T @ self.loadings)
    
    def _estimate_factor_var(self):
        """Estimate VAR model for extracted factors."""
        if self.factors is None:
            raise ValueError("Factors must be extracted first")
        
        T, r = self.factors.shape
        
        # Create lagged factor matrix
        F_lagged = np.zeros((T - self.n_lags, r * self.n_lags + 1))
        
        for lag in range(self.n_lags):
            start_col = lag * r
            end_col = (lag + 1) * r
            F_lagged[:, start_col:end_col] = self.factors[self.n_lags - 1 - lag : T - 1 - lag]
        
        # Add constant
        F_lagged[:, -1] = 1
        
        # Dependent variable
        F_current = self.factors[self.n_lags:]
        
        # OLS estimation
        self.factor_var_coeffs = np.linalg.lstsq(F_lagged, F_current, rcond=None)[0]
        
        # Residual covariance
        residuals = F_current - F_lagged @ self.factor_var_coeffs
        self.factor_var_sigma = (residuals.T @ residuals) / (residuals.shape[0] - F_lagged.shape[1])
    
    def _select_factors(self, X: np.ndarray, max_factors: int) -> int:
        """Select optimal number of factors using information criteria."""
        T, N = X.shape
        ic_values = {}
        
        for r in range(1, min(max_factors + 1, min(T, N) // 2)):
            # Fit model with r factors
            pca = PCA(n_components=r)
            F_r = pca.fit_transform(X)
            Lambda_r = pca.components_.T * np.sqrt(T - 1)
            
            # Compute fit
            X_fitted = F_r @ Lambda_r.T
            residuals = X - X_fitted
            sigma2 = np.mean(residuals**2)
            
            # Information criteria (Bai & Ng, 2002)
            penalty_1 = (N + T) / (N * T) * np.log(N * T / (N + T))
            penalty_2 = (N + T) / (N * T) * np.log(min(N, T))
            
            ic_values[r] = {
                'IC1': np.log(sigma2) + r * penalty_1,
                'IC2': np.log(sigma2) + r * penalty_2,
                'sigma2': sigma2
            }
        
        # Select based on IC2 (more parsimonious)
        optimal_factors = min(ic_values.keys(), key=lambda x: ic_values[x]['IC2'])
        self.ic_criteria = ic_values
        
        return optimal_factors
    
    def _compute_diagnostics(self, X: np.ndarray):
        """Compute model diagnostics."""
        if self.factors is None or self.loadings is None:
            return
        
        # Reconstruction error
        X_fitted = self.factors @ self.loadings.T
        mse = np.mean((X - X_fitted)**2)
        self.reconstruction_mse = mse
        
        # Factor loadings significance (rough approximation)
        T = X.shape[0]
        loading_std_errors = np.sqrt(np.diag(np.linalg.inv(self.factors.T @ self.factors)) * mse)
        self.loading_t_stats = self.loadings.T / loading_std_errors[:, np.newaxis]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data to factor space.
        
        Parameters:
        -----------
        X : np.ndarray
            New data to transform
            
        Returns:
        --------
        np.ndarray
            Factor scores
        """
        if self.loadings is None:
            raise ValueError("Model must be fitted first")
        
        if self.standardize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Project onto factor space
        factors = X_scaled @ self.loadings @ np.linalg.inv(self.loadings.T @ self.loadings)
        return factors
    
    def forecast_factors(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast factors using the estimated VAR model.
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Factor forecasts and confidence intervals
        """
        if self.factor_var_coeffs is None:
            raise ValueError("Factor VAR must be estimated first")
        
        r = self.factors.shape[1]
        
        # Get last observations for initialization
        F_last = self.factors[-self.n_lags:]
        
        # Forecast recursively
        forecasts = np.zeros((steps, r))
        forecast_vars = np.zeros((steps, r))
        
        # Build companion form
        A_companion = self._build_companion_matrix()
        
        # Current state (stacked lags + constant)
        state = np.concatenate([F_last.flatten(), [1]])
        
        for h in range(steps):
            # One-step forecast
            forecast_state = A_companion @ state
            forecasts[h] = forecast_state[:r]
            
            # Update state for next forecast
            if self.n_lags > 1:
                state = np.concatenate([forecast_state[:r * self.n_lags], [1]])
            else:
                state = np.concatenate([forecast_state[:r], [1]])
            
            # Forecast variance (accumulated)
            forecast_vars[h] = np.diag(self.factor_var_sigma) * (h + 1)
        
        # Confidence intervals (95%)
        confidence_intervals = 1.96 * np.sqrt(forecast_vars)
        
        return forecasts, confidence_intervals
    
    def _build_companion_matrix(self) -> np.ndarray:
        """Build companion form matrix for factor VAR."""
        r = self.factors.shape[1]
        
        if self.n_lags == 1:
            return self.factor_var_coeffs
        
        # Build companion form for higher-order VAR
        companion_size = r * self.n_lags + 1
        A_companion = np.zeros((companion_size, companion_size))
        
        # First block row
        A_companion[:r, :] = self.factor_var_coeffs.T
        
        # Identity blocks for lagged terms
        for i in range(1, self.n_lags):
            start_row = i * r
            end_row = (i + 1) * r
            start_col = (i - 1) * r
            end_col = i * r
            A_companion[start_row:end_row, start_col:end_col] = np.eye(r)
        
        # Constant term handling
        A_companion[-1, -1] = 1
        
        return A_companion
    
    def forecast_variables(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast original variables using factor forecasts.
        
        Parameters:
        -----------
        steps : int
            Forecast horizon
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Variable forecasts and confidence intervals
        """
        factor_forecasts, factor_conf_int = self.forecast_factors(steps)
        
        # Transform back to variable space
        var_forecasts = factor_forecasts @ self.loadings.T
        
        # Approximate confidence intervals
        # (This is a rough approximation - proper intervals require more complex calculation)
        var_conf_int = factor_conf_int @ np.abs(self.loadings.T)
        
        if self.standardize:
            var_forecasts = self.scaler.inverse_transform(var_forecasts)
            var_conf_int = var_conf_int * self.scaler.scale_
        
        return var_forecasts, var_conf_int


class SparseDynamicFactorModel(DynamicFactorModel):
    """
    Sparse dynamic factor model with automatic variable selection.
    
    This model allows for sparse loadings, automatically determining
    which variables load on which factors.
    """
    
    def __init__(self, n_factors: int, n_lags: int = 1, 
                 sparsity_penalty: float = 0.1, **kwargs):
        """
        Initialize sparse dynamic factor model.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors
        n_lags : int
            VAR lags for factors
        sparsity_penalty : float
            L1 penalty parameter for sparse loadings
        """
        super().__init__(n_factors, n_lags, **kwargs)
        self.sparsity_penalty = sparsity_penalty
        self.sparse_loadings = None
    
    def fit(self, X: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> 'SparseDynamicFactorModel':
        """
        Fit sparse dynamic factor model using EM algorithm.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
        max_iter : int
            Maximum iterations for EM
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        SparseDynamicFactorModel
            Fitted sparse model
        """
        T, N = X.shape
        
        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Initialize with regular factor model
        super().fit(X_scaled)
        
        # Sparse loading estimation via coordinate descent
        self.sparse_loadings = self._estimate_sparse_loadings(
            X_scaled, max_iter, tol
        )
        
        # Re-estimate factors with sparse loadings
        self.factors = self._estimate_factors_given_loadings(X_scaled, self.sparse_loadings)
        
        # Re-estimate factor VAR
        self._estimate_factor_var()
        
        return self
    
    def _estimate_sparse_loadings(self, X: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        """Estimate sparse loadings using coordinate descent."""
        T, N = X.shape
        r = self.n_factors
        
        # Initialize with dense loadings
        loadings = self.loadings.copy()
        
        for iteration in range(max_iter):
            loadings_old = loadings.copy()
            
            for i in range(N):
                for j in range(r):
                    # Partial residual for variable i, factor j
                    partial_resid = X[:, i] - np.sum(
                        self.factors[:, k] * loadings[i, k] 
                        for k in range(r) if k != j
                    )
                    
                    # Soft thresholding
                    numerator = np.dot(self.factors[:, j], partial_resid)
                    denominator = np.dot(self.factors[:, j], self.factors[:, j])
                    
                    if denominator > 0:
                        loadings[i, j] = self._soft_threshold(
                            numerator / denominator, 
                            self.sparsity_penalty / denominator
                        )
                    else:
                        loadings[i, j] = 0
            
            # Check convergence
            if np.max(np.abs(loadings - loadings_old)) < tol:
                break
        
        return loadings
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator for L1 penalty."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def _estimate_factors_given_loadings(self, X: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        """Re-estimate factors given sparse loadings."""
        # Weighted least squares
        Lambda_Lambda = loadings.T @ loadings
        if np.linalg.cond(Lambda_Lambda) < 1e12:
            factors = X @ loadings @ np.linalg.inv(Lambda_Lambda)
        else:
            # Use pseudo-inverse for ill-conditioned case
            factors = X @ loadings @ np.linalg.pinv(Lambda_Lambda)
        
        return factors
    
    def get_active_loadings(self, threshold: float = 1e-3) -> Dict[int, List[int]]:
        """
        Get active (non-zero) loadings for interpretation.
        
        Parameters:
        -----------
        threshold : float
            Threshold for considering loadings as non-zero
            
        Returns:
        --------
        Dict[int, List[int]]
            Dictionary mapping factors to lists of active variables
        """
        if self.sparse_loadings is None:
            raise ValueError("Sparse model must be fitted first")
        
        active_loadings = {}
        for j in range(self.n_factors):
            active_vars = np.where(np.abs(self.sparse_loadings[:, j]) > threshold)[0]
            active_loadings[j] = active_vars.tolist()
        
        return active_loadings


def estimate_factors_bai_ng(X: np.ndarray, max_factors: int = 10) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Estimate number of factors using Bai & Ng (2002) criteria.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix (T x N)
    max_factors : int
        Maximum number of factors to consider
        
    Returns:
    --------
    Tuple[int, np.ndarray, np.ndarray]
        Optimal number of factors, factors, and loadings
    """
    T, N = X.shape
    
    # Standardize data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    ic_values = {}
    
    for r in range(1, min(max_factors + 1, min(T, N) // 4)):
        # Extract r factors
        pca = PCA(n_components=r)
        F_r = pca.fit_transform(X_std)
        Lambda_r = pca.components_.T * np.sqrt(T - 1)
        
        # Residual sum of squares
        X_fitted = F_r @ Lambda_r.T
        residuals = X_std - X_fitted
        V_NT = np.sum(residuals**2) / (N * T)
        
        # Penalty functions
        g_NT = min(N, T)
        penalty_functions = {
            'IC1': ((N + T) / (N * T)) * np.log((N * T) / (N + T)),
            'IC2': ((N + T) / (N * T)) * np.log(g_NT),
            'IC3': (np.log(g_NT) / g_NT),
            'BIC': (np.log(N * T) / (N * T))
        }
        
        ic_values[r] = {
            criterion: np.log(V_NT) + r * penalty
            for criterion, penalty in penalty_functions.items()
        }
        ic_values[r]['V_NT'] = V_NT
    
    # Select optimal number based on IC2 (recommended by Bai & Ng)
    optimal_r = min(ic_values.keys(), key=lambda x: ic_values[x]['IC2'])
    
    # Re-estimate with optimal number
    pca_final = PCA(n_components=optimal_r)
    factors_final = pca_final.fit_transform(X_std)
    loadings_final = pca_final.components_.T * np.sqrt(T - 1)
    
    return optimal_r, factors_final, loadings_final