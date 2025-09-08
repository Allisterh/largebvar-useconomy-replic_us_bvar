"""
Matrix Normal Distribution utilities for efficient VAR computation.

This module implements efficient sampling and density evaluation for matrix normal
distributions, which are central to Bayesian VAR estimation.
"""

import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.stats import matrix_normal as scipy_matrix_normal
from typing import Tuple, Optional
import numba
from numba import jit, prange


class MatrixNormal:
    """
    Efficient implementation of Matrix Normal distribution operations.
    
    A matrix X follows a Matrix Normal distribution MN(M, U, V) if:
    vec(X) ~ N(vec(M), V ⊗ U)
    
    where:
    - M is the mean matrix (m x n)
    - U is the row covariance matrix (m x m) 
    - V is the column covariance matrix (n x n)
    """
    
    def __init__(self, mean: np.ndarray, row_cov: np.ndarray, col_cov: np.ndarray):
        """
        Initialize Matrix Normal distribution.
        
        Parameters:
        -----------
        mean : np.ndarray
            Mean matrix (m x n)
        row_cov : np.ndarray
            Row covariance matrix (m x m)
        col_cov : np.ndarray  
            Column covariance matrix (n x n)
        """
        self.mean = mean
        self.row_cov = row_cov
        self.col_cov = col_cov
        
        # Precompute Cholesky factors for efficiency
        self.row_chol = cholesky(row_cov, lower=True)
        self.col_chol = cholesky(col_cov, lower=True)
        
        # Cache inverse and determinant for density evaluation
        self._row_cov_inv = None
        self._col_cov_inv = None
        self._log_det_cache = None
    
    @property
    def row_cov_inv(self) -> np.ndarray:
        """Cached inverse of row covariance."""
        if self._row_cov_inv is None:
            self._row_cov_inv = np.linalg.inv(self.row_cov)
        return self._row_cov_inv
    
    @property
    def col_cov_inv(self) -> np.ndarray:
        """Cached inverse of column covariance."""
        if self._col_cov_inv is None:
            self._col_cov_inv = np.linalg.inv(self.col_cov)
        return self._col_cov_inv
    
    @property
    def log_det(self) -> float:
        """Cached log determinant for density evaluation."""
        if self._log_det_cache is None:
            # log|V ⊗ U| = n*log|U| + m*log|V|
            m, n = self.mean.shape
            log_det_row = 2 * np.sum(np.log(np.diag(self.row_chol)))
            log_det_col = 2 * np.sum(np.log(np.diag(self.col_chol)))
            self._log_det_cache = n * log_det_row + m * log_det_col
        return self._log_det_cache
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate samples from the Matrix Normal distribution.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray
            Samples with shape (size, m, n)
        """
        m, n = self.mean.shape
        
        if size == 1:
            # Generate standard normal matrix
            Z = np.random.standard_normal((m, n))
            # Transform: M + L_U @ Z @ L_V^T
            return self.mean + self.row_chol @ Z @ self.col_chol.T
        else:
            # Batch sampling for efficiency
            return self._batch_sample(size)
    
    def _batch_sample(self, size: int) -> np.ndarray:
        """Efficient batch sampling."""
        m, n = self.mean.shape
        samples = np.zeros((size, m, n))
        
        for i in range(size):
            Z = np.random.standard_normal((m, n))
            samples[i] = self.mean + self.row_chol @ Z @ self.col_chol.T
            
        return samples
    
    def logpdf(self, X: np.ndarray) -> float:
        """
        Compute log probability density.
        
        Parameters:
        -----------
        X : np.ndarray
            Matrix to evaluate (m x n)
            
        Returns:
        --------
        float
            Log probability density
        """
        m, n = self.mean.shape
        
        # Center the matrix
        X_centered = X - self.mean
        
        # Compute quadratic form: tr(V^{-1} @ X_centered.T @ U^{-1} @ X_centered)
        temp = solve_triangular(self.row_chol, X_centered, lower=True)
        temp2 = solve_triangular(self.col_chol, temp.T, lower=True)
        quad_form = np.sum(temp2**2)
        
        # Log density
        const = -0.5 * m * n * np.log(2 * np.pi)
        log_det_term = -0.5 * self.log_det
        quad_term = -0.5 * quad_form
        
        return const + log_det_term + quad_term
    
    def pdf(self, X: np.ndarray) -> float:
        """
        Compute probability density.
        
        Parameters:
        -----------
        X : np.ndarray
            Matrix to evaluate
            
        Returns:
        --------
        float
            Probability density
        """
        return np.exp(self.logpdf(X))


class EfficientMatrixNormalSampler:
    """
    High-performance sampler for Matrix Normal distributions in VAR context.
    
    This class is optimized for repeated sampling scenarios common in Gibbs sampling,
    where the covariance structure may change but the dimensions remain fixed.
    """
    
    def __init__(self, m: int, n: int):
        """
        Initialize sampler for fixed dimensions.
        
        Parameters:
        -----------
        m : int
            Number of rows
        n : int
            Number of columns
        """
        self.m = m
        self.n = n
        self._cached_chol_row = None
        self._cached_chol_col = None
        self._last_row_cov_hash = None
        self._last_col_cov_hash = None
    
    def _hash_matrix(self, A: np.ndarray) -> int:
        """Simple hash for matrix caching."""
        return hash(A.tobytes())
    
    def _get_cholesky_factors(self, row_cov: np.ndarray, 
                            col_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Cholesky factors with caching.
        
        Parameters:
        -----------
        row_cov : np.ndarray
            Row covariance matrix
        col_cov : np.ndarray
            Column covariance matrix
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Cholesky factors (L_row, L_col)
        """
        row_hash = self._hash_matrix(row_cov)
        col_hash = self._hash_matrix(col_cov)
        
        # Check cache for row covariance
        if (self._cached_chol_row is None or 
            self._last_row_cov_hash != row_hash):
            self._cached_chol_row = cholesky(row_cov, lower=True)
            self._last_row_cov_hash = row_hash
        
        # Check cache for column covariance  
        if (self._cached_chol_col is None or
            self._last_col_cov_hash != col_hash):
            self._cached_chol_col = cholesky(col_cov, lower=True)
            self._last_col_cov_hash = col_hash
            
        return self._cached_chol_row, self._cached_chol_col
    
    def sample(self, mean: np.ndarray, row_cov: np.ndarray, 
              col_cov: np.ndarray, size: int = 1) -> np.ndarray:
        """
        Generate samples efficiently with caching.
        
        Parameters:
        -----------
        mean : np.ndarray
            Mean matrix
        row_cov : np.ndarray
            Row covariance
        col_cov : np.ndarray
            Column covariance
        size : int
            Number of samples
            
        Returns:
        --------
        np.ndarray
            Samples
        """
        L_row, L_col = self._get_cholesky_factors(row_cov, col_cov)
        
        if size == 1:
            Z = np.random.standard_normal((self.m, self.n))
            return mean + L_row @ Z @ L_col.T
        else:
            samples = np.zeros((size, self.m, self.n))
            for i in range(size):
                Z = np.random.standard_normal((self.m, self.n))
                samples[i] = mean + L_row @ Z @ L_col.T
            return samples


@jit(nopython=True)
def matrix_normal_conjugate_update(prior_mean: np.ndarray, prior_row_cov: np.ndarray,
                                 prior_col_cov: np.ndarray, data: np.ndarray,
                                 data_row_cov: np.ndarray, data_col_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conjugate update for Matrix Normal distribution.
    
    Given:
    - Prior: X ~ MN(M0, U0, V0)
    - Data: Y|X ~ MN(X, U_data, V_data)
    
    Returns posterior parameters for X|Y ~ MN(M1, U1, V1)
    
    Parameters:
    -----------
    prior_mean : np.ndarray
        Prior mean matrix
    prior_row_cov : np.ndarray
        Prior row covariance
    prior_col_cov : np.ndarray
        Prior column covariance
    data : np.ndarray
        Observed data matrix
    data_row_cov : np.ndarray
        Data row covariance (likelihood)
    data_col_cov : np.ndarray
        Data column covariance (likelihood)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Posterior (mean, row_cov, col_cov)
    """
    # Compute precision matrices
    prior_row_prec = np.linalg.inv(prior_row_cov)
    prior_col_prec = np.linalg.inv(prior_col_cov)
    data_row_prec = np.linalg.inv(data_row_cov)
    data_col_prec = np.linalg.inv(data_col_cov)
    
    # Posterior precision matrices
    post_row_prec = prior_row_prec + data_row_prec
    post_col_prec = prior_col_prec + data_col_prec
    
    # Posterior covariance matrices
    post_row_cov = np.linalg.inv(post_row_prec)
    post_col_cov = np.linalg.inv(post_col_prec)
    
    # Posterior mean
    post_mean = post_row_cov @ (prior_row_prec @ prior_mean @ prior_col_prec + 
                               data_row_prec @ data @ data_col_prec) @ post_col_cov
    
    return post_mean, post_row_cov, post_col_cov


def efficient_matrix_normal_logpdf(X: np.ndarray, mean: np.ndarray,
                                 row_chol: np.ndarray, col_chol: np.ndarray) -> float:
    """
    Efficient log PDF computation using precomputed Cholesky factors.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrix to evaluate
    mean : np.ndarray
        Mean matrix  
    row_chol : np.ndarray
        Cholesky factor of row covariance (lower triangular)
    col_chol : np.ndarray
        Cholesky factor of column covariance (lower triangular)
        
    Returns:
    --------
    float
        Log probability density
    """
    m, n = X.shape
    
    # Center the matrix
    X_centered = X - mean
    
    # Solve triangular systems efficiently
    temp1 = solve_triangular(row_chol, X_centered, lower=True)
    temp2 = solve_triangular(col_chol, temp1.T, lower=True)
    
    # Quadratic form
    quad_form = np.sum(temp2**2)
    
    # Log determinant terms
    log_det_row = 2 * np.sum(np.log(np.diag(row_chol)))
    log_det_col = 2 * np.sum(np.log(np.diag(col_chol)))
    log_det = n * log_det_row + m * log_det_col
    
    # Constant term
    const = -0.5 * m * n * np.log(2 * np.pi)
    
    return const - 0.5 * log_det - 0.5 * quad_form