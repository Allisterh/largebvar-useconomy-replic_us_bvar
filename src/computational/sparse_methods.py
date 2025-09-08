"""
Sparse methods for very large Bayesian VAR systems.

This module implements sparse matrix methods for handling VAR systems with
200+ variables where dense methods become computationally prohibitive.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import solve_triangular
from typing import Tuple, Optional, Dict, List
import warnings


class SparseVAR:
    """
    Sparse implementation of Large Bayesian VAR for very high-dimensional systems.
    
    This class uses sparse matrix methods to handle VAR systems with hundreds
    of variables by exploiting sparsity in the coefficient matrices and 
    implementing specialized algorithms for sparse Bayesian inference.
    """
    
    def __init__(self, n_vars: int, n_lags: int, sparsity_threshold: float = 1e-6):
        """
        Initialize sparse VAR system.
        
        Parameters:
        -----------
        n_vars : int
            Number of variables
        n_lags : int  
            Number of lags
        sparsity_threshold : float
            Threshold below which coefficients are considered zero
        """
        self.n_vars = n_vars
        self.n_lags = n_lags
        self.sparsity_threshold = sparsity_threshold
        
        # Dimensions
        self.n_coeffs = n_vars * n_lags + 1  # +1 for constant
        self.n_total_coeffs = self.n_coeffs * n_vars
        
        # Sparse structures (to be initialized)
        self.sparse_design = None
        self.sparse_precision = None
        self.sparse_coeffs = None
    
    def create_sparse_design_matrix(self, Y: np.ndarray) -> csr_matrix:
        """
        Create sparse design matrix for VAR regression.
        
        Parameters:
        -----------
        Y : np.ndarray
            Time series data (T x n_vars)
            
        Returns:
        --------
        csr_matrix
            Sparse design matrix
        """
        T, n_vars = Y.shape
        T_eff = T - self.n_lags
        
        # Initialize sparse matrix components
        row_indices = []
        col_indices = []
        data_values = []
        
        # Build design matrix efficiently
        for t in range(T_eff):
            row_base = t
            
            # Lagged variables
            for lag in range(self.n_lags):
                for var in range(n_vars):
                    if abs(Y[self.n_lags - 1 - lag + t, var]) > self.sparsity_threshold:
                        row_indices.append(row_base)
                        col_indices.append(lag * n_vars + var)
                        data_values.append(Y[self.n_lags - 1 - lag + t, var])
            
            # Constant term
            row_indices.append(row_base)
            col_indices.append(self.n_lags * n_vars)
            data_values.append(1.0)
        
        # Create sparse matrix
        X_sparse = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(T_eff, self.n_coeffs)
        )
        
        return X_sparse
    
    def sparse_minnesota_prior(self, sigma_ols: np.ndarray, 
                             lambda1: float = 0.1, lambda2: float = 0.5, 
                             lambda3: float = 1.0) -> Tuple[np.ndarray, csr_matrix]:
        """
        Create sparse Minnesota prior precision matrix.
        
        The Minnesota prior imposes that most cross-variable effects are zero,
        making it naturally sparse.
        
        Parameters:
        -----------
        sigma_ols : np.ndarray
            OLS residual variances for scaling
        lambda1 : float
            Overall shrinkage parameter
        lambda2 : float
            Cross-variable shrinkage parameter  
        lambda3 : float
            Exogenous variable shrinkage parameter
            
        Returns:
        --------
        Tuple[np.ndarray, csr_matrix]
            Prior mean and sparse precision matrix
        """
        # Prior mean (sparse: only own lags have non-zero means)
        prior_mean = np.zeros(self.n_total_coeffs)
        
        # Sparse precision matrix components
        row_indices = []
        col_indices = []
        precision_values = []
        
        for eq in range(self.n_vars):
            eq_offset = eq * self.n_coeffs
            sigma_eq = sigma_ols[eq]
            
            # Own lags (diagonal dominance)
            for lag in range(1, self.n_lags + 1):
                for var in range(self.n_vars):
                    coeff_idx = eq_offset + (lag - 1) * self.n_vars + var
                    
                    if var == eq:  # Own lag
                        # Prior mean: AR(1) assumption
                        if lag == 1:
                            prior_mean[coeff_idx] = 1.0
                        
                        # Prior precision
                        precision_val = 1.0 / ((lambda1 / lag)**2 * sigma_eq)
                    else:  # Cross lag
                        precision_val = 1.0 / ((lambda1 * lambda2 / lag)**2 * sigma_eq * sigma_ols[var])
                    
                    # Add to sparse structure only if significant
                    if precision_val > self.sparsity_threshold:
                        row_indices.append(coeff_idx)
                        col_indices.append(coeff_idx)
                        precision_values.append(precision_val)
            
            # Constant term
            const_idx = eq_offset + self.n_lags * self.n_vars
            precision_val = 1.0 / (lambda3**2 * sigma_eq)
            
            row_indices.append(const_idx)
            col_indices.append(const_idx)
            precision_values.append(precision_val)
        
        # Create sparse precision matrix
        sparse_precision = csr_matrix(
            (precision_values, (row_indices, col_indices)),
            shape=(self.n_total_coeffs, self.n_total_coeffs)
        )
        
        return prior_mean, sparse_precision
    
    def sparse_posterior_moments(self, Y: np.ndarray, X_sparse: csr_matrix,
                                Sigma: np.ndarray, prior_mean: np.ndarray,
                                prior_precision: csr_matrix) -> Tuple[np.ndarray, csr_matrix]:
        """
        Compute posterior moments using sparse methods.
        
        Parameters:
        -----------
        Y : np.ndarray
            Response matrix
        X_sparse : csr_matrix
            Sparse design matrix
        Sigma : np.ndarray
            Error covariance matrix
        prior_mean : np.ndarray
            Prior mean vector
        prior_precision : csr_matrix
            Sparse prior precision matrix
            
        Returns:
        --------
        Tuple[np.ndarray, csr_matrix]
            Posterior mean and sparse precision matrix
        """
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Compute X'X efficiently for sparse X
        XtX = X_sparse.T @ X_sparse
        
        # Posterior precision: Sigma^{-1} ⊗ X'X + prior_precision
        # Use sparse Kronecker product
        data_precision = kron(Sigma_inv, XtX, format='csr')
        post_precision = data_precision + prior_precision
        
        # Compute X'Y
        XtY = X_sparse.T @ Y
        XtY_vec = XtY.flatten('F')
        
        # Prior contribution
        prior_term = prior_precision @ prior_mean
        
        # Solve for posterior mean using sparse solver
        post_mean = spsolve(post_precision, XtY_vec + prior_term)
        
        return post_mean, post_precision
    
    def sparse_cholesky_factor(self, A_sparse: csr_matrix) -> csr_matrix:
        """
        Compute sparse Cholesky factorization with fill-reducing ordering.
        
        Parameters:
        -----------
        A_sparse : csr_matrix
            Sparse positive definite matrix
            
        Returns:
        --------
        csr_matrix
            Sparse lower triangular Cholesky factor
        """
        try:
            from scikits.sparse.cholmod import cholesky
            factor = cholesky(A_sparse)
            return factor.L()
        except ImportError:
            # Fallback to scipy's incomplete Cholesky
            warnings.warn("CHOLMOD not available, using incomplete Cholesky")
            return self._incomplete_cholesky(A_sparse)
    
    def _incomplete_cholesky(self, A_sparse: csr_matrix, 
                           max_rank: Optional[int] = None) -> csr_matrix:
        """
        Incomplete Cholesky factorization for large sparse matrices.
        
        Parameters:
        -----------
        A_sparse : csr_matrix
            Sparse matrix to factor
        max_rank : int, optional
            Maximum rank for approximation
            
        Returns:
        --------
        csr_matrix
            Incomplete Cholesky factor
        """
        if max_rank is None:
            max_rank = min(A_sparse.shape[0] // 10, 1000)
        
        n = A_sparse.shape[0]
        
        # Use randomized methods for very large matrices
        if n > 5000:
            return self._randomized_cholesky(A_sparse, max_rank)
        
        # Standard incomplete Cholesky for moderate sizes
        L_data = []
        L_indices = []
        L_indptr = [0]
        
        for i in range(min(n, max_rank)):
            # Extract column
            col = A_sparse[:, i].toarray().flatten()
            
            # Apply previous factors
            for j in range(i):
                L_j = L_data[L_indptr[j]:L_indptr[j+1]]
                L_j_indices = L_indices[L_indptr[j]:L_indptr[j+1]]
                
                # Update column
                for k, idx in enumerate(L_j_indices):
                    if idx <= i:
                        col[idx:] -= L_j[k] * L_j[k:]
            
            # Normalize
            if col[i] > 0:
                L_ii = np.sqrt(col[i])
                col[i:] /= L_ii
                
                # Store non-zeros
                nonzero_mask = np.abs(col[i:]) > self.sparsity_threshold
                nonzero_indices = np.where(nonzero_mask)[0] + i
                nonzero_values = col[nonzero_indices]
                
                L_data.extend(nonzero_values)
                L_indices.extend(nonzero_indices)
            
            L_indptr.append(len(L_data))
        
        return csr_matrix((L_data, L_indices, L_indptr), shape=(n, max_rank))
    
    def _randomized_cholesky(self, A_sparse: csr_matrix, rank: int) -> csr_matrix:
        """
        Randomized incomplete Cholesky for very large matrices.
        
        Parameters:
        -----------
        A_sparse : csr_matrix
            Sparse matrix
        rank : int
            Target rank
            
        Returns:
        --------
        csr_matrix
            Randomized Cholesky factor
        """
        n = A_sparse.shape[0]
        
        # Random sampling matrix
        Omega = np.random.randn(n, rank * 2)
        
        # Power iteration for accuracy
        Y = A_sparse @ Omega
        for _ in range(2):
            Y = A_sparse @ (A_sparse.T @ Y)
        
        # QR decomposition
        Q, _ = np.linalg.qr(Y)
        
        # Compute B = Q^T A Q
        B = Q.T @ A_sparse @ Q
        
        # Cholesky of reduced system
        try:
            L_B = np.linalg.cholesky(B[:rank, :rank])
            L = Q[:, :rank] @ L_B
            return csr_matrix(L)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition
            eigvals, eigvecs = eigsh(B, k=min(rank, B.shape[0] - 1))
            L = Q @ eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
            return csr_matrix(L)


class AdaptiveSparsity:
    """
    Adaptive sparsity methods for automatic relevance determination in large VARs.
    
    This class implements methods to automatically determine which coefficients
    should be set to zero, reducing model complexity and improving interpretation.
    """
    
    def __init__(self, sparsity_prior: str = 'horseshoe'):
        """
        Initialize adaptive sparsity model.
        
        Parameters:
        -----------
        sparsity_prior : str
            Type of sparsity prior ('horseshoe', 'spike_slab', 'lasso')
        """
        self.sparsity_prior = sparsity_prior
    
    def horseshoe_prior_precision(self, n_coeffs: int, tau: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate precision matrix for Horseshoe prior.
        
        The Horseshoe prior provides adaptive shrinkage where small coefficients
        are shrunk aggressively while large coefficients are preserved.
        
        Parameters:
        -----------
        n_coeffs : int
            Number of coefficients
        tau : float
            Global shrinkage parameter
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Local shrinkage parameters and precision diagonal
        """
        # Sample local shrinkage parameters
        lambda_local = np.random.exponential(1.0, n_coeffs)
        
        # Compute precision diagonal
        precision_diag = 1.0 / (tau**2 * lambda_local**2)
        
        return lambda_local, precision_diag
    
    def spike_and_slab_prior(self, n_coeffs: int, 
                           spike_precision: float = 100.0,
                           slab_precision: float = 0.1,
                           inclusion_prob: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Spike-and-Slab prior for variable selection.
        
        Parameters:
        -----------
        n_coeffs : int
            Number of coefficients
        spike_precision : float
            Precision for spike (near-zero) component
        slab_precision : float
            Precision for slab (non-zero) component  
        inclusion_prob : float
            Prior probability of inclusion
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Inclusion indicators and precision values
        """
        # Sample inclusion indicators
        indicators = np.random.binomial(1, inclusion_prob, n_coeffs)
        
        # Compute precision values
        precision_vals = indicators * slab_precision + (1 - indicators) * spike_precision
        
        return indicators, precision_vals
    
    def adaptive_lasso_weights(self, coeffs_ols: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Compute adaptive LASSO penalty weights.
        
        Parameters:
        -----------
        coeffs_ols : np.ndarray
            OLS coefficient estimates
        gamma : float
            Adaptation parameter
            
        Returns:
        --------
        np.ndarray
            Adaptive penalty weights
        """
        weights = 1.0 / (np.abs(coeffs_ols) + 1e-8)**gamma
        return weights


def sparse_gibbs_step(Y: np.ndarray, X_sparse: csr_matrix, 
                     Sigma: np.ndarray, prior_mean: np.ndarray,
                     prior_precision: csr_matrix) -> np.ndarray:
    """
    Single Gibbs step for sparse VAR coefficient sampling.
    
    Parameters:
    -----------
    Y : np.ndarray
        Response matrix
    X_sparse : csr_matrix
        Sparse design matrix
    Sigma : np.ndarray
        Current error covariance
    prior_mean : np.ndarray
        Prior mean
    prior_precision : csr_matrix
        Sparse prior precision
        
    Returns:
    --------
    np.ndarray
        New coefficient sample
    """
    # Compute posterior moments
    sparse_var = SparseVAR(Y.shape[1], X_sparse.shape[1] // Y.shape[1] - 1)
    post_mean, post_precision = sparse_var.sparse_posterior_moments(
        Y, X_sparse, Sigma, prior_mean, prior_precision
    )
    
    # Sample from posterior using sparse methods
    # For very large systems, use approximate sampling
    if post_precision.shape[0] > 10000:
        return approximate_sparse_sample(post_mean, post_precision)
    else:
        # Exact sampling for moderate sizes
        post_chol = sparse_var.sparse_cholesky_factor(post_precision)
        z = np.random.standard_normal(len(post_mean))
        return post_mean + spsolve(post_chol, z)


def approximate_sparse_sample(mean: np.ndarray, precision: csr_matrix,
                            n_iterations: int = 100) -> np.ndarray:
    """
    Approximate sampling from high-dimensional sparse Gaussian using conjugate gradients.
    
    Parameters:
    -----------
    mean : np.ndarray
        Mean vector
    precision : csr_matrix
        Sparse precision matrix
    n_iterations : int
        Number of CG iterations
        
    Returns:
    --------
    np.ndarray
        Approximate sample
    """
    from scipy.sparse.linalg import cg
    
    # Generate random right-hand side
    z = np.random.standard_normal(len(mean))
    
    # Solve precision @ x = z using conjugate gradients
    x, _ = cg(precision, z, maxiter=n_iterations, atol=1e-6)
    
    return mean + x