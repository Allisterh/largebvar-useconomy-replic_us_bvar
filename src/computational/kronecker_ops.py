"""
Efficient Kronecker product operations for large-scale VAR computation.

This module implements optimized Kronecker product operations that are essential
for computational efficiency in large Bayesian VAR models.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix
from typing import Union, Tuple
import numba
from numba import jit, prange


class KroneckerOperations:
    """
    Efficient Kronecker product operations for VAR computations.
    
    These operations are critical for computational efficiency when dealing with
    large VAR systems where traditional matrix operations become prohibitive.
    """
    
    @staticmethod
    def kron_vec_fast(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute (A ⊗ B) @ v efficiently without forming the full Kronecker product.
        
        This is much more memory-efficient than forming kron(A, B) explicitly.
        
        Parameters:
        -----------
        A : np.ndarray
            First matrix (m x n)
        B : np.ndarray  
            Second matrix (p x q)
        v : np.ndarray
            Vector to multiply (nq x 1)
            
        Returns:
        --------
        np.ndarray
            Result vector (mp x 1)
        """
        m, n = A.shape
        p, q = B.shape
        
        # Reshape v to matrix form
        V = v.reshape((n, q), order='F')
        
        # Compute B @ V.T @ A.T efficiently
        result = B @ V @ A.T
        
        # Flatten back to vector
        return result.flatten('F')
    
    @staticmethod
    def kron_solve_fast(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Solve (A ⊗ B) @ X = C efficiently using the matrix equation AXB^T = C.
        
        Parameters:
        -----------
        A : np.ndarray
            First matrix (m x m)
        B : np.ndarray
            Second matrix (n x n)  
        C : np.ndarray
            Right-hand side (mn x k)
            
        Returns:
        --------
        np.ndarray
            Solution X (mn x k)
        """
        m, _ = A.shape
        n, _ = B.shape
        k = C.shape[1]
        
        # Solve using the Sylvester equation approach
        X = np.zeros((m * n, k))
        
        for i in prange(k):
            c_mat = C[:, i].reshape((m, n), order='F')
            # Solve A @ x @ B.T = c_mat
            x_mat = np.linalg.solve(A, c_mat @ np.linalg.inv(B.T))
            X[:, i] = x_mat.flatten('F')
            
        return X
    
    @staticmethod
    def efficient_var_posterior(Y: np.ndarray, X: np.ndarray, 
                              Sigma: np.ndarray, prior_mean: np.ndarray, 
                              prior_precision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute VAR posterior moments efficiently using Kronecker structure.
        
        The VAR posterior has the form:
        vec(B) | Y, Sigma ~ N(vec(B_post), Sigma ⊗ V_post)
        
        where V_post^{-1} = X'X + prior_precision
        
        Parameters:
        -----------
        Y : np.ndarray
            Response matrix (T x n)
        X : np.ndarray  
            Design matrix (T x k)
        Sigma : np.ndarray
            Error covariance (n x n)
        prior_mean : np.ndarray
            Prior mean vec(B_prior)
        prior_precision : np.ndarray
            Prior precision matrix
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Posterior mean and covariance
        """
        T, n = Y.shape
        T, k = X.shape
        
        # Compute precision matrix efficiently
        XtX = X.T @ X
        post_precision = XtX + prior_precision
        post_precision_inv = np.linalg.inv(post_precision)
        
        # Compute posterior mean using Kronecker structure
        # vec(B_post) = (Sigma^{-1} ⊗ V_post) @ (vec(X'Y) + (I ⊗ prior_precision) @ vec(B_prior))
        Sigma_inv = np.linalg.inv(Sigma)
        
        XtY_vec = (X.T @ Y).flatten('F')
        prior_term = prior_precision @ prior_mean
        
        post_mean = KroneckerOperations.kron_vec_fast(
            post_precision_inv, Sigma_inv, XtY_vec + prior_term
        )
        
        # Posterior covariance is Sigma ⊗ V_post
        # We don't form this explicitly but return the factors
        return post_mean, (Sigma, post_precision_inv)
    
    @staticmethod
    def block_wise_inverse(A: np.ndarray, block_size: int = 50) -> np.ndarray:
        """
        Compute matrix inverse using block-wise operations for memory efficiency.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix to invert
        block_size : int
            Size of blocks for computation
            
        Returns:
        --------
        np.ndarray
            Inverse of A
        """
        n = A.shape[0]
        if n <= block_size:
            return np.linalg.inv(A)
        
        # Use Schur complement for block inversion
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        
        # Recursive inversion
        A11_inv = KroneckerOperations.block_wise_inverse(A11, block_size)
        S = A22 - A21 @ A11_inv @ A12  # Schur complement
        S_inv = KroneckerOperations.block_wise_inverse(S, block_size)
        
        # Assemble result
        result = np.zeros_like(A)
        result[:mid, :mid] = A11_inv + A11_inv @ A12 @ S_inv @ A21 @ A11_inv
        result[:mid, mid:] = -A11_inv @ A12 @ S_inv
        result[mid:, :mid] = -S_inv @ A21 @ A11_inv
        result[mid:, mid:] = S_inv
        
        return result


class SparseKroneckerOps:
    """
    Sparse Kronecker operations for very large VAR systems.
    
    When dealing with 200+ variables, even the efficient dense operations
    can become memory-intensive. These sparse methods handle such cases.
    """
    
    @staticmethod
    def sparse_kron_vec(A_sparse: csr_matrix, B_sparse: csr_matrix, v: np.ndarray) -> np.ndarray:
        """
        Sparse version of Kronecker-vector multiplication.
        
        Parameters:
        -----------
        A_sparse : csr_matrix
            Sparse matrix A
        B_sparse : csr_matrix
            Sparse matrix B
        v : np.ndarray
            Dense vector
            
        Returns:
        --------
        np.ndarray
            Result of (A ⊗ B) @ v
        """
        m, n = A_sparse.shape
        p, q = B_sparse.shape
        
        # Reshape v and compute efficiently
        V = v.reshape((n, q), order='F')
        result = B_sparse @ V @ A_sparse.T
        
        return result.flatten('F')
    
    @staticmethod
    def construct_sparse_var_precision(X: np.ndarray, Sigma_inv: np.ndarray, 
                                     prior_precision: csr_matrix) -> csr_matrix:
        """
        Construct sparse precision matrix for VAR posterior.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        Sigma_inv : np.ndarray
            Inverse error covariance
        prior_precision : csr_matrix
            Sparse prior precision
            
        Returns:
        --------
        csr_matrix
            Sparse posterior precision matrix
        """
        XtX = X.T @ X
        
        # Create sparse Kronecker product structure
        n_vars = Sigma_inv.shape[0]
        k_eq = XtX.shape[0]
        
        # Use efficient sparse Kronecker construction
        data_precision = sparse.kron(Sigma_inv, XtX, format='csr')
        
        return data_precision + prior_precision


@jit(nopython=True)
def fast_cholesky_update(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Fast Cholesky update for rank-1 modifications.
    
    Updates L such that L_new @ L_new.T = L @ L.T + v @ v.T
    
    Parameters:
    -----------
    L : np.ndarray
        Lower triangular Cholesky factor
    v : np.ndarray
        Update vector
        
    Returns:
    --------
    np.ndarray
        Updated Cholesky factor
    """
    n = L.shape[0]
    L_new = L.copy()
    v_work = v.copy()
    
    for i in range(n):
        r = np.sqrt(L_new[i, i]**2 + v_work[i]**2)
        c = r / L_new[i, i]
        s = v_work[i] / L_new[i, i]
        
        L_new[i, i] = r
        
        if i < n - 1:
            L_new[i+1:, i] = (L_new[i+1:, i] + s * v_work[i+1:]) / c
            v_work[i+1:] = c * v_work[i+1:] - s * L_new[i+1:, i]
    
    return L_new


def batch_matrix_normal_sample(mean_matrices: np.ndarray, 
                              U: np.ndarray, V: np.ndarray,
                              n_samples: int) -> np.ndarray:
    """
    Efficient batch sampling from matrix normal distribution.
    
    X ~ MN(M, U, V) where vec(X) ~ N(vec(M), V ⊗ U)
    
    Parameters:
    -----------
    mean_matrices : np.ndarray
        Mean matrices (n_samples x m x n)
    U : np.ndarray
        Row covariance (m x m)
    V : np.ndarray
        Column covariance (n x n)
    n_samples : int
        Number of samples
        
    Returns:
    --------
    np.ndarray
        Samples (n_samples x m x n)
    """
    m, n = U.shape[0], V.shape[0]
    samples = np.zeros((n_samples, m, n))
    
    # Precompute Cholesky factors
    L_U = np.linalg.cholesky(U)
    L_V = np.linalg.cholesky(V)
    
    for i in prange(n_samples):
        # Generate standard normal matrix
        Z = np.random.standard_normal((m, n))
        
        # Transform to matrix normal: M + L_U @ Z @ L_V.T
        samples[i] = mean_matrices[i] + L_U @ Z @ L_V.T
    
    return samples