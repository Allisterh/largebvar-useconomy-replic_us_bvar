import numpy as np
import numba
from numba import jit

def normal_posterior_moments(Y, X, Sigma, prior_mean, prior_cov):
    """
    Calculates the moments of the conditional posterior distribution for β.
    """
    n_vars = Sigma.shape[0]
    T, k_eq = X.shape
    k_sys = n_vars * k_eq

    prior_precision = np.linalg.inv(prior_cov)
    sigma_inv = np.linalg.inv(Sigma)

    # --- Construct posterior_precision (V_bar⁻¹) manually ---
    # V_bar⁻¹ = V⁻¹ + (Σ⁻¹ ⊗ X'X)
    # This is a (k_sys, k_sys) matrix.
    # It's a block matrix where the (i,j)-th block is sigma_inv[i,j] * (X.T @ X)
    
    XtX = X.T @ X # (k_eq, k_eq)
    
    kron_term = np.zeros((k_sys, k_sys))
    for i in range(n_vars):
        for j in range(n_vars):
            # Block (i,j) of kron_term is sigma_inv[i,j] * XtX
            row_start = i * k_eq
            row_end = (i + 1) * k_eq
            col_start = j * k_eq
            col_end = (j + 1) * k_eq
            kron_term[row_start:row_end, col_start:col_end] = sigma_inv[i,j] * XtX

    posterior_precision = prior_precision + kron_term
    posterior_cov = np.linalg.inv(posterior_precision)

    # --- Construct data_term (vec(X'YΣ⁻¹)) manually ---
    # data_term = (Σ⁻¹ ⊗ X') @ vec(Y)
    # This is a (k_sys,) vector.
    # It's a stack of (X'YΣ⁻¹)_i where _i means the i-th column of the matrix.
    
    # First, calculate X'Y (k_eq, n_vars)
    XtY = X.T @ Y
    
    # Then, calculate XtY @ sigma_inv (k_eq, n_vars)
    XtY_sigma_inv = XtY @ sigma_inv
    
    # Now, vectorize XtY_sigma_inv column-wise (Fortran order)
    data_term = XtY_sigma_inv.reshape(-1, order='F')

    # --- Construct prior_term (V⁻¹b) ---
    prior_term = prior_precision @ prior_mean

    # --- Calculate posterior_mean ---
    posterior_mean = posterior_cov @ (prior_term + data_term)

    return posterior_mean, posterior_cov

def inv_wishart_posterior_params(Y, X, beta_matrix, prior_scale, prior_dof):
    """
    Calculates the parameters of the conditional posterior distribution for Σ.
    """
    T = Y.shape[0]

    # Posterior degrees of freedom: d_bar = d + T
    posterior_dof = prior_dof + T

    # Posterior scale matrix: Ψ_bar = Ψ + (Y - Xβ)'(Y - Xβ)
    residuals = Y - X @ beta_matrix
    posterior_scale = prior_scale + residuals.T @ residuals

    return posterior_scale, posterior_dof