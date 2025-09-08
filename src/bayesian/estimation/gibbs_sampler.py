import numpy as np
from tqdm import tqdm

from . import conjugate_posterior as cp

class GibbsSampler:
    """
    Implements a Gibbs sampler for estimating the parameters of a Bayesian VAR.

    The sampler iteratively draws from the conditional posterior distributions of
    the VAR coefficients (β) and the error covariance matrix (Σ), assuming a
    Normal-Inverse-Wishart prior structure.
    """
    def __init__(self, n_draws=5000, n_burnin=2000):
        """
        Initializes the GibbsSampler.

        Args:
            n_draws (int): The total number of draws to generate from the posterior.
            n_burnin (int): The number of initial draws to discard as burn-in.
        """
        if n_burnin >= n_draws:
            raise ValueError("Burn-in draws must be less than total draws.")
        self.n_draws = n_draws
        self.n_burnin = n_burnin
        self.n_retained_draws = n_draws - n_burnin

    def sample(self, Y, X, prior_mean, prior_cov, prior_scale, prior_dof):
        """
        Runs the Gibbs sampler to draw from the posterior of (β, Σ).

        Args:
            Y (np.ndarray): The matrix of dependent variables (T x n_vars).
            X (np.ndarray): The matrix of independent variables (T x k_eq).
            prior_mean (np.ndarray): The prior mean for β.
            prior_cov (np.ndarray): The prior covariance matrix for β.
            prior_scale (np.ndarray): The prior scale matrix (Ψ) for Σ.
            prior_dof (int): The prior degrees of freedom (d) for Σ.

        Returns:
            tuple: A tuple containing:
                - beta_draws (np.ndarray): Retained draws for β (n_retained_draws, k_sys).
                - sigma_draws (np.ndarray): Retained draws for Σ (n_retained_draws, n_vars, n_vars).
        """
        T, n_vars = Y.shape
        _ , k_eq = X.shape
        k_sys = n_vars * k_eq

        # Initialize storage for draws
        beta_draws = np.zeros((self.n_retained_draws, k_sys))
        sigma_draws = np.zeros((self.n_retained_draws, n_vars, n_vars))

        # --- Initial values ---
        # Start with OLS estimates for β and Σ for simplicity.
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ Y
        
        beta_current = beta_ols.flatten('F')  # Vectorize column-by-column
        
        residuals = Y - X @ beta_ols
        sigma_current = (residuals.T @ residuals) / (T - k_eq)

        # --- Gibbs sampling loop ---
        for i in tqdm(range(self.n_draws), desc="Gibbs Sampling"):
            # 1. Draw β from p(β | Y, Σ)
            post_mean_beta, post_cov_beta = cp.normal_posterior_moments(
                Y, X, sigma_current, prior_mean, prior_cov
            )
            beta_current = np.random.multivariate_normal(post_mean_beta, post_cov_beta)

            # 2. Draw Σ from p(Σ | Y, β)
            beta_matrix = beta_current.reshape((k_eq, n_vars), order='F')
            post_scale_sigma, post_dof_sigma = cp.inv_wishart_posterior_params(
                Y, X, beta_matrix, prior_scale, prior_dof
            )
            
            # Draw S_inv from Wishart(post_scale⁻¹, post_dof)
            # Then Σ = (S_inv)⁻¹
            post_scale_inv = np.linalg.inv(post_scale_sigma)
            sigma_inv_draw = np.random.standard_normal((post_dof_sigma, n_vars)) @ np.linalg.cholesky(post_scale_inv).T
            sigma_inv_draw = sigma_inv_draw.T @ sigma_inv_draw

            sigma_current = np.linalg.inv(sigma_inv_draw)

            # Store draws after burn-in period
            if i >= self.n_burnin:
                draw_idx = i - self.n_burnin
                beta_draws[draw_idx, :] = beta_current
                sigma_draws[draw_idx, :, :] = sigma_current
        
        return beta_draws, sigma_draws
