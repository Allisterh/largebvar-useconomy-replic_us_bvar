import numpy as np

class MinnesotaPrior:
    """
    Implements the Minnesota prior for a Bayesian VAR model.

    This prior is based on the work of Litterman (1979) and is centered on the
    assumption that most time series are well-described by a random walk. It
    shrinks the VAR coefficients towards that behavior, preventing overfitting
    in large models.

    The prior is defined for the vectorized VAR coefficients, β = vec(B),
    where B is the matrix of coefficients for all equations.
    """
    def __init__(self, lambda1=0.1, lambda2=0.5, lambda3=1.0, center_at_1=True):
        """
        Initializes the MinnesotaPrior.

        Args:
            lambda1 (float): Shrinkage parameter for own-lag coefficients.
            lambda2 (float): Shrinkage parameter for cross-variable lag coefficients.
            lambda3 (float): Shrinkage parameter for exogenous variable coefficients.
            center_at_1 (bool): If True, sets the prior mean for the first own-lag
                                coefficient to 1 (random walk). If False, centers
                                all coefficients at 0 (white noise).
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.center_at_1 = center_at_1

    def get_prior_moments(self, n_vars, n_lags, n_exog, sigmas_sq):
        """
        Builds the prior mean vector and prior covariance matrix for the VAR coefficients.

        This implementation assumes the coefficients are vectorized equation by equation,
        i.e., β = [β₁, β₂, ..., βₙ]' where βᵢ are the coefficients for equation i.

        Args:
            n_vars (int): The number of endogenous variables in the VAR.
            n_lags (int): The number of lags in the VAR.
            n_exog (int): The number of exogenous variables (e.g., constant).
            sigmas_sq (np.ndarray): A (n_vars,) array of OLS residual variances
                                    for each variable, used for scaling the prior.

        Returns:
            tuple: A tuple containing:
                - prior_mean (np.ndarray): The prior mean vector (n_vars * (n_vars*n_lags + n_exog),).
                - prior_cov (np.ndarray): The prior covariance matrix (diagonal).
        """
        # Total number of coefficients per equation
        k_eq = n_vars * n_lags + n_exog
        # Total number of coefficients in the system
        k_sys = n_vars * k_eq

        # --- Prior Mean (b) ---
        prior_mean = np.zeros(k_sys)
        if self.center_at_1:
            # Set prior mean of first own lag to 1 for each equation
            for i in range(n_vars):
                # Index for the first lag of variable `i` in equation `i`
                idx = i * k_eq + i
                prior_mean[idx] = 1

        # --- Prior Covariance (V) ---
        # We construct a diagonal covariance matrix. The inverse of this is the precision matrix.
        prior_var_diag = np.zeros(k_sys)

        # Iterate through each equation to build the diagonal of the covariance matrix
        for i in range(n_vars):  # Equation `i`
            sigma_i_sq = sigmas_sq[i]
            
            # --- Endogenous variables' coefficients ---
            for lag in range(1, n_lags + 1):
                lag_decay = 1.0 / (lag**2)
                
                for j in range(n_vars): # Variable `j` providing the lag
                    sigma_j_sq = sigmas_sq[j]
                    
                    # Index for coefficient of var `j` at `lag` in equation `i`
                    idx = i * k_eq + (lag - 1) * n_vars + j

                    if i == j:
                        # Variance for own lag
                        variance = (self.lambda1**2) * lag_decay
                    else:
                        # Variance for cross-variable lag
                        variance = (self.lambda2**2) * lag_decay * (sigma_i_sq / sigma_j_sq)
                    
                    prior_var_diag[idx] = variance

            # --- Exogenous variables' coefficients ---
            for j in range(n_exog):
                idx = i * k_eq + n_vars * n_lags + j
                variance = (self.lambda3**2) * sigma_i_sq
                prior_var_diag[idx] = variance

        prior_cov = np.diag(prior_var_diag)

        return prior_mean, prior_cov
