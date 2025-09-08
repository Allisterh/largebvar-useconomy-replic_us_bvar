import numpy as np

def posterior_predictive_check(Y, X, beta_draws, sigma_draws):
    """
    Performs a posterior predictive check by simulating new data from the model.

    For each posterior draw of (β, Σ), we simulate a new dataset Y_rep of the
    same size as the original Y. We can then compare the distribution of
    statistics from Y_rep to the same statistics from Y to check for model fit.

    Args:
        Y (np.ndarray): The original matrix of dependent variables (T x n_vars).
        X (np.ndarray): The original matrix of independent variables (T x k_eq).
        beta_draws (np.ndarray): Posterior draws for β (n_draws, k_sys).
        sigma_draws (np.ndarray): Posterior draws for Σ (n_draws, n_vars, n_vars).

    Returns:
        np.ndarray: An array of replicated datasets (n_draws, T, n_vars).
    """
    n_draws = beta_draws.shape[0]
    T, n_vars = Y.shape
    _ , k_eq = X.shape
    
    replicated_Y = np.zeros((n_draws, T, n_vars))

    for i in range(n_draws):
        # Reshape the vectorized beta to its matrix form (k_eq x n_vars)
        beta_i = beta_draws[i, :].reshape((k_eq, n_vars), order='F')
        sigma_i = sigma_draws[i, :, :]
        
        # Generate errors from the posterior predictive distribution
        # which is a multivariate normal with covariance Σ from the current draw.
        errors = np.random.multivariate_normal(np.zeros(n_vars), sigma_i, size=T)
        
        # Generate one replicated dataset Y_rep = Xβ + errors
        replicated_Y[i, :, :] = X @ beta_i + errors

    return replicated_Y

def bayesian_p_value(replicated_Y, original_Y, test_statistic_func):
    """
    Computes a Bayesian p-value for a given test statistic.

    The Bayesian p-value is the proportion of replicated datasets for which the
    test statistic is greater than or equal to the test statistic of the
    original dataset. A p-value close to 0 or 1 suggests model misfit.

    Args:
        replicated_Y (np.ndarray): Array of replicated datasets from the posterior
                                   predictive distribution (n_draws, T, n_vars).
        original_Y (np.ndarray): The original data matrix (T, n_vars).
        test_statistic_func (function): A function that takes a data matrix (T, n_vars)
                                        and returns a scalar test statistic.

    Returns:
        float: The Bayesian p-value.
    """
    n_draws = replicated_Y.shape[0]

    # Calculate the test statistic for the original data
    t_original = test_statistic_func(original_Y)

    # Calculate the test statistic for each replicated dataset
    t_replicated = np.array([test_statistic_func(y_rep) for y_rep in replicated_Y])

    # The p-value is the proportion of times the replicated statistic is more extreme
    p_value = np.mean(t_replicated >= t_original)

    return p_value
