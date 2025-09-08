import numpy as np

def effective_sample_size(draws):
    """
    Computes the Effective Sample Size (ESS) for a set of posterior draws.

    ESS is a measure of how many independent samples the posterior draws are
    equivalent to. It is useful for assessing the efficiency of the MCMC sampler.
    A low ESS indicates high autocorrelation in the draws, meaning the sampler
    is exploring the posterior space inefficiently.

    This implementation uses Geyer's initial positive sequence estimator.

    Args:
        draws (np.ndarray): A 1D array of posterior draws for a single parameter.

    Returns:
        float: The effective sample size.
    """
    n_draws = len(draws)
    if n_draws < 2:
        return np.nan

    mean_draw = np.mean(draws)
    var_draw = np.var(draws, ddof=1)

    if var_draw == 0:
        return np.inf # No variance, effectively infinite ESS

    # Autocovariance calculation using FFT for efficiency
    fft_len = 2**(n_draws - 1).bit_length()
    draws_centered = draws - mean_draw
    autocov_fft = np.fft.ifft(np.fft.fft(draws_centered, n=fft_len) * np.conj(np.fft.fft(draws_centered, n=fft_len))).real
    autocov = autocov_fft[:n_draws] / n_draws

    # Geyer's initial positive sequence
    rho_sum = 0
    # t is the lag
    for t in range(1, n_draws - 1):
        rho = autocov[t] / var_draw
        # Sum pairs of autocorrelations
        if t % 2 == 1:
            rho_pair_sum = rho + (autocov[t+1] / var_draw if t+1 < n_draws else 0)
            if rho_pair_sum <= 0:
                break
            rho_sum += rho_pair_sum

    # Guard against negative rho_sum which can happen with noisy estimates
    if rho_sum < 0:
        rho_sum = 0

    ess = n_draws / (1 + 2 * rho_sum)
    return ess

def calculate_ess_for_all_params(beta_draws):
    """
    Calculates ESS for all VAR coefficients.

    Args:
        beta_draws (np.ndarray): Posterior draws for β (n_draws, k_sys).

    Returns:
        np.ndarray: An array of ESS values, one for each parameter.
    """
    n_draws, n_params = beta_draws.shape
    ess_values = np.zeros(n_params)
    for i in range(n_params):
        ess_values[i] = effective_sample_size(beta_draws[:, i])
    return ess_values
