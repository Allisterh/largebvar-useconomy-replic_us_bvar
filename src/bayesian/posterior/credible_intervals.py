import numpy as np

def get_credible_intervals(draws, interval_levels=[0.68, 0.90]):
    """
    Computes credible intervals for a set of posterior draws.

    Args:
        draws (np.ndarray): A 2D array of posterior draws, where rows are
                            draws and columns are parameters or time periods.
        interval_levels (list): A list of desired interval levels (e.g., 0.68 for 68%).

    Returns:
        dict: A dictionary where keys are the interval levels and values are
              dicts containing the 'lower' and 'upper' bounds.
    """
    intervals = {}
    for level in interval_levels:
        lower_quantile = (1.0 - level) / 2.0
        upper_quantile = 1.0 - lower_quantile
        
        lower_bound = np.quantile(draws, lower_quantile, axis=0)
        upper_bound = np.quantile(draws, upper_quantile, axis=0)
        
        intervals[level] = {'lower': lower_bound, 'upper': upper_bound}
        
    return intervals

def get_posterior_summary(draws):
    """
    Computes summary statistics (mean, median, std dev) for posterior draws.

    Args:
        draws (np.ndarray): A 2D array of posterior draws.

    Returns:
        dict: A dictionary with 'mean', 'median', and 'std' as keys.
    """
    summary = {
        'mean': np.mean(draws, axis=0),
        'median': np.median(draws, axis=0),
        'std': np.std(draws, axis=0)
    }
    return summary
