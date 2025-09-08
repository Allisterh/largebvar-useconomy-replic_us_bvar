import numpy as np

def construct_companion_matrix(beta_draw, n_vars, n_lags, k_eq):
    """Helper to build the VAR(p) companion matrix from a vectorized coefficient draw.
    
    The companion matrix is used to represent a VAR(p) process as a VAR(1).
    
    Args:
        beta_draw: Vectorized coefficient draw (k_sys,) where k_sys = n_vars * k_eq
                  Vectorized in Fortran order: vec([B1, B2, ..., Bn]) where Bi are equation coefficients
        n_vars: Number of variables
        n_lags: Number of lags
        k_eq: Number of coefficients per equation (n_vars * n_lags + n_exog)
        
    Returns:
        companion_matrix: (n_vars * n_lags, n_vars * n_lags) companion form matrix
    """
    # Reshape beta_draw to coefficient matrix: (k_eq, n_vars)
    # Each column represents one equation's coefficients
    beta_matrix = beta_draw.reshape((k_eq, n_vars), order='F')
    
    # Extract only the endogenous lag coefficients (first n_vars * n_lags rows)
    # Shape: (n_vars * n_lags, n_vars)
    B_endog_coeffs = beta_matrix[:n_vars * n_lags, :]
    
    # Transpose to get the format expected for companion matrix construction
    # We want B_endog where each row i contains coefficients for equation i
    B_endog = B_endog_coeffs.T  # Shape: (n_vars, n_vars * n_lags)
    
    # Build the companion form matrix
    companion_matrix = np.zeros((n_vars * n_lags, n_vars * n_lags))
    
    # First n_vars rows: coefficients from the VAR equations
    companion_matrix[:n_vars, :] = B_endog
    
    # Remaining rows: identity matrix for the lag structure
    if n_lags > 1:
        companion_matrix[n_vars:, :-n_vars] = np.eye(n_vars * (n_lags - 1))
        
    return companion_matrix

def compute_irf(beta_draws, sigma_draws, n_vars, n_lags, n_exog, horizon, identification='cholesky'):
    """
    Computes Impulse Response Functions (IRFs) for each posterior draw.

    Args:
        beta_draws (np.ndarray): Posterior draws for β (n_draws, k_sys).
        sigma_draws (np.ndarray): Posterior draws for Σ (n_draws, n_vars, n_vars).
        n_vars (int): Number of endogenous variables.
        n_lags (int): Number of lags.
        n_exog (int): Number of exogenous variables.
        horizon (int): The number of periods for the IRF.
        identification (str): The identification scheme, 'cholesky' is currently supported.

    Returns:
        np.ndarray: An array of IRF draws (n_draws, n_vars, n_vars, horizon).
                    Dimensions are (draw, impulse_var, response_var, time_period).
    """
    n_draws = beta_draws.shape[0]
    k_eq = n_vars * n_lags + n_exog
    irf_draws = np.zeros((n_draws, n_vars, n_vars, horizon))
    
    # Debug: Check input statistics
    print(f"IRF computation - Beta draws shape: {beta_draws.shape}")
    print(f"IRF computation - Sigma draws shape: {sigma_draws.shape}")
    print(f"IRF computation - Beta statistics: min={np.nanmin(beta_draws):.6f}, "
          f"max={np.nanmax(beta_draws):.6f}, std={np.nanstd(beta_draws):.6f}")
    
    n_problematic_draws = 0
    n_flat_irfs = 0

    for i in range(n_draws):
        beta_i = beta_draws[i, :]
        sigma_i = sigma_draws[i, :, :]
        
        # Check for numerical issues
        if not np.all(np.isfinite(beta_i)) or not np.all(np.isfinite(sigma_i)):
            irf_draws[i, :, :, :] = np.nan
            n_problematic_draws += 1
            continue
            
        # Construct companion matrix
        B_companion = construct_companion_matrix(beta_i, n_vars, n_lags, k_eq)
        
        # Check companion matrix for stability
        max_eigenval = np.max(np.abs(np.linalg.eigvals(B_companion)))
        if max_eigenval > 0.999:  # Near unit root
            if i < 5:  # Log only for first few draws
                print(f"Draw {i}: Large eigenvalue in companion matrix: {max_eigenval:.6f}")
        
        # Identify the shock using Cholesky decomposition
        if identification == 'cholesky':
            try:
                chol_factor = np.linalg.cholesky(sigma_i)
            except np.linalg.LinAlgError:
                # If Sigma is not positive definite, skip this draw
                irf_draws[i, :, :, :] = np.nan
                n_problematic_draws += 1
                continue
        else:
            raise NotImplementedError(f"Identification scheme '{identification}' not supported.")

        # Create a shock for each variable
        draw_max_irf = 0.0  # Track maximum IRF for this draw
        
        for impulse_var_idx in range(n_vars):
            shock_vector = chol_factor[:, impulse_var_idx]
            
            # Check shock magnitude
            shock_magnitude = np.linalg.norm(shock_vector)
            if shock_magnitude < 1e-10:
                if i < 5:  # Log only for first few draws
                    print(f"Draw {i}, Impulse {impulse_var_idx}: Very small shock magnitude: {shock_magnitude:.2e}")
            
            # Initialize state vector for the VAR(1) companion form
            state = np.zeros(n_vars * n_lags)
            state[:n_vars] = shock_vector
            
            # Trace the impulse through time
            for h in range(horizon):
                # The response of all variables is in the first n_vars elements of the state
                response = state[:n_vars]
                irf_draws[i, impulse_var_idx, :, h] = response
                
                # Track maximum response for diagnostics
                response_max = np.max(np.abs(response))
                draw_max_irf = max(draw_max_irf, response_max)
                
                # Check for explosive behavior and truncate if necessary
                if response_max > 1e10:  # Explosive threshold
                    if i < 5 and h < 5:  # Log for first few draws/periods
                        print(f"Draw {i}, Period {h}: Explosive IRF detected ({response_max:.2e}), truncating")
                    # Set remaining periods to NaN for this draw
                    irf_draws[i, impulse_var_idx, :, h:] = np.nan
                    break
                
                # Update state for the next period: S_t+1 = B_companion * S_t
                state = B_companion @ state
        
        # Check if this draw produced flat IRFs
        if draw_max_irf < 1e-8:
            n_flat_irfs += 1
            if i < 5:  # Log details for first few flat draws
                print(f"Draw {i}: Flat IRF detected (max response: {draw_max_irf:.2e})")
                print(f"  - Companion matrix max eigenval: {max_eigenval:.6f}")
                print(f"  - Sigma matrix condition number: {np.linalg.cond(sigma_i):.2f}")
                print(f"  - Beta coefficients range: [{np.nanmin(beta_i):.6f}, {np.nanmax(beta_i):.6f}]")

    # Final diagnostics
    print(f"IRF computation completed:")
    print(f"  - Total draws: {n_draws}")
    print(f"  - Problematic draws (NaN): {n_problematic_draws}")
    print(f"  - Flat IRF draws: {n_flat_irfs}")
    print(f"  - Overall IRF statistics: min={np.nanmin(irf_draws):.6f}, max={np.nanmax(irf_draws):.6f}")
    
    return irf_draws

def compute_fevd(irf_draws):
    """
    Computes Forecast Error Variance Decomposition (FEVD) from IRF draws.

    The FEVD measures the fraction of the variance of the h-step ahead forecast
    error for variable j that is accounted for by shocks to variable k.

    Args:
        irf_draws (np.ndarray): An array of IRF draws from compute_irf, with shape
                                (n_draws, n_impulse, n_response, horizon).

    Returns:
        np.ndarray: An array of FEVD draws (n_draws, n_response, n_impulse, horizon).
                    The dimensions are swapped for easier interpretation, showing
                    for each response variable, the contribution of each impulse variable.
    """
    n_draws, n_impulse, n_response, horizon = irf_draws.shape
    
    # FEVD array, dimensions swapped for intuitive access
    fevd_draws = np.zeros((n_draws, n_response, n_impulse, horizon))
    
    # Square the IRFs to get the variance contribution of each shock at each horizon
    squared_irfs = irf_draws**2
    
    # Cumulative sum of squared IRFs over the horizon
    # This gives the cumulative variance contribution at each step h
    cumulative_squared_irfs = np.cumsum(squared_irfs, axis=3)
    
    # Total forecast error variance for each response variable at each horizon
    # Sum the cumulative squared IRFs across all shocks (n_impulse)
    total_fev = np.sum(cumulative_squared_irfs, axis=1) # Shape: (n_draws, n_response, horizon)
    
    # Avoid division by zero for the first period if there's no variance
    total_fev[total_fev == 0] = 1e-12

    # Calculate the decomposition
    for i in range(n_impulse): # The variable whose shock we're considering
        for j in range(n_response): # The variable whose forecast error we're decomposing
            # Contribution of shock i to variance of variable j
            contribution = cumulative_squared_irfs[:, i, j, :]
            fevd_draws[:, j, i, :] = contribution / total_fev[:, j, :]
            
    return fevd_draws
