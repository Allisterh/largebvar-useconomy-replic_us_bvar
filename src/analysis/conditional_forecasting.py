import numpy as np
import statsmodels.api as sm
from bayesian.posterior.posterior_analysis import construct_companion_matrix

def _construct_ss_matrices(beta_draw, sigma_draw, n_vars, n_lags, n_exog):
    """
    Constructs the state-space matrices for a given VAR parameter draw.
    """
    k_eq = n_vars * n_lags + n_exog
    k_states = n_vars * n_lags

    # Transition matrix (T)
    T_ss = construct_companion_matrix(beta_draw, n_vars, n_lags, k_eq)
    
    # Design/Observation matrix (Z)
    Z = np.zeros((n_vars, k_states))
    Z[:, :n_vars] = np.eye(n_vars)
    
    # State disturbance loading matrix (R)
    R = np.zeros((k_states, n_vars))
    R[:n_vars, :] = np.eye(n_vars)
    
    # State disturbance covariance matrix (Q)
    Q = sigma_draw
    
    return Z, T_ss, R, Q

def conditional_forecast(beta_draws, sigma_draws, data, n_lags, n_exog, horizon, constraints):
    """
    Generates conditional forecasts using the statsmodels state-space framework.
    """
    n_draws = beta_draws.shape[0]
    T, n_vars = data.shape
    k_states = n_vars * n_lags

    forecast_draws = np.zeros((n_draws, horizon, n_vars))

    # Prepare the observation matrix for the forecast period
    Y_forecast = np.full((horizon, n_vars), np.nan)
    for t_future, constr in constraints.items():
        if 1 <= t_future <= horizon:
            for var_idx, value in constr.items():
                Y_forecast[t_future - 1, var_idx] = value
    
    Y_full = np.vstack([data, Y_forecast])

    for i in range(n_draws):
        beta_i = beta_draws[i, :]
        sigma_i = sigma_draws[i, :, :]
        
        Z, T_ss, R, Q = _construct_ss_matrices(beta_i, sigma_i, n_vars, n_lags, n_exog)

        # Define a custom state-space model using statsmodels' base class
        class VARStateSpace(sm.tsa.statespace.MLEModel):
            def __init__(self, endog):
                super().__init__(endog, k_states=k_states, k_posdef=n_vars, initialization='stationary')
                self.ssm['design'] = Z
                self.ssm['transition'] = T_ss
                self.ssm['selection'] = R
                self.ssm['state_cov'] = Q

        try:
            # Instantiate and run the simulation smoother
            model = VARStateSpace(Y_full)
            sim_results = model.simulation_smoother()
            # The simulate() method modifies the results object in place
            sim_results.simulate()
            sim_states = sim_results.simulated_state

            # Extract the forecast period
            forecast_draws[i, :, :] = sim_states[-horizon:, :n_vars]

        except np.linalg.LinAlgError:
            forecast_draws[i, :, :] = np.nan

    return forecast_draws