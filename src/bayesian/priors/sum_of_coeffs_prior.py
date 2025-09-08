import numpy as np

class SumOfCoefficientsPrior:
    """
    Implements the Sum-of-Coefficients prior for a Bayesian VAR model.

    This prior, also known as the "co-persistence" prior, is due to Doan,
    Litterman, and Sims (1984). It is implemented by adding dummy observations
    to the data and pushes the model towards a VAR in first differences,
    reflecting a belief in the presence of unit roots in the data.
    """
    def __init__(self, mu=1.0):
        """
        Initializes the SumOfCoefficientsPrior.

        Args:
            mu (float): The hyperparameter controlling the tightness of the prior.
                        As mu -> inf, the prior is imposed exactly (VAR in differences).
                        As mu -> 0, the prior has no effect.
        """
        self.mu = mu

    def create_dummy_observations(self, data, n_lags, n_exog):
        """
        Creates the dummy observations for the sum-of-coefficients prior.

        Args:
            data (np.ndarray): The original training data, with shape (T, n_vars).
            n_lags (int): The number of lags in the VAR.
            n_exog (int): The number of exogenous variables in the model.

        Returns:
            tuple: A tuple containing:
                - y_dummy (np.ndarray): The dummy observations for the dependent variables (Y).
                - x_dummy (np.ndarray): The dummy observations for the independent variables (X).
        """
        if self.mu == 0:
            # If mu is zero, the prior has no effect.
            return np.array([]).reshape(0, data.shape[1]), np.array([]).reshape(0, data.shape[1] * n_lags + n_exog)

        n_vars = data.shape[1]
        
        # Take the average of the first `n_lags` observations as the initial condition
        initial_conditions = np.mean(data[:n_lags, :], axis=0)

        # The dummy observation for the dependent variables (y_dummy)
        y_dummy = self.mu * initial_conditions
        y_dummy = y_dummy.reshape(1, n_vars)

        # The dummy observation for the independent variables (x_dummy)
        # Part 1: Lags of endogenous variables
        x_dummy_lags = np.tile(self.mu * initial_conditions, n_lags)
        
        # Part 2: Exogenous variables
        # The prior does not apply to exogenous variables, so their dummy values are 0.
        x_dummy_exog = np.zeros(n_exog)

        x_dummy = np.concatenate([x_dummy_lags, x_dummy_exog])
        x_dummy = x_dummy.reshape(1, -1)

        return y_dummy, x_dummy
