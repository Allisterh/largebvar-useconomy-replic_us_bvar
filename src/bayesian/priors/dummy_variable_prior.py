import numpy as np

class DummyInitialObservationPrior:
    """
    Implements the dummy initial observation prior (or single-unit-root prior)
    of Sims and Zha (1998).

    This prior reflects a belief in a common stochastic trend among the
    variables in the system. It is implemented by adding a single dummy
    observation based on the initial values of the series.
    """
    def __init__(self, delta=1.0):
        """
        Initializes the DummyInitialObservationPrior.

        Args:
            delta (float): The hyperparameter controlling the tightness of the prior.
                           As delta -> inf, the prior is imposed exactly.
                           As delta -> 0, the prior has no effect.
        """
        self.delta = delta

    def create_dummy_observations(self, data, n_lags, n_exog):
        """
        Creates the dummy observations for the single-unit-root prior.

        Args:
            data (np.ndarray): The original training data, with shape (T, n_vars).
            n_lags (int): The number of lags in the VAR.
            n_exog (int): The number of exogenous variables in the model.

        Returns:
            tuple: A tuple containing:
                - y_dummy (np.ndarray): The dummy observations for the dependent variables (Y).
                - x_dummy (np.ndarray): The dummy observations for the independent variables (X).
        """
        if self.delta == 0:
            return np.array([]).reshape(0, data.shape[1]), np.array([]).reshape(0, data.shape[1] * n_lags + n_exog)

        n_vars = data.shape[1]
        
        # Use the first observation as the initial condition
        initial_observation = data[0, :]

        # y_dummy: delta * initial_observation
        y_dummy = self.delta * initial_observation
        y_dummy = y_dummy.reshape(1, n_vars)

        # x_dummy:
        # Part 1: Lags of endogenous variables
        # The first lag is delta * initial_observation, others are zero.
        x_dummy_lag1 = self.delta * initial_observation
        x_dummy_other_lags = np.zeros(n_vars * (n_lags - 1))
        x_dummy_lags = np.concatenate([x_dummy_lag1, x_dummy_other_lags])
        
        # Part 2: Exogenous variables
        # The prior does not apply to exogenous variables.
        x_dummy_exog = np.zeros(n_exog)

        x_dummy = np.concatenate([x_dummy_lags, x_dummy_exog])
        x_dummy = x_dummy.reshape(1, -1)

        return y_dummy, x_dummy
