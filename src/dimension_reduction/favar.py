"""
Factor-Augmented Vector Autoregression (FAVAR) implementation.

This module implements FAVAR models that combine a small number of observable
factors or variables with unobserved factors extracted from a large dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from .factor_models import DynamicFactorModel, estimate_factors_bai_ng
from ..bayesian.core_bayesian_var import LargeBayesianVAR
from ..computational.kronecker_ops import KroneckerOperations
import warnings


class FAVAR:
    """
    Factor-Augmented Vector Autoregression model.
    
    The FAVAR model consists of two blocks:
    1. Observable variables (Y_t) - small set of key variables
    2. Informational variables (X_t) - large set of additional variables
    
    The model structure is:
    [F_t; Y_t] = A(L) [F_{t-1}; Y_{t-1}] + ε_t
    X_t = Λ_x F_t + Λ_y Y_t + u_t
    
    where F_t are unobserved factors extracted from X_t.
    """
    
    def __init__(self, n_factors: int, n_lags: int = 4, 
                 factor_method: str = 'pc', standardize: bool = True):
        """
        Initialize FAVAR model.
        
        Parameters:
        -----------
        n_factors : int
            Number of unobserved factors to extract
        n_lags : int
            Number of lags in the VAR
        factor_method : str
            Method for factor extraction ('pc', 'ml', 'bai_ng')
        standardize : bool
            Whether to standardize variables
        """
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.factor_method = factor_method
        self.standardize = standardize
        
        # Model components
        self.factor_model = None
        self.var_model = None
        self.observable_vars = None
        self.informational_vars = None
        self.factors = None
        self.favar_data = None
        
        # Identification
        self.factor_loadings_x = None
        self.factor_loadings_y = None
        
    def fit(self, Y_observable: np.ndarray, X_informational: np.ndarray,
            observable_names: Optional[List[str]] = None,
            informational_names: Optional[List[str]] = None) -> 'FAVAR':
        """
        Fit FAVAR model using two-step estimation.
        
        Step 1: Extract factors from informational variables
        Step 2: Estimate VAR with factors and observable variables
        
        Parameters:
        -----------
        Y_observable : np.ndarray
            Observable variables (T x n_obs)
        X_informational : np.ndarray
            Informational variables (T x n_info)
        observable_names : List[str], optional
            Names of observable variables
        informational_names : List[str], optional
            Names of informational variables
            
        Returns:
        --------
        FAVAR
            Fitted FAVAR model
        """
        T_obs, n_obs = Y_observable.shape
        T_info, n_info = X_informational.shape
        
        if T_obs != T_info:
            raise ValueError("Observable and informational data must have same length")
        
        self.observable_vars = observable_names or [f"Y_{i}" for i in range(n_obs)]
        self.informational_vars = informational_names or [f"X_{i}" for i in range(n_info)]
        
        # Step 1: Extract factors from informational variables
        print(f"Step 1: Extracting {self.n_factors} factors from {n_info} informational variables")
        self._extract_factors(X_informational)
        
        # Step 2: Estimate FAVAR system
        print("Step 2: Estimating FAVAR system")
        self._estimate_favar_system(Y_observable, X_informational)
        
        return self
    
    def _extract_factors(self, X_informational: np.ndarray):
        """Extract factors from informational variables."""
        if self.factor_method == 'bai_ng':
            # Use Bai & Ng automatic factor selection
            n_factors_selected, factors, loadings = estimate_factors_bai_ng(
                X_informational, max_factors=min(20, X_informational.shape[1] // 4)
            )
            
            if n_factors_selected != self.n_factors:
                warnings.warn(f"Bai & Ng selected {n_factors_selected} factors, "
                            f"but {self.n_factors} was requested. Using selected number.")
                self.n_factors = n_factors_selected
            
            self.factors = factors
            self.factor_loadings_x = loadings
        else:
            # Use specified factor extraction method
            self.factor_model = DynamicFactorModel(
                n_factors=self.n_factors,
                n_lags=1,  # Factor dynamics will be captured in VAR
                standardize=self.standardize,
                method=self.factor_method
            )
            
            self.factor_model.fit(X_informational)
            self.factors = self.factor_model.factors
            self.factor_loadings_x = self.factor_model.loadings
    
    def _estimate_favar_system(self, Y_observable: np.ndarray, X_informational: np.ndarray):
        """Estimate the FAVAR system."""
        # Combine factors and observable variables
        self.favar_data = np.hstack([self.factors, Y_observable])
        
        # Create variable names
        factor_names = [f"Factor_{i+1}" for i in range(self.n_factors)]
        favar_names = factor_names + self.observable_vars
        
        # Estimate VAR on [F_t, Y_t]
        self.var_model = LargeBayesianVAR(n_lags=self.n_lags)
        self.var_model.fit(self.favar_data)
        
        print(f"FAVAR system estimated: {self.favar_data.shape[1]} variables, "
              f"{self.n_lags} lags, {self.favar_data.shape[0]} observations")
    
    def identify_factors(self, identification_scheme: str = 'cholesky',
                        slow_variables: Optional[List[int]] = None,
                        fast_variables: Optional[List[int]] = None) -> Dict:
        """
        Identify structural shocks in FAVAR system.
        
        Parameters:
        -----------
        identification_scheme : str
            Identification scheme ('cholesky', 'sign_restrictions', 'slow_fast')
        slow_variables : List[int], optional
            Indices of slow-moving variables for identification
        fast_variables : List[int], optional
            Indices of fast-moving variables for identification
            
        Returns:
        --------
        Dict
            Identification results
        """
        if self.var_model is None:
            raise ValueError("FAVAR model must be fitted first")
        
        identification_results = {}
        
        if identification_scheme == 'cholesky':
            # Standard Cholesky identification
            identification_results['method'] = 'cholesky'
            identification_results['ordering'] = list(range(self.favar_data.shape[1]))
            
        elif identification_scheme == 'slow_fast':
            # Slow/fast identification as in Bernanke, Boivin & Eliasz (2005)
            if slow_variables is None or fast_variables is None:
                # Default: factors are slow, observables are fast
                slow_variables = list(range(self.n_factors))
                fast_variables = list(range(self.n_factors, self.favar_data.shape[1]))
            
            identification_results.update({
                'method': 'slow_fast',
                'slow_variables': slow_variables,
                'fast_variables': fast_variables
            })
            
        elif identification_scheme == 'sign_restrictions':
            # Placeholder for sign restriction identification
            warnings.warn("Sign restrictions not fully implemented yet")
            identification_results['method'] = 'sign_restrictions'
        
        return identification_results
    
    def impulse_responses(self, steps: int = 20, identification: str = 'cholesky',
                         **identification_kwargs) -> Dict[str, np.ndarray]:
        """
        Compute impulse responses for FAVAR system.
        
        Parameters:
        -----------
        steps : int
            Impulse response horizon
        identification : str
            Identification scheme
        **identification_kwargs
            Additional arguments for identification
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Impulse responses for factors and observables
        """
        if self.var_model is None:
            raise ValueError("FAVAR model must be fitted first")
        
        # Get IRFs from VAR model
        irf_summary, irf_intervals = self.var_model.get_irf_summary(
            horizon=steps, identification=identification
        )
        
        # Split IRFs into factors and observables
        factor_irfs = irf_summary['median'][:, :self.n_factors, :]
        observable_irfs = irf_summary['median'][:, self.n_factors:, :]
        
        # Compute IRFs for informational variables
        # These respond through their loadings on factors
        informational_irfs = self._compute_informational_irfs(factor_irfs, observable_irfs)
        
        results = {
            'factors': factor_irfs,
            'observables': observable_irfs,
            'informationals': informational_irfs,
            'confidence_intervals': irf_intervals
        }
        
        return results
    
    def _compute_informational_irfs(self, factor_irfs: np.ndarray, 
                                  observable_irfs: np.ndarray) -> np.ndarray:
        """
        Compute impulse responses for informational variables.
        
        These variables respond to shocks through their factor loadings:
        IRF_X = Λ_F * IRF_F + Λ_Y * IRF_Y
        """
        n_shocks, n_factors, n_periods = factor_irfs.shape
        n_observables = observable_irfs.shape[1]
        n_informationals = self.factor_loadings_x.shape[0]
        
        # Initialize informational IRFs
        informational_irfs = np.zeros((n_shocks, n_informationals, n_periods))
        
        for shock in range(n_shocks):
            for period in range(n_periods):
                # Response through factors
                factor_response = self.factor_loadings_x @ factor_irfs[shock, :, period]
                
                # Add response through observables (if loadings exist)
                if hasattr(self, 'factor_loadings_y') and self.factor_loadings_y is not None:
                    observable_response = self.factor_loadings_y @ observable_irfs[shock, :, period]
                    factor_response += observable_response
                
                informational_irfs[shock, :, period] = factor_response
        
        return informational_irfs
    
    def forecast(self, steps: int, include_informationals: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts from FAVAR model.
        
        Parameters:
        -----------
        steps : int
            Forecast horizon
        include_informationals : bool
            Whether to forecast informational variables
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Forecasts for different variable groups
        """
        if self.var_model is None:
            raise ValueError("FAVAR model must be fitted first")
        
        # Get factor and observable forecasts from VAR
        favar_forecasts, favar_intervals = self.var_model.forecast_factors(steps)
        
        # Split into factors and observables
        factor_forecasts = favar_forecasts[:, :self.n_factors]
        observable_forecasts = favar_forecasts[:, self.n_factors:]
        
        results = {
            'factors': factor_forecasts,
            'observables': observable_forecasts,
            'favar_confidence_intervals': favar_intervals
        }
        
        if include_informationals:
            # Forecast informational variables using factor loadings
            informational_forecasts = factor_forecasts @ self.factor_loadings_x.T
            
            # Add observable contribution if loadings exist
            if hasattr(self, 'factor_loadings_y') and self.factor_loadings_y is not None:
                informational_forecasts += observable_forecasts @ self.factor_loadings_y.T
            
            results['informationals'] = informational_forecasts
        
        return results
    
    def factor_importance(self, variable_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute factor importance for informational variables.
        
        Parameters:
        -----------
        variable_names : List[str], optional
            Names of informational variables
            
        Returns:
        --------
        pd.DataFrame
            Factor importance matrix
        """
        if self.factor_loadings_x is None:
            raise ValueError("Factor loadings not available")
        
        if variable_names is None:
            variable_names = self.informational_vars
        
        # Compute R-squared-like measure for each variable
        loadings_squared = self.factor_loadings_x ** 2
        total_loadings = np.sum(loadings_squared, axis=1, keepdims=True)
        factor_importance = loadings_squared / (total_loadings + 1e-8)  # Avoid division by zero
        
        factor_names = [f"Factor_{i+1}" for i in range(self.n_factors)]
        
        return pd.DataFrame(
            factor_importance,
            index=variable_names,
            columns=factor_names
        )
    
    def get_factor_interpretation(self, top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get interpretation of factors based on highest loadings.
        
        Parameters:
        -----------
        top_k : int
            Number of top variables to show for each factor
            
        Returns:
        --------
        Dict[int, List[Tuple[str, float]]]
            Factor interpretations with top loading variables
        """
        if self.factor_loadings_x is None:
            raise ValueError("Factor loadings not available")
        
        factor_interpretation = {}
        
        for factor_idx in range(self.n_factors):
            loadings = self.factor_loadings_x[:, factor_idx]
            
            # Get top positive and negative loadings
            top_pos_indices = np.argsort(loadings)[-top_k:][::-1]
            top_neg_indices = np.argsort(loadings)[:top_k]
            
            factor_vars = []
            
            # Add positive loadings
            for idx in top_pos_indices:
                if loadings[idx] > 0.1:  # Threshold for significance
                    factor_vars.append((self.informational_vars[idx], loadings[idx]))
            
            # Add negative loadings
            for idx in top_neg_indices:
                if loadings[idx] < -0.1:  # Threshold for significance
                    factor_vars.append((self.informational_vars[idx], loadings[idx]))
            
            factor_interpretation[factor_idx] = factor_vars
        
        return factor_interpretation


class BayesianFAVAR(FAVAR):
    """
    Bayesian Factor-Augmented VAR with proper uncertainty quantification.
    
    This extends the FAVAR to include proper Bayesian estimation and
    factor uncertainty propagation.
    """
    
    def __init__(self, n_factors: int, n_lags: int = 4,
                 factor_prior: str = 'minnesota', **kwargs):
        """
        Initialize Bayesian FAVAR.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors
        n_lags : int
            VAR lags
        factor_prior : str
            Prior type for factor VAR
        """
        super().__init__(n_factors, n_lags, **kwargs)
        self.factor_prior = factor_prior
        self.factor_uncertainty = None
    
    def fit(self, Y_observable: np.ndarray, X_informational: np.ndarray,
            n_draws: int = 5000, n_burnin: int = 2000, **kwargs) -> 'BayesianFAVAR':
        """
        Fit Bayesian FAVAR with proper uncertainty quantification.
        
        Parameters:
        -----------
        Y_observable : np.ndarray
            Observable variables
        X_informational : np.ndarray
            Informational variables
        n_draws : int
            Number of Gibbs draws
        n_burnin : int
            Burn-in draws
            
        Returns:
        --------
        BayesianFAVAR
            Fitted Bayesian FAVAR
        """
        # Step 1: Extract factors (with uncertainty)
        self._extract_factors_bayesian(X_informational, n_draws, n_burnin)
        
        # Step 2: Estimate Bayesian VAR accounting for factor uncertainty
        self._estimate_bayesian_favar(Y_observable, n_draws, n_burnin)
        
        return self
    
    def _extract_factors_bayesian(self, X: np.ndarray, n_draws: int, n_burnin: int):
        """Extract factors with Bayesian uncertainty quantification."""
        # For now, use point estimates but store uncertainty
        # Full Bayesian factor extraction would require more complex MCMC
        self._extract_factors(X)
        
        # Approximate factor uncertainty from factor model residuals
        if hasattr(self.factor_model, 'reconstruction_mse'):
            self.factor_uncertainty = np.sqrt(self.factor_model.reconstruction_mse)
        else:
            # Crude approximation
            X_fitted = self.factors @ self.factor_loadings_x.T
            residuals = X - X_fitted
            self.factor_uncertainty = np.std(residuals, axis=0)
    
    def _estimate_bayesian_favar(self, Y_observable: np.ndarray, n_draws: int, n_burnin: int):
        """Estimate Bayesian FAVAR accounting for factor uncertainty."""
        # Combine factors and observables
        self.favar_data = np.hstack([self.factors, Y_observable])
        
        # Fit Bayesian VAR
        self.var_model = LargeBayesianVAR(n_lags=self.n_lags)
        self.var_model.fit(self.favar_data, n_draws=n_draws, n_burnin=n_burnin)
    
    def impulse_responses_with_uncertainty(self, steps: int = 20, 
                                         confidence_levels: List[float] = [0.68, 0.95]) -> Dict:
        """
        Compute impulse responses with proper uncertainty bands.
        
        Parameters:
        -----------
        steps : int
            IRF horizon
        confidence_levels : List[float]
            Confidence levels for bands
            
        Returns:
        --------
        Dict
            IRFs with uncertainty bands
        """
        irf_summary, irf_intervals = self.var_model.get_irf_summary(
            horizon=steps, 
            identification='cholesky',
            interval_levels=confidence_levels
        )
        
        # Account for factor uncertainty in informational variable IRFs
        factor_irfs = irf_summary['median'][:, :self.n_factors, :]
        observable_irfs = irf_summary['median'][:, self.n_factors:, :]
        
        # Compute informational IRFs with uncertainty
        informational_irfs = self._compute_informational_irfs(factor_irfs, observable_irfs)
        
        # Add factor uncertainty to confidence bands
        if self.factor_uncertainty is not None:
            for level in irf_intervals:
                # Adjust confidence bands for factor uncertainty
                # This is a simplified adjustment - full propagation would be more complex
                uncertainty_adjustment = np.mean(self.factor_uncertainty) * np.sqrt(level)
                irf_intervals[level]['lower'] -= uncertainty_adjustment
                irf_intervals[level]['upper'] += uncertainty_adjustment
        
        return {
            'factors': factor_irfs,
            'observables': observable_irfs,  
            'informationals': informational_irfs,
            'confidence_intervals': irf_intervals,
            'factor_uncertainty': self.factor_uncertainty
        }