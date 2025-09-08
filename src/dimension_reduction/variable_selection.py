"""
Variable selection methods for large VAR systems.

This module implements various approaches for automatic variable selection
in high-dimensional VAR models, including spike-and-slab priors, LASSO-type
methods, and information-theoretic approaches.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from scipy.special import gammaln
from scipy.stats import invgamma, bernoulli
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings


class SpikeAndSlabVAR:
    """
    Spike-and-Slab variable selection for VAR models.
    
    This implements the George & McCulloch (1993) stochastic search
    variable selection (SSVS) approach adapted for VAR models.
    """
    
    def __init__(self, n_lags: int, spike_variance: float = 0.01,
                 slab_variance: float = 1.0, inclusion_prob: float = 0.5):
        """
        Initialize Spike-and-Slab VAR.
        
        Parameters:
        -----------
        n_lags : int
            Number of VAR lags
        spike_variance : float
            Variance for spike (near-zero) component
        slab_variance : float  
            Variance for slab (non-zero) component
        inclusion_prob : float
            Prior inclusion probability
        """
        self.n_lags = n_lags
        self.spike_variance = spike_variance
        self.slab_variance = slab_variance
        self.inclusion_prob = inclusion_prob
        
        # Model results
        self.inclusion_indicators = None
        self.coefficients = None
        self.inclusion_probabilities = None
        self.selected_variables = None
        
    def fit(self, Y: np.ndarray, n_draws: int = 10000, n_burnin: int = 5000,
            thin: int = 5) -> 'SpikeAndSlabVAR':
        """
        Fit Spike-and-Slab VAR using Gibbs sampling.
        
        Parameters:
        -----------
        Y : np.ndarray
            VAR data (T x n_vars)
        n_draws : int
            Total Gibbs draws
        n_burnin : int
            Burn-in draws
        thin : int
            Thinning interval
            
        Returns:
        --------
        SpikeAndSlabVAR
            Fitted model
        """
        T, n_vars = Y.shape
        n_coeffs = n_vars * self.n_lags + 1  # +1 for constant
        
        # Create design matrix
        X = self._create_design_matrix(Y)
        T_eff = X.shape[0]
        
        # Initialize storage
        n_retained = (n_draws - n_burnin) // thin
        coefficient_draws = np.zeros((n_retained, n_vars, n_coeffs))
        indicator_draws = np.zeros((n_retained, n_vars, n_coeffs - 1), dtype=int)  # No selection for constant
        sigma_draws = np.zeros((n_retained, n_vars, n_vars))
        
        # Initialize values
        gamma = np.random.binomial(1, self.inclusion_prob, (n_vars, n_coeffs - 1))
        beta = np.random.normal(0, 0.1, (n_coeffs, n_vars))
        Sigma = np.eye(n_vars)
        
        print(f"Starting Gibbs sampling for Spike-and-Slab VAR...")
        
        draw_idx = 0
        for draw in range(n_draws):
            if draw % 1000 == 0:
                print(f"Draw {draw}/{n_draws}")
            
            # Sample coefficients and indicators equation by equation
            for eq in range(n_vars):
                beta[:, eq], gamma[eq, :] = self._sample_equation_coeffs(
                    Y[:, eq], X, Sigma[eq, eq], gamma[eq, :], eq
                )
            
            # Sample error covariance
            residuals = Y[self.n_lags:] - X @ beta
            Sigma = self._sample_covariance(residuals, T_eff, n_vars)
            
            # Store draws after burn-in
            if draw >= n_burnin and (draw - n_burnin) % thin == 0:
                coefficient_draws[draw_idx] = beta.T
                indicator_draws[draw_idx] = gamma
                sigma_draws[draw_idx] = Sigma
                draw_idx += 1
        
        # Compute posterior summaries
        self.coefficients = np.mean(coefficient_draws, axis=0)
        self.inclusion_probabilities = np.mean(indicator_draws, axis=0)
        self.inclusion_indicators = (self.inclusion_probabilities > 0.5).astype(int)
        
        # Store draws for inference
        self.coefficient_draws = coefficient_draws
        self.sigma_draws = sigma_draws
        
        print("Gibbs sampling completed!")
        return self
    
    def _create_design_matrix(self, Y: np.ndarray) -> np.ndarray:
        """Create VAR design matrix."""
        T, n_vars = Y.shape
        T_eff = T - self.n_lags
        
        X = np.zeros((T_eff, n_vars * self.n_lags + 1))
        
        # Lagged variables
        for lag in range(self.n_lags):
            start_col = lag * n_vars
            end_col = (lag + 1) * n_vars
            X[:, start_col:end_col] = Y[self.n_lags - 1 - lag : T - 1 - lag]
        
        # Constant
        X[:, -1] = 1
        
        return X
    
    def _sample_equation_coeffs(self, y: np.ndarray, X: np.ndarray, 
                               sigma2: float, gamma_current: np.ndarray,
                               eq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample coefficients and indicators for one equation."""
        n_coeffs = X.shape[1]
        beta_new = np.zeros(n_coeffs)
        gamma_new = gamma_current.copy()
        
        # Sample constant term (always included)
        X_const = X[:, [-1]]
        prior_var = self.slab_variance
        posterior_var = 1 / (1/prior_var + (X_const.T @ X_const)[0, 0] / sigma2)
        posterior_mean = posterior_var * (X_const.T @ y)[0] / sigma2
        beta_new[-1] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
        
        # Sample coefficients and indicators for lagged terms
        for j in range(n_coeffs - 1):  # Exclude constant
            X_j = X[:, [j]]
            
            # Current inclusion state
            if gamma_current[j] == 1:
                # Currently included - sample from slab
                prior_var = self.slab_variance
                posterior_var = 1 / (1/prior_var + (X_j.T @ X_j)[0, 0] / sigma2)
                posterior_mean = posterior_var * (X_j.T @ y)[0] / sigma2
                beta_candidate = np.random.normal(posterior_mean, np.sqrt(posterior_var))
                
                # Probability of staying included
                log_prob_include = self._log_marginal_likelihood(
                    y, X_j, sigma2, self.slab_variance
                ) + np.log(self.inclusion_prob)
                
                log_prob_exclude = self._log_marginal_likelihood(
                    y, X_j, sigma2, self.spike_variance
                ) + np.log(1 - self.inclusion_prob)
                
                # Sample inclusion indicator
                prob_include = 1 / (1 + np.exp(log_prob_exclude - log_prob_include))
                gamma_new[j] = np.random.binomial(1, prob_include)
                
                if gamma_new[j] == 1:
                    beta_new[j] = beta_candidate
                else:
                    # Sample from spike
                    prior_var = self.spike_variance
                    posterior_var = 1 / (1/prior_var + (X_j.T @ X_j)[0, 0] / sigma2)
                    posterior_mean = posterior_var * (X_j.T @ y)[0] / sigma2
                    beta_new[j] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
            
            else:
                # Currently excluded - similar logic with spike as current
                prior_var = self.spike_variance
                posterior_var = 1 / (1/prior_var + (X_j.T @ X_j)[0, 0] / sigma2)
                posterior_mean = posterior_var * (X_j.T @ y)[0] / sigma2
                beta_candidate = np.random.normal(posterior_mean, np.sqrt(posterior_var))
                
                # Probability of inclusion
                log_prob_include = self._log_marginal_likelihood(
                    y, X_j, sigma2, self.slab_variance
                ) + np.log(self.inclusion_prob)
                
                log_prob_exclude = self._log_marginal_likelihood(
                    y, X_j, sigma2, self.spike_variance
                ) + np.log(1 - self.inclusion_prob)
                
                prob_include = 1 / (1 + np.exp(log_prob_exclude - log_prob_include))
                gamma_new[j] = np.random.binomial(1, prob_include)
                
                if gamma_new[j] == 1:
                    # Sample from slab
                    prior_var = self.slab_variance
                    posterior_var = 1 / (1/prior_var + (X_j.T @ X_j)[0, 0] / sigma2)
                    posterior_mean = posterior_var * (X_j.T @ y)[0] / sigma2
                    beta_new[j] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
                else:
                    beta_new[j] = beta_candidate
        
        return beta_new, gamma_new
    
    def _log_marginal_likelihood(self, y: np.ndarray, X: np.ndarray, 
                                sigma2: float, prior_var: float) -> float:
        """Compute log marginal likelihood for variable inclusion."""
        n = len(y)
        XtX = (X.T @ X)[0, 0]
        Xty = (X.T @ y)[0]
        
        # Marginal likelihood components
        log_ml = -0.5 * np.log(2 * np.pi * sigma2) * n
        log_ml += -0.5 * np.log(1 + XtX * prior_var / sigma2)
        log_ml += -0.5 * (y.T @ y - (Xty**2 * prior_var) / (sigma2 + XtX * prior_var)) / sigma2
        
        return log_ml
    
    def _sample_covariance(self, residuals: np.ndarray, T: int, n_vars: int) -> np.ndarray:
        """Sample error covariance matrix."""
        # Inverse Wishart prior
        nu_prior = n_vars + 2
        S_prior = np.eye(n_vars)
        
        # Posterior parameters
        nu_post = nu_prior + T
        S_post = S_prior + residuals.T @ residuals
        
        # Sample from inverse Wishart
        # Using the fact that if W ~ Wishart(S^{-1}, nu) then W^{-1} ~ InvWishart(S, nu)
        L = np.linalg.cholesky(S_post)
        Z = np.random.standard_normal((nu_post, n_vars))
        W = Z.T @ Z
        Sigma_inv = L @ W @ L.T
        
        return np.linalg.inv(Sigma_inv)
    
    def get_selected_variables(self, threshold: float = 0.5) -> Dict[int, List[int]]:
        """
        Get selected variables for each equation.
        
        Parameters:
        -----------
        threshold : float
            Inclusion probability threshold
            
        Returns:
        --------
        Dict[int, List[int]]
            Selected variables by equation
        """
        if self.inclusion_probabilities is None:
            raise ValueError("Model must be fitted first")
        
        selected_vars = {}
        n_vars = self.inclusion_probabilities.shape[0]
        
        for eq in range(n_vars):
            selected_indices = np.where(self.inclusion_probabilities[eq] > threshold)[0]
            selected_vars[eq] = selected_indices.tolist()
        
        return selected_vars
    
    def plot_inclusion_probabilities(self, variable_names: Optional[List[str]] = None):
        """Plot inclusion probabilities heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if variable_names is None:
                variable_names = [f"Var_{i+1}" for i in range(self.inclusion_probabilities.shape[0])]
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.inclusion_probabilities, 
                       xticklabels=[f"Lag_{i+1}_Var_{j+1}" 
                                  for i in range(self.n_lags) 
                                  for j in range(len(variable_names))],
                       yticklabels=variable_names,
                       cmap='viridis', vmin=0, vmax=1)
            plt.title('Variable Inclusion Probabilities')
            plt.xlabel('Lagged Variables')
            plt.ylabel('Equations')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib/Seaborn not available for plotting")


class LassoVAR:
    """
    LASSO-based variable selection for VAR models.
    
    Uses penalized regression with cross-validation to select
    relevant variables in each VAR equation.
    """
    
    def __init__(self, n_lags: int, alpha: Optional[float] = None,
                 cv_folds: int = 5, standardize: bool = True):
        """
        Initialize LASSO VAR.
        
        Parameters:
        -----------
        n_lags : int
            Number of VAR lags
        alpha : float, optional
            LASSO penalty parameter (if None, use CV)
        cv_folds : int
            Cross-validation folds
        standardize : bool
            Whether to standardize variables
        """
        self.n_lags = n_lags
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.standardize = standardize
        
        # Model results
        self.coefficients = None
        self.selected_variables = None
        self.cv_scores = None
        self.scaler = StandardScaler() if standardize else None
        
    def fit(self, Y: np.ndarray) -> 'LassoVAR':
        """
        Fit LASSO VAR model.
        
        Parameters:
        -----------
        Y : np.ndarray
            VAR data (T x n_vars)
            
        Returns:
        --------
        LassoVAR
            Fitted model
        """
        T, n_vars = Y.shape
        
        # Standardize if requested
        if self.standardize:
            Y_scaled = self.scaler.fit_transform(Y)
        else:
            Y_scaled = Y.copy()
        
        # Create design matrix
        X = self._create_design_matrix(Y_scaled)
        Y_dep = Y_scaled[self.n_lags:]
        
        # Fit LASSO for each equation
        self.coefficients = np.zeros((n_vars, X.shape[1]))
        self.cv_scores = {}
        self.selected_variables = {}
        
        print(f"Fitting LASSO VAR with {n_vars} equations...")
        
        for eq in range(n_vars):
            print(f"Equation {eq + 1}/{n_vars}")
            
            if self.alpha is None:
                # Use cross-validation to select alpha
                lasso_cv = LassoCV(cv=self.cv_folds, random_state=42, max_iter=2000)
                lasso_cv.fit(X, Y_dep[:, eq])
                
                self.coefficients[eq] = lasso_cv.coef_
                self.cv_scores[eq] = {
                    'alpha_': lasso_cv.alpha_,
                    'mse_path_': lasso_cv.mse_path_,
                    'alphas_': lasso_cv.alphas_
                }
            else:
                # Use fixed alpha
                from sklearn.linear_model import Lasso
                lasso = Lasso(alpha=self.alpha, max_iter=2000)
                lasso.fit(X, Y_dep[:, eq])
                self.coefficients[eq] = lasso.coef_
            
            # Record selected variables
            selected_indices = np.where(np.abs(self.coefficients[eq]) > 1e-6)[0]
            self.selected_variables[eq] = selected_indices.tolist()
        
        print("LASSO VAR fitting completed!")
        return self
    
    def _create_design_matrix(self, Y: np.ndarray) -> np.ndarray:
        """Create VAR design matrix."""
        T, n_vars = Y.shape
        T_eff = T - self.n_lags
        
        X = np.zeros((T_eff, n_vars * self.n_lags + 1))
        
        # Lagged variables
        for lag in range(self.n_lags):
            start_col = lag * n_vars
            end_col = (lag + 1) * n_vars
            X[:, start_col:end_col] = Y[self.n_lags - 1 - lag : T - 1 - lag]
        
        # Constant
        X[:, -1] = 1
        
        return X
    
    def get_selection_summary(self, variable_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get variable selection summary.
        
        Parameters:
        -----------
        variable_names : List[str], optional
            Variable names
            
        Returns:
        --------
        pd.DataFrame
            Selection summary
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        n_vars = self.coefficients.shape[0]
        
        if variable_names is None:
            variable_names = [f"Var_{i+1}" for i in range(n_vars)]
        
        # Create coefficient names
        coeff_names = []
        for lag in range(self.n_lags):
            for var_name in variable_names:
                coeff_names.append(f"{var_name}_lag{lag+1}")
        coeff_names.append("Constant")
        
        # Create summary dataframe
        summary_data = []
        
        for eq in range(n_vars):
            for coeff_idx, coeff_val in enumerate(self.coefficients[eq]):
                if abs(coeff_val) > 1e-6:  # Non-zero coefficient
                    summary_data.append({
                        'Equation': variable_names[eq],
                        'Variable': coeff_names[coeff_idx],
                        'Coefficient': coeff_val,
                        'Selected': True
                    })
        
        return pd.DataFrame(summary_data)
    
    def plot_regularization_path(self, equation: int = 0):
        """Plot LASSO regularization path for specific equation."""
        try:
            import matplotlib.pyplot as plt
            
            if self.cv_scores is None or equation not in self.cv_scores:
                warnings.warn("CV scores not available for plotting")
                return
            
            alphas = self.cv_scores[equation]['alphas_']
            mse_path = self.cv_scores[equation]['mse_path_']
            optimal_alpha = self.cv_scores[equation]['alpha_']
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(np.log10(alphas), np.mean(mse_path, axis=1))
            plt.axvline(np.log10(optimal_alpha), color='red', linestyle='--', 
                       label=f'Optimal α = {optimal_alpha:.4f}')
            plt.xlabel('log₁₀(α)')
            plt.ylabel('Cross-Validation MSE')
            plt.title(f'Equation {equation + 1}: CV Error')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.fill_between(np.log10(alphas), 
                           np.mean(mse_path, axis=1) - np.std(mse_path, axis=1),
                           np.mean(mse_path, axis=1) + np.std(mse_path, axis=1),
                           alpha=0.3)
            plt.plot(np.log10(alphas), np.mean(mse_path, axis=1))
            plt.axvline(np.log10(optimal_alpha), color='red', linestyle='--')
            plt.xlabel('log₁₀(α)')
            plt.ylabel('Cross-Validation MSE')
            plt.title(f'Equation {equation + 1}: CV Error with std')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")


class InformationTheoreticSelection:
    """
    Information-theoretic variable selection for VAR models.
    
    Uses mutual information and transfer entropy to identify
    relevant variables and their causal relationships.
    """
    
    def __init__(self, n_lags: int = 4, significance_level: float = 0.05):
        """
        Initialize information-theoretic selection.
        
        Parameters:
        -----------
        n_lags : int
            Maximum lags to consider
        significance_level : float
            Significance level for tests
        """
        self.n_lags = n_lags
        self.significance_level = significance_level
        
        # Results
        self.mutual_information_matrix = None
        self.transfer_entropy_matrix = None
        self.selected_connections = None
        
    def fit(self, Y: np.ndarray, variable_names: Optional[List[str]] = None) -> 'InformationTheoreticSelection':
        """
        Fit information-theoretic variable selection.
        
        Parameters:
        -----------
        Y : np.ndarray
            VAR data (T x n_vars)
        variable_names : List[str], optional
            Variable names
            
        Returns:
        --------
        InformationTheoreticSelection
            Fitted selection model
        """
        T, n_vars = Y.shape
        
        if variable_names is None:
            variable_names = [f"Var_{i+1}" for i in range(n_vars)]
        
        self.variable_names = variable_names
        
        # Compute mutual information matrix
        print("Computing mutual information matrix...")
        self.mutual_information_matrix = self._compute_mutual_information(Y)
        
        # Compute transfer entropy matrix
        print("Computing transfer entropy matrix...")
        self.transfer_entropy_matrix = self._compute_transfer_entropy(Y)
        
        # Select significant connections
        self.selected_connections = self._select_connections()
        
        print(f"Information-theoretic selection completed!")
        print(f"Selected {len(self.selected_connections)} significant connections")
        
        return self
    
    def _compute_mutual_information(self, Y: np.ndarray) -> np.ndarray:
        """Compute mutual information between all variable pairs."""
        n_vars = Y.shape[1]
        mi_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    mi_matrix[i, j] = mutual_info_regression(
                        Y[:, [i]], Y[:, j], random_state=42
                    )[0]
        
        return mi_matrix
    
    def _compute_transfer_entropy(self, Y: np.ndarray) -> np.ndarray:
        """Compute transfer entropy matrix."""
        T, n_vars = Y.shape
        te_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Compute TE from j to i
                    te_matrix[i, j] = self._transfer_entropy(Y[:, i], Y[:, j])
        
        return te_matrix
    
    def _transfer_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute transfer entropy from y to x.
        
        TE(Y->X) = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, Y_{t-1})
        """
        # This is a simplified implementation
        # For proper transfer entropy, more sophisticated estimation is needed
        
        # Create lagged data
        T = len(x)
        X_t = x[self.n_lags:]
        X_lag = np.column_stack([x[self.n_lags-i-1:T-i-1] for i in range(self.n_lags)])
        Y_lag = np.column_stack([y[self.n_lags-i-1:T-i-1] for i in range(self.n_lags)])
        
        # Conditional mutual information approximation
        # MI(X_t, Y_{t-1} | X_{t-1})
        
        # Simplified using regression-based approach
        from sklearn.metrics import mutual_info_score
        
        # Discretize for mutual information calculation
        X_t_disc = self._discretize(X_t)
        X_lag_disc = self._discretize(X_lag.mean(axis=1))  # Simplified
        Y_lag_disc = self._discretize(Y_lag.mean(axis=1))  # Simplified
        
        # Approximate transfer entropy
        mi_full = mutual_info_score(X_t_disc, Y_lag_disc)
        
        return mi_full
    
    def _discretize(self, x: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretize continuous variable for MI calculation."""
        return np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    
    def _select_connections(self) -> List[Tuple[int, int, str, float]]:
        """Select significant connections based on information criteria."""
        connections = []
        n_vars = self.mutual_information_matrix.shape[0]
        
        # Threshold based on empirical distribution
        mi_threshold = np.percentile(self.mutual_information_matrix[self.mutual_information_matrix > 0], 
                                   (1 - self.significance_level) * 100)
        te_threshold = np.percentile(self.transfer_entropy_matrix[self.transfer_entropy_matrix > 0], 
                                   (1 - self.significance_level) * 100)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Check mutual information
                    if self.mutual_information_matrix[i, j] > mi_threshold:
                        connections.append((i, j, 'MI', self.mutual_information_matrix[i, j]))
                    
                    # Check transfer entropy
                    if self.transfer_entropy_matrix[i, j] > te_threshold:
                        connections.append((j, i, 'TE', self.transfer_entropy_matrix[i, j]))
        
        return connections
    
    def get_connection_summary(self) -> pd.DataFrame:
        """Get summary of selected connections."""
        if self.selected_connections is None:
            raise ValueError("Model must be fitted first")
        
        summary_data = []
        for connection in self.selected_connections:
            from_var, to_var, conn_type, strength = connection
            summary_data.append({
                'From': self.variable_names[from_var],
                'To': self.variable_names[to_var],
                'Type': conn_type,
                'Strength': strength
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_network(self):
        """Plot variable network based on selected connections."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, name in enumerate(self.variable_names):
                G.add_node(i, label=name)
            
            # Add edges
            for connection in self.selected_connections:
                from_var, to_var, conn_type, strength = connection
                G.add_edge(from_var, to_var, weight=strength, type=conn_type)
            
            # Plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, alpha=0.7)
            
            # Draw edges with different colors for MI and TE
            mi_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'MI']
            te_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'TE']
            
            nx.draw_networkx_edges(G, pos, edgelist=mi_edges, 
                                 edge_color='blue', alpha=0.5, label='Mutual Info')
            nx.draw_networkx_edges(G, pos, edgelist=te_edges, 
                                 edge_color='red', alpha=0.5, label='Transfer Entropy')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, 
                                  {i: name for i, name in enumerate(self.variable_names)},
                                  font_size=10)
            
            plt.title('Variable Network based on Information Theory')
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("NetworkX/Matplotlib not available for network plotting")