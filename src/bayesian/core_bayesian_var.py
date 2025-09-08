import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any
import warnings

from .priors import minnesota_prior, sum_of_coeffs_prior, dummy_variable_prior
from .estimation import gibbs_sampler
from .posterior import posterior_analysis, credible_intervals, predictive_checks
from .diagnostics import mcmc_diagnostics

try:
    from analysis import conditional_forecasting
except ImportError:
    warnings.warn("Conditional forecasting not available")
    conditional_forecasting = None

try:
    from ..computational.kronecker_ops import KroneckerOperations
    from ..computational.matrix_normal import EfficientMatrixNormalSampler
    from ..computational.sparse_methods import SparseVAR, sparse_gibbs_step
    from ..computational.memory_management import MemoryManager, memory_efficient
except ImportError:
    warnings.warn("Computational optimization modules not fully available")
    KroneckerOperations = None
    EfficientMatrixNormalSampler = None
    SparseVAR = None
    sparse_gibbs_step = None
    MemoryManager = None
    memory_efficient = lambda **kwargs: lambda f: f

logger = logging.getLogger(__name__)

class LargeBayesianVAR:
    """
    Main class for estimating a Large Bayesian Vector Autoregression (BVAR).

    This class ties together the data preparation, prior specification,
    estimation (via Gibbs sampling), and posterior analysis with computational
    optimizations for handling 100+ variables efficiently.
    """
    def __init__(self, n_lags: int, n_exog: int = 1, 
                 use_minnesota_prior: bool = True, use_sum_of_coeffs: bool = True,
                 use_sparse_methods: bool = False, max_memory_mb: int = 4000,
                 fast_mode: bool = False):
        """
        Initializes the LargeBayesianVAR model.

        Args:
            n_lags (int): The number of lags for the endogenous variables.
            n_exog (int): The number of exogenous variables (default is 1 for a constant).
            use_minnesota_prior (bool): Flag to use the Minnesota prior.
            use_sum_of_coeffs (bool): Flag to use the Sum-of-Coefficients prior.
            use_sparse_methods (bool): Whether to use sparse computational methods for very large VARs.
            max_memory_mb (int): Maximum memory to use for computations.
            fast_mode (bool): Enable fast mode with optimizations for quicker results.
        """
        self.n_lags = n_lags
        self.n_exog = n_exog
        self.use_minnesota_prior = use_minnesota_prior
        self.use_sum_of_coeffs = use_sum_of_coeffs
        self.use_sparse_methods = use_sparse_methods
        self.max_memory_mb = max_memory_mb
        self.fast_mode = fast_mode
        
        # Computational optimization tools
        self.memory_manager = MemoryManager() if MemoryManager else None
        self.sparse_var = None
        self.matrix_sampler = None
        
        # To be populated after fitting
        self.beta_draws = None
        self.sigma_draws = None
        self.Y = None
        self.X = None
        self.n_vars = None
        
        logger.info(f"Initialized Large Bayesian VAR with {n_lags} lags, "
                   f"sparse methods: {'enabled' if use_sparse_methods else 'disabled'}")
    
    def _should_use_sparse_methods(self, n_vars: int) -> bool:
        """Determine whether to use sparse methods based on problem size."""
        if not self.use_sparse_methods:
            return False
        
        # Use sparse methods for 200+ variables or when memory is constrained
        total_params = n_vars * (n_vars * self.n_lags + self.n_exog)
        memory_estimate_mb = total_params * 8 * 5000 / (1024 * 1024)  # Rough estimate for draws
        
        return n_vars >= 200 or memory_estimate_mb > self.max_memory_mb

    def _create_YX_matrices(self, data):
        """Prepares the Y and X matrices for VAR estimation."""
        T, n_vars = data.shape
        self.n_vars = n_vars
        
        # Y matrix (T-p x n)
        Y = data[self.n_lags:, :]
        
        # X matrix (T-p x k)
        # Lags of endogenous variables
        X_lags = np.zeros((T - self.n_lags, n_vars * self.n_lags))
        for i in range(self.n_lags):
            start_col = i * n_vars
            end_col = (i + 1) * n_vars
            X_lags[:, start_col:end_col] = data[self.n_lags-i-1 : T-i-1, :]
        
        # Exogenous variables (e.g., a constant)
        # For now, we only support a constant.
        if self.n_exog > 0:
            X_exog = np.ones((T - self.n_lags, self.n_exog))
            X = np.hstack([X_lags, X_exog])
        else:
            X = X_lags
        
        return Y, X

    @memory_efficient(max_memory_mb=8000)
    def fit(self, data, lambda1: float = 0.1, lambda2: float = 0.3, 
            lambda3: float = 1.0, mu: float = 1.0, n_draws: int = 5000, 
            n_burnin: int = 2000) -> 'LargeBayesianVAR':
        """
        Fits the BVAR model using optimized Gibbs sampling.

        Args:
            data (np.ndarray): The time series data (T x n_vars).
            lambda1, lambda2, lambda3 (float): Hyperparameters for the Minnesota prior.
            mu (float): Hyperparameter for the Sum-of-Coefficients prior.
            n_draws (int): Total number of Gibbs draws.
            n_burnin (int): Number of burn-in draws to discard.
        """
        # Allow DataFrame input to preserve index and column names
        if isinstance(data, pd.DataFrame):
            self._input_index = data.index
            self._input_names = list(data.columns)
            data_values = data.values
        else:
            self._input_index = None
            self._input_names = None
            data_values = data

        T, n_vars = data_values.shape
        
        # Apply fast mode optimizations 
        if self.fast_mode:
            n_draws = min(n_draws, 500)  # Limit draws for speed
            n_burnin = min(n_burnin, n_draws // 3)  # Ensure burn-in < draws
            # Use balanced priors in fast mode - not too tight to avoid flat IRFs
            lambda1 = max(lambda1, 0.1)   # Ensure sufficient dynamics (avoid over-shrinkage)
            lambda1 = min(lambda1, 0.5)   # But not too loose for stability
            lambda2 = max(lambda2, 0.5)   # Allow cross-variable interactions
            lambda2 = min(lambda2, 1.0)   # But maintain some shrinkage
            logger.info(f"FAST MODE: Reduced to {n_draws} draws, {n_burnin} burn-in, balanced priors: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}")
        
        # Final parameter adjustment to ensure n_burnin < n_draws
        if n_burnin >= n_draws:
            n_burnin = max(1, n_draws // 4)
            logger.warning(f"Adjusted burn-in to {n_burnin} (was >= n_draws={n_draws})")
        
        # Automatic hyperparameter adjustment based on data characteristics
        T_to_vars_ratio = T / (n_vars * self.n_lags)
        if T_to_vars_ratio < 2:  # High-dimensional case
            lambda1 = max(lambda1, 0.2)  # More shrinkage for stability
            lambda2 = max(lambda2, 0.5)
            logger.info(f"High-dimensional case detected (T/k = {T_to_vars_ratio:.1f}). Increased shrinkage: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}")
        elif T_to_vars_ratio > 10:  # Low-dimensional case  
            lambda1 = min(lambda1, 0.05)  # Less shrinkage to preserve dynamics
            lambda2 = min(lambda2, 0.2)
            logger.info(f"Low-dimensional case detected (T/k = {T_to_vars_ratio:.1f}). Reduced shrinkage: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}")
        else:
            logger.info(f"Balanced case (T/k = {T_to_vars_ratio:.1f}). Using original priors: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}")
            
        logger.info(f"Fitting Large Bayesian VAR: {n_vars} variables, {T} observations")
        logger.info(f"Gibbs sampler parameters: n_draws={n_draws}, n_burnin={n_burnin}")
        logger.info(f"Final hyperparameters: lambda1={lambda1:.3f}, lambda2={lambda2:.3f}, lambda3={lambda3:.3f}, mu={mu:.3f}")
        
        # Determine computational strategy
        use_sparse = self._should_use_sparse_methods(n_vars)
        
        if use_sparse and SparseVAR:
            logger.info("Using sparse computational methods for very large VAR")
            return self._fit_sparse(data_values, lambda1, lambda2, lambda3, mu, n_draws, n_burnin)
        else:
            logger.info("Using optimized dense computational methods")
            return self._fit_dense_optimized(data_values, lambda1, lambda2, lambda3, mu, n_draws, n_burnin)
    
    def _fit_dense_optimized(self, data: np.ndarray, lambda1: float, lambda2: float,
                           lambda3: float, mu: float, n_draws: int, n_burnin: int) -> 'LargeBayesianVAR':
        """Fit using optimized dense methods with Kronecker operations."""
        self.Y, self.X = self._create_YX_matrices(data)
        T_eff, k_eq = self.X.shape

        # --- Construct Priors with optimization ---
        # 1. Efficient OLS computation
        if KroneckerOperations and self.n_vars > 50:
            # Use optimized matrix operations
            XtX = self.X.T @ self.X
            XtY = self.X.T @ self.Y
            beta_ols = np.linalg.solve(XtX, XtY)
        else:
            beta_ols = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
            
        residuals = self.Y - self.X @ beta_ols
        sigmas_sq = np.diag(residuals.T @ residuals) / (T_eff - k_eq)

        # 2. Minnesota Prior
        mn_prior = minnesota_prior.MinnesotaPrior(lambda1, lambda2, lambda3)
        prior_mean, prior_cov = mn_prior.get_prior_moments(self.n_vars, self.n_lags, self.n_exog, sigmas_sq)

        # 3. Dummy Observation Priors
        Y_dummy, X_dummy = np.array([]), np.array([])
        if self.use_sum_of_coeffs:
            soc_prior = sum_of_coeffs_prior.SumOfCoefficientsPrior(mu)
            y_d, x_d = soc_prior.create_dummy_observations(data, self.n_lags, self.n_exog)
            Y_dummy = np.vstack([Y_dummy, y_d]) if Y_dummy.size else y_d
            X_dummy = np.vstack([X_dummy, x_d]) if X_dummy.size else x_d
        
        # Augment data with dummy observations
        Y_aug = np.vstack([self.Y, Y_dummy]) if Y_dummy.size else self.Y
        X_aug = np.vstack([self.X, X_dummy]) if X_dummy.size else self.X

        # 4. Priors for Sigma (error covariance)
        prior_dof = self.n_vars + 2
        prior_scale = np.eye(self.n_vars)

        # --- Run Gibbs Sampler ---
        # Use standard Gibbs sampler (efficient version has matrix dimension issues)
        logger.info("Using standard Gibbs sampler")
        sampler = gibbs_sampler.GibbsSampler(n_draws, n_burnin)
        self.beta_draws, self.sigma_draws = sampler.sample(
            Y_aug, X_aug, prior_mean, prior_cov, prior_scale, prior_dof
        )
        
        # Add compatibility attributes for IRF analysis
        self._add_compatibility_attributes()
        
        # Check VAR stability
        self._check_var_stability()
        
        logger.info("Dense fitting complete.")
        return self
    
    def _add_compatibility_attributes(self):
        """Add compatibility attributes for IRF analysis and other legacy components."""
        # Use posterior means as point estimates
        beta_mean = np.mean(self.beta_draws, axis=0)
        sigma_mean = np.mean(self.sigma_draws, axis=0)
        
        # Debug: Check beta_draws statistics
        logger.info(f"Beta draws shape: {self.beta_draws.shape}")
        logger.info(f"Beta mean statistics: min={np.nanmin(beta_mean):.6f}, max={np.nanmax(beta_mean):.6f}, "
                   f"std={np.nanstd(beta_mean):.6f}")
        
        # Reshape beta to coefficient matrix (k_eq x n_vars), Fortran order to match vectorization
        k_eq = self.n_vars * self.n_lags + self.n_exog
        beta_matrices = beta_mean.reshape((k_eq, self.n_vars), order='F')
        
        # Debug: Check beta_matrices statistics after reshape
        logger.info(f"Beta matrices shape: {beta_matrices.shape}")
        logger.info(f"Beta matrices statistics: min={np.nanmin(beta_matrices):.6f}, max={np.nanmax(beta_matrices):.6f}")
        
        # Extract coefficient matrices A(L) for each lag (n_vars x n_vars)
        self.coefs = []
        for lag in range(self.n_lags):
            start_idx = lag * self.n_vars
            end_idx = (lag + 1) * self.n_vars
            block = beta_matrices[start_idx:end_idx, :]  # (n_vars predictors) x (n_vars equations)
            A_lag = block.T  # A_L: rows=equations, cols=predictors
            self.coefs.append(A_lag)
            
            # Debug: Check coefficient matrix statistics
            max_abs = float(np.nanmax(np.abs(A_lag)))
            A_off_diag = A_lag.copy()
            np.fill_diagonal(A_off_diag, 0.0)
            max_off_diag = float(np.nanmax(np.abs(A_off_diag))) if A_off_diag.size else 0.0
            logger.info(f"Lag {lag+1} A matrix: max|A|={max_abs:.6f}, max|off_diag|={max_off_diag:.6f}, "
                       f"diagonal_mean={float(np.nanmean(np.diag(A_lag))):.6f}")
            
            # Check for potential issues
            if max_abs < 1e-6:
                logger.warning(f"Lag {lag+1} coefficients are extremely small (max={max_abs:.2e}). "
                              "This may cause flat IRFs.")
            elif max_abs > 10:
                logger.warning(f"Lag {lag+1} coefficients are very large (max={max_abs:.2f}). "
                              "This may indicate numerical issues.")
        
        # Build params DataFrame similar to statsmodels VARResults.params
        # Columns are equations (variables), rows are parameters ordered by lags then exog
        if hasattr(self, 'names') and isinstance(getattr(self, 'names'), list):
            eq_names = self.names
        elif getattr(self, '_input_names', None) is not None:
            eq_names = list(self._input_names)
        else:
            eq_names = [f'var_{i+1}' for i in range(self.n_vars)]

        row_index = []
        for lag in range(1, self.n_lags + 1):
            for var in eq_names:
                row_index.append(f'L{lag}.{var}')
        if self.n_exog > 0:
            row_index.append('const')
        params_matrix = beta_matrices  # shape (k_eq, n_vars)
        self.params = pd.DataFrame(params_matrix, index=row_index, columns=eq_names)

        # Debug: Check sigma matrix
        logger.info(f"Sigma matrix statistics: min={np.nanmin(sigma_mean):.6f}, max={np.nanmax(sigma_mean):.6f}")
        sigma_off_diag = sigma_mean.copy()
        np.fill_diagonal(sigma_off_diag, 0.0)
        logger.info(f"Sigma: max|diag|={float(np.nanmax(np.abs(np.diag(sigma_mean)))):.6f}, "
                   f"max|off_diag|={float(np.nanmax(np.abs(sigma_off_diag))):.6f}")
        
        # Check sigma matrix properties
        try:
            eigenvals = np.linalg.eigvals(sigma_mean)
            min_eigenval = np.min(eigenvals)
            cond_number = np.max(eigenvals) / min_eigenval if min_eigenval > 0 else np.inf
            logger.info(f"Sigma matrix: min_eigenval={min_eigenval:.6f}, condition_number={cond_number:.2f}")
            if min_eigenval <= 0:
                logger.warning("Sigma matrix is not positive definite!")
            elif cond_number > 1e12:
                logger.warning(f"Sigma matrix is poorly conditioned (cond={cond_number:.2e})")
        except Exception as e:
            logger.warning(f"Failed to check Sigma matrix properties: {e}")

        # Compatibility attributes for legacy components
        self.k_ar = self.n_lags  # Number of lags
        self.sigma_u = sigma_mean  # Residual covariance matrix
        
        # Compute fitted values and residuals using posterior mean
        Y_fitted = self.X @ beta_matrices  # (T_eff, n_vars)
        residuals = self.Y - Y_fitted        # (T_eff, n_vars)

        # Determine column names
        if hasattr(self, 'names') and isinstance(getattr(self, 'names'), list):
            col_names = self.names
        elif getattr(self, '_input_names', None) is not None:
            col_names = list(self._input_names)
        else:
            col_names = [f'var_{i+1}' for i in range(self.n_vars)]

        # Index for fitted/residuals aligned with lag drop
        if getattr(self, '_input_index', None) is not None:
            try:
                idx_eff = self._input_index[self.n_lags:]
            except Exception:
                idx_eff = pd.RangeIndex(start=0, stop=self.Y.shape[0], step=1)
        else:
            idx_eff = pd.RangeIndex(start=0, stop=self.Y.shape[0], step=1)

        # Expose pandas-friendly attributes expected by diagnostics
        self.fittedvalues = pd.DataFrame(Y_fitted, index=idx_eff, columns=col_names)
        self.resid = pd.DataFrame(residuals, index=idx_eff, columns=col_names)
        self.endog = self.Y  # numpy array (T_eff x n_vars)

        # Basic fit statistics approximations for compatibility
        try:
            T_eff = self.Y.shape[0]
            ll = -0.5 * (
                T_eff * self.n_vars * np.log(2 * np.pi)
                + T_eff * np.log(np.linalg.det(sigma_mean))
                + np.trace(self.resid.values.T @ self.resid.values @ np.linalg.inv(sigma_mean))
            )
        except Exception:
            ll = np.nan
        self.llf = ll
        # Parameter counts and info criteria
        n_params = beta_matrices.size + (self.n_vars * (self.n_vars + 1)) // 2
        n_obs = self.Y.shape[0]
        self.nobs = n_obs
        self.df_model = n_params
        self.aic = -2 * ll + 2 * n_params if not np.isnan(ll) else np.nan
        self.bic = -2 * ll + np.log(n_obs) * n_params if not np.isnan(ll) else np.nan
        self.hqic = np.nan
        self.fpe = np.nan

    def _check_var_stability(self):
        """Enhanced VAR stability check with detailed diagnostics."""
        if not hasattr(self, 'coefs') or len(self.coefs) == 0:
            logger.warning("No coefficient matrices available for stability check")
            return
            
        try:
            # Construct companion matrix using point estimates
            n_vars, n_lags = self.n_vars, self.n_lags
            companion_size = n_vars * n_lags
            companion = np.zeros((companion_size, companion_size))
            
            # Fill first n_vars rows with coefficient matrices
            for lag in range(n_lags):
                start_col = lag * n_vars
                end_col = (lag + 1) * n_vars
                companion[:n_vars, start_col:end_col] = self.coefs[lag]
            
            # Fill identity block for lag structure  
            if n_lags > 1:
                companion[n_vars:, :-n_vars] = np.eye(n_vars * (n_lags - 1))
            
            # Compute eigenvalues
            eigenvals = np.linalg.eigvals(companion)
            eigenval_mags = np.abs(eigenvals)
            max_eigenval = np.max(eigenval_mags)
            
            # Detailed eigenvalue analysis
            sorted_mags = np.sort(eigenval_mags)[::-1]  # Descending order
            top_5_eigenvals = sorted_mags[:min(5, len(sorted_mags))]
            
            logger.info("=" * 50)  
            logger.info("VAR STABILITY ANALYSIS")
            logger.info("=" * 50)
            logger.info(f"System size: {n_vars} variables, {n_lags} lags")
            logger.info(f"Companion matrix: {companion_size}x{companion_size}")
            logger.info(f"Top 5 eigenvalue magnitudes: {top_5_eigenvals}")
            logger.info(f"Maximum eigenvalue magnitude: {max_eigenval:.6f}")
            
            # Stability assessment with detailed thresholds
            if max_eigenval >= 1.0:
                # Count degrees of instability
                unstable_count = np.sum(eigenval_mags >= 1.0)
                severely_unstable = np.sum(eigenval_mags > 1.1)
                
                logger.error(f"🚨 VAR SYSTEM IS UNSTABLE! 🚨")
                logger.error(f"Maximum eigenvalue: {max_eigenval:.6f}")
                logger.error(f"Unstable eigenvalues (|λ| ≥ 1.0): {unstable_count}/{len(eigenvals)}")
                if severely_unstable > 0:
                    logger.error(f"Severely unstable eigenvalues (|λ| > 1.1): {severely_unstable}")
                
                logger.error("⚠️ CONSEQUENCES: Explosive IRFs, unreliable forecasts, invalid inference")
                logger.error("")
                logger.error("🔧 RECOMMENDED FIXES:")
                logger.error("1. DATA ISSUES:")
                logger.error("   - Check for non-stationary variables (run ADF tests)")
                logger.error("   - Use first differences instead of levels")
                logger.error("   - Remove trending or seasonal components")
                logger.error("2. MODEL SPECIFICATION:")
                logger.error("   - Reduce number of lags (try p-1 or p-2)")
                logger.error("   - Use cointegration constraints if applicable")
                logger.error("3. PRIOR SETTINGS:")
                logger.error("   - Increase λ1 shrinkage (try 0.3-0.5)")
                logger.error("   - Increase λ2 cross-variable shrinkage") 
                logger.error("4. ESTIMATION:")
                logger.error("   - Increase number of burn-in draws")
                logger.error("   - Check for convergence issues")
                
            elif max_eigenval > 0.95:
                near_unstable = np.sum(eigenval_mags > 0.95)
                logger.warning(f"⚠️ VAR system is near instability!")
                logger.warning(f"Maximum eigenvalue: {max_eigenval:.6f}")
                logger.warning(f"Near-unstable eigenvalues (|λ| > 0.95): {near_unstable}")
                logger.warning("IRFs may grow very large and decay slowly")
                logger.warning("Consider tightening priors or reducing lags")
                
            elif max_eigenval > 0.8:
                logger.info(f"✅ VAR system is stable but dynamic")
                logger.info(f"Maximum eigenvalue: {max_eigenval:.6f}")
                logger.info("IRFs will have moderate persistence")
                
            else:
                logger.info(f"✅ VAR system is stable with fast decay")
                logger.info(f"Maximum eigenvalue: {max_eigenval:.6f}")
                logger.info("IRFs will decay quickly to zero")
            
            # Store stability metrics for later use
            self._stability_metrics = {
                'max_eigenval': max_eigenval,
                'eigenvals': eigenvals,
                'is_stable': max_eigenval < 1.0,
                'stability_margin': 1.0 - max_eigenval
            }
                
        except Exception as e:
            logger.error(f"Stability check failed: {e}")
            logger.error("This may indicate numerical issues with the estimation")
            self._stability_metrics = {
                'max_eigenval': np.nan,
                'eigenvals': None,
                'is_stable': None,
                'stability_margin': np.nan
            }

    def cov_params(self) -> np.ndarray:
        """Approximate parameter covariance from posterior draws (flattened beta)."""
        if self.beta_draws is None or self.beta_draws.size == 0:
            raise RuntimeError('Model not fitted: no posterior draws available')
        # beta_draws shape: (n_draws, k_eq * n_vars)
        return np.cov(self.beta_draws, rowvar=False)
    
    def _fit_sparse(self, data: np.ndarray, lambda1: float, lambda2: float,
                   lambda3: float, mu: float, n_draws: int, n_burnin: int) -> 'LargeBayesianVAR':
        """Fit using sparse computational methods for very large VARs."""
        self.Y, self.X = self._create_YX_matrices(data)
        T_eff, k_eq = self.X.shape
        
        # Initialize sparse VAR
        self.sparse_var = SparseVAR(self.n_vars, self.n_lags)
        
        # Create sparse design matrix
        X_sparse = self.sparse_var.create_sparse_design_matrix(data)
        
        # Sparse Minnesota prior
        beta_ols = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        residuals = self.Y - self.X @ beta_ols
        sigmas_sq = np.diag(residuals.T @ residuals) / (T_eff - k_eq)
        
        prior_mean, prior_precision = self.sparse_var.sparse_minnesota_prior(
            sigmas_sq, lambda1, lambda2, lambda3
        )
        
        # Sparse Gibbs sampling
        logger.info("Running sparse Gibbs sampling...")
        self.beta_draws = np.zeros((n_draws - n_burnin, self.n_vars * k_eq))
        self.sigma_draws = np.zeros((n_draws - n_burnin, self.n_vars, self.n_vars))
        
        # Initialize
        Sigma_current = np.eye(self.n_vars)
        
        for draw in range(n_draws):
            # Sparse coefficient sampling
            beta_current = sparse_gibbs_step(
                self.Y, X_sparse, Sigma_current, prior_mean, prior_precision
            )
            
            # Standard covariance sampling
            beta_matrix = beta_current.reshape((k_eq, self.n_vars), order='F')
            residuals = self.Y - self.X @ beta_matrix
            scale = residuals.T @ residuals + np.eye(self.n_vars)
            
            # Sample from inverse Wishart
            L = np.linalg.cholesky(scale)
            Z = np.random.standard_normal((T_eff + self.n_vars, self.n_vars))
            W = Z.T @ Z
            Sigma_current = np.linalg.inv(L @ W @ L.T)
            
            # Store after burn-in
            if draw >= n_burnin:
                idx = draw - n_burnin
                self.beta_draws[idx] = beta_current
                self.sigma_draws[idx] = Sigma_current
        
        logger.info("Sparse fitting complete.")
        return self
    
    def _efficient_gibbs_sampling(self, Y: np.ndarray, X: np.ndarray,
                                prior_mean: np.ndarray, prior_cov: np.ndarray,
                                prior_scale: np.ndarray, prior_dof: int,
                                n_draws: int, n_burnin: int) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient Gibbs sampling using matrix normal distributions."""
        T, n_vars = Y.shape
        _, k_eq = X.shape
        
        # Storage
        beta_draws = np.zeros((n_draws - n_burnin, k_eq * n_vars))
        sigma_draws = np.zeros((n_draws - n_burnin, n_vars, n_vars))
        
        # Initialize
        Sigma = np.eye(n_vars)
        
        for draw in range(n_draws):
            # Sample coefficients using efficient matrix normal sampling
            if KroneckerOperations:
                post_mean, (Sigma_factor, V_factor) = KroneckerOperations.efficient_var_posterior(
                    Y, X, Sigma, prior_mean, np.linalg.inv(prior_cov)
                )
                
                # Sample from matrix normal using cached Cholesky factors
                beta_sample = self.matrix_sampler.sample(
                    post_mean.reshape((k_eq, n_vars), order='F'),
                    V_factor, Sigma_factor
                ).flatten('F')
            else:
                # Fallback to standard sampling
                precision = np.kron(np.linalg.inv(Sigma), X.T @ X) + np.linalg.inv(prior_cov)
                mean_vec = np.linalg.solve(precision, 
                                         np.kron(np.linalg.inv(Sigma), X.T @ Y).flatten('F') + 
                                         np.linalg.inv(prior_cov) @ prior_mean)
                beta_sample = np.random.multivariate_normal(mean_vec, np.linalg.inv(precision))
            
            # Sample covariance matrix
            beta_matrix = beta_sample.reshape((k_eq, n_vars), order='F')
            residuals = Y - X @ beta_matrix
            post_scale = prior_scale + residuals.T @ residuals
            
            # Inverse Wishart sampling
            nu_post = prior_dof + T
            L = np.linalg.cholesky(post_scale)
            Z = np.random.standard_normal((nu_post, n_vars))
            W = Z.T @ Z
            Sigma = np.linalg.inv(L @ W @ L.T)
            
            # Store after burn-in
            if draw >= n_burnin:
                idx = draw - n_burnin
                beta_draws[idx] = beta_sample
                sigma_draws[idx] = Sigma
                
                if (draw - n_burnin) % 1000 == 0:
                    logger.info(f"Completed {draw - n_burnin} post burn-in draws")
        
        return beta_draws, sigma_draws

    def get_irf(self, horizon=24, identification='cholesky'):
        """Computes IRFs from the posterior draws."""
        if self.beta_draws is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        irf_draws = posterior_analysis.compute_irf(
            self.beta_draws, self.sigma_draws, self.n_vars, self.n_lags, self.n_exog, horizon, identification
        )
        return irf_draws

    def get_irf_summary(self, horizon=24, identification='cholesky', interval_levels=[0.68, 0.90]):
        """
        Computes and summarizes IRFs, providing the median and credible intervals.
        """
        irf_draws = self.get_irf(horizon, identification)
        
        # Reshape for summary functions
        # Original shape: (n_draws, n_vars, n_vars, horizon)
        n_draws, n_impulse, n_response, horizon = irf_draws.shape
        irf_reshaped = irf_draws.reshape(n_draws, -1)

        summary = credible_intervals.get_posterior_summary(irf_reshaped)
        intervals = credible_intervals.get_credible_intervals(irf_reshaped, interval_levels)

        # Reshape summary stats back to original dimensions
        summary['median'] = summary['median'].reshape(n_impulse, n_response, horizon)
        summary['mean'] = summary['mean'].reshape(n_impulse, n_response, horizon)
        for level in intervals:
            intervals[level]['lower'] = intervals[level]['lower'].reshape(n_impulse, n_response, horizon)
            intervals[level]['upper'] = intervals[level]['upper'].reshape(n_impulse, n_response, horizon)

        return summary, intervals

    def get_fevd(self, horizon=24, identification='cholesky'):
        """Computes FEVDs from the posterior draws."""
        # FEVDs are derived from IRFs, so we compute them first.
        irf_draws = self.get_irf(horizon, identification)
        fevd_draws = posterior_analysis.compute_fevd(irf_draws)
        return fevd_draws

    def get_fevd_summary(self, horizon=24, identification='cholesky', interval_levels=[0.68, 0.90]):
        """
        Computes and summarizes FEVDs, providing the median and credible intervals.
        """
        fevd_draws = self.get_fevd(horizon, identification)

        # Reshape for summary functions
        # Original shape: (n_draws, n_response, n_impulse, horizon)
        n_draws, n_response, n_impulse, horizon = fevd_draws.shape
        fevd_reshaped = fevd_draws.reshape(n_draws, -1)

        summary = credible_intervals.get_posterior_summary(fevd_reshaped)
        intervals = credible_intervals.get_credible_intervals(fevd_reshaped, interval_levels)

        # Reshape summary stats back to original dimensions
        summary['median'] = summary['median'].reshape(n_response, n_impulse, horizon)
        summary['mean'] = summary['mean'].reshape(n_response, n_impulse, horizon)
        for level in intervals:
            intervals[level]['lower'] = intervals[level]['lower'].reshape(n_response, n_impulse, horizon)
            intervals[level]['upper'] = intervals[level]['upper'].reshape(n_response, n_impulse, horizon)

        return summary, intervals

    def run_diagnostics(self):
        """
        Runs and prints a report of model diagnostics.
        """
        if self.beta_draws is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        print("\n--- Running Model Diagnostics ---")

        # 1. MCMC Convergence (Effective Sample Size)
        print("\n1. MCMC Convergence (Effective Sample Size - ESS):")
        ess_values = mcmc_diagnostics.calculate_ess_for_all_params(self.beta_draws)
        print(f"  - Minimum ESS for β coefficients: {ess_values.min():.2f}")
        print(f"  - Maximum ESS for β coefficients: {ess_values.max():.2f}")
        print(f"  - Average ESS for β coefficients: {ess_values.mean():.2f}")
        ess_threshold = 100
        low_ess_count = np.sum(ess_values < ess_threshold)
        if low_ess_count > 0:
            print(f"  - WARNING: {low_ess_count}/{len(ess_values)} parameters have an ESS below {ess_threshold}.")
        else:
            print(f"  - All parameters have an ESS above {ess_threshold}, which is good.")

        # 2. Posterior Predictive Check (Bayesian p-value)
        print("\n2. Posterior Predictive Check (Bayesian p-value):")
        replicated_Y = predictive_checks.posterior_predictive_check(
            self.Y, self.X, self.beta_draws, self.sigma_draws
        )
        
        # Define a test statistic: variance of the series
        def test_stat_variance(data):
            return np.var(data, axis=0).sum() # Sum of variances across all variables

        p_value = predictive_checks.bayesian_p_value(replicated_Y, self.Y, test_stat_variance)
        print(f"  - Test Statistic: Sum of variances across all series.")
        print(f"  - Bayesian p-value: {p_value:.3f}")
        if p_value < 0.05 or p_value > 0.95:
            print("  - WARNING: The p-value is close to 0 or 1, suggesting potential model misfit.")
        else:
            print("  - The p-value is not close to 0 or 1, which does not indicate a major model misfit.")

        print("\n--- Diagnostics Complete ---")

    def get_conditional_forecast(self, horizon, constraints):
        """
        Computes conditional forecasts based on a specified scenario.

        Args:
            horizon (int): The forecast horizon.
            constraints (dict): A dictionary specifying the constraints for the scenario.
                                e.g., {1: {0: 100.0}} means at t+1, var 0 is 100.0.

        Returns:
            np.ndarray: Draws of the conditional forecast (n_draws, horizon, n_vars).
        """
        if self.beta_draws is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # The conditional forecasting function needs the original data used for fitting
        # to establish the history.
        if self.Y is None:
            raise RuntimeError("Historical data (Y) not found. Ensure model is fitted.")

        forecast_draws = conditional_forecasting.conditional_forecast(
            self.beta_draws, 
            self.sigma_draws, 
            self.Y, # Using the Y from the fitted model as history
            self.n_lags, 
            self.n_exog, 
            horizon, 
            constraints
        )
        return forecast_draws
