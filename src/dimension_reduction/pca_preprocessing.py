"""
Principal Component Analysis preprocessing for VAR data.

This module provides PCA-based dimension reduction and preprocessing
utilities specifically designed for VAR applications.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional, List, Dict, Union
import warnings


class VARPCAPreprocessor:
    """
    PCA preprocessing specifically designed for VAR applications.
    
    This class provides dimension reduction capabilities while preserving
    the temporal structure needed for VAR modeling.
    """
    
    def __init__(self, n_components: Union[int, float] = 0.95, 
                 method: str = 'standard', scaling: str = 'standard',
                 preserve_temporal: bool = True):
        """
        Initialize VAR PCA preprocessor.
        
        Parameters:
        -----------
        n_components : int or float
            Number of components (int) or explained variance ratio (float)
        method : str
            PCA method ('standard', 'incremental', 'sparse')
        scaling : str
            Scaling method ('standard', 'robust', 'none')
        preserve_temporal : bool
            Whether to preserve temporal ordering in components
        """
        self.n_components = n_components
        self.method = method
        self.scaling = scaling
        self.preserve_temporal = preserve_temporal
        
        # Model components
        self.pca_model = None
        self.scaler = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.original_variables = None
        self.selected_components = None
        
        # Temporal structure preservation
        self.temporal_weights = None
        
    def fit(self, X: np.ndarray, variable_names: Optional[List[str]] = None) -> 'VARPCAPreprocessor':
        """
        Fit PCA preprocessor to VAR data.
        
        Parameters:
        -----------
        X : np.ndarray
            VAR data (T x n_vars)
        variable_names : List[str], optional
            Variable names
            
        Returns:
        --------
        VARPCAPreprocessor
            Fitted preprocessor
        """
        T, n_vars = X.shape
        
        if variable_names is None:
            variable_names = [f"Var_{i+1}" for i in range(n_vars)]
        
        self.original_variables = variable_names
        
        # Apply scaling
        X_scaled = self._apply_scaling(X)
        
        # Fit PCA model
        self._fit_pca(X_scaled)
        
        # Select components based on criteria
        self._select_components()
        
        # Preserve temporal structure if requested
        if self.preserve_temporal:
            self._adjust_for_temporal_structure(X_scaled)
        
        print(f"PCA preprocessing completed:")
        print(f"  Original variables: {n_vars}")
        print(f"  Selected components: {len(self.selected_components)}")
        print(f"  Explained variance: {np.sum(self.explained_variance_ratio_[self.selected_components]):.3f}")
        
        return self
    
    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply specified scaling to data."""
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")
        
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        return X_scaled
    
    def _fit_pca(self, X: np.ndarray):
        """Fit PCA model using specified method."""
        if self.method == 'standard':
            # Determine number of components
            if isinstance(self.n_components, float) and self.n_components < 1.0:
                # Find components needed for desired explained variance
                pca_full = PCA()
                pca_full.fit(X)
                cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = np.argmax(cumsum_var >= self.n_components) + 1
            else:
                n_comp = int(self.n_components)
            
            self.pca_model = PCA(n_components=n_comp)
            
        elif self.method == 'incremental':
            # For memory-efficient processing of large datasets
            n_comp = int(self.n_components) if isinstance(self.n_components, int) else min(X.shape)
            batch_size = min(1000, X.shape[0] // 5)
            self.pca_model = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
            
        elif self.method == 'sparse':
            # For sparse loadings
            n_comp = int(self.n_components) if isinstance(self.n_components, int) else min(20, X.shape[1]//2)
            self.pca_model = SparsePCA(n_components=n_comp, alpha=0.1)
            
        else:
            raise ValueError(f"Unknown PCA method: {self.method}")
        
        # Fit the model
        self.pca_model.fit(X)
        self.components_ = self.pca_model.components_
        
        # Get explained variance ratio
        if hasattr(self.pca_model, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.pca_model.explained_variance_ratio_
        else:
            # For sparse PCA, compute explained variance manually
            X_transformed = self.pca_model.transform(X)
            total_var = np.var(X, axis=0).sum()
            component_vars = np.var(X_transformed, axis=0)
            self.explained_variance_ratio_ = component_vars / total_var
    
    def _select_components(self):
        """Select components based on various criteria."""
        n_components = self.components_.shape[0]
        
        # Initialize selection with all components
        self.selected_components = list(range(n_components))
        
        # Apply selection criteria
        
        # 1. Minimum explained variance threshold
        min_var_threshold = 0.01  # Components must explain at least 1% of variance
        significant_components = np.where(self.explained_variance_ratio_ > min_var_threshold)[0]
        
        # 2. Cumulative explained variance
        if isinstance(self.n_components, float) and self.n_components < 1.0:
            cumsum_var = np.cumsum(self.explained_variance_ratio_)
            enough_components = np.where(cumsum_var >= self.n_components)[0]
            if len(enough_components) > 0:
                max_components = enough_components[0] + 1
                significant_components = significant_components[significant_components < max_components]
        
        self.selected_components = significant_components.tolist()
    
    def _adjust_for_temporal_structure(self, X: np.ndarray):
        """Adjust components to preserve temporal structure."""
        # Compute temporal autocorrelation for each original variable
        temporal_autocorr = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            if X.shape[0] > 1:
                autocorr = np.corrcoef(X[:-1, i], X[1:, i])[0, 1]
                temporal_autocorr[i] = autocorr if not np.isnan(autocorr) else 0
        
        # Weight components by their correlation with high-autocorrelation variables
        component_temporal_scores = np.zeros(len(self.selected_components))
        
        for i, comp_idx in enumerate(self.selected_components):
            component = self.components_[comp_idx]
            # Weight by absolute loadings on high-autocorrelation variables
            weights = np.abs(component) * temporal_autocorr
            component_temporal_scores[i] = np.sum(weights)
        
        self.temporal_weights = component_temporal_scores / np.sum(component_temporal_scores)
        
        # Optionally reorder components by temporal relevance
        temporal_order = np.argsort(component_temporal_scores)[::-1]
        self.selected_components = [self.selected_components[i] for i in temporal_order]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform
            
        Returns:
        --------
        np.ndarray
            Transformed data with selected components
        """
        if self.pca_model is None:
            raise ValueError("PCA model must be fitted first")
        
        # Apply same scaling as during fit
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Transform using PCA
        X_transformed = self.pca_model.transform(X_scaled)
        
        # Select only the chosen components
        return X_transformed[:, self.selected_components]
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Inverse transform from PCA space back to original space.
        
        Parameters:
        -----------
        X_pca : np.ndarray
            Data in PCA space
            
        Returns:
        --------
        np.ndarray
            Reconstructed data in original space
        """
        if self.pca_model is None:
            raise ValueError("PCA model must be fitted first")
        
        # Create full PCA representation (with zeros for unselected components)
        n_total_components = self.components_.shape[0]
        X_full_pca = np.zeros((X_pca.shape[0], n_total_components))
        X_full_pca[:, self.selected_components] = X_pca
        
        # Inverse transform
        X_reconstructed = self.pca_model.inverse_transform(X_full_pca)
        
        # Inverse scaling
        if self.scaler is not None:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        
        return X_reconstructed
    
    def get_component_interpretation(self, top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get interpretation of principal components.
        
        Parameters:
        -----------
        top_k : int
            Number of top variables to show for each component
            
        Returns:
        --------
        Dict[int, List[Tuple[str, float]]]
            Component interpretations
        """
        if self.components_ is None:
            raise ValueError("PCA model must be fitted first")
        
        interpretations = {}
        
        for i, comp_idx in enumerate(self.selected_components):
            loadings = self.components_[comp_idx]
            
            # Get top positive and negative loadings
            top_pos_indices = np.argsort(loadings)[-top_k:][::-1]
            top_neg_indices = np.argsort(loadings)[:top_k]
            
            component_vars = []
            
            # Add positive loadings
            for idx in top_pos_indices:
                if loadings[idx] > 0.1:  # Threshold for significance
                    component_vars.append((self.original_variables[idx], loadings[idx]))
            
            # Add negative loadings
            for idx in top_neg_indices:
                if loadings[idx] < -0.1:  # Threshold for significance
                    component_vars.append((self.original_variables[idx], loadings[idx]))
            
            interpretations[i] = component_vars
        
        return interpretations
    
    def plot_explained_variance(self):
        """Plot explained variance by component."""
        try:
            import matplotlib.pyplot as plt
            
            n_components = len(self.explained_variance_ratio_)
            
            plt.figure(figsize=(12, 5))
            
            # Individual explained variance
            plt.subplot(1, 2, 1)
            plt.bar(range(n_components), self.explained_variance_ratio_)
            plt.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='1% threshold')
            
            # Highlight selected components
            selected_mask = np.zeros(n_components, dtype=bool)
            selected_mask[self.selected_components] = True
            plt.bar(np.where(selected_mask)[0], 
                   self.explained_variance_ratio_[selected_mask], 
                   color='orange', alpha=0.7, label='Selected')
            
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Individual Component Variance')
            plt.legend()
            
            # Cumulative explained variance
            plt.subplot(1, 2, 2)
            cumsum_var = np.cumsum(self.explained_variance_ratio_)
            plt.plot(range(n_components), cumsum_var, 'b-o', markersize=4)
            
            if isinstance(self.n_components, float) and self.n_components < 1.0:
                plt.axhline(self.n_components, color='red', linestyle='--', 
                          label=f'Target: {self.n_components:.1%}')
            
            # Mark selected components
            if len(self.selected_components) > 0:
                last_selected = max(self.selected_components)
                plt.axvline(last_selected, color='orange', linestyle='--', 
                          label=f'Selected: {len(self.selected_components)} components')
            
            plt.xlabel('Component')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Variance Explained')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
    
    def plot_component_loadings(self, component_idx: int = 0, top_k: int = 10):
        """Plot loadings for a specific component."""
        try:
            import matplotlib.pyplot as plt
            
            if component_idx >= len(self.selected_components):
                raise ValueError(f"Component index {component_idx} out of range")
            
            actual_comp_idx = self.selected_components[component_idx]
            loadings = self.components_[actual_comp_idx]
            
            # Get top loadings by absolute value
            top_indices = np.argsort(np.abs(loadings))[-top_k:][::-1]
            top_loadings = loadings[top_indices]
            top_variables = [self.original_variables[i] for i in top_indices]
            
            plt.figure(figsize=(10, 6))
            colors = ['red' if x < 0 else 'blue' for x in top_loadings]
            plt.barh(range(len(top_loadings)), top_loadings, color=colors, alpha=0.7)
            plt.yticks(range(len(top_loadings)), top_variables)
            plt.xlabel('Loading')
            plt.title(f'Component {component_idx + 1} Loadings (Top {top_k})')
            plt.grid(axis='x', alpha=0.3)
            
            # Add explained variance info
            var_explained = self.explained_variance_ratio_[actual_comp_idx]
            plt.text(0.02, 0.98, f'Explained Variance: {var_explained:.1%}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")


def adaptive_pca_components(X: np.ndarray, max_components: int = 50, 
                          information_criteria: str = 'bic') -> int:
    """
    Select optimal number of PCA components using information criteria.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    max_components : int
        Maximum components to consider
    information_criteria : str
        Criteria to use ('aic', 'bic', 'hqic')
        
    Returns:
    --------
    int
        Optimal number of components
    """
    T, n_vars = X.shape
    max_comp = min(max_components, min(T, n_vars) - 1)
    
    ic_values = []
    
    for k in range(1, max_comp + 1):
        pca = PCA(n_components=k)
        X_pca = pca.fit_transform(X)
        
        # Reconstruction error
        X_reconstructed = pca.inverse_transform(X_pca)
        mse = np.mean((X - X_reconstructed)**2)
        
        # Log-likelihood (Gaussian assumption)
        log_likelihood = -0.5 * T * n_vars * np.log(2 * np.pi * mse) - 0.5 * T * n_vars
        
        # Information criteria
        n_params = k * n_vars + k  # loadings + component variances
        
        if information_criteria == 'aic':
            ic = -2 * log_likelihood + 2 * n_params
        elif information_criteria == 'bic':
            ic = -2 * log_likelihood + n_params * np.log(T)
        elif information_criteria == 'hqic':
            ic = -2 * log_likelihood + 2 * n_params * np.log(np.log(T))
        else:
            raise ValueError(f"Unknown information criteria: {information_criteria}")
        
        ic_values.append(ic)
    
    # Select components with minimum IC
    optimal_k = np.argmin(ic_values) + 1
    
    return optimal_k


class RollingPCA:
    """
    Rolling PCA for time-varying dimension reduction.
    
    This class implements rolling window PCA to capture time-varying
    relationships in high-dimensional VAR data.
    """
    
    def __init__(self, window_size: int, n_components: int = 10, 
                 step_size: int = 1):
        """
        Initialize rolling PCA.
        
        Parameters:
        -----------
        window_size : int
            Size of rolling window
        n_components : int
            Number of components to extract
        step_size : int
            Step size for rolling window
        """
        self.window_size = window_size
        self.n_components = n_components
        self.step_size = step_size
        
        # Results storage
        self.component_history = []
        self.explained_variance_history = []
        self.time_indices = []
    
    def fit(self, X: np.ndarray) -> 'RollingPCA':
        """
        Fit rolling PCA to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Time series data (T x n_vars)
            
        Returns:
        --------
        RollingPCA
            Fitted rolling PCA
        """
        T, n_vars = X.shape
        
        # Initialize storage
        n_windows = (T - self.window_size) // self.step_size + 1
        self.component_history = np.zeros((n_windows, self.n_components, n_vars))
        self.explained_variance_history = np.zeros((n_windows, self.n_components))
        self.time_indices = []
        
        scaler = StandardScaler()
        
        print(f"Computing rolling PCA with {n_windows} windows...")
        
        window_idx = 0
        for t in range(0, T - self.window_size + 1, self.step_size):
            # Extract window data
            X_window = X[t:t + self.window_size]
            
            # Standardize
            X_scaled = scaler.fit_transform(X_window)
            
            # Fit PCA
            pca = PCA(n_components=min(self.n_components, X_scaled.shape[0] - 1))
            pca.fit(X_scaled)
            
            # Store results
            n_comp_actual = pca.components_.shape[0]
            self.component_history[window_idx, :n_comp_actual] = pca.components_
            self.explained_variance_history[window_idx, :n_comp_actual] = pca.explained_variance_ratio_
            self.time_indices.append(t + self.window_size // 2)  # Center of window
            
            window_idx += 1
        
        print("Rolling PCA completed!")
        return self
    
    def get_component_stability(self) -> np.ndarray:
        """
        Compute stability of components over time.
        
        Returns:
        --------
        np.ndarray
            Component stability scores
        """
        n_windows = len(self.time_indices)
        stability_scores = np.zeros(self.n_components)
        
        for comp in range(self.n_components):
            # Compute correlation between consecutive component vectors
            correlations = []
            
            for t in range(1, n_windows):
                comp_prev = self.component_history[t-1, comp]
                comp_curr = self.component_history[t, comp]
                
                # Handle sign ambiguity in PCA
                corr_pos = np.corrcoef(comp_prev, comp_curr)[0, 1]
                corr_neg = np.corrcoef(comp_prev, -comp_curr)[0, 1]
                
                best_corr = corr_pos if abs(corr_pos) > abs(corr_neg) else corr_neg
                correlations.append(abs(best_corr))
            
            stability_scores[comp] = np.mean(correlations)
        
        return stability_scores
    
    def plot_component_evolution(self, component_idx: int = 0, top_variables: int = 5):
        """Plot evolution of component loadings over time."""
        try:
            import matplotlib.pyplot as plt
            
            if component_idx >= self.n_components:
                raise ValueError(f"Component index {component_idx} out of range")
            
            # Get component loadings over time
            component_loadings = self.component_history[:, component_idx, :]
            
            # Find top variables by average absolute loading
            avg_abs_loadings = np.mean(np.abs(component_loadings), axis=0)
            top_var_indices = np.argsort(avg_abs_loadings)[-top_variables:][::-1]
            
            plt.figure(figsize=(12, 8))
            
            # Plot loadings evolution
            for i, var_idx in enumerate(top_var_indices):
                plt.subplot(2, 1, 1)
                plt.plot(self.time_indices, component_loadings[:, var_idx], 
                        label=f'Var_{var_idx+1}', marker='o', markersize=3)
            
            plt.xlabel('Time')
            plt.ylabel('Component Loading')
            plt.title(f'Component {component_idx + 1}: Loading Evolution')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Plot explained variance evolution
            plt.subplot(2, 1, 2)
            plt.plot(self.time_indices, self.explained_variance_history[:, component_idx], 
                    'r-o', markersize=4)
            plt.xlabel('Time')
            plt.ylabel('Explained Variance Ratio')
            plt.title(f'Component {component_idx + 1}: Explained Variance Evolution')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")