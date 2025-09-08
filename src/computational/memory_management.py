"""
Memory management utilities for large-scale VAR systems.

This module provides memory-efficient algorithms and data structures for
handling very large VAR models that approach system memory limits.
"""

import numpy as np
import psutil
import gc
from typing import Generator, Tuple, Optional, Dict, Any
from functools import wraps
import warnings
import os
from contextlib import contextmanager


class MemoryManager:
    """
    Centralized memory management for large-scale VAR computations.
    
    This class monitors memory usage and implements strategies to keep
    computations within available system memory.
    """
    
    def __init__(self, max_memory_fraction: float = 0.8):
        """
        Initialize memory manager.
        
        Parameters:
        -----------
        max_memory_fraction : float
            Maximum fraction of system memory to use
        """
        self.max_memory_fraction = max_memory_fraction
        self.total_memory = psutil.virtual_memory().total
        self.max_memory = int(self.total_memory * max_memory_fraction)
        
        # Memory usage tracking
        self.memory_log = []
        self.peak_usage = 0
    
    def get_available_memory(self) -> int:
        """Get currently available memory in bytes."""
        return psutil.virtual_memory().available
    
    def get_current_usage(self) -> int:
        """Get current process memory usage in bytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def log_memory_usage(self, operation: str):
        """Log current memory usage for an operation."""
        usage = self.get_current_usage()
        self.memory_log.append((operation, usage))
        self.peak_usage = max(self.peak_usage, usage)
    
    def estimate_array_memory(self, shape: Tuple[int, ...], dtype=np.float64) -> int:
        """
        Estimate memory usage of an array.
        
        Parameters:
        -----------
        shape : Tuple[int, ...]
            Array shape
        dtype : np.dtype
            Array data type
            
        Returns:
        --------
        int
            Estimated memory usage in bytes
        """
        return np.prod(shape) * np.dtype(dtype).itemsize
    
    def can_allocate(self, shape: Tuple[int, ...], dtype=np.float64, 
                    safety_margin: float = 0.1) -> bool:
        """
        Check if an array can be safely allocated.
        
        Parameters:
        -----------
        shape : Tuple[int, ...]
            Desired array shape
        dtype : np.dtype
            Array data type
        safety_margin : float
            Safety margin as fraction of available memory
            
        Returns:
        --------
        bool
            True if allocation is safe
        """
        required_memory = self.estimate_array_memory(shape, dtype)
        available_memory = self.get_available_memory()
        safe_memory = available_memory * (1 - safety_margin)
        
        return required_memory <= safe_memory
    
    def suggest_batch_size(self, total_size: int, min_batch: int = 1) -> int:
        """
        Suggest optimal batch size for processing.
        
        Parameters:
        -----------
        total_size : int
            Total number of items to process
        min_batch : int
            Minimum batch size
            
        Returns:
        --------
        int
            Suggested batch size
        """
        available_memory = self.get_available_memory()
        memory_per_item = available_memory // (total_size * 10)  # Conservative estimate
        
        batch_size = max(min_batch, available_memory // (memory_per_item * 2))
        return min(batch_size, total_size)


def memory_efficient(max_memory_mb: int = 1000):
    """
    Decorator to ensure function uses limited memory.
    
    Parameters:
    -----------
    max_memory_mb : int
        Maximum memory usage in MB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = MemoryManager()
            initial_usage = manager.get_current_usage()
            
            try:
                result = func(*args, **kwargs)
                
                final_usage = manager.get_current_usage()
                usage_mb = (final_usage - initial_usage) / (1024 * 1024)
                
                if usage_mb > max_memory_mb:
                    warnings.warn(f"Function {func.__name__} used {usage_mb:.1f} MB, "
                                f"exceeding limit of {max_memory_mb} MB")
                
                return result
                
            except MemoryError:
                gc.collect()  # Force garbage collection
                raise MemoryError(f"Function {func.__name__} exceeded available memory")
                
        return wrapper
    return decorator


class StreamingMatrix:
    """
    Memory-efficient matrix operations using streaming/chunked processing.
    
    This class allows operations on matrices that don't fit in memory by
    processing them in chunks.
    """
    
    def __init__(self, data_source, shape: Tuple[int, int], chunk_size: int = 1000):
        """
        Initialize streaming matrix.
        
        Parameters:
        -----------
        data_source : callable or array-like
            Source of matrix data (can be file, generator, or array)
        shape : Tuple[int, int]
            Full matrix shape
        chunk_size : int
            Size of chunks for processing
        """
        self.data_source = data_source
        self.shape = shape
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager()
    
    def chunks(self, axis: int = 0) -> Generator[np.ndarray, None, None]:
        """
        Generate chunks along specified axis.
        
        Parameters:
        -----------
        axis : int
            Axis along which to chunk (0 for rows, 1 for columns)
            
        Yields:
        -------
        np.ndarray
            Matrix chunks
        """
        if callable(self.data_source):
            # Data source is a function that generates chunks
            yield from self.data_source(self.chunk_size, axis)
        else:
            # Data source is array-like
            if axis == 0:
                for i in range(0, self.shape[0], self.chunk_size):
                    end_idx = min(i + self.chunk_size, self.shape[0])
                    yield self.data_source[i:end_idx, :]
            else:
                for i in range(0, self.shape[1], self.chunk_size):
                    end_idx = min(i + self.chunk_size, self.shape[1])
                    yield self.data_source[:, i:end_idx]
    
    def streaming_transpose(self) -> 'StreamingMatrix':
        """Create transposed streaming matrix."""
        def transposed_source(chunk_size, axis):
            # Swap axis for transposed access
            opposite_axis = 1 - axis
            for chunk in self.chunks(opposite_axis):
                yield chunk.T
        
        transposed_shape = (self.shape[1], self.shape[0])
        return StreamingMatrix(transposed_source, transposed_shape, self.chunk_size)
    
    def streaming_matmul(self, other: 'StreamingMatrix') -> np.ndarray:
        """
        Streaming matrix multiplication for large matrices.
        
        Parameters:
        -----------
        other : StreamingMatrix
            Other matrix to multiply
            
        Returns:
        --------
        np.ndarray
            Result matrix (or streaming matrix if result is also large)
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        result_shape = (self.shape[0], other.shape[1])
        
        # Check if result fits in memory
        if not self.memory_manager.can_allocate(result_shape):
            return self._streaming_matmul_chunked(other)
        
        # Result fits in memory
        result = np.zeros(result_shape)
        
        for i, self_chunk in enumerate(self.chunks(axis=0)):
            chunk_start = i * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.shape[0])
            
            for j, other_chunk in enumerate(other.chunks(axis=1)):
                other_start = j * other.chunk_size
                other_end = min(other_start + other.chunk_size, other.shape[1])
                
                # Accumulate partial products
                result[chunk_start:chunk_end, other_start:other_end] += \
                    self_chunk @ other_chunk
        
        return result
    
    def _streaming_matmul_chunked(self, other: 'StreamingMatrix') -> 'StreamingMatrix':
        """Matrix multiplication where result doesn't fit in memory."""
        def result_generator(chunk_size, axis):
            if axis == 0:  # Row chunks
                for self_chunk in self.chunks(axis=0):
                    # Compute full row result for this chunk
                    chunk_result = np.zeros((self_chunk.shape[0], other.shape[1]))
                    for other_chunk in other.chunks(axis=1):
                        chunk_result += self_chunk @ other_chunk
                    yield chunk_result
            else:  # Column chunks  
                for other_chunk in other.chunks(axis=1):
                    # Compute full column result for this chunk
                    chunk_result = np.zeros((self.shape[0], other_chunk.shape[1]))
                    for self_chunk in self.chunks(axis=0):
                        chunk_result += self_chunk @ other_chunk
                    yield chunk_result
        
        result_shape = (self.shape[0], other.shape[1])
        return StreamingMatrix(result_generator, result_shape, self.chunk_size)


class MemoryEfficientVAR:
    """
    Memory-efficient VAR operations for very large systems.
    
    This class implements VAR operations that can handle datasets approaching
    system memory limits by using streaming and chunked computations.
    """
    
    def __init__(self, max_memory_mb: int = 4000):
        """
        Initialize memory-efficient VAR.
        
        Parameters:
        -----------
        max_memory_mb : int
            Maximum memory to use in MB
        """
        self.max_memory_mb = max_memory_mb
        self.memory_manager = MemoryManager(max_memory_fraction=0.7)
    
    @memory_efficient(max_memory_mb=2000)
    def streaming_ols(self, Y: np.ndarray, X: np.ndarray, 
                     chunk_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OLS estimates using streaming for large datasets.
        
        Parameters:
        -----------
        Y : np.ndarray
            Response matrix
        X : np.ndarray
            Design matrix
        chunk_size : int
            Chunk size for processing
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            OLS coefficients and residual covariance
        """
        T, n_vars = Y.shape
        T, k_vars = X.shape
        
        # Check if we can do everything in memory
        total_memory_needed = self.memory_manager.estimate_array_memory(
            (k_vars, k_vars)) + self.memory_manager.estimate_array_memory((k_vars, n_vars))
        
        if total_memory_needed < self.memory_manager.get_available_memory() * 0.5:
            # Standard computation
            XtX = X.T @ X
            XtY = X.T @ Y
            beta = np.linalg.solve(XtX, XtY)
            residuals = Y - X @ beta
            sigma = (residuals.T @ residuals) / (T - k_vars)
            return beta, sigma
        
        # Streaming computation
        XtX = np.zeros((k_vars, k_vars))
        XtY = np.zeros((k_vars, n_vars))
        
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            X_chunk = X[i:end_idx]
            Y_chunk = Y[i:end_idx]
            
            XtX += X_chunk.T @ X_chunk
            XtY += X_chunk.T @ Y_chunk
            
            # Periodic garbage collection
            if i % (chunk_size * 10) == 0:
                gc.collect()
        
        beta = np.linalg.solve(XtX, XtY)
        
        # Compute residual covariance in chunks
        sigma = np.zeros((n_vars, n_vars))
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            X_chunk = X[i:end_idx]
            Y_chunk = Y[i:end_idx]
            
            residuals_chunk = Y_chunk - X_chunk @ beta
            sigma += residuals_chunk.T @ residuals_chunk
        
        sigma /= (T - k_vars)
        return beta, sigma
    
    def chunked_gibbs_sampling(self, Y: np.ndarray, X: np.ndarray,
                              n_draws: int = 1000, n_burnin: int = 500,
                              chunk_size: int = 100) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Memory-efficient Gibbs sampling using chunked processing.
        
        Parameters:
        -----------
        Y : np.ndarray
            Response matrix
        X : np.ndarray
            Design matrix  
        n_draws : int
            Total number of draws
        n_burnin : int
            Burn-in draws
        chunk_size : int
            Chunk size for processing
            
        Yields:
        -------
        Tuple[np.ndarray, np.ndarray]
            Coefficient and covariance samples
        """
        T, n_vars = Y.shape
        T, k_vars = X.shape
        
        # Initialize with OLS
        beta_current, sigma_current = self.streaming_ols(Y, X, chunk_size)
        
        for draw in range(n_draws):
            # Sample coefficients (may need chunking for very large k_vars)
            if k_vars > 5000:
                beta_current = self._chunked_beta_sample(
                    Y, X, sigma_current, beta_current, chunk_size
                )
            else:
                # Standard sampling
                precision = np.kron(np.linalg.inv(sigma_current), X.T @ X)
                mean_vec = np.linalg.solve(precision, 
                                         np.kron(np.linalg.inv(sigma_current), X.T @ Y).flatten('F'))
                beta_current = np.random.multivariate_normal(
                    mean_vec, np.linalg.inv(precision)
                ).reshape((k_vars, n_vars), order='F')
            
            # Sample covariance matrix
            residuals = Y - X @ beta_current
            scale = residuals.T @ residuals
            sigma_current = self._sample_inverse_wishart(scale, T)
            
            if draw >= n_burnin:
                yield beta_current.copy(), sigma_current.copy()
            
            # Memory cleanup
            if draw % 100 == 0:
                gc.collect()
    
    def _chunked_beta_sample(self, Y: np.ndarray, X: np.ndarray, 
                           sigma: np.ndarray, beta_current: np.ndarray,
                           chunk_size: int) -> np.ndarray:
        """Sample coefficients using chunked operations for very large k_vars."""
        T, n_vars = Y.shape
        T, k_vars = X.shape
        
        # Block Gibbs sampling for coefficient blocks
        beta_new = beta_current.copy()
        
        for block_start in range(0, k_vars, chunk_size):
            block_end = min(block_start + chunk_size, k_vars)
            block_indices = slice(block_start, block_end)
            
            # Conditional distribution for this block
            X_block = X[:, block_indices]
            
            # Residual excluding current block contribution
            residual = Y - X @ beta_new
            residual += X_block @ beta_new[block_indices, :]
            
            # Block posterior
            precision_block = np.kron(np.linalg.inv(sigma), X_block.T @ X_block)
            mean_vec = np.linalg.solve(
                precision_block,
                np.kron(np.linalg.inv(sigma), X_block.T @ residual).flatten('F')
            )
            
            # Sample block
            beta_block = np.random.multivariate_normal(
                mean_vec, np.linalg.inv(precision_block)
            ).reshape((-1, n_vars), order='F')
            
            beta_new[block_indices, :] = beta_block
        
        return beta_new
    
    def _sample_inverse_wishart(self, scale: np.ndarray, df: int) -> np.ndarray:
        """Sample from inverse Wishart distribution."""
        n = scale.shape[0]
        
        # Sample from Wishart
        A = np.random.standard_normal((df, n))
        W = A.T @ A + scale
        
        return np.linalg.inv(W)


@contextmanager
def memory_limit(max_memory_mb: int):
    """
    Context manager to enforce memory limits.
    
    Parameters:
    -----------
    max_memory_mb : int
        Maximum memory to use in MB
    """
    initial_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    
    try:
        yield
    finally:
        current_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        usage_increase = current_usage - initial_usage
        
        if usage_increase > max_memory_mb:
            warnings.warn(f"Memory usage exceeded limit: {usage_increase:.1f} MB > {max_memory_mb} MB")
            gc.collect()  # Force cleanup


def optimize_chunk_size(matrix_shape: Tuple[int, int], 
                       target_memory_mb: int = 500) -> int:
    """
    Optimize chunk size based on available memory and matrix dimensions.
    
    Parameters:
    -----------
    matrix_shape : Tuple[int, int]
        Shape of matrix to process
    target_memory_mb : int
        Target memory usage per chunk in MB
        
    Returns:
    --------
    int
        Optimal chunk size
    """
    rows, cols = matrix_shape
    bytes_per_row = cols * 8  # Assuming float64
    target_bytes = target_memory_mb * 1024 * 1024
    
    optimal_chunk_size = target_bytes // bytes_per_row
    
    # Ensure reasonable bounds
    optimal_chunk_size = max(10, min(optimal_chunk_size, rows))
    
    return optimal_chunk_size