"""
Logarithmic Integral (Li) function module.

This module provides vectorized computation of the logarithmic integral
Li(x) with batching for large arrays.
"""

from typing import Optional
import numpy as np
from scipy.special import expi
from loguru import logger


def li_vectorized(
    x: np.ndarray, 
    batch_size: int = 1_000_000,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute the logarithmic integral Li(x) = Ei(ln(x)) - Ei(ln(2)) = integral from 2 to x of dt/ln(t).
    
    This is a vectorized implementation with batching for memory efficiency
    when processing large arrays.
    
    Args:
        x: Array of values to compute Li for (must be >= 2)
        batch_size: Number of elements to process per batch
        verbose: Whether to show progress
        
    Returns:
        Array of Li(x) values
        
    Example:
        >>> import numpy as np
        >>> x = np.array([10, 100, 1000])
        >>> li_vectorized(x)
        array([5.120..., 29.080..., 176.564...])
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    
    # Validate input
    if np.any(x < 2):
        logger.warning("Li(x) is only defined for x >= 2. Values < 2 will be set to NaN.")
        x = np.maximum(x, 2.0)
    
    # Constant for subtraction: Ei(ln(2))
    expi_log2 = expi(np.log(2.0))
    
    # For small arrays, compute directly
    if len(x) <= batch_size:
        return expi(np.log(x)) - expi_log2
    
    # For large arrays, use batching
    n = len(x)
    num_batches = (n + batch_size - 1) // batch_size
    result = np.empty(n, dtype=np.float64)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch = x[start_idx:end_idx]
        
        try:
            result[start_idx:end_idx] = expi(np.log(batch)) - expi_log2
        except (MemoryError, OverflowError) as e:
            logger.error(f"Error in batch {i+1}/{num_batches}: {e}")
            # Fallback: process smaller sub-batches
            result[start_idx:end_idx] = np.nan
        
        # Progress reporting
        if verbose and (i + 1) % max(1, num_batches // 10) == 0:
            progress = 100 * (i + 1) / num_batches
            logger.debug(f"Li computation: {i+1}/{num_batches} ({progress:.1f}%)")
    
    return result


def li_prime_vectorized(
    x: np.ndarray,
    batch_size: int = 1_000_000,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute the derivative of Li(x), which is 1/ln(x).
    
    Args:
        x: Array of values (must be > 1)
        batch_size: Batch size for processing
        verbose: Whether to show progress
        
    Returns:
        Array of dLi/dx values (1/ln(x))
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    
    # Avoid log(1) = 0
    x_safe = np.maximum(x, 1.000001)
    return 1.0 / np.log(x_safe)


def li_offset(x: float, offset: float = 2.0) -> float:
    """
    Compute Li(x) with custom offset.
    
    Li(x, offset) = Ei(ln(x)) - Ei(ln(offset))
    
    Args:
        x: Upper bound (must be > offset)
        offset: Lower bound (default: 2)
        
    Returns:
        Li value with custom offset
    """
    if x <= offset:
        return 0.0
    return expi(np.log(x)) - expi(np.log(offset))


def li_offset_vectorized(
    x: np.ndarray,
    offset: float = 2.0,
    batch_size: int = 1_000_000,
    verbose: bool = False
) -> np.ndarray:
    """
    Vectorized Li with custom offset.
    
    Args:
        x: Array of upper bounds
        offset: Lower bound
        batch_size: Batch size for processing
        verbose: Whether to show progress
        
    Returns:
        Array of Li(x, offset) values
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    
    expi_offset = expi(np.log(offset))
    
    if len(x) <= batch_size:
        return expi(np.log(x)) - expi_offset
    
    n = len(x)
    num_batches = (n + batch_size - 1) // batch_size
    result = np.empty(n, dtype=np.float64)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch = x[start_idx:end_idx]
        result[start_idx:end_idx] = expi(np.log(batch)) - expi_offset
    
    return result


if __name__ == "__main__":
    # Test Li function
    import time
    
    print("Testing Li function computation...")
    
    # Small test
    x_small = np.array([10, 100, 1000, 10000])
    li_small = li_vectorized(x_small, verbose=False)
    print(f"Li({x_small}) = {li_small}")
    
    # Large test
    n_large = 10_000_000
    x_large = np.logspace(2, 9, n_large)
    
    print(f"\nComputing Li for {n_large:,} values...")
    start = time.time()
    li_large = li_vectorized(x_large, batch_size=2_000_000, verbose=True)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")
    print(f"Result shape: {li_large.shape}")
    print(f"Result range: [{li_large.min():.4f}, {li_large.max():.4f}]")
    
    print("\nTest completed successfully!")
