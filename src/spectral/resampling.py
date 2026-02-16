"""
Resampling module for FFT preparation.

This module provides functions for resampling R(p) onto a uniform
grid for proper FFT computation.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
from loguru import logger


def resample_to_uniform_grid(
    x: np.ndarray,
    y: np.ndarray,
    n_points: int,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data to a uniform grid using linear interpolation.
    
    Args:
        x: Input x values (prime numbers)
        y: Input y values (R(p) residuals)
        n_points: Number of points in the output grid
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        Tuple of (x_resampled, y_resampled)
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    if n_points > len(x):
        logger.warning(
            f"Requested {n_points} points but only {len(x)} available. "
            f"Using original data size."
        )
        n_points = len(x)
    
    # Create uniform grid in index space
    indices = np.linspace(0, len(x) - 1, n_points)
    
    # Interpolate
    interpolator = interp1d(
        x, y, kind=method,
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    x_new = x[indices.astype(int)]
    y_new = interpolator(x_new)
    
    return x_new, y_new


def resample_to_log_uniform_grid(
    prime_N: np.ndarray,
    R_values: np.ndarray,
    n_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample R(p) to a uniform grid in ln(p) space.
    
    This is the preferred method for prime number analysis because
    primes are logarithmically distributed. Uniform sampling in ln(p)
    preserves the structure better than uniform sampling in p.
    
    Args:
        prime_N: Array of prime numbers
        R_values: Array of residual values
        n_points: Number of points in the output grid
        
    Returns:
        Tuple of (ln_p_grid, R_resampled, prime_N_resampled)
    """
    if len(prime_N) != len(R_values):
        raise ValueError("prime_N and R_values must have the same length")
    
    # Compute ln(p)
    ln_p = np.log(prime_N.astype(np.float64))
    
    # Create uniform grid in ln(p) space
    ln_p_min = ln_p.min()
    ln_p_max = ln_p.max()
    ln_p_grid = np.linspace(ln_p_min, ln_p_max, n_points)
    
    # Interpolate R values onto the log-uniform grid
    interpolator = interp1d(
        ln_p, R_values, kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    R_resampled = interpolator(ln_p_grid)
    
    # Convert back to prime values for reference
    prime_N_resampled = np.exp(ln_p_grid)
    
    return ln_p_grid, R_resampled, prime_N_resampled


def adaptive_resample(
    prime_N: np.ndarray,
    R_values: np.ndarray,
    max_points: int = 10_000_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptively resample data based on size.
    
    For small datasets, use all points.
    For large datasets, resample to max_points.
    
    Args:
        prime_N: Array of prime numbers
        R_values: Array of residual values
        max_points: Maximum number of points to keep
        
    Returns:
        Tuple of (prime_N_processed, R_processed, ln_p_grid)
    """
    n = len(prime_N)
    
    if n <= max_points:
        # Use all points with log-uniform resampling
        ln_p_grid, R_processed, prime_N_processed = resample_to_log_uniform_grid(
            prime_N, R_values, n
        )
        return prime_N_processed, R_processed, ln_p_grid
    else:
        # Downsample to max_points
        ln_p_grid, R_processed, prime_N_processed = resample_to_log_uniform_grid(
            prime_N, R_values, max_points
        )
        return prime_N_processed, R_processed, ln_p_grid


def center_and_normalize(
    R_values: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Center and normalize residual values.
    
    Args:
        R_values: Array of residual values
        
    Returns:
        Tuple of (centered_normalized, mean, std)
    """
    mean = R_values.mean()
    std = R_values.std()
    
    if std == 0:
        logger.warning("Residuals have zero standard deviation")
        centered = R_values - mean
        return centered, mean, 1.0
    
    centered = (R_values - mean) / std
    return centered, mean, std


def prepare_for_fft(
    prime_N: np.ndarray,
    R_values: np.ndarray,
    max_points: int = 10_000_000,
    use_memmap: bool = False,
    memmap_dir: str | None = None,
    memmap_filename: str = "fft_signal.dat",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Complete preprocessing pipeline for FFT.
    
    1. Resample to log-uniform grid if needed
    2. Center to zero mean
    3. Normalize to unit variance
    
    Args:
        prime_N: Array of prime numbers
        R_values: Array of residual values
        max_points: Maximum number of points for resampling
        
    Returns:
        Tuple of (prime_N_processed, R_centered_normalized, ln_p_grid, mean, std)
    """
    logger.info(f"Preparing {len(prime_N):,} points for FFT...")
    
    # Adaptive resampling
    prime_N_proc, R_proc, ln_p_grid = adaptive_resample(
        prime_N, R_values, max_points
    )
    
    logger.info(f"After resampling: {len(prime_N_proc):,} points")
    
    # Center and normalize
    R_centered, mean, std = center_and_normalize(R_proc)
    
    logger.info(
        f"Preprocessing complete: mean={mean:.6f}, std={std:.6f}"
    )

    if use_memmap:
        import os

        if memmap_dir is None:
            memmap_dir = "."
        os.makedirs(memmap_dir, exist_ok=True)
        memmap_path = os.path.join(memmap_dir, memmap_filename)

        logger.info(f"Transferring signal for FFT to memmap file: {memmap_path}")

        mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=R_centered.shape)
        mm[:] = R_centered[:]
        # Free regular array; caller will work with memmap
        del R_centered

        return prime_N_proc, mm, ln_p_grid, mean, std

    return prime_N_proc, R_centered, ln_p_grid, mean, std


def compute_frequency_grid(
    n_points: int,
    delta_x: float
) -> np.ndarray:
    """
    Compute frequency grid for FFT output.
    
    Args:
        n_points: Number of points in the signal
        delta_x: Spacing between points in x-space
        
    Returns:
        Array of frequencies
    """
    from scipy.fft import fftfreq
    
    return fftfreq(n_points, d=delta_x)


def get_frequency_resolution(
    n_points: int,
    x_range: Tuple[float, float]
) -> float:
    """
    Compute the frequency resolution of the FFT.
    
    Args:
        n_points: Number of points in the signal
        x_range: Tuple of (min_x, max_x) in x-space
        
    Returns:
        Frequency resolution (delta_f)
    """
    x_min, x_max = x_range
    total_range = x_max - x_min
    delta_x = total_range / (n_points - 1)
    return 1.0 / total_range


if __name__ == "__main__":
    # Test resampling
    import matplotlib.pyplot as plt
    
    print("Testing resampling functions...")
    
    # Create test data
    n = 1000
    x = np.linspace(2, 1000, n)
    y = np.sin(x * 0.1) + np.random.randn(n) * 0.1
    
    # Test linear resampling
    x_res, y_res = resample_to_uniform_grid(x, y, 100)
    print(f"Linear resampling: {len(x)} -> {len(x_res)} points")
    
    # Test log-uniform resampling
    x_log = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    y_log = np.sin(x_log * 0.1)
    ln_grid, y_log_res, x_log_res = resample_to_log_uniform_grid(x_log, y_log, 50)
    print(f"Log-uniform resampling: {len(x_log)} -> {len(x_log_res)} points")
    
    # Test preprocessing for FFT
    prime_N = np.linspace(2, 10000, 5000)
    R = np.sin(np.log(prime_N) * 5) + np.random.randn(5000) * 0.1
    prime_N_proc, R_proc, ln_p, mean, std = prepare_for_fft(prime_N, R, max_points=1000)
    print(f"FFT preprocessing: {len(prime_N)} -> {len(prime_N_proc)} points")
    print(f"Mean: {mean:.6f}, Std: {std:.6f}")
    
    print("\nTest completed successfully!")
