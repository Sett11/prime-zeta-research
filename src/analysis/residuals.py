"""
Residuals computation module.

This module provides functions for computing the residual term R(p)
after removing the smooth trend (k*Li(p) + b) from CN(p).
"""

from typing import Dict, Tuple
import numpy as np
from loguru import logger

from .li_function import li_vectorized


def compute_residuals(
    prime_N: np.ndarray,
    CN_values: np.ndarray,
    k: float,
    b: float,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residual term R(p) = CN(p) - (k*Li(p) + b).
    
    This is the main function for extracting the oscillatory part
    of the cumulative need function.
    
    Args:
        prime_N: Array of prime numbers
        CN_values: Array of CN(p) values
        k: Slope coefficient from linear regression
        b: Intercept from linear regression
        verbose: Whether to show progress
        
    Returns:
        Tuple of (prime_N, R_values)
    """
    if verbose:
        logger.info("Computing residuals R(p) = CN(p) - (k*Li(p) + b)")
    
    # Compute Li(p)
    li_values = li_vectorized(prime_N, verbose=False)
    
    # Compute trend
    trend = k * li_values + b
    
    # Compute residuals
    R_values = CN_values - trend
    
    if verbose:
        logger.info(
            f"Residuals computed: "
            f"mean={R_values.mean():.6f}, "
            f"std={R_values.std():.4f}, "
            f"min={R_values.min():.4f}, "
            f"max={R_values.max():.4f}"
        )
    
    return prime_N, R_values


def compute_residuals_dict(
    need_data: Dict[str, np.ndarray],
    fit_result: Dict[str, float],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute residuals for need data using fit results.
    
    Args:
        need_data: Dictionary with 'prime_N', 'CN' keys
        fit_result: Dictionary with 'k', 'b' keys
        verbose: Whether to show progress
        
    Returns:
        Dictionary with 'prime_N', 'R' keys
    """
    prime_N, R_values = compute_residuals(
        need_data["prime_N"],
        need_data["CN"],
        fit_result["k"],
        fit_result["b"],
        verbose=verbose
    )
    
    return {
        "prime_N": prime_N,
        "R": R_values
    }


def center_residuals(R_values: np.ndarray) -> np.ndarray:
    """
    Center residuals to have zero mean.
    
    Args:
        R_values: Array of residual values
        
    Returns:
        Centered residual array
    """
    return R_values - R_values.mean()


def normalize_residuals(R_values: np.ndarray) -> np.ndarray:
    """
    Normalize residuals to have unit variance.
    
    Args:
        R_values: Array of residual values
        
    Returns:
        Normalized residual array (zero mean, unit variance)
    """
    centered = R_values - R_values.mean()
    return centered / centered.std()


def compute_residual_statistics(R_values: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for residual values.
    
    Args:
        R_values: Array of residual values
        
    Returns:
        Dictionary of statistics
    """
    return {
        "count": len(R_values),
        "mean": float(np.mean(R_values)),
        "std": float(np.std(R_values)),
        "variance": float(np.var(R_values)),
        "min": float(np.min(R_values)),
        "max": float(np.max(R_values)),
        "range": float(np.max(R_values) - np.min(R_values)),
        "median": float(np.median(R_values)),
        "skewness": float(np.mean(((R_values - R_values.mean()) / R_values.std()) ** 3)),
        "kurtosis": float(np.mean(((R_values - R_values.mean()) / R_values.std()) ** 4)) - 3,
        "percentile_1": float(np.percentile(R_values, 1)),
        "percentile_5": float(np.percentile(R_values, 5)),
        "percentile_25": float(np.percentile(R_values, 25)),
        "percentile_75": float(np.percentile(R_values, 75)),
        "percentile_95": float(np.percentile(R_values, 95)),
        "percentile_99": float(np.percentile(R_values, 99)),
    }


def detect_outliers(
    R_values: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in residual values.
    
    Args:
        R_values: Array of residual values
        threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of (outlier_indices, outlier_values)
    """
    mean = np.mean(R_values)
    std = np.std(R_values)
    
    z_scores = np.abs((R_values - mean) / std)
    outlier_mask = z_scores > threshold
    
    return np.where(outlier_mask)[0], R_values[outlier_mask]


if __name__ == "__main__":
    # Test residuals computation
    from src.primes.sieve import generate_primes_sieve_optimized
    from src.primes.need import compute_all_need_data
    from .regression import fit_cn_to_li
    
    print("Testing residuals computation...")
    
    # Generate small dataset
    MAX_N = 100_000
    C = 10.0
    
    primes, _ = generate_primes_sieve_optimized(MAX_N)
    print(f"Generated {len(primes)} primes")
    
    # Compute Need and CN
    need_data = compute_all_need_data(primes, C, MAX_N, verbose=True)
    
    # Fit to Li
    fit_result = fit_cn_to_li(need_data["prime_N"], need_data["CN"])
    
    # Compute residuals
    res_data = compute_residuals_dict(need_data, fit_result, verbose=True)
    
    # Statistics
    stats = compute_residual_statistics(res_data["R"])
    print("\nResidual statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Detect outliers
    outliers_idx, outliers_values = detect_outliers(res_data["R"], threshold=3.0)
    print(f"\nOutliers (>{3}sigma): {len(outliers_idx)}")
    if len(outliers_idx) > 0:
        print(f"  Max outlier: {outliers_values.max():.4f}")
    
    print("\nTest completed successfully!")
