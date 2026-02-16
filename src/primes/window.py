"""
Window functions module.

This module provides functions for calculating window sizes and managing
the local analysis window around each prime number.
"""

import math
from typing import Tuple
import numpy as np


def window_size(N: int, C: float) -> int:
    """
    Calculate window half-width W(N) = max(1, round(C * ln(N))).
    
    The window size grows logarithmically with N to maintain approximately
    constant expected number of primes in the window.
    
    Args:
        N: Prime number (or any positive integer)
        C: Scaling parameter (typically around 10.0)
        
    Returns:
        Window half-width as integer
        
    Example:
        >>> window_size(100, 10.0)
        46  # round(10 * ln(100)) = round(10 * 4.605) = 46
    """
    if N < 3:
        return 1
    w = int(round(C * math.log(N)))
    return max(w, 1)


def window_size_vectorized(primes: np.ndarray, C: float) -> np.ndarray:
    """
    Vectorized calculation of window sizes for all primes.
    
    Args:
        primes: Array of prime numbers
        C: Scaling parameter
        
    Returns:
        Array of window half-widths
    """
    W = np.round(C * np.log(primes.astype(np.float64))).astype(np.int64)
    W = np.maximum(W, 1)
    return W


def calculate_window_bounds(p: int, W: int, max_n: int) -> Tuple[int, int]:
    """
    Calculate the bounds of the window around prime p.
    
    Args:
        p: Prime number (center of window)
        W: Window half-width
        max_n: Maximum value in the dataset (for boundary checking)
        
    Returns:
        Tuple of (left_bound, right_bound)
    """
    left = max(2, p - W)
    right = min(max_n, p + W)
    return left, right


def expected_primes_in_window(p: int, C: float) -> float:
    """
    Calculate expected number of primes in the window around p.
    
    Based on the prime number theorem, density near p is ~1/ln(p).
    The window length is 2*W(p) = 2*C*ln(p), so expected count is:
    E = 2*C*ln(p) / ln(p) = 2*C
    
    Args:
        p: Prime number
        C: Window scaling parameter
        
    Returns:
        Expected number of primes in window (should be ~2*C)
    """
    return 2 * C


def validate_window_for_prime(p: int, W: int, max_n: int, min_margin: int = 10) -> bool:
    """
    Check if the window for prime p is valid (not too close to boundaries).
    
    Args:
        p: Prime number
        W: Window half-width
        max_n: Maximum value in dataset
        min_margin: Minimum distance from boundary required
        
    Returns:
        True if window is valid
    """
    left, right = calculate_window_bounds(p, W, max_n)
    return (p - left >= min_margin) and (right - p >= min_margin)


def filter_valid_primes(
    primes: np.ndarray, W_values: np.ndarray, max_n: int, min_margin: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter primes to keep only those with valid windows.
    
    Args:
        primes: Array of prime numbers
        W_values: Array of window sizes for each prime
        max_n: Maximum value in dataset
        min_margin: Minimum distance from boundary
        
    Returns:
        Tuple of (valid_primes, valid_W)
    """
    left_bounds = np.maximum(2, primes - W_values)
    right_bounds = np.minimum(max_n, primes + W_values)
    
    left_margin = primes - left_bounds
    right_margin = right_bounds - primes
    
    valid_mask = (left_margin >= min_margin) & (right_margin >= min_margin)
    
    return primes[valid_mask], W_values[valid_mask]


if __name__ == "__main__":
    # Test window functions
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    C = 10.0
    
    W = window_size_vectorized(primes, C)
    print(f"Primes: {primes}")
    print(f"Window sizes: {W}")
    
    # Test expected count
    for p in [10, 100, 1000, 10000]:
        expected = expected_primes_in_window(p, C)
        print(f"Expected primes around {p}: {expected:.1f}")
    
    # Validate windows
    valid_primes, valid_W = filter_valid_primes(primes, W, 100)
    print(f"Valid primes: {valid_primes}")
    print(f"Valid windows: {valid_W}")
