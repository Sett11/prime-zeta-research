"""
Prime sieve module.

This module provides optimized implementations of the Sieve of Eratosthenes
for generating prime numbers up to a specified limit.
"""

import math
import time
from typing import Tuple
import numpy as np
from loguru import logger

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed. Progress bar will not be available.")


def generate_primes_sieve_optimized(limit: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized Sieve of Eratosthenes for generating primes up to limit.
    
    For limit < 10^9, uses standard sieve which is sufficiently fast.
    For larger limits, segmented sieve can be used (not implemented here yet).
    
    Args:
        limit: Upper bound for prime generation
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (primes_array, is_prime_bool_array)
        - primes_array: All prime numbers from 2 to limit as np.int64
        - is_prime_bool_array: Boolean array where True indicates prime
        
    Example:
        >>> primes, is_prime = generate_primes_sieve_optimized(100)
        >>> primes
        array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
               61, 67, 71, 73, 79, 83, 89, 97])
    """
    if limit < 2:
        logger.warning(f"Limit {limit} is less than 2, returning empty arrays")
        is_prime = np.zeros(limit + 1, dtype=bool)
        return np.array([], dtype=np.int64), is_prime
    
    logger.info(f"Starting prime generation up to {limit:,}")
    start_time = time.time()
    
    # Standard optimized Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    
    sqrt_limit = int(limit ** 0.5) + 1
    
    # Only sieve odd numbers for optimization
    primes_found = []
    
    if verbose and HAS_TQDM:
        pbar = tqdm(total=sqrt_limit, desc="Sieving primes", unit="iteration")
        last_logged = 0
    
    for p in range(2, sqrt_limit):
        if is_prime[p]:
            # Start sieving from p^2
            is_prime[p * p : limit + 1 : p] = False
            
            if verbose and HAS_TQDM:
                pbar.update(p - last_logged)
                last_logged = p
                
                # Log ETA every 1000 iterations
                if p > 0 and p % 1000 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / p) * (sqrt_limit - p)
                    logger.debug(f"Progress: {p:,}/{sqrt_limit:,} ({100*p/sqrt_limit:.1f}%) - ETA: {eta/60:.1f} min")
    
    if verbose and HAS_TQDM:
        pbar.close()
    
    primes = np.nonzero(is_prime)[0].astype(np.int64)
    
    elapsed = time.time() - start_time
    prime_count = len(primes)
    theoretical = limit / math.log(limit) if limit > 1 else 0
    
    logger.success(
        f"Prime generation completed in {elapsed/60:.2f} minutes. "
        f"Found {prime_count:,} primes. "
        f"Relative deviation: {(prime_count - theoretical) / theoretical * 100:.3f}%"
    )
    
    return primes, is_prime


def generate_primes_odd_only(limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-optimized sieve using only odd numbers.

    This reduces memory usage by ~50% by not storing even numbers.
    However, the returned arrays include all numbers for compatibility.

    Args:
        limit: Upper bound for prime generation

    Returns:
        Tuple of (primes_array, is_prime_bool_array)
    """
    if limit < 2:
        return np.array([], dtype=np.int64), np.zeros(limit + 1, dtype=bool)

    # Only store odd numbers in the sieve (indices represent odd numbers: 1, 3, 5, ...)
    sieve_size = (limit // 2) + 1  # Index i represents number 2*i+1
    is_prime_odd = np.ones(sieve_size, dtype=bool)
    is_prime_odd[0] = False  # 1 is not prime (index 0 = number 1)

    sqrt_limit = int(limit ** 0.5) + 1

    # Only sieve odd numbers starting from 3
    for p in range(3, sqrt_limit + 1, 2):
        if is_prime_odd[p // 2]:
            # For odd p, the starting point is p*p, which is odd
            start = p * p
            # Step by 2p to skip even multiples
            is_prime_odd[start // 2 : sieve_size : p] = False

    # Convert back to full array (including even numbers)
    is_prime = np.zeros(limit + 1, dtype=bool)
    is_prime[2] = True

    # Build full is_prime array by checking odd positions one by one
    # This is slower but guaranteed to work correctly
    for i in range(1, len(is_prime_odd)):
        number = 2 * i + 1
        if number > limit:
            break
        if is_prime_odd[i]:
            is_prime[number] = True

    primes = np.nonzero(is_prime)[0].astype(np.int64)

    return primes, is_prime


def segmented_sieve(limit: int, segment_size: int = 10_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmented Sieve of Eratosthenes for very large limits.
    
    Processes the sieve in segments to reduce memory usage.
    Useful for limit > 10^9.
    
    Args:
        limit: Upper bound for prime generation
        segment_size: Size of each segment (default: 10M)
        
    Returns:
        Tuple of (primes_array, is_prime_bool_array)
    """
    import sys
    
    if limit < 2:
        return np.array([], dtype=np.int64), np.zeros(limit + 1, dtype=bool)
    
    logger.info(f"Starting segmented prime generation up to {limit:,}")
    start_time = time.time()
    
    # Generate small primes up to sqrt(limit) for sieving segments
    sqrt_limit = int(limit ** 0.5) + 1
    small_primes, _ = generate_primes_sieve_optimized(sqrt_limit)
    
    all_primes = [2] if limit >= 2 else []
    
    # Process segments
    low = 3
    high = min(segment_size, limit)
    
    while low <= limit:
        segment = np.ones(high - low + 1, dtype=bool)
        
        for p in small_primes[1:]:  # Skip 2
            # Find the first multiple of p in [low, high]
            start = max(p * p, ((low + p - 1) // p) * p)
            segment[start - low : high - low + 1 : p] = False
        
        # Add primes from this segment
        segment_primes = np.nonzero(segment)[0] + low
        all_primes.extend(segment_primes.tolist())
        
        low = high + 1
        high = min(low + segment_size - 1, limit)
        
        if low % (segment_size * 10) == 0:
            logger.info(f"Progress: {low:,}/{limit:,} ({100*low/limit:.1f}%)")
    
    primes = np.array(all_primes, dtype=np.int64)
    is_prime = np.zeros(limit + 1, dtype=bool)
    is_prime[primes] = True
    
    elapsed = time.time() - start_time
    logger.info(
        f"Segmented prime generation completed in {elapsed:.2f}s. "
        f"Found {len(primes):,} primes."
    )
    
    return primes, is_prime


def count_primes_estimate(limit: int) -> float:
    """
    Estimate the number of primes up to limit using Li(limit).
    
    Args:
        limit: Upper bound
        
    Returns:
        Estimated count of primes
    """
    if limit < 2:
        return 0
    return limit / math.log(limit)


if __name__ == "__main__":
    # Quick test
    test_limit = 100
    primes, is_prime = generate_primes_sieve_optimized(test_limit)
    print(f"Primes up to {test_limit}: {primes}")
    print(f"Count: {len(primes)}")
    
    # Verify with known count
    known_count = 25  # Primes <= 100
    assert len(primes) == known_count, f"Expected {known_count}, got {len(primes)}"
    print("Test passed!")
