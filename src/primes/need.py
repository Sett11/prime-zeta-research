"""
Need(p) computation module.

This module provides functions for calculating the local "need" value
for each prime number and the cumulative need CN(p).
"""

import time
from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger

from .window import window_size_vectorized, filter_valid_primes, calculate_window_bounds


def compute_need_single(
    p: int, 
    primes: np.ndarray, 
    C: float, 
    max_n: int
) -> float:
    """
    Compute Need(p) for a single prime number.
    
    Need(p) = (L(p) - k_primes(p)) / L(p)
    where L(p) is the window length and k_primes(p) is the count of primes in the window.
    
    Args:
        p: Prime number to compute Need for
        primes: Array of all primes up to max_n
        C: Window scaling parameter
        max_n: Maximum value in the dataset
        
    Returns:
        Need(p) value (fraction of composite numbers in the window)
    """
    W = int(round(C * np.log(p)))
    W = max(W, 1)
    
    left, right = calculate_window_bounds(p, W, max_n)
    L = right - left + 1
    
    if L <= 0:
        return 0.0
    
    # Binary search for count of primes in window
    left_idx = np.searchsorted(primes, left, side='left')
    right_idx = np.searchsorted(primes, right, side='right')
    k_primes = right_idx - left_idx
    
    return (L - k_primes) / L


def compute_need_vectorized(
    primes: np.ndarray, 
    C: float, 
    max_n: int,
    batch_size: int = 100000,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Need(p) for all primes using vectorized operations.
    
    This is the main function for Need(p) computation, optimized for
    large arrays of primes.
    
    Args:
        primes: Array of prime numbers
        C: Window scaling parameter
        max_n: Maximum value in the dataset
        batch_size: Batch size for progress reporting
        verbose: Whether to show progress
        
    Returns:
        Tuple of (valid_primes, need_values)
        - valid_primes: Primes with valid windows
        - need_values: Corresponding Need(p) values
    """
    n = len(primes)
    if verbose:
        logger.info(f"Computing Need(p) for {n:,} primes with C={C}")
        start_time = time.time()
    
    # Compute window sizes for all primes
    W_values = window_size_vectorized(primes, C)
    
    # Filter out primes too close to boundaries
    valid_primes, valid_W = filter_valid_primes(primes, W_values, max_n)
    
    if verbose:
        logger.info(f"Valid primes after filtering: {len(valid_primes):,} / {n:,}")
    
    # Compute Need(p) for each valid prime
    need_values = np.zeros(len(valid_primes), dtype=np.float64)
    
    for i, p in enumerate(valid_primes):
        W = valid_W[i]
        left, right = calculate_window_bounds(p, W, max_n)
        L = right - left + 1
        
        if L > 0:
            left_idx = np.searchsorted(primes, left, side='left')
            right_idx = np.searchsorted(primes, right, side='right')
            k_primes = right_idx - left_idx
            need_values[i] = (L - k_primes) / L
        else:
            need_values[i] = 0.0
        
        # Progress reporting
        if verbose and (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(valid_primes) - i - 1) / rate if rate > 0 else 0
            logger.debug(
                f"  Progress: {i+1:,}/{len(valid_primes):,} "
                f"({100*(i+1)/len(valid_primes):.1f}%) - "
                f"ETA: {remaining:.0f}s"
            )
    
    if verbose:
        elapsed = time.time() - start_time
        logger.info(
            f"Need(p) computation completed in {elapsed:.2f}s. "
            f"Mean Need: {need_values.mean():.6f}, Std: {need_values.std():.6f}"
        )
    
    return valid_primes, need_values


def compute_need_batched(
    primes: np.ndarray,
    C: float,
    max_n: int,
    batch_size: int = 10000,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Need(p) using batched processing for memory efficiency.
    
    This version processes primes in batches to reduce memory usage
    when dealing with very large prime arrays.
    
    Args:
        primes: Array of prime numbers
        C: Window scaling parameter
        max_n: Maximum value in the dataset
        batch_size: Number of primes to process per batch
        verbose: Whether to show progress
        
    Returns:
        Tuple of (valid_primes, need_values)
    """
    n = len(primes)
    if verbose:
        logger.info(f"Batched Need(p) computation for {n:,} primes")
        start_time = time.time()
    
    all_valid_primes = []
    all_need_values = []
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_primes = primes[batch_start:batch_end]
        
        batch_W = window_size_vectorized(batch_primes, C)
        valid_batch_primes, valid_batch_W = filter_valid_primes(
            batch_primes, batch_W, max_n
        )
        
        # Compute Need for this batch
        for i, p in enumerate(valid_batch_primes):
            W = valid_batch_W[i]
            left, right = calculate_window_bounds(p, W, max_n)
            L = right - left + 1
            
            if L > 0:
                left_idx = np.searchsorted(primes, left, side='left')
                right_idx = np.searchsorted(primes, right, side='right')
                k_primes = right_idx - left_idx
                need_val = (L - k_primes) / L
            else:
                need_val = 0.0
            
            all_valid_primes.append(p)
            all_need_values.append(need_val)
        
        if verbose:
            batch_progress = batch_end / n * 100
            print(f"\r  Progress: {batch_progress:.1f}%", end="", flush=True)
    
    if verbose:
        elapsed = time.time() - start_time
        print()  # New line after progress
        logger.info(
            f"Batched Need(p) computation completed in {elapsed:.2f}s"
        )
    
    return (
        np.array(all_valid_primes, dtype=np.int64),
        np.array(all_need_values, dtype=np.float64)
    )


def iter_need_batches(
    primes: np.ndarray,
    C: float,
    max_n: int,
    batch_size: int = 100000,
    verbose: bool = True,
):
    """
    Iterator over Need(p) batches for streaming processing.

    Unlike ``compute_need_batched``, does not accumulate all values in memory,
    returns one batch at a time:
        yield valid_batch_primes, need_batch_values

    This allows:
    - immediate database writing;
    - incremental CN(p) calculation;
    - not holding entire Need/CN array in memory for max_n ~ 1e9+.

    Args:
        primes: Array of all primes up to max_n
        C: Window parameter
        max_n: Maximum value in the dataset
        batch_size: Batch size by indices in primes array
        verbose: Log progress

    Yields:
        (valid_primes_batch, need_values_batch)
    """
    n = len(primes)
    if verbose:
        logger.info(
            f"Streaming batched Need(p) computation for {n:,} primes "
            f"with C={C}, batch_size={batch_size}"
        )
        start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_primes = primes[batch_start:batch_end]

        # Window sizes and filtering only for current batch
        batch_W = window_size_vectorized(batch_primes, C)
        valid_batch_primes, valid_batch_W = filter_valid_primes(
            batch_primes, batch_W, max_n
        )

        if len(valid_batch_primes) == 0:
            if verbose:
                batch_progress = batch_end / n * 100
                logger.debug(
                    f"  Batch {batch_start:,}-{batch_end:,} "
                    f"({batch_progress:.1f}%) -> 0 valid primes"
                )
            continue

        need_batch = np.zeros(len(valid_batch_primes), dtype=np.float64)

        for i, p in enumerate(valid_batch_primes):
            W = valid_batch_W[i]
            left, right = calculate_window_bounds(p, W, max_n)
            L = right - left + 1

            if L > 0:
                left_idx = np.searchsorted(primes, left, side="left")
                right_idx = np.searchsorted(primes, right, side="right")
                k_primes = right_idx - left_idx
                need_batch[i] = (L - k_primes) / L
            else:
                need_batch[i] = 0.0

        if verbose:
            batch_progress = batch_end / n * 100
            logger.debug(
                f"  Batch {batch_start:,}-{batch_end:,} "
                f"({batch_progress:.1f}%) -> "
                f"{len(valid_batch_primes):,} valid primes"
            )

        yield valid_batch_primes, need_batch

    if verbose:
        elapsed = time.time() - start_time
        logger.info(
            f"Streaming Need(p) computation finished in {elapsed:.2f}s"
        )


def iter_need_batches_segment(
    primes: np.ndarray,
    C: float,
    max_n: int,
    start_idx: int,
    end_idx: int,
    batch_size: int = 100000,
    verbose: bool = True,
):
    """
    Iterator over Need(p) batches for segment [start_idx, end_idx).

    Used in multiprocessing mode: worker processes its index range,
    but when calculating windows uses full primes array (searchsorted),
    so windows may extend beyond segment boundaries - result is correct.

    Args:
        primes: Array of all primes up to max_n (full, read-only).
        C: Window parameter.
        max_n: Maximum value in the dataset.
        start_idx: Start index of segment (inclusive).
        end_idx: End index of segment (exclusive).
        batch_size: Batch size by indices.
        verbose: Log progress.

    Yields:
        (valid_primes_batch, need_values_batch) only for primes in [start_idx, end_idx).
    """
    n = len(primes)
    end_idx = min(end_idx, n)
    start_idx = max(0, start_idx)
    if start_idx >= end_idx:
        return

    if verbose:
        logger.info(
            f"Streaming Need(p) segment [{start_idx:,}, {end_idx:,}) "
            f"of {n:,} primes, C={C}, batch_size={batch_size}"
        )
        start_time = time.time()

    for batch_start in range(start_idx, end_idx, batch_size):
        batch_end = min(batch_start + batch_size, end_idx)
        batch_primes = primes[batch_start:batch_end]

        batch_W = window_size_vectorized(batch_primes, C)
        valid_batch_primes, valid_batch_W = filter_valid_primes(
            batch_primes, batch_W, max_n
        )

        if len(valid_batch_primes) == 0:
            if verbose:
                logger.debug(
                    f"  Segment batch {batch_start:,}-{batch_end:,} -> 0 valid"
                )
            continue

        need_batch = np.zeros(len(valid_batch_primes), dtype=np.float64)
        for i, p in enumerate(valid_batch_primes):
            W = valid_batch_W[i]
            left, right = calculate_window_bounds(p, W, max_n)
            L = right - left + 1
            if L > 0:
                left_idx = np.searchsorted(primes, left, side="left")
                right_idx = np.searchsorted(primes, right, side="right")
                k_primes = right_idx - left_idx
                need_batch[i] = (L - k_primes) / L
            else:
                need_batch[i] = 0.0

        if verbose:
            logger.debug(
                f"  Segment batch {batch_start:,}-{batch_end:,} -> "
                f"{len(valid_batch_primes):,} valid"
            )
        yield valid_batch_primes, need_batch

    if verbose:
        elapsed = time.time() - start_time
        logger.info(f"Segment Need(p) finished in {elapsed:.2f}s")


def compute_cumulative_need(need_values: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative need CN(p) = sum of Need(p) up to each prime.
    
    Args:
        need_values: Array of Need(p) values
        
    Returns:
        Array of CN(p) values
    """
    return np.cumsum(need_values)


def compute_all_need_data(
    primes: np.ndarray,
    C: float,
    max_n: int,
    use_batched: bool = False,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute all need-related data for a given C value.
    
    This is the main entry point for need computation, returning
    all related arrays in a single call.
    
    Args:
        primes: Array of all primes up to max_n
        C: Window scaling parameter
        max_n: Maximum value in the dataset
        use_batched: Whether to use batched processing
        verbose: Whether to show progress
        
    Returns:
        Dictionary with keys:
        - 'prime_N': Valid primes
        - 'need': Need(p) values
        - 'CN': Cumulative CN(p) values
    """
    if use_batched:
        prime_N, need_arr = compute_need_batched(
            primes, C, max_n, verbose=verbose
        )
    else:
        prime_N, need_arr = compute_need_vectorized(
            primes, C, max_n, verbose=verbose
        )
    
    CN_arr = compute_cumulative_need(need_arr)
    
    if verbose:
        logger.info(
            f"C={C}: Computed {len(prime_N):,} primes, "
            f"CN range: [{CN_arr[0]:.4f}, {CN_arr[-1]:.4f}]"
        )
    
    return {
        "prime_N": prime_N,
        "need": need_arr,
        "CN": CN_arr
    }


def compute_need_statistics(need_values: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for Need(p) values.
    
    Args:
        need_values: Array of Need(p) values
        
    Returns:
        Dictionary of statistics
    """
    return {
        "count": len(need_values),
        "mean": float(np.mean(need_values)),
        "std": float(np.std(need_values)),
        "min": float(np.min(need_values)),
        "max": float(np.max(need_values)),
        "median": float(np.median(need_values)),
        "percentile_25": float(np.percentile(need_values, 25)),
        "percentile_75": float(np.percentile(need_values, 75)),
    }


if __name__ == "__main__":
    # Test need computation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from primes.sieve import generate_primes_sieve_optimized
    
    print("Testing Need(p) computation...")
    
    # Generate small set of primes
    MAX_N = 10000
    primes, _ = generate_primes_sieve_optimized(MAX_N)
    print(f"Generated {len(primes)} primes up to {MAX_N}")
    
    # Compute Need(p)
    C = 10.0
    prime_N, need_arr = compute_need_vectorized(primes, C, MAX_N, verbose=True)
    print(f"Computed Need(p) for {len(prime_N)} primes")
    print(f"Mean Need: {need_arr.mean():.6f}")
    
    # Compute CN(p)
    CN_arr = compute_cumulative_need(need_arr)
    print(f"CN range: [{CN_arr[0]:.4f}, {CN_arr[-1]:.4f}]")
    
    # Statistics
    stats = compute_need_statistics(need_arr)
    print(f"Statistics: {stats}")
    
    print("\nTest completed successfully!")
