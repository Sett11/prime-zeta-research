#!/usr/bin/env python3
"""
Stage 0: Prime generation.

Run separately to generate primes and save to DB.
Parameters are read from config.yaml.

Usage:
    python scripts/0_generate_primes.py
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from src.config_loader import get_config
from src.primes.sieve import generate_primes_sieve_optimized
from src.database.schema import PrimesMetadata, create_database, get_session
from src.database.operations import DatabaseManager


def setup_logging():
    """Configure logging."""
    config = get_config()
    log_level = config.log_level
    log_file = config.get("logging", "file", "research.log")

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(log_file, level=log_level, rotation="10 MB", retention=5)


def check_existing_primes(db_manager: DatabaseManager, max_n: int) -> bool:
    """
    Check if primes already exist for given max_n.

    Args:
        db_manager: Database manager
        max_n: Maximum number

    Returns:
        True if primes already exist
    """
    metadata = db_manager.primes_metadata.get_by_max_n(max_n)
    return metadata is not None


def estimate_time_remaining(current_iter: int, total_iter: int, elapsed_seconds: float) -> str:
    """Estimate remaining time."""
    if current_iter == 0:
        return "N/A"
    rate = current_iter / elapsed_seconds
    remaining = total_iter - current_iter
    eta_seconds = remaining / rate

    if eta_seconds > 3600:
        return f"{eta_seconds/3600:.1f} hours"
    elif eta_seconds > 60:
        return f"{eta_seconds/60:.1f} minutes"
    else:
        return f"{eta_seconds:.1f} seconds"


def main():
    """Main function for prime generation."""
    setup_logging()

    # Load configuration
    config = get_config()
    max_n = config.max_n
    db_path = config.db_path

    # Calculate expected runtime (safe for max_n <= 1)
    theoretical_primes = max_n / np.log(max_n) if max_n > 1 else 0
    sqrt_n = int(max_n ** 0.5) if max_n > 0 else 0

    logger.info("=" * 70)
    logger.info("STAGE 0: Prime Generation")
    logger.info("=" * 70)
    logger.info(f"Limit max_n: {max_n:,}")
    logger.info(f"Expected primes (x/ln x): {theoretical_primes:,.0f}")
    logger.info(f"Upper limit for checking (sqrt(n)): {sqrt_n:,}")
    logger.info(f"db_path: {db_path}")

    # Check existing data
    with DatabaseManager(db_path) as db:
        if check_existing_primes(db, max_n):
            logger.warning(f"Primes for max_n={max_n:,} already exist in DB!")
            logger.info("Skipping generation - data already exists.")
            return

        # Generate primes
        logger.info("-" * 70)
        logger.info("Starting prime generation...")
        logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()

        primes, is_prime = generate_primes_sieve_optimized(max_n, verbose=True)

        elapsed = time.time() - start_time
        prime_count = len(primes)
        theoretical = max_n / np.log(max_n) if max_n > 1 else 0
        deviation = (prime_count - theoretical) / theoretical * 100 if theoretical > 0 else 0

        logger.info("-" * 70)
        logger.info("Generation complete!")
        logger.info(f"Runtime: {elapsed/60:.2f} minutes")
        logger.info(f"Primes found: {prime_count:,}")
        logger.info(f"Theoretical (x/ln(x)): {theoretical:,.2f}")
        logger.info(f"Deviation: {deviation:.2f}%")

        # Save to DB
        logger.info("-" * 70)
        logger.info("Saving data to database...")

        # Create generation record
        primes_file = os.path.join("data", "primes", f"primes_{max_n}.npy")
        os.makedirs(os.path.dirname(primes_file), exist_ok=True)

        # Save primes to binary file
        np.save(primes_file, primes)
        logger.info(f"Primes saved to: {primes_file}")

        # Create experiment
        experiment = db.experiments.get_or_create(
            max_n, 0.0,  # C=0 for metadata-only record
            description=f"Prime generation for max_n={max_n}"
        )

        # Save metadata
        db.primes_metadata.save(
            experiment=experiment,
            max_n=max_n,
            prime_count=prime_count,
            data_file=primes_file,
            generation_time=elapsed
        )

        db.experiments.update_status(experiment, "primes_ready")

        logger.info("-" * 70)
        logger.success("Data saved to database successfully!")
        logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("=" * 70)
    logger.info("STAGE 0: Prime Generation - COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
