#!/usr/bin/env python3
"""
Stage 1: Compute Need(p), CN(p), Li(p), R(p).

Computes all necessary functions and saves to DB.
Parameters are read from config.yaml.

Usage:
    python scripts/1_compute_need_cn.py
"""

import sys
import time
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, cpu_count

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from loguru import logger
from src.config_loader import get_config
from src.primes.sieve import generate_primes_sieve_optimized
from src.primes.need import compute_all_need_data, iter_need_batches, iter_need_batches_segment
from src.analysis.li_function import li_vectorized
from src.analysis.regression import linear_regression
from src.analysis.residuals import compute_residuals
from src.database.operations import DatabaseManager


def _worker_need_cn_segment(
    primes_file: str,
    start_idx: int,
    end_idx: int,
    max_n: int,
    c_value: float,
    out_dir: str,
    batch_size: int = 300_000,
) -> None:
    """
    Worker: computes Need/CN for segment [start_idx, end_idx) and saves to npz.
    Called in separate process.
    """
    primes = np.load(primes_file, mmap_mode="r")
    if primes.dtype != np.int64:
        primes = np.asarray(primes, dtype=np.int64)
    primes_list = []
    need_list = []
    cn_list = []
    running_cn = 0.0
    for batch_primes, need_batch in iter_need_batches_segment(
        primes,
        c_value,
        max_n,
        start_idx,
        end_idx,
        batch_size=batch_size,
        verbose=False,
    ):
        if len(batch_primes) == 0:
            continue
        cn_batch = np.cumsum(need_batch) + running_cn
        running_cn = float(cn_batch[-1])
        primes_list.append(np.asarray(batch_primes, dtype=np.int64))
        need_list.append(need_batch)
        cn_list.append(cn_batch)
    seg_primes = np.concatenate(primes_list) if primes_list else np.array([], dtype=np.int64)
    seg_need = np.concatenate(need_list) if need_list else np.array([], dtype=np.float64)
    seg_cn_local = np.concatenate(cn_list) if cn_list else np.array([], dtype=np.float64)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path / f"need_cn_{start_idx}_{end_idx}.npz",
        primes=seg_primes,
        need=seg_need,
        cn_local=seg_cn_local,
    )


def setup_logging():
    """Configure logging."""
    config = get_config()
    log_level = config.log_level
    log_file = config.get("logging", "file", "research.log")

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(log_file, level=log_level, rotation="10 MB", retention=5)


def check_existing_data(db_manager: DatabaseManager, max_n: int, c_value: float) -> bool:
    """
    Check if computations already exist for given parameters.

    Args:
        db_manager: Database manager
        max_n: Maximum number
        c_value: C parameter

    Returns:
        True if data already exists
    """
    # Try to find existing experiment
    experiment = None
    if hasattr(db_manager.experiments, "get_by_params"):
        experiment = db_manager.experiments.get_by_params(
            max_n, c_value
        )
    else:
        experiment = db_manager.experiments.get_or_create(max_n, c_value)

    if experiment is None:
        return False

    results = db_manager.results.get_by_experiment(experiment.id, limit=1)
    return len(results) > 0


def ensure_results_dir(base_dir: str) -> Path:
    """Create results directory."""
    results_dir = Path(base_dir) / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_need_cn_data(
    primes: np.ndarray,
    need_values: np.ndarray,
    cn_values: np.ndarray,
    save_path: Path,
    max_n: int,
    c_value: float
):
    """
    Plot Need(p) and CN(p) in linear and logarithmic scales.

    Args:
        primes: Array of prime numbers
        need_values: Array of Need(p) values
        cn_values: Array of CN(p) values
        save_path: Path to save plot
        max_n: Maximum number
        c_value: C parameter
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Need(p) and CN(p) for max_n={max_n:,}, C={c_value}', fontsize=14, fontweight='bold')

    # Plot 1: Need(p) vs p (linear)
    ax1 = axes[0, 0]
    ax1.plot(primes, need_values, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('p (prime number)', fontsize=10)
    ax1.set_ylabel('Need(p)', fontsize=10)
    ax1.set_title('Need(p) vs p (linear)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Need(p) vs p (log scale)
    ax2 = axes[0, 1]
    ax2.semilogx(primes, need_values, 'b-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('p (log scale)', fontsize=10)
    ax2.set_ylabel('Need(p)', fontsize=10)
    ax2.set_title('Need(p) vs p (log p axis)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: CN(p) vs p (linear)
    ax3 = axes[1, 0]
    ax3.plot(primes, cn_values, 'r-', linewidth=0.5, alpha=0.7)
    ax3.set_xlabel('p (prime number)', fontsize=10)
    ax3.set_ylabel('CN(p)', fontsize=10)
    ax3.set_title('CN(p) vs p (linear)', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: CN(p) vs p (log scale)
    ax4 = axes[1, 1]
    ax4.semilogx(primes, cn_values, 'r-', linewidth=0.5, alpha=0.7)
    ax4.set_xlabel('p (log scale)', fontsize=10)
    ax4.set_ylabel('CN(p)', fontsize=10)
    ax4.set_title('CN(p) vs p (log p axis)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved: {save_path}")


def save_need_cn_data(
    primes: np.ndarray,
    need_values: np.ndarray,
    cn_values: np.ndarray,
    results_dir: Path,
    max_n: int,
    c_value: float
):
    """
    Save Need and CN data in CSV and JSON formats.

    Args:
        primes: Array of prime numbers
        need_values: Array of Need(p) values
        cn_values: Array of CN(p) values
        results_dir: Directory to save
        max_n: Maximum number
        c_value: C parameter
    """
    # Compute statistics
    need_stats = {
        "min": float(np.min(need_values)),
        "max": float(np.max(need_values)),
        "mean": float(np.mean(need_values)),
        "median": float(np.median(need_values)),
        "std": float(np.std(need_values))
    }

    cn_stats = {
        "min": float(np.min(cn_values)),
        "max": float(np.max(cn_values)),
        "mean": float(np.mean(cn_values)),
        "median": float(np.median(cn_values)),
        "std": float(np.std(cn_values)),
        "initial": float(cn_values[0]) if len(cn_values) > 0 else None,
        "final": float(cn_values[-1]) if len(cn_values) > 0 else None
    }

    # Save CSV
    csv_path = results_dir / "need_cn_data.csv"
    header = "prime,need,cn"
    data_to_save = np.column_stack([primes, need_values, cn_values])
    np.savetxt(csv_path, data_to_save, delimiter=',', header=header,
               fmt='%.10f', comments='')
    logger.info(f"CSV data saved: {csv_path}")

    # Save JSON
    json_path = results_dir / "need_cn_data.json"
    json_data = {
        "experiment": {
            "max_n": int(max_n),
            "c_value": float(c_value),
            "timestamp": datetime.now().isoformat(),
            "n_points": int(len(primes))
        },
        "statistics": {
            "need": need_stats,
            "cn": cn_stats
        },
        "data": {
            "primes": primes.tolist() if len(primes) <= 1000000 else "too_large",
            "need": need_values.tolist() if len(need_values) <= 1000000 else "too_large",
            "cn": cn_values.tolist() if len(cn_values) <= 1000000 else "too_large"
        }
    }

    # For large data, save only statistics and metadata
    if len(primes) > 1000000:
        json_data["data"]["note"] = "Data too large for JSON. Use CSV file."
        json_data["data"]["csv_file"] = str(csv_path)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"JSON data saved: {json_path}")

    return csv_path, json_path


def main():
    """Main function for Need/CN/Li/R computation."""
    parser = argparse.ArgumentParser(description="Stage 1: Need(p), CN(p)")
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Multiprocess mode: segments by indices, CN merge, DB write",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=None,
        help="Number of processes for --mp (default: min(cpu_count(), 8))",
    )
    args = parser.parse_args()
    use_mp = args.mp
    n_procs = args.n_procs

    setup_logging()

    # Load configuration
    config = get_config()
    max_n = config.max_n
    c_value = config.c_value
    db_path = config.db_path

    logger.info("=" * 60)
    logger.info("STAGE 1: Need(p), CN(p), Li(p), R(p)")
    logger.info("=" * 60)
    logger.info(f"max_n: {max_n:,}")
    logger.info(f"c_value: {c_value}")
    logger.info(f"db_path: {db_path}")
    if use_mp:
        logger.info("Mode: multiprocess (--mp)")

    with DatabaseManager(db_path) as db:
        # Check if data exists
        if check_existing_data(db, max_n, c_value):
            logger.warning(f"Data for max_n={max_n:,}, c={c_value} already exists!")
            logger.info("Skipping computation - data already exists.")
            return

        # Check primes exist
        primes_metadata = db.primes_metadata.get_by_max_n(max_n)
        if primes_metadata is None:
            logger.error(f"Primes for max_n={max_n:,} not found!")
            logger.info("First run: python scripts/0_generate_primes.py")
            return

        primes_file = primes_metadata.data_file
        if not os.path.isabs(primes_file):
            # Relative path - resolve from project root (where config.yaml is)
            root = Path(config.config_path).parent if getattr(config, "config_path", None) else Path.cwd()
            primes_file = str(root / primes_file)

        if use_mp:
            # Multiprocess mode: memmap, segments, workers, merge, DB
            primes_mm = np.load(primes_file, mmap_mode="r")
            n_primes = len(primes_mm)
            n_procs = n_procs or min(cpu_count() or 4, 8)
            segment_size = (n_primes + n_procs - 1) // n_procs
            segments = []
            for i in range(n_procs):
                start_idx = i * segment_size
                end_idx = min(start_idx + segment_size, n_primes)
                if start_idx < end_idx:
                    segments.append((start_idx, end_idx))
            logger.info(f"Segments: {len(segments)}, primes: {n_primes:,}")

            # data/need_cn next to data/primes
            need_cn_dir = Path(primes_file).resolve().parent.parent / "need_cn"
            need_cn_dir.mkdir(parents=True, exist_ok=True)

            procs = []
            for start_idx, end_idx in segments:
                p = Process(
                    target=_worker_need_cn_segment,
                    args=(primes_file, start_idx, end_idx, max_n, c_value, str(need_cn_dir)),
                    kwargs={"batch_size": 300_000},
                )
                p.start()
                procs.append((p, start_idx, end_idx))
            for p, _, _ in procs:
                p.join()

            experiment = db.experiments.get_or_create(
                max_n,
                c_value,
                description=f"Analysis with max_n={max_n}, c={c_value}",
            )
            db.experiments.update_status(experiment, "computing")

            cn_offset = 0.0
            batch_size_db = 500_000
            start_time = time.time()
            total_written = 0
            log_every = 3_000_000  # log every 3M rows
            logger.info(f"Merge and DB write (batch {batch_size_db:,})...")
            for seg_idx, (start_idx, end_idx) in enumerate(segments):
                npz_path = need_cn_dir / f"need_cn_{start_idx}_{end_idx}.npz"
                if not npz_path.exists():
                    logger.error(f"Worker file not found: {npz_path}")
                    continue
                data = np.load(npz_path)
                seg_primes = data["primes"]
                seg_need = data["need"]
                seg_cn_local = data["cn_local"]
                if len(seg_cn_local) > 0:
                    cn_global = seg_cn_local + cn_offset
                    cn_offset = float(cn_global[-1])
                else:
                    cn_global = seg_cn_local
                for i in range(0, len(seg_primes), batch_size_db):
                    sl = slice(i, min(i + batch_size_db, len(seg_primes)))
                    db.results.save_batch(
                        experiment=experiment,
                        primes=seg_primes[sl],
                        need_values=seg_need[sl],
                        cn_values=cn_global[sl],
                        li_values=None,
                        r_values=None,
                        batch_size=batch_size_db,
                    )
                    n_batch = sl.stop - sl.start
                    total_written += n_batch
                    prev = total_written - n_batch
                    if (total_written // log_every) > (prev // log_every):
                        logger.info(f"  Written to DB: {total_written:,} rows")
            need_elapsed = time.time() - start_time
            logger.info(f"Need/CN (--mp) merge and DB write in {need_elapsed:.2f} s")
            db.experiments.update_status(experiment, "data_ready")
            logger.info("-" * 40)
            logger.info("Stage 1 (--mp) completed.")
            return

        # Single-threaded mode
        logger.info("Loading prime numbers...")
        primes = np.load(primes_file)
        logger.info(f"Loaded {len(primes):,} primes")

        # Create or get experiment
        experiment = db.experiments.get_or_create(
            max_n,
            c_value,
            description=f"Analysis with max_n={max_n}, c={c_value}",
        )
        db.experiments.update_status(experiment, "computing")

        # Stage 1: Need(p) and CN(p) with streaming DB write
        logger.info("-" * 40)
        logger.info("Step 1: Computing Need(p) and CN(p) (batches, immediate DB write)")
        start_time = time.time()

        n_primes = len(primes)

        # For small experiments, store full arrays for plots/regression.
        # For very large (tens of millions), work strictly streaming.
        full_arrays_threshold = 5_000_000
        keep_full_arrays = n_primes <= full_arrays_threshold

        if keep_full_arrays:
            logger.info(
                f"Will store full Need/CN arrays for plots and regression "
                f"(primes {n_primes:,} ≤ threshold {full_arrays_threshold:,})."
            )
            all_primes = []
            all_need = []
            all_cn = []
        else:
            logger.warning(
                f"Number of primes {n_primes:,} exceeds threshold {full_arrays_threshold:,}. "
                f"Switching to streaming mode without storing all Need/CN values in memory. "
                f"Need/CN plots and CN~Li regression on this stage will be skipped."
            )

        running_cn = 0.0
        processed = 0

        for batch_primes, batch_need in iter_need_batches(
            primes,
            c_value,
            max_n,
            batch_size=300_000,
            verbose=True,
        ):
            if len(batch_primes) == 0:
                continue

            # Incremental CN(p)
            cn_batch = np.cumsum(batch_need) + running_cn
            running_cn = float(cn_batch[-1])

            # Immediate batch write to DB
            db.results.save_batch(
                experiment=experiment,
                primes=batch_primes,
                need_values=batch_need,
                cn_values=cn_batch,
                li_values=None,
                r_values=None,
                batch_size=300000,
            )

            processed += len(batch_primes)

            if keep_full_arrays:
                all_primes.append(batch_primes)
                all_need.append(batch_need)
                all_cn.append(cn_batch)

            if processed % 1_000_000 <= len(batch_primes):
                logger.info(
                    f"  Processed {processed:,} primes "
                    f"({processed / n_primes * 100:.1f}%); "
                    f"current CN(p) ≈ {running_cn:.4e}"
                )

        need_elapsed = time.time() - start_time
        logger.info(f"Need/CN (with DB write) computed in {need_elapsed:.2f} seconds")

        # If we didn't store full arrays - finish here without plots/regression
        if not keep_full_arrays:
            logger.info(
                "Streaming mode: Need/CN plots, CSV/JSON and CN~Li regression "
                "will be done later (if needed) on separate, reduced samples."
            )
            db.experiments.update_status(experiment, "data_ready")
            total_elapsed = time.time() - start_time
            logger.info("-" * 40)
            logger.info(f"Stage 1 completed in {total_elapsed:.2f} seconds")
            return

        # Restore full arrays for analysis/visualization
        need_data = {
            "prime_N": np.concatenate(all_primes) if all_primes else np.array([], dtype=np.int64),
            "need": np.concatenate(all_need) if all_need else np.array([], dtype=np.float64),
            "CN": np.concatenate(all_cn) if all_cn else np.array([], dtype=np.float64),
        }

        # Save Need/CN plots and data
        logger.info("-" * 40)
        logger.info("Step 1.1: Saving Need/CN plots and data")
        save_plots_start = time.time()

        # Create results directory
        results_base_dir = config.results_dir
        results_dir = ensure_results_dir(results_base_dir)
        logger.info(f"Results directory: {results_dir}")

        # Build and save plots
        plot_path = results_dir / "need_cn_plots.png"
        plot_need_cn_data(
            need_data["prime_N"],
            need_data["need"],
            need_data["CN"],
            plot_path,
            max_n,
            c_value
        )

        # Save data to CSV and JSON
        csv_path, json_path = save_need_cn_data(
            need_data["prime_N"],
            need_data["need"],
            need_data["CN"],
            results_dir,
            max_n,
            c_value
        )

        save_plots_elapsed = time.time() - save_plots_start
        logger.info(f"Plots and data saved in {save_plots_elapsed:.2f} seconds")
        logger.info(f"  - Plots: {plot_path}")
        logger.info(f"  - CSV: {csv_path}")
        logger.info(f"  - JSON: {json_path}")

        # Stages 2-4 (Li, CN~Li regression, R(p)) moved to Stage 2
        # of spectral analysis. Here we limit to Need/CN computation
        # and their saving to DB and files.
        save_elapsed = time.time() - save_plots_start
        logger.info(f"Additional analysis (plots/statistics Need/CN) took {save_elapsed:.2f} seconds")

        # Update status
        db.experiments.update_status(experiment, "data_ready")

        total_elapsed = time.time() - start_time
        logger.info("-" * 40)
        logger.info(f"Stage 1 completed in {total_elapsed:.2f} seconds")

    logger.info("=" * 60)
    logger.info("STAGE 1: Need/CN/Li/R - COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
