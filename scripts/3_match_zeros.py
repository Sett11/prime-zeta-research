#!/usr/bin/env python3
"""
Stage 3: Matching with zeta function zeros.

Matches spectrum peaks with theoretical zeta function zeros.
Results are saved to JSON and PNG files.

Usage:
    python scripts/3_match_zeros.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from loguru import logger
from src.config_loader import get_config
from src.database.operations import DatabaseManager


def setup_logging():
    """Configure logging."""
    config = get_config()
    log_level = config.log_level
    log_file = config.get("logging", "file", "research.log")

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(log_file, level=log_level, rotation="10 MB", retention=5)


def load_gamma_k(filepath: str) -> np.ndarray:
    """Load zeta function zeros from file."""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.txt'):
        return np.loadtxt(filepath)
    else:
        # Try to determine format
        try:
            return np.loadtxt(filepath)
        except:
            logger.error(f"Could not load gamma_k from {filepath}")
            return np.array([])


def gamma_to_frequency(gamma: np.ndarray) -> np.ndarray:
    """Convert gamma_k to frequencies: f_k = gamma_k / (2*pi)."""
    return gamma / (2 * np.pi)


def match_peaks_to_zeros(peak_freqs: np.ndarray, peak_amps: np.ndarray,
                         gamma_k: np.ndarray) -> list:
    """
    Vectorized matching of spectrum peaks with zeta function zeros.

    Uses np.searchsorted for O(n log m) complexity instead of O(n*m).

    Returns:
        List of matching results sorted by error
    """
    if len(peak_freqs) == 0:
        logger.warning("No peaks to match!")
        return []

    f_theory = gamma_to_frequency(gamma_k)

    # Sort peaks by frequency for fast search
    sorted_indices = np.argsort(peak_freqs)
    sorted_peak_freqs = peak_freqs[sorted_indices]
    sorted_peak_amps = peak_amps[sorted_indices]

    # Vectorized search for nearest peaks for all theoretical frequencies
    # searchsorted finds insertion position, then check neighboring elements
    insert_positions = np.searchsorted(sorted_peak_freqs, f_theory)

    # Handle boundary cases
    insert_positions = np.clip(insert_positions, 0, len(sorted_peak_freqs) - 1)

    # Find nearest peaks (check element at position and previous if exists)
    idx1 = insert_positions
    idx2 = np.maximum(0, insert_positions - 1)

    # For elements at beginning (insert_positions == 0), use only idx1
    # For others, check both candidates
    use_both = insert_positions > 0

    # Compute distances to both candidates
    dist1 = np.abs(sorted_peak_freqs[idx1] - f_theory)
    dist2 = np.where(use_both, np.abs(sorted_peak_freqs[idx2] - f_theory), np.inf)

    # Choose nearest
    use_idx1 = dist1 <= dist2
    closest_indices = np.where(use_idx1, idx1, idx2)

    # Get frequencies and amplitudes of nearest peaks
    closest_freqs = sorted_peak_freqs[closest_indices]
    closest_amps = sorted_peak_amps[closest_indices]

    # Compute relative errors vectorized
    distances = np.abs(closest_freqs - f_theory)
    relative_errors = np.where(f_theory > 0, distances / f_theory, np.inf)

    # Create results array
    matches = []
    for i in range(len(gamma_k)):
        matches.append({
            "gamma_index": i + 1,
            "gamma_value": float(gamma_k[i]),
            "f_theory": float(f_theory[i]),
            "f_peak": float(closest_freqs[i]),
            "relative_error": float(relative_errors[i]),
            "amplitude": float(closest_amps[i])
        })

    # Sort by error
    matches.sort(key=lambda x: x["relative_error"])

    return matches


def compute_statistics(matches: list) -> dict:
    """Compute match statistics."""
    if not matches:
        return {"total": 0}

    errors = [m["relative_error"] for m in matches]
    good_matches = [m for m in matches if m["relative_error"] < 0.01]

    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    threshold_counts = {}
    for t in thresholds:
        threshold_counts[f"< {t*100:.1f}%"] = sum(1 for e in errors if e < t)

    return {
        "total": len(matches),
        "good_matches_1percent": len(good_matches),
        "good_percentage": 100 * len(good_matches) / len(matches),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "min_error": float(min(errors)),
        "max_error": float(max(errors)),
        "thresholds": threshold_counts,
        "percentiles": {
            "1%": float(np.percentile(errors, 1)),
            "5%": float(np.percentile(errors, 5)),
            "10%": float(np.percentile(errors, 10)),
            "50%": float(np.percentile(errors, 50)),
            "90%": float(np.percentile(errors, 90))
        }
    }


def save_matches_to_txt(matches: list, save_path: Path, stats: dict):
    """Save all matches to text file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("MATCHING SPECTRUM PEAKS WITH ZETA FUNCTION ZEROS\n")
        f.write("=" * 80 + "\n\n")

        # Statistics
        f.write("STATISTICS:\n")
        f.write(f"  Total zeros: {stats['total']}\n")
        f.write(f"  Good matches (<1%): {stats['good_matches_1percent']} ({stats['good_percentage']:.1f}%)\n")
        f.write(f"  Mean error: {stats['mean_error']*100:.4f}%\n")
        f.write(f"  Median error: {stats['median_error']*100:.4f}%\n")
        f.write(f"  Min error: {stats['min_error']*100:.4f}%\n")
        f.write(f"  Max error: {stats['max_error']*100:.4f}%\n\n")

        # Thresholds
        f.write("ERROR THRESHOLD DISTRIBUTION:\n")
        for threshold, count in stats['thresholds'].items():
            f.write(f"  {threshold}: {count}\n")
        f.write("\n")

        # Percentiles
        f.write("ERROR PERCENTILES:\n")
        for percentile, value in stats['percentiles'].items():
            f.write(f"  {percentile}: {value*100:.4f}%\n")
        f.write("\n")

        # Divider
        f.write("=" * 80 + "\n")
        f.write("ALL MATCHES (sorted by error, best to worst):\n")
        f.write("=" * 80 + "\n\n")

        # Table header
        f.write(f"{'#':<6} {'gamma_k':<18} {'f_theory':<18} {'f_peak':<18} "
                f"{'Error (%)':<12} {'Amplitude':<12}\n")
        f.write("-" * 80 + "\n")

        # All matches
        for i, match in enumerate(matches, 1):
            f.write(f"{match['gamma_index']:<6} "
                   f"{match['gamma_value']:<18.10f} "
                   f"{match['f_theory']:<18.10f} "
                   f"{match['f_peak']:<18.10f} "
                   f"{match['relative_error']*100:<12.6f} "
                   f"{match['amplitude']:<12.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total records: {len(matches)}\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Text file saved: {save_path}")


def plot_matches(matches: list, gamma_k: np.ndarray, peak_freqs: np.ndarray,
                 save_path: Path, title: str):
    """Save matches plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error histogram
    ax1 = axes[0, 0]
    errors = [m["relative_error"] * 100 for m in matches]  # in percent
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Relative error (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    ax1.set_xlim(0, 10)
    ax1.grid(True, alpha=0.3)

    # Best matches
    ax2 = axes[0, 1]
    best_matches = matches[:100]
    gamma_vals = [m["gamma_value"] for m in best_matches]
    errors_vals = [m["relative_error"] * 100 for m in best_matches]
    ax2.scatter(gamma_vals, errors_vals, c='blue', s=10, alpha=0.5)
    ax2.set_xlabel('gamma_k')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Best 100 matches')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Frequency comparison (best 50)
    ax3 = axes[1, 0]
    best_50 = matches[:50]
    f_theory = [m["f_theory"] for m in best_50]
    f_peak = [m["f_peak"] for m in best_50]
    ax3.scatter(f_theory, f_peak, c='green', s=20, alpha=0.7)
    ax3.plot([min(f_theory), max(f_theory)],
             [min(f_theory), max(f_theory)], 'r--', label='y=x')
    ax3.set_xlabel('Theoretical frequency')
    ax3.set_ylabel('Found frequency')
    ax3.set_title('Frequency comparison (top 50)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Cumulative function
    ax4 = axes[1, 1]
    sorted_errors = sorted(errors)
    cumsum = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax4.plot(sorted_errors, cumsum, 'b-', linewidth=2)
    ax4.axhline(y=80, color='r', linestyle='--', label='80%')
    ax4.axhline(y=90, color='orange', linestyle='--', label='90%')
    ax4.set_xlabel('Relative error (%)')
    ax4.set_ylabel('Match percentage')
    ax4.set_title('Cumulative match quality function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 10)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved: {save_path}")


def main():
    """Main matching function."""
    setup_logging()

    # Load configuration
    config = get_config()
    max_n = config.max_n
    c_value = config.c_value
    db_path = config.db_path
    gamma_k_file = config.gamma_k_file
    results_base_dir = config.results_dir

    logger.info("=" * 60)
    logger.info("STAGE 3: Matching with Zeta Function Zeros")
    logger.info("=" * 60)
    logger.info(f"max_n: {max_n:,}")
    logger.info(f"c_value: {c_value}")
    logger.info(f"gamma_k file: {gamma_k_file}")

    # Create results directory (use existing)
    results_dir = Path(results_base_dir)
    if not results_dir.exists():
        results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load gamma_k
    logger.info("-" * 60)
    logger.info("STEP 1: Load zeta function zeros")
    logger.info("-" * 60)
    gamma_start = time.time()
    gamma_k = load_gamma_k(gamma_k_file)
    gamma_load_elapsed = time.time() - gamma_start

    if len(gamma_k) == 0:
        logger.error(f"Could not load gamma_k from {gamma_k_file}")
        return

    logger.info(f"Loaded {len(gamma_k):,} zeta function zeros in {gamma_load_elapsed:.2f} seconds")
    logger.info(f"gamma_k range: [{gamma_k[0]:.4f}, {gamma_k[-1]:.4f}]")

    # Find spectrum file
    logger.info("-" * 60)
    logger.info("STEP 2: Load spectrum")
    logger.info("-" * 60)
    spectrum_start = time.time()
    spectrum_dir = Path(results_base_dir)
    spectrum_files = list(spectrum_dir.glob("**/spectrum_data.npz"))

    if not spectrum_files:
        logger.error("File spectrum_data.npz not found!")
        logger.info("First run stage 2: python scripts/2_spectral_analysis.py")
        return

    latest_spectrum = max(spectrum_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using spectrum: {latest_spectrum}")

    # Load spectrum
    data = np.load(latest_spectrum)
    peak_freqs = data["peak_frequencies"]
    peak_amps = data["peak_amplitudes"]

    spectrum_load_elapsed = time.time() - spectrum_start
    logger.info(f"Loaded {len(peak_freqs):,} peaks in {spectrum_load_elapsed:.2f} seconds")
    logger.info(f"Peak frequency range: [{peak_freqs.min():.6f}, {peak_freqs.max():.6f}]")
    logger.info(f"Peak amplitude range: [{peak_amps.min():.6f}, {peak_amps.max():.6f}]")

    # Matching
    logger.info("-" * 60)
    logger.info("STEP 3: Match peaks with zeta function zeros")
    logger.info("-" * 60)
    logger.info(f"Matching {len(gamma_k)} zeros with {len(peak_freqs)} peaks...")
    start_time = time.time()

    matches = match_peaks_to_zeros(peak_freqs, peak_amps, gamma_k)

    elapsed = time.time() - start_time
    logger.info(f"Matching completed in {elapsed:.2f} seconds")
    logger.info(f"Processed {len(matches)} matches")

    # Statistics
    logger.info("-" * 60)
    logger.info("STEP 4: Compute statistics")
    logger.info("-" * 60)
    stats_start = time.time()
    stats = compute_statistics(matches)
    stats_elapsed = time.time() - stats_start

    logger.info(f"Statistics computed in {stats_elapsed:.2f} seconds")
    logger.info(f"Total matched: {stats['total']}")
    logger.info(f"Good matches (<1%): {stats['good_matches_1percent']} ({stats['good_percentage']:.1f}%)")
    logger.info(f"Mean error: {stats['mean_error']*100:.4f}%")
    logger.info(f"Median error: {stats['median_error']*100:.4f}%")
    logger.info(f"Min error: {stats['min_error']*100:.4f}%")
    logger.info(f"Max error: {stats['max_error']*100:.4f}%")

    # Results
    results = {
        "experiment": {
            "max_n": int(max_n),
            "c_value": float(c_value),
            "timestamp": datetime.now().isoformat(),
            "gamma_k_file": gamma_k_file,
            "spectrum_file": str(latest_spectrum)
        },
        "matching": {
            "total_zeros": len(gamma_k),
            "total_peaks": len(peak_freqs),
            "time_seconds": float(elapsed)
        },
        "statistics": stats,
        "top_matches": matches[:100],  # Best 100
        "all_matches": matches  # All matches
    }

    # Save results
    logger.info("-" * 60)
    logger.info("STEP 5: Save results")
    logger.info("-" * 60)
    save_start = time.time()

    # Save JSON
    json_path = results_dir / "zeta_matching_results.json"
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"JSON saved: {json_path}")

    # Save TXT (all matches)
    txt_path = results_dir / "zeta_matching_results.txt"
    save_matches_to_txt(matches, txt_path, stats)

    # Save PNG
    if "png" in config.output_format:
        png_path = results_dir / "zeta_matching_plot.png"
        plot_matches(matches, gamma_k, peak_freqs, png_path,
                    f"Zeta Zero Matching (max_n={max_n:,}, c={c_value})")

    save_elapsed = time.time() - save_start
    logger.info(f"Saving completed in {save_elapsed:.2f} seconds")

    logger.info("=" * 60)
    logger.info("STAGE 3: Matching - COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
