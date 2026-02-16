#!/usr/bin/env python3
"""
Stage 6: Compare FFT(R) with theoretical zeta function zero frequencies.

This script:
    1. Loads R_resampled from R_dynamics.npz
    2. Computes FFT from R_resampled
    3. Loads gamma_k from zeros file
    4. Converts gamma_k -> frequencies f_k = gamma_k/(2π)
    5. Matches FFT peaks with theoretical frequencies
    6. Builds detailed comparison plots
    7. Saves all results

Usage:
    python scripts/6_compare_fft_with_zeros.py

Results saved to: data/result/fft_vs_gamma_[timestamp]/

Author: Prime Zeta Research Team
Date: 2026-02-11
"""

import sys
import os
import json
import argparse
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
from scipy import stats

import mpmath as mp
from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'n_processes': None,              # Auto: min(cpu_count(), 8)
    'sample_size_zeta': 100000,      # Points for ζ(s) computation (optional)
    'mp_dps': 25,                   # mpmath precision
    'batch_report': 1000,           # Progress bar every N points
    'random_seed': 42,
    'peak_percentile': 90,         # Percentile for peak finding
    'peak_distance': 10,            # Min distance between peaks
    'match_tolerance': 0.01,       # Matching tolerance (1%)
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: str = "fft_comparison.log") -> logger:
    """
    Configure logging.
    """
    logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}:{line}</cyan> | "
        "<level>{message}</level>"
    )

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{message}"
    )

    logger.add(sys.stderr, level="INFO", format=console_format, colorize=True)
    logger.add(log_file, level="DEBUG", format=file_format,
               rotation="100 MB", retention="3 days", encoding="utf-8")

    return logger


def log_section(title: str, width: int = 80) -> None:
    """Output section header."""
    border = "=" * width
    padding = " " * ((width - len(title) - 2) // 2)
    logger.info(border)
    logger.info(f"{padding}{title}")
    logger.info(border)


def log_table(headers: list, rows: list) -> None:
    """Output table to log."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    row_format = " | ".join(f"{{:<{w}}}" for w in col_widths)

    logger.info(row_format.format(*headers))
    logger.info("-" * (sum(col_widths) + 3 * (len(headers) - 1)))

    for row in rows:
        formatted = []
        for i, val in enumerate(row):
            if isinstance(val, float):
                formatted.append(f"{val:.6f}")
            elif isinstance(val, int):
                formatted.append(f"{val:,}")
            else:
                formatted.append(str(val))
        logger.info(row_format.format(*formatted))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_r_dynamics(data_dir: Path, logger) -> tuple:
    """
    Load R_dynamics.npz.

    Returns:
        (R_resampled, ln_p_grid, delta_ln_p)
    """
    log_section("LOADING R DYNAMICS")

    # Search for R_dynamics.npz
    search_paths = [
        data_dir / "R_dynamics.npz",
        data_dir / "dataset_sources" / "R_dynamics.npz",
        Path(data_dir).parent / "dataset_sources" / "R_dynamics.npz",
    ]

    r_path = None
    for path in search_paths:
        logger.debug(f"Checking: {path}")
        if path.exists():
            r_path = path
            logger.success(f"Found: {r_path}")
            break

    if r_path is None:
        raise FileNotFoundError(f"R_dynamics.npz not found! Searched: {search_paths}")

    logger.info(f"Loading: {r_path}")
    logger.info(f"Size: {r_path.stat().st_size / (1024**3):.2f} GB")

    data = np.load(str(r_path), mmap_mode='r')

    R_resampled = data['R_resampled']
    ln_p_grid = data['ln_p_grid']

    if len(ln_p_grid) < 2:
        raise ValueError(
            f"ln_p_grid has only {len(ln_p_grid)} element(s). "
            "Need at least 2 points to compute delta_ln_p. "
            "Re-run stage 2 (scripts/2_spectral_analysis.py) to generate valid data."
        )
    delta_ln_p = (ln_p_grid[-1] - ln_p_grid[0]) / (len(ln_p_grid) - 1)

    logger.success(f"Loaded: {len(R_resampled):,} R points")
    logger.info(f"  ln(p) range: [{ln_p_grid.min():.6f}, {ln_p_grid.max():.6f}]")
    logger.info(f"  delta_ln_p: {delta_ln_p:.10f}")
    logger.info(f"  R range: [{R_resampled.min():.6f}, {R_resampled.max():.6f}]")

    # R statistics
    logger.info("R statistics:")
    log_table(
        ['Metric', 'Value'],
        [
            ['Mean', R_resampled.mean()],
            ['Std', R_resampled.std()],
            ['Min', R_resampled.min()],
            ['Max', R_resampled.max()],
            ['Median', np.median(R_resampled)],
        ]
    )

    return R_resampled, ln_p_grid, delta_ln_p


def load_gamma_k(data_dir: Path, logger) -> np.ndarray:
    """
    Load zeta function zeros from file.

    Returns:
        gamma_k: array of imaginary parts of zeros
    """
    log_section("LOADING ZETA FUNCTION ZEROS")

    # Search for zeros file
    search_paths = [
        data_dir.parent / "data" / "gamma_k_5000000.txt",
        data_dir.parent.parent / "data" / "gamma_k_5000000.txt",
        Path("data") / "gamma_k_5000000.txt",
        Path(__file__).parent.parent.parent / "data" / "gamma_k_5000000.txt",
    ]

    gamma_path = None
    for path in search_paths:
        logger.debug(f"Checking: {path}")
        if path.exists():
            gamma_path = path
            logger.success(f"Found: {gamma_path}")
            break

    if gamma_path is None:
        raise FileNotFoundError("gamma_k file not found!")

    logger.info(f"Loading: {gamma_path}")

    gamma_k = np.loadtxt(str(gamma_path))

    logger.success(f"Loaded: {len(gamma_k):,} zeros")
    logger.info(f"  First: γ₁ = {gamma_k[0]:.6f}")
    logger.info(f"  Last: γ₂₀₀₀₀₀₀ = {gamma_k[-1]:.6f}")
    logger.info(f"  Range: [{gamma_k.min():.2f}, {gamma_k.max():.2f}]")

    # Interval statistics
    intervals = np.diff(gamma_k)
    logger.info("Zero interval statistics:")
    log_table(
        ['Metric', 'Value'],
        [
            ['Mean interval', intervals.mean()],
            ['Min interval', intervals.min()],
            ['Max interval', intervals.max()],
            ['Std', intervals.std()],
        ]
    )

    return gamma_k


# ============================================================================
# FFT ANALYSIS
# ============================================================================

def compute_fft(R: np.ndarray, delta_ln_p: float, logger) -> tuple:
    """
    Compute FFT from R and find peaks.

    Returns:
        (frequencies, amplitudes, peak_freqs, peak_amps)
    """
    log_section("COMPUTING FFT")

    n = len(R)
    logger.info(f"FFT size: {n:,} points")

    # FFT
    start_time = time.time()
    fft_result = fft(R)
    elapsed = time.time() - start_time
    logger.info(f"FFT computed in {elapsed:.2f} sec")

    # Only positive frequencies
    n_freq = n // 2 + 1
    frequencies = fftfreq(n, delta_ln_p)[:n_freq]
    amplitudes = 2.0 / n * np.abs(fft_result[:n_freq])
    amplitudes[0] = amplitudes[0] / 2  # DC component

    # DC component separately
    logger.info(f"Frequency range: [{frequencies[0]:.6f}, {frequencies[-1]:.2f}]")
    logger.info(f"Frequency resolution: {frequencies[1] - frequencies[0]:.10f}")
    logger.info(f"DC amplitude: {amplitudes[0]:.6f}")
    logger.info(f"Max amplitude: {amplitudes.max():.6f}")
    logger.info(f"Mean amplitude (no DC): {amplitudes[1:].mean():.6f}")

    # Peak finding
    log_section("PEAK FINDING IN SPECTRUM")

    # Find ALL local maxima
    peak_indices, _ = find_peaks(amplitudes, height=0, distance=1)

    peak_freqs = frequencies[peak_indices]
    peak_amps = amplitudes[peak_indices]

    logger.success(f"Peaks found: {len(peak_freqs):,}")
    logger.info(f"  Frequency range: [{peak_freqs.min():.2f}, {peak_freqs.max():.2f}] Hz")
    logger.info(f"  Amplitude range: [{peak_amps.min():.6f}, {peak_amps.max():.6f}]")

    # Also find "significant" peaks - significantly above average level
    mean_amp = amplitudes[amplitudes > 0].mean()
    significant_threshold = mean_amp * 1000  # Peaks 1000x above mean

    significant_indices, _ = find_peaks(
        amplitudes,
        height=significant_threshold,
        distance=1
    )
    significant_peak_freqs = frequencies[significant_indices]
    significant_peak_amps = amplitudes[significant_indices]

    logger.info(f"  Significant peaks (>1000×mean): {len(significant_peak_freqs):,}")

    return frequencies, amplitudes, peak_freqs, peak_amps, significant_peak_freqs, significant_peak_amps


def gamma_to_frequency(gamma_k: np.ndarray) -> np.ndarray:
    """
    Convert gamma_k -> frequencies f_k = gamma_k / (2π).
    """
    return gamma_k / (2 * np.pi)


# ============================================================================
# PEAK MATCHING WITH ZEROS
# ============================================================================

def match_peaks_to_zeros(peak_freqs: np.ndarray, peak_amps: np.ndarray,
                        gamma_k: np.ndarray, logger) -> dict:
    """
    Match gamma_k with nearest FFT peaks.
    """
    log_section("MATCHING GAMMA_K WITH FFT PEAKS")

    # Theoretical frequencies
    f_theory = gamma_to_frequency(gamma_k)

    logger.info(f"FFT peaks for matching: {len(peak_freqs):,}")
    logger.info(f"Theoretical gamma_k frequencies: {len(f_theory):,}")

    # Sort peaks by frequency
    sorted_idx = np.argsort(peak_freqs)
    sorted_freqs = peak_freqs[sorted_idx]
    sorted_amps = peak_amps[sorted_idx]

    # Vectorized matching - find nearest peak for each gamma_k
    logger.info("Matching gamma_k with nearest peaks...")
    start_time = time.time()

    # For each theoretical frequency, find nearest peak
    insert_positions = np.searchsorted(sorted_freqs, f_theory)
    insert_positions = np.clip(insert_positions, 0, len(sorted_freqs) - 1)

    # Check neighbors
    idx1 = insert_positions
    idx2 = np.maximum(0, insert_positions - 1)

    use_idx1 = insert_positions > 0
    dist1 = np.abs(sorted_freqs[idx1] - f_theory)
    dist2 = np.where(use_idx1, np.abs(sorted_freqs[idx2] - f_theory), np.inf)

    closest_idx = np.where(dist1 <= dist2, idx1, idx2)
    closest_freqs = sorted_freqs[closest_idx]
    closest_amps = sorted_amps[closest_idx]

    # Errors
    abs_errors = np.abs(closest_freqs - f_theory)
    rel_errors = np.where(f_theory > 0, abs_errors / f_theory, 0)

    elapsed = time.time() - start_time
    logger.info(f"Matching completed in {elapsed:.2f} sec")

    # Error statistics
    log_section("ERROR STATISTICS")

    log_table(
        ['Metric', 'Value'],
        [
            ['Mean abs error', abs_errors.mean()],
            ['Median abs error', np.median(abs_errors)],
            ['Max abs error', abs_errors.max()],
            ['Mean rel error (%)', rel_errors.mean() * 100],
            ['Median rel error (%)', np.median(rel_errors) * 100],
        ]
    )

    # Percentiles
    logger.info("Relative error percentiles:")
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    rows = []
    for p in percentiles:
        val = np.percentile(rel_errors, p) * 100
        rows.append([f"{p}%", f"{val:.6f}%"])
    log_table(['Percentile', 'Error'], rows)

    # Good matches
    good_matches = rel_errors < 0.01  # 1%
    good_count = good_matches.sum()
    good_ratio = good_count / len(rel_errors) * 100

    logger.success(f"Good matches (<1%): {good_count:,} ({good_ratio:.4f}%)")

    # Top best and worst
    best_idx = np.argsort(rel_errors)[:10]
    worst_idx = np.argsort(rel_errors)[-10:]

    logger.info("Top-10 best matches:")
    rows = []
    for i in best_idx:
        rows.append([i+1, f"{gamma_k[i]:.2f}", f"{f_theory[i]:.6f}",
                     f"{closest_freqs[i]:.6f}", f"{rel_errors[i]*100:.8f}%"])
    log_table(['#', 'γ_k', 'f_theory', 'f_peak', 'Error'], rows)

    logger.info("Top-10 worst matches:")
    rows = []
    for i in worst_idx:
        rows.append([i+1, f"{gamma_k[i]:.2f}", f"{f_theory[i]:.6f}",
                     f"{closest_freqs[i]:.6f}", f"{rel_errors[i]*100:.4f}%"])
    log_table(['#', 'γ_k', 'f_theory', 'f_peak', 'Error'], rows)

    return {
        'gamma_k': gamma_k,
        'f_theory': f_theory,
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps,
        'matched_freqs': closest_freqs,
        'matched_amps': closest_amps,
        'abs_errors': abs_errors,
        'rel_errors': rel_errors,
        'good_matches': good_matches,
        'good_count': int(good_count),
        'good_ratio': float(good_ratio),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(frequencies: np.ndarray, amplitudes: np.ndarray,
                         peak_freqs: np.ndarray, peak_amps: np.ndarray,
                         significant_peak_freqs: np.ndarray, significant_peak_amps: np.ndarray,
                         match_results: dict,
                         output_dir: Path, logger) -> None:
    """
    Create detailed visualizations of FFT and gamma_k comparison.
    """
    log_section("CREATING VISUALIZATIONS")

    logger.info(f"Directory: {output_dir}")
    logger.info(f"Total peaks: {len(peak_freqs):,}")
    logger.info(f"Significant peaks (>1000×mean): {len(significant_peak_freqs):,}")

    gamma_k = match_results['gamma_k']
    f_theory = match_results['f_theory']
    rel_errors = match_results['rel_errors']

    # Main plot: Spectrum with theoretical frequencies
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 16))

    # 1.1 Full spectrum
    ax = axes1[0, 0]
    ax.semilogy(frequencies[1:], amplitudes[1:], 'b-', linewidth=0.3, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Amplitude (log)', fontsize=10)
    ax.set_title('Full FFT(R) Spectrum\n(log scale)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, frequencies[-1]])

    # 1.2 Spectrum with ALL peaks and significant
    ax = axes1[0, 1]
    ax.semilogy(frequencies[1:], amplitudes[1:], 'b-', linewidth=0.3, alpha=0.5, label='Spectrum')
    # All peaks - small red dots
    ax.scatter(peak_freqs, peak_amps, c='red', s=1, alpha=0.3, label=f'All peaks ({len(peak_freqs):,})')
    # Significant peaks - large yellow markers
    ax.scatter(significant_peak_freqs, significant_peak_amps, c='gold', s=10, marker='*',
              alpha=0.8, label=f'Significant ({len(significant_peak_freqs):,})')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Amplitude (log)', fontsize=10)
    ax.set_title(f'FFT(R) Spectrum with Peaks\n(red: {len(peak_freqs):,} all, gold: {len(significant_peak_freqs):,} significant)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, frequencies[-1]])

    # 1.3 Error distribution
    ax = axes1[1, 0]
    valid_errors = rel_errors[~np.isinf(rel_errors)]
    errors_percent = valid_errors * 100
    errors_percent_clipped = np.clip(errors_percent, 0, 10)

    ax.hist(errors_percent_clipped, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='1% threshold')
    ax.set_xlabel('Relative error (%)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Matching Error Distribution\n(n={len(errors_percent):,}, median={np.median(errors_percent):.4f}%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.4 Good match percentage by frequency
    ax = axes1[1, 1]
    bins = np.linspace(0, f_theory.max(), 100)
    bin_indices = np.digitize(f_theory, bins)
    good_by_bin = np.array([np.mean(rel_errors[bin_indices == i] < 0.01)
                           for i in range(1, len(bins))])
    good_by_bin = np.nan_to_num(good_by_bin, nan=0)

    ax.fill_between(bins[:-1], 0, good_by_bin * 100, alpha=0.5, color='green')
    ax.plot(bins[:-1], good_by_bin * 100, 'g-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Good matches (%)', fontsize=10)
    ax.set_title('Match Quality by Frequency Range', fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = output_dir / "fft_spectrum_overview.png"
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved: {path1}")

    # Plot 2: Low frequency detail
    fig2, axes2 = plt.subplots(2, 1, figsize=(20, 12))

    # 2.1 Low frequencies (zoom)
    ax = axes2[0]
    f_max_zoom = 1000  # Hz

    mask = f_theory <= f_max_zoom
    f_theory_zoom = f_theory[mask]
    rel_errors_zoom = rel_errors[mask]

    ax.scatter(f_theory_zoom, rel_errors_zoom * 100, s=1, alpha=0.5, c='blue')
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='1% threshold')
    ax.set_xlabel('Theoretical frequency f = γ/(2π) (Hz)', fontsize=10)
    ax.set_ylabel('Matching error (%)', fontsize=10)
    ax.set_title(f'Matching Errors at Low Frequencies (0-{f_max_zoom} Hz)\n(n={len(f_theory_zoom):,})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.2 Error vs frequency (all data)
    ax = axes2[1]
    ax.scatter(f_theory, rel_errors * 100, s=0.5, alpha=0.3, c='purple')
    ax.axhline(1, color='red', linestyle='--', linewidth=1, label='1% threshold')
    ax.set_xlabel('Theoretical frequency f = γ/(2π) (Hz)', fontsize=10)
    ax.set_ylabel('Error (%)', fontsize=10)
    ax.set_title('Matching Errors by All Frequencies', fontsize=12)
    ax.set_ylim([0, 10])  # Clip for clarity
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = output_dir / "fft_error_analysis.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved: {path2}")

    # Plot 3: Amplitude analysis
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 16))

    # 3.1 Peak amplitudes vs frequency
    ax = axes3[0, 0]
    ax.scatter(peak_freqs, peak_amps, s=1, alpha=0.3, c='blue')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('FFT(R) Peak Amplitudes', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3.2 Peak amplitude vs matching error (correlation)
    ax = axes3[0, 1]
    matched_amps = match_results['matched_amps']
    ax.scatter(matched_amps, rel_errors * 100, s=1, alpha=0.3, c='green')
    ax.set_xlabel('Nearest peak amplitude', fontsize=10)
    ax.set_ylabel('Relative error (%)', fontsize=10)
    ax.set_title('Peak Amplitude vs Match Accuracy', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3.3 Amplitude histogram
    ax = axes3[1, 0]
    ax.hist(peak_amps, bins=100, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Amplitude', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Peak Amplitude Distribution', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3.4 Theoretical vs found frequency
    ax = axes3[1, 1]
    matched_freqs = match_results['matched_freqs']
    ax.scatter(f_theory, matched_freqs, s=1, alpha=0.3, c='steelblue')
    ax.plot([f_theory.min(), f_theory.max()], [f_theory.min(), f_theory.max()],
            'r--', linewidth=2, label='y=x (perfect match)')
    ax.set_xlabel('Theoretical frequency f = γ/(2π) (Hz)', fontsize=10)
    ax.set_ylabel('Found frequency (Hz)', fontsize=10)
    ax.set_title('Theoretical vs Found Frequencies', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = output_dir / "fft_amplitude_analysis.png"
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved: {path3}")

    # Plot 4: Cumulative function
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))

    sorted_errors = np.sort(rel_errors) * 100
    cumsum = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

    ax4.plot(sorted_errors, cumsum, 'b-', linewidth=2)
    ax4.axhline(90, color='orange', linestyle='--', label='90%')
    ax4.axhline(95, color='green', linestyle='--', label='95%')
    ax4.axhline(99, color='red', linestyle='--', label='99%')
    ax4.axvline(1, color='purple', linestyle='--', alpha=0.7, label='1% threshold')

    # Find percentage at 1%
    pct_at_1 = np.searchsorted(sorted_errors, 1) / len(sorted_errors) * 100

    ax4.set_xlabel('Relative error (%)', fontsize=12)
    ax4.set_ylabel('Match percentage', fontsize=12)
    ax4.set_title(f'Cumulative Match Quality Function\n({pct_at_1:.2f}% at 1% error)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 5])
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    path4 = output_dir / "cumulative_quality.png"
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved: {path4}")

    logger.success("All visualizations created")


# ============================================================================
# SAVING RESULTS
# ============================================================================

def save_results(frequencies: np.ndarray, amplitudes: np.ndarray,
                 peak_freqs: np.ndarray, peak_amps: np.ndarray,
                 significant_peak_freqs: np.ndarray, significant_peak_amps: np.ndarray,
                 match_results: dict,
                 config: dict,
                 output_dir: Path, logger) -> dict:
    """
    Save all results to files.
    """
    log_section("SAVING RESULTS")

    gamma_k = match_results['gamma_k']
    f_theory = match_results['f_theory']
    rel_errors = match_results['rel_errors']
    abs_errors = match_results['abs_errors']

    # Main results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config': config,
        },
        'fft_info': {
            'n_points': int(len(frequencies)),
            'frequency_range': [float(frequencies.min()), float(frequencies.max())],
            'frequency_resolution': float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0,
            'n_all_peaks': int(len(peak_freqs)),
            'n_significant_peaks': int(len(significant_peak_freqs)),
            'significant_peak_threshold': '>1000×mean amplitude',
        },
        'gamma_k_info': {
            'n_zeros': int(len(gamma_k)),
            'gamma_range': [float(gamma_k.min()), float(gamma_k.max())],
            'frequency_range': [float(f_theory.min()), float(f_theory.max())],
        },
        'matching': {
            'total_matches': len(rel_errors),
            'good_matches_1percent': int(match_results['good_count']),
            'good_ratio_percent': float(match_results['good_ratio']),
        },
        'statistics': {
            'abs_errors': {
                'mean': float(abs_errors.mean()),
                'median': float(np.median(abs_errors)),
                'std': float(abs_errors.std()),
                'max': float(abs_errors.max()),
            },
            'rel_errors_percent': {
                'mean': float(rel_errors.mean() * 100),
                'median': float(np.median(rel_errors) * 100),
                'std': float(rel_errors.std() * 100),
                'max': float(rel_errors.max() * 100),
            },
        },
    }

    # Save JSON
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved: {json_path}")

    # Save subset for analysis
    n_export = min(100000, len(gamma_k))

    export_idx = np.linspace(0, len(gamma_k)-1, n_export, dtype=int)

    export_data = {
        'gamma_k': gamma_k[export_idx].tolist(),
        'f_theory': f_theory[export_idx].tolist(),
        'matched_freqs': match_results['matched_freqs'][export_idx].tolist(),
        'rel_errors_percent': (rel_errors[export_idx] * 100).tolist(),
    }

    export_path = output_dir / "matching_export.json"
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    logger.success(f"Saved: {export_path}")

    # Save CSV with full data (if not too large)
    csv_path = output_dir / "matching_full.csv"
    if len(gamma_k) <= 500000:
        header = "gamma_k,f_theory,matched_freq,abs_error,rel_error_percent\n"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(header)
            for i in range(len(gamma_k)):
                f.write(f"{gamma_k[i]},{f_theory[i]},{match_results['matched_freqs'][i]},"
                       f"{abs_errors[i]},{rel_errors[i]*100:.8f}\n")
        logger.success(f"Saved: {csv_path}")

    # Save numpy arrays
    npz_path = output_dir / "matching_arrays.npz"
    np.savez(npz_path,
              gamma_k=gamma_k,
              f_theory=f_theory,
              matched_freqs=match_results['matched_freqs'],
              abs_errors=abs_errors,
              rel_errors=rel_errors)
    logger.success(f"Saved: {npz_path}")

    # Text summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FFT(R) COMPARISON WITH THEORETICAL ZETA FUNCTION ZERO FREQUENCIES\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {results['metadata']['timestamp']}\n\n")

        f.write("PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"FFT points: {results['fft_info']['n_points']:,}\n")
        f.write(f"Total peaks: {results['fft_info']['n_all_peaks']:,}\n")
        f.write(f"Significant peaks (>1000×mean): {results['fft_info']['n_significant_peaks']:,}\n")
        f.write(f"Zeta function zeros: {results['gamma_k_info']['n_zeros']:,}\n\n")

        f.write("MATCH QUALITY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Good matches (<1%): {results['matching']['good_matches_1percent']:,} "
               f"({results['matching']['good_ratio_percent']:.4f}%)\n\n")

        f.write("ERROR STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Median error: {results['statistics']['rel_errors_percent']['median']:.6f}%\n")
        f.write(f"Mean error: {results['statistics']['rel_errors_percent']['mean']:.6f}%\n")
        f.write(f"Max error: {results['statistics']['rel_errors_percent']['max']:.4f}%\n\n")

        f.write("RANGES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"FFT frequencies: [{results['fft_info']['frequency_range'][0]:.2f}, "
               f"{results['fft_info']['frequency_range'][1]:.2f}] Hz\n")
        f.write(f"γ/(2π) frequencies: [{results['gamma_k_info']['frequency_range'][0]:.2f}, "
               f"{results['gamma_k_info']['frequency_range'][1]:.2f}] Hz\n")

    logger.success(f"Saved: {summary_path}")

    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description='Compare FFT(R) with theoretical zeta function zero frequencies'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory (default: data/)')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes')

    args = parser.parse_args()

    # Initialize
    logger = setup_logging("fft_comparison.log")

    log_section("COMPARING FFT(R) WITH γ_k/(2π)")
    logger.info(f"Start time: {datetime.now().isoformat()}")

    # Directories
    if args.data_dir is None:
        data_dir = Path("data")
    else:
        data_dir = Path(args.data_dir)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = data_dir / "result" / f"fft_vs_gamma_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Configuration
    config = DEFAULT_CONFIG.copy()
    config['n_processes'] = args.n_processes or min(cpu_count(), 8)

    log_section("CONFIGURATION")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    try:
        # Load data
        R, ln_p_grid, delta_ln_p = load_r_dynamics(data_dir, logger)
        gamma_k = load_gamma_k(data_dir, logger)

        # FFT - returns ALL peaks and significant peaks
        frequencies, amplitudes, peak_freqs, peak_amps, significant_peak_freqs, significant_peak_amps = compute_fft(R, delta_ln_p, logger)

        # Free memory
        del R
        gc.collect()

        # Matching - ALL gamma_k with FFT spectrum
        match_results = match_peaks_to_zeros(peak_freqs, peak_amps, gamma_k, logger)

        # Visualization
        create_visualizations(frequencies, amplitudes, peak_freqs, peak_amps,
                           significant_peak_freqs, significant_peak_amps,
                           match_results, output_dir, logger)

        # Save
        results = save_results(frequencies, amplitudes, peak_freqs, peak_amps,
                             significant_peak_freqs, significant_peak_amps,
                             match_results, config, output_dir, logger)

        # Final report
        log_section("FINAL REPORT")
        logger.success("COMPARISON COMPLETE!")
        logger.info(f"Good matches (<1%): {results['matching']['good_matches_1percent']:,} "
                   f"({results['matching']['good_ratio_percent']:.4f}%)")
        logger.info(f"Median error: {results['statistics']['rel_errors_percent']['median']:.6f}%")
        logger.info(f"All files in: {output_dir}")
        logger.info(f"End time: {datetime.now().isoformat()}")

        # File list
        logger.info("\nCREATED FILES:")
        for f in sorted(output_dir.iterdir()):
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            logger.info(f"  {f.name}: {size_str}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
