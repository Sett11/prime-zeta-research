#!/usr/bin/env python3
"""
Stage 2: Spectral analysis (FFT/Wavelet).

Performs FFT and optional Wavelet analysis on residual term R(p).
Results are saved to JSON and PNG files.

Usage:
    python scripts/2_spectral_analysis.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config_loader import get_config
from src.database.operations import DatabaseManager
from src.spectral.resampling import prepare_for_fft
from src.spectral.fft_analysis import compute_real_fft
from src.spectral.wavelet_analysis import compute_wavelet_analysis, PYWT_AVAILABLE
from src.analysis.streaming_cn_li import stream_regression_for_C, build_R_signal_for_fft
# Scalogram is built via scripts/5_analyze_scalogram.py


def setup_logging():
    """Configure logging."""
    config = get_config()
    log_level = config.log_level
    log_file = config.get("logging", "file", "research.log")

    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(log_file, level=log_level, rotation="10 MB", retention=5)


def ensure_results_dir(base_dir: str) -> Path:
    """Create results directory."""
    results_dir = Path(base_dir) / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_json(data: dict, filepath: Path):
    """Save data to JSON file."""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def plot_spectrum(frequencies: np.ndarray, amplitudes: np.ndarray,
                  peaks_freqs: np.ndarray, peaks_amps: np.ndarray,
                  save_path: Path, title: str):
    """Save spectrum plot."""
    plt.figure(figsize=(12, 6))

    # Main spectrum
    plt.subplot(1, 2, 1)
    plt.semilogy(frequencies, amplitudes)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude (log)')
    plt.title('Power Spectrum')
    plt.grid(True, alpha=0.3)

    # Marked peaks
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, amplitudes, 'b-', alpha=0.5, label='Spectrum')

    # Limit peaks for visualization (no more than 10000)
    max_display_peaks = 10000
    if len(peaks_freqs) > max_display_peaks:
        # Select top-N peaks by amplitude
        top_indices = np.argsort(peaks_amps)[-max_display_peaks:]
        display_freqs = peaks_freqs[top_indices]
        display_amps = peaks_amps[top_indices]
        plt.scatter(display_freqs, display_amps, c='red', s=20, alpha=0.5)
        plt.title(f'{title}\nShowing top-{max_display_peaks} of {len(peaks_freqs)} peaks')
    else:
        plt.scatter(peaks_freqs, peaks_amps, c='red', s=20, label=f'Peaks ({len(peaks_freqs)})')
        plt.title(f'{title}\nPeaks found: {len(peaks_freqs)}')
        # Legend only if peaks are few
        if len(peaks_freqs) <= 1000:
            plt.legend()

    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved: {save_path}")


def main():
    """Main spectral analysis function."""
    setup_logging()

    # Load configuration
    config = get_config()
    max_n = config.max_n
    c_value = config.c_value
    db_path = config.db_path
    results_base_dir = config.results_dir

    logger.info("=" * 60)
    logger.info("STAGE 2: Spectral Analysis (FFT)")
    logger.info("=" * 60)
    logger.info(f"max_n: {max_n:,}")
    logger.info(f"c_value: {c_value}")

    # Create results directory
    results_dir = ensure_results_dir(results_base_dir)
    logger.info(f"Results directory: {results_dir}")

    primes_file = Path(config.get_primes_file())
    use_file_based = primes_file.exists()
    if use_file_based:
        logger.info("Load mode: from file (primes.npy + streaming Need/CN/R calculation)")
    else:
        logger.info("Load mode: from DB (get_arrays)")

    with DatabaseManager(db_path) as db:
        # Get experiment
        experiment = db.experiments.get_or_create(max_n, c_value)

        if not use_file_based and experiment.status not in ["data_ready", "completed"]:
            logger.error("Data not ready! Run stage 1 first.")
            return

        load_start = time.time()
        if use_file_based:
            # File mode: primes from .npy, streaming regression and R(p)
            logger.info("-" * 60)
            logger.info("STEP 1: Load primes from file and streaming CN/R calculation")
            logger.info("-" * 60)
            primes_mm = np.load(str(primes_file), mmap_mode="r")
            if primes_mm.dtype != np.int64:
                primes_mm = np.asarray(primes_mm, dtype=np.int64)
            logger.info(f"Opened memmap of primes: {len(primes_mm):,} points")

            reg = stream_regression_for_C(
                primes_mm, max_n, c_value,
                li_batch_size=200_000,
                skip_initial=500,
                verbose=True,
            )
            k, b, corr = reg.k, reg.b, reg.correlation
            logger.info(f"Regression results: k={k:.6f}, b={b:.6f}, correlation={corr:.6f}")

            valid_primes, R_values = build_R_signal_for_fft(
                primes_mm, max_n, c_value, k, b,
                li_batch_size=200_000,
                verbose=True,
            )
            input_primes = valid_primes
        else:
            # DB mode: load prime/Need/CN from DB
            logger.info("-" * 60)
            logger.info("STEP 1: Load data from database")
            logger.info("-" * 60)
            primes, need_vals, cn_vals, _, _ = db.results.get_arrays(experiment.id, verbose=True)
            load_elapsed = time.time() - load_start
            logger.info(f"Loaded {len(primes):,} points (prime, Need, CN) in {load_elapsed:.2f} seconds")
            logger.info(f"CN size in memory: ~{cn_vals.nbytes / 1024 / 1024:.1f} MB")

            if len(primes) == 0:
                logger.error("No data for analysis!")
                return

            logger.info("Computing Li(p) and CN(p) ~ Li(p) regression")
            from src.analysis.li_function import li_vectorized
            from src.analysis.regression import linear_regression
            from src.analysis.residuals import compute_residuals

            li_values = li_vectorized(primes, verbose=True)
            k, b, corr = linear_regression(li_values, cn_vals, skip_initial=500)
            logger.info(f"Regression results: k={k:.6f}, b={b:.6f}, correlation={corr:.6f}")

            _, R_values = compute_residuals(
                primes,
                cn_vals,
                k,
                b,
                verbose=True,
            )
            input_primes = primes

        load_elapsed = time.time() - load_start
        logger.info(f"Data preparation took {load_elapsed:.2f} seconds")

        if len(input_primes) == 0:
            logger.error("No data for analysis!")
            return

        # FFT preparation
        logger.info("-" * 60)
        logger.info("STEP 2: FFT preparation (resampling)")
        logger.info("-" * 60)
        prep_start = time.time()
        max_resample = config.get_experiment_param("max_resample_points", 10_000_000)
        logger.info(f"Maximum points for resampling: {max_resample:,}")
        logger.info(f"Input points: {len(input_primes):,}")

        prime_proc, R_proc, ln_p_grid, R_mean, R_std = prepare_for_fft(
            input_primes,
            R_values,
            max_points=max_resample,
            use_memmap=True,
            memmap_dir=str(results_dir),
            memmap_filename="fft_signal.dat",
        )

        num_input_points = len(input_primes)

        # Free memory
        logger.info("Freeing memory from source arrays...")
        import gc
        if use_file_based:
            del valid_primes, R_values
        else:
            del primes, need_vals, cn_vals, li_values, R_values
        gc.collect()

        # Log memory usage after freeing
        try:
            from src.utils.memory import get_memory_usage
            mem = get_memory_usage()
            logger.info(f"Memory after freeing: RSS={mem['rss_mb']:.0f}MB, available={mem['available_mb']:.0f}MB")
        except ImportError:
            pass

        prep_elapsed = time.time() - prep_start
        logger.info(f"Preparation completed in {prep_elapsed:.2f} seconds")
        logger.info(f"Points after resampling: {len(R_proc):,}")
        logger.info(f"R(p) statistics: mean={R_mean:.6f}, std={R_std:.6f}")

        # FFT
        logger.info("-" * 60)
        logger.info("STEP 3: Compute FFT")
        logger.info("-" * 60)
        fft_start = time.time()
        logger.info(f"Input signal size: {len(R_proc):,} points")
        logger.info(f"Expected output spectrum size: {len(R_proc)//2 + 1:,} frequencies")

        if len(ln_p_grid) < 2:
            logger.error("ln_p_grid has fewer than 2 points - cannot compute FFT")
            return
        delta_ln_p = (ln_p_grid[-1] - ln_p_grid[0]) / (len(ln_p_grid) - 1)
        logger.info(f"Step by ln(p): {delta_ln_p:.10f}")

        freq_pos, amp_pos = compute_real_fft(R_proc, delta_ln_p)

        fft_elapsed = time.time() - fft_start
        logger.info(f"FFT computed in {fft_elapsed:.2f} seconds")
        logger.info(f"Spectrum size: {len(freq_pos):,} frequencies")
        logger.info(f"Frequency range: [{freq_pos.min():.6f}, {freq_pos.max():.6f}]")
        logger.info(f"Frequency resolution: {freq_pos[1] - freq_pos[0]:.10f}")
        logger.info(f"Max amplitude: {amp_pos.max():.6f}, Mean: {amp_pos.mean():.6f}")

        # Peak finding
        logger.info("-" * 60)
        logger.info("STEP 4: Peak finding in spectrum")
        logger.info("-" * 60)
        peak_start = time.time()
        percentile = config.get_analysis_param("percentile_threshold", 90.0)
        distance = config.get_analysis_param("peak_distance", 10)
        logger.info(f"Search parameters: threshold={percentile}%, min distance={distance}")

        threshold = np.percentile(amp_pos, percentile)
        logger.info(f"Amplitude threshold: {threshold:.6f}")

        peak_indices, _ = find_peaks(amp_pos, height=threshold, distance=distance)

        peak_freqs = freq_pos[peak_indices]
        peak_amps = amp_pos[peak_indices]

        peak_elapsed = time.time() - peak_start
        logger.info(f"Peak finding completed in {peak_elapsed:.2f} seconds")
        logger.info(f"Found {len(peak_freqs)} peaks (threshold {percentile}%)")
        if len(peak_freqs) > 0:
            logger.info(f"Peak frequency range: [{peak_freqs.min():.6f}, {peak_freqs.max():.6f}]")
            logger.info(f"Peak amplitude range: [{peak_amps.min():.6f}, {peak_amps.max():.6f}]")

        # Statistics
        stats = {
            "experiment": {
                "max_n": int(max_n),
                "c_value": float(c_value),
                "timestamp": datetime.now().isoformat()
            },
            "preprocessing": {
                "input_points": num_input_points,
                "resampled_points": len(R_proc),
                "mean": float(R_mean),
                "std": float(R_std),
                "time_seconds": float(prep_elapsed)
            },
            "fft": {
                "n_frequencies": len(freq_pos),
                "frequency_range": [float(freq_pos.min()), float(freq_pos.max())],
                "delta_frequency": float(freq_pos[1] - freq_pos[0]) if len(freq_pos) > 1 else 0,
                "time_seconds": float(fft_elapsed)
            },
            "peaks": {
                "count": len(peak_freqs),
                "percentile_threshold": percentile,
                "distance": distance,
                "threshold_height": float(threshold)
            },
            "peak_data": {
                "frequencies": [float(f) for f in peak_freqs],
                "amplitudes": [float(a) for a in peak_amps]
            }
        }

        # Save JSON
        json_path = results_dir / "spectrum_analysis.json"
        save_json(stats, json_path)

        # Save PNG
        if "png" in config.output_format:
            png_path = results_dir / "spectrum_plot.png"
            try:
                plot_spectrum(freq_pos, amp_pos, peak_freqs, peak_amps, png_path,
                             f"FFT Spectrum (max_n={max_n:,}, c={c_value})")
            except Exception as e:
                logger.warning(f"Could not create PNG plot: {e}")
                logger.warning("Continuing without PNG - data will be saved to npz")

        # Wavelet analysis (optional)
        wavelet_results = None
        wavelet_config = config._config.get("analysis", {}).get("wavelet", {}) if hasattr(config, '_config') else {}
        wavelet_enabled = wavelet_config.get("enabled", False)

        if wavelet_enabled:
            if not PYWT_AVAILABLE:
                logger.warning("Wavelet analysis enabled but PyWavelets not installed. Skipping.")
                logger.info("Install with: pip install PyWavelets")
            else:
                logger.info("-" * 60)
                logger.info("STEP 5: Wavelet analysis (CWT)")
                logger.info("-" * 60)
                wavelet_start = time.time()

                try:
                    # Get parameters from config
                    max_points = wavelet_config.get("max_points")
                    num_scales = wavelet_config.get("num_scales", 100)
                    wavelet_name = wavelet_config.get("wavelet", "cmor1.5-1.0")
                    scale_range = wavelet_config.get("scale_range", [0.1, 0.45])
                    percentile = wavelet_config.get("percentile_threshold", 90.0)

                    logger.info(f"Wavelet analysis parameters:")
                    logger.info(f"  Wavelet: {wavelet_name}")
                    logger.info(f"  Number of scales: {num_scales}")
                    logger.info(f"  Scale range: {scale_range}")
                    if max_points:
                        logger.info(f"  Max points: {max_points:,}")
                    else:
                        logger.info(f"  No decimation (using all {len(R_proc):,} points)")

                    # Run wavelet analysis
                    gamma_k_file = config.get_experiment_param("gamma_k_file")
                    wavelet_results = compute_wavelet_analysis(
                        R_proc,
                        ln_p_grid,
                        wavelet=wavelet_name,
                        num_scales=num_scales,
                        scale_range=tuple(scale_range) if scale_range else None,
                        max_points=max_points,
                        percentile=percentile,
                        gamma_k_file=gamma_k_file,
                        verbose=True
                    )

                    wavelet_elapsed = time.time() - wavelet_start
                    logger.info(f"Wavelet analysis completed in {wavelet_elapsed:.2f} seconds")
                    logger.info(f"Found {len(wavelet_results['peak_frequencies'])} significant frequencies")

                except Exception as e:
                    logger.error(f"Error during wavelet analysis: {e}")
                    logger.warning("Continuing without wavelet analysis")
                    wavelet_results = None
        else:
            logger.info("Wavelet analysis disabled in configuration")

        # Save binary data
        logger.info("-" * 60)
        logger.info("STEP 6: Save results")
        logger.info("-" * 60)
        save_start = time.time()

        # Update statistics with wavelet results
        if wavelet_results:
            stats["wavelet"] = {
                "enabled": True,
                "num_scales": wavelet_results["num_scales"],
                "wavelet": wavelet_results["wavelet"],
                "n_peak_frequencies": len(wavelet_results["peak_frequencies"]),
                "peak_frequencies": [float(f) for f in wavelet_results["peak_frequencies"]],
                "peak_powers": [float(p) for p in wavelet_results["peak_powers"]],
                "frequency_range": [
                    float(wavelet_results["frequencies"].min()),
                    float(wavelet_results["frequencies"].max())
                ]
            }
        else:
            stats["wavelet"] = {"enabled": False}

        # Save R dynamics on resampled grid (copy from R_proc, may be memmap)
        R_resampled = np.asarray(R_proc)
        logger.info(f"Preparing R_resampled for saving: {len(R_resampled):,} points")

        # Save FFT data
        save_dict = {
            "frequencies": freq_pos,
            "amplitudes": amp_pos,
            "peak_frequencies": peak_freqs,
            "peak_amplitudes": peak_amps,
            "ln_p_grid": ln_p_grid,
            "R_resampled": R_resampled,
        }

        # Add wavelet data if available
        if wavelet_results:
            save_dict["wavelet_coefficients"] = wavelet_results["coefficients"]
            save_dict["wavelet_frequencies"] = wavelet_results["frequencies"]
            save_dict["wavelet_scales"] = wavelet_results["scales"]
            save_dict["wavelet_power"] = wavelet_results["power"]
            save_dict["wavelet_peak_frequencies"] = wavelet_results["peak_frequencies"]
            save_dict["wavelet_peak_powers"] = wavelet_results["peak_powers"]

        spectrum_npz_path = results_dir / "spectrum_data.npz"
        np.savez(
            spectrum_npz_path,
            **save_dict,
        )
        logger.info(f"Binary data saved: {spectrum_npz_path}")

        # Scalogram is built via scripts/5_analyze_scalogram.py
        # This avoids double loading data from DB

        save_elapsed = time.time() - save_start
        logger.info(f"Saving completed in {save_elapsed:.2f} seconds")

        # Update status
        db.experiments.update_status(experiment, "spectrum_ready")

    total_elapsed = time.time() - load_start
    logger.info("-" * 40)
    logger.info(f"Stage 2 completed in {total_elapsed:.2f} seconds")
    logger.info(f"Results saved to: {results_dir}")

    logger.info("=" * 60)
    logger.info("STAGE 2: Spectral Analysis - COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
