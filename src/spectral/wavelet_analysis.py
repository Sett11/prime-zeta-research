"""
Wavelet analysis module for Prime Zeta Research.

This module provides Continuous Wavelet Transform (CWT) analysis
using PyWavelets for time-frequency localization of prime number patterns.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not available. Wavelet analysis will be disabled.")


def compute_cwt(
    signal: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    sampling_period: float = 1.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform (CWT) of a signal.
    
    Args:
        signal: Input signal (1D array, already centered/normalized)
        scales: Array of scales for CWT analysis
        wavelet: Wavelet name (default: "cmor1.5-1.0" - complex Morlet)
        sampling_period: Sampling period of the signal
        verbose: Whether to log progress
        
    Returns:
        Tuple of (coefficients, frequencies)
        - coefficients: Complex CWT coefficients (scales x time)
        - frequencies: Corresponding frequencies for each scale
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required for wavelet analysis. Install with: pip install PyWavelets")
    
    if verbose:
        logger.info(f"Computing CWT for {len(signal):,} points with {len(scales)} scales...")
        logger.info(f"Wavelet: {wavelet}, Sampling period: {sampling_period:.10f}")
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        wavelet,
        sampling_period=sampling_period
    )
    
    if verbose:
        logger.info(f"CWT computed. Coefficient matrix shape: {coefficients.shape}")
        logger.info(f"Frequency range: [{frequencies.min():.6f}, {frequencies.max():.6f}]")
    
    return coefficients, frequencies


def compute_wavelet_power(
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Compute wavelet power spectrum from CWT coefficients.
    
    Args:
        coefficients: Complex CWT coefficients
        
    Returns:
        Power spectrum (absolute value squared)
    """
    return np.abs(coefficients) ** 2


def find_wavelet_peaks(
    power: np.ndarray,
    frequencies: np.ndarray,
    percentile: float = 90.0,
    min_power: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in the wavelet power spectrum.
    
    Args:
        power: Wavelet power spectrum (scales x time)
        frequencies: Frequency array for scales
        percentile: Percentile threshold for peak detection
        min_power: Minimum absolute power (overrides percentile)
        
    Returns:
        Tuple of (peak_frequencies, peak_powers, peak_time_indices)
    """
    # Compute maximum power across time for each scale
    max_power_per_scale = np.max(power, axis=1)
    
    # Determine threshold
    if min_power is None:
        threshold = np.percentile(max_power_per_scale, percentile)
    else:
        threshold = min_power
    
    # Find scales with significant power
    significant_scales = max_power_per_scale >= threshold
    peak_frequencies = frequencies[significant_scales]
    peak_powers = max_power_per_scale[significant_scales]
    
    # Find time indices where peaks occur
    peak_time_indices = []
    for scale_idx in np.where(significant_scales)[0]:
        time_idx = np.argmax(power[scale_idx, :])
        peak_time_indices.append(time_idx)
    peak_time_indices = np.array(peak_time_indices)
    
    logger.info(
        f"Found {len(peak_frequencies)} significant frequencies "
        f"(threshold={threshold:.6f}, percentile={percentile}%)"
    )
    
    return peak_frequencies, peak_powers, peak_time_indices


def compute_wavelet_analysis(
    signal: np.ndarray,
    ln_p_grid: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    num_scales: int = 100,
    scale_range: Optional[Tuple[float, float]] = None,
    max_points: Optional[int] = None,
    percentile: float = 90.0,
    gamma_k_file: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Complete wavelet analysis pipeline.
    
    Args:
        signal: Input signal (R(p) values)
        ln_p_grid: Grid in ln(p) space
        wavelet: Wavelet name
        num_scales: Number of scales for analysis
        scale_range: Optional tuple (min_scale, max_scale)
        max_points: Optional maximum points (for downsampling)
        percentile: Percentile threshold for peak detection
        verbose: Whether to log progress
        
    Returns:
        Dictionary with analysis results
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required for wavelet analysis.")
    
    # Optional downsampling
    if max_points is not None and len(signal) > max_points:
        if verbose:
            logger.info(f"Downsampling signal: {len(signal):,} -> {max_points:,} points")
        step = len(signal) // max_points
        signal = signal[::step]
        ln_p_grid = ln_p_grid[::step]
    
    # Save downsampled ln_p_grid for use in scalogram
    ln_p_grid_downsampled = ln_p_grid.copy()
    
    # Compute sampling period
    if len(ln_p_grid) > 1:
        delta_ln_p = ln_p_grid[1] - ln_p_grid[0]
    else:
        delta_ln_p = 1.0
    
    # Determine scale range
    if scale_range is None:
        # Automatically compute scale range based on zeta function zero frequencies
        # Formula from original notebooks: scale = 1 / (frequency * sampling_period)

        # Try to load gamma_k for accurate frequency range determination
        if gamma_k_file:
            try:
                # Use general loader that accounts for additional zero files
                from src.spectral.matching import load_gamma_k_values

                gamma_k = load_gamma_k_values(gamma_k_file)
                f_theory = gamma_k / (2 * np.pi)
                f_min = float(f_theory.min())
                f_max = float(f_theory.max())
                if verbose:
                    logger.info(
                        f"Loaded frequency range from gamma_k (including additional zeros if available): "
                        f"[{f_min:.2f}, {f_max:.2f}]"
                    )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Failed to load gamma_k from {gamma_k_file}: {e}. "
                        f"Using default values."
                    )
                f_min = 2.0  # Minimum frequency (gamma_k ≈ 14 → f ≈ 2.25)
                f_max = 900.0  # Maximum frequency (for first ~5000-10000 zeros)
        else:
            f_min = 2.0  # Minimum frequency (gamma_k ≈ 14 → f ≈ 2.25)
            f_max = 900.0  # Maximum frequency (gamma_k ≈ 5447 → f ≈ 866.7)
        
        # Compute scales from original notebooks formula
        # scale = 1 / (frequency * sampling_period)
        # Larger scale for lower frequency, smaller scale for higher frequency
        scale_max = 1.0 / (f_min * delta_ln_p)  # For minimum frequency (larger scale)
        scale_min = 1.0 / (f_max * delta_ln_p)   # For maximum frequency (smaller scale)

        # Limit minimum scale (PyWavelets requires minimum ~0.1)
        # But for very small delta_ln_p, scales can be huge
        # Limit maximum scale to reasonable value (e.g., 10000)
        scale_min = max(0.1, scale_min)
        scale_max = min(10000.0, scale_max)

        # If scales are still too large or out of bounds, use alternative approach
        if scale_min >= scale_max or scale_max > 10000.0:
            # Use more conservative range that covers main frequencies
            # Focus on first 100-200 zeros (more important for analysis)
            f_max_limited = 200.0 / (2 * np.pi)  # Approximately for gamma_k ≈ 200
            scale_min = max(0.1, 1.0 / (f_max_limited * delta_ln_p))
            scale_max = min(10000.0, 1.0 / (f_min * delta_ln_p))
            if verbose:
                logger.warning(
                    f"Scales out of bounds. Using limited range "
                    f"for frequencies [{f_min:.2f}, {f_max_limited:.2f}]: "
                    f"scales [{scale_min:.2f}, {scale_max:.2f}]"
                )
        
        scale_range = (scale_min, scale_max)
        
        if verbose:
            # Compute expected frequencies for verification
            expected_freq_min = 1.0 / (scale_max * delta_ln_p)
            expected_freq_max = 1.0 / (scale_min * delta_ln_p)
            logger.info(
                f"Automatically computed scale range: [{scale_min:.4f}, {scale_max:.4f}]"
            )
            logger.info(
                f"Expected frequency range: [{expected_freq_min:.2f}, {expected_freq_max:.2f}]"
            )
    
    # Generate scales
    scales = np.logspace(
        np.log10(scale_range[0]),
        np.log10(scale_range[1]),
        num_scales
    )
    
    if verbose:
        logger.info(f"Scale range: [{scales[0]:.4f}, {scales[-1]:.4f}]")
        logger.info(f"Number of scales: {num_scales}")
    
    # Compute CWT
    coefficients, frequencies = compute_cwt(
        signal,
        scales,
        wavelet,
        sampling_period=delta_ln_p,
        verbose=verbose
    )
    
    # Compute power spectrum
    power = compute_wavelet_power(coefficients)
    
    # Find peaks
    peak_freqs, peak_powers, peak_time_indices = find_wavelet_peaks(
        power,
        frequencies,
        percentile=percentile
    )
    
    return {
        "coefficients": coefficients,
        "frequencies": frequencies,
        "scales": scales,
        "power": power,
        "peak_frequencies": peak_freqs,
        "peak_powers": peak_powers,
        "peak_time_indices": peak_time_indices,
        "signal_length": len(signal),
        "num_scales": num_scales,
        "wavelet": wavelet,
        "sampling_period": delta_ln_p,
        "ln_p_grid": ln_p_grid_downsampled  # Downsampled ln_p_grid for scalogram
    }


def analyze_wavelet_continuity(
    power: np.ndarray,
    frequencies: np.ndarray,
    target_frequencies: np.ndarray,
    threshold_percentile: float = 80.0
) -> Dict:
    """
    Analyze continuity of frequencies in wavelet spectrum.
    
    Checks if target frequencies (e.g., from zeta zeros) appear
    as continuous horizontal bands in the wavelet scalogram.
    
    Args:
        power: Wavelet power spectrum (scales x time)
        frequencies: Frequency array for scales
        target_frequencies: Target frequencies to check
        threshold_percentile: Percentile for power threshold
        
    Returns:
        Dictionary with continuity analysis results
    """
    threshold = np.percentile(power, threshold_percentile)
    
    continuity_results = []
    for target_freq in target_frequencies:
        # Find closest scale
        scale_idx = np.argmin(np.abs(frequencies - target_freq))
        
        # Check power along this scale
        power_at_scale = power[scale_idx, :]
        significant_points = np.sum(power_at_scale >= threshold)
        continuity_ratio = significant_points / len(power_at_scale)
        
        continuity_results.append({
            "target_frequency": float(target_freq),
            "scale_index": int(scale_idx),
            "scale_frequency": float(frequencies[scale_idx]),
            "continuity_ratio": float(continuity_ratio),
            "significant_points": int(significant_points),
            "mean_power": float(np.mean(power_at_scale)),
            "max_power": float(np.max(power_at_scale))
        })
    
    return {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "continuity_results": continuity_results
    }


if __name__ == "__main__":
    # Test wavelet analysis
    print("Testing wavelet analysis...")
    
    if not PYWT_AVAILABLE:
        print("PyWavelets not available. Install with: pip install PyWavelets")
    else:
        # Create test signal
        n = 10000
        ln_p = np.linspace(0, 10, n)
        signal = np.sin(2 * np.pi * 0.5 * ln_p) + 0.5 * np.sin(2 * np.pi * 2.0 * ln_p)
        signal = signal - signal.mean()
        
        # Run analysis
        result = compute_wavelet_analysis(
            signal,
            ln_p,
            num_scales=50,
            verbose=True
        )
        
        print(f"Analysis complete!")
        print(f"Found {len(result['peak_frequencies'])} significant frequencies")
        print(f"Power spectrum shape: {result['power'].shape}")
