"""
FFT analysis module.

This module provides FFT computation and peak detection for spectral analysis.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from scipy.fft import fft, fftfreq, rfft
from scipy.signal import find_peaks
from loguru import logger


def compute_fft(
    signal: np.ndarray,
    delta_x: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of a signal.
    
    Args:
        signal: Input signal (1D array, already centered)
        delta_x: Spacing between points
        
    Returns:
        Tuple of (frequencies, amplitudes)
        - frequencies: Array of frequency values
        - amplitudes: Array of FFT amplitudes (absolute values)
    """
    n = len(signal)
    
    # Compute FFT
    fft_result = fft(signal)
    
    # Compute frequencies
    frequencies = fftfreq(n, d=delta_x)
    
    # Get amplitudes (absolute values)
    amplitudes = np.abs(fft_result)
    
    return frequencies, amplitudes


def compute_real_fft(
    signal: np.ndarray,
    delta_x: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT for real-valued signals (more efficient).

    Args:
        signal: Real-valued input signal
        delta_x: Spacing between points

    Returns:
        Tuple of (frequencies, amplitudes) for positive frequencies only
    """
    n = len(signal)

    # Compute real FFT
    fft_result = rfft(signal)

    # Compute frequencies (positive only) - rfft returns n//2+1 coefficients
    # so we need to slice rfftfreq to match
    frequencies = np.fft.rfftfreq(n, d=delta_x)[:len(fft_result)]

    # Get amplitudes
    amplitudes = np.abs(fft_result)

    return frequencies, amplitudes


def find_spectrum_peaks(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    percentile: float = 90.0,
    distance: int = 10,
    height: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in the frequency spectrum.
    
    Args:
        frequencies: Array of frequency values
        amplitudes: Array of amplitude values
        percentile: Minimum percentile threshold for peak detection
        distance: Minimum distance between peaks
        height: Minimum absolute height for peaks (overrides percentile)
        
    Returns:
        Tuple of (peak_frequencies, peak_amplitudes, peak_indices)
    """
    # Determine threshold
    if height is None:
        height = np.percentile(amplitudes, percentile)
    
    # Find peaks
    peak_indices, properties = find_peaks(
        amplitudes,
        height=height,
        distance=distance
    )
    
    peak_frequencies = frequencies[peak_indices]
    peak_amplitudes = amplitudes[peak_indices]
    
    logger.info(
        f"Found {len(peak_frequencies)} peaks "
        f"(threshold={height:.2f}, distance={distance})"
    )
    
    return peak_frequencies, peak_amplitudes, peak_indices


def analyze_spectrum(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    percentile: float = 90.0,
    distance: int = 10
) -> Dict:
    """
    Complete spectrum analysis pipeline.
    
    Args:
        frequencies: Array of frequency values
        amplitudes: Array of amplitude values
        percentile: Percentile threshold for peak detection
        distance: Minimum distance between peaks
        
    Returns:
        Dictionary with analysis results
    """
    # Find peaks
    peak_freqs, peak_amps, peak_idx = find_spectrum_peaks(
        frequencies, amplitudes, percentile, distance
    )
    
    # Filter to positive frequencies only
    pos_mask = frequencies > 0
    freq_pos = frequencies[pos_mask]
    amp_pos = amplitudes[pos_mask]
    
    # Basic statistics
    total_power = np.sum(amplitudes ** 2)
    peak_power = np.sum(peak_amps ** 2)
    power_ratio = peak_power / total_power if total_power > 0 else 0
    
    # Dominant frequency
    if len(peak_amps) > 0:
        dominant_idx = np.argmax(peak_amps)
        dominant_freq = peak_freqs[dominant_idx]
        dominant_amp = peak_amps[dominant_idx]
    else:
        dominant_freq = 0.0
        dominant_amp = 0.0
    
    # Frequency range
    freq_min = freq_pos.min() if len(freq_pos) > 0 else 0
    freq_max = freq_pos.max() if len(freq_pos) > 0 else 0
    
    return {
        "n_points": len(frequencies),
        "n_peaks": len(peak_freqs),
        "peak_frequencies": peak_freqs,
        "peak_amplitudes": peak_amps,
        "peak_indices": peak_idx,
        "dominant_frequency": dominant_freq,
        "dominant_amplitude": dominant_amp,
        "power_ratio": power_ratio,
        "frequency_range": (freq_min, freq_max),
        "threshold_height": np.percentile(amp_pos, percentile) if len(amp_pos) > 0 else 0,
        "mean_amplitude": np.mean(amp_pos) if len(amp_pos) > 0 else 0,
        "max_amplitude": np.max(amp_pos) if len(amp_pos) > 0 else 0,
    }


def compute_spectrum_statistics(
    amplitudes: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistics for spectrum amplitudes.
    
    Args:
        amplitudes: Array of spectrum amplitudes
        
    Returns:
        Dictionary of statistics
    """
    return {
        "mean": float(np.mean(amplitudes)),
        "std": float(np.std(amplitudes)),
        "min": float(np.min(amplitudes)),
        "max": float(np.max(amplitudes)),
        "median": float(np.median(amplitudes)),
        "dynamic_range_db": 20 * np.log10(np.max(amplitudes) / np.maximum(np.min(amplitudes), 1e-10)),
    }


def compute_nyquist_frequency(
    n_points: int,
    x_range: Tuple[float, float]
) -> float:
    """
    Compute the Nyquist frequency (maximum detectable frequency).
    
    Args:
        n_points: Number of points in the signal
        x_range: Tuple of (min_x, max_x)
        
    Returns:
        Nyquist frequency
    """
    x_min, x_max = x_range
    total_range = x_max - x_min
    delta_x = total_range / (n_points - 1)
    return 1.0 / (2 * delta_x)


def compute_frequency_from_period(
    period: float,
    x_range: Tuple[float, float]
) -> float:
    """
    Convert a period to frequency given the data range.
    
    Args:
        period: Period value in x-space
        x_range: Tuple of (min_x, max_x)
        
    Returns:
        Corresponding frequency
    """
    return 1.0 / period


if __name__ == "__main__":
    # Test FFT analysis
    import matplotlib.pyplot as plt
    
    print("Testing FFT analysis...")
    
    # Create test signal with known frequencies
    n = 10000
    x = np.linspace(0, 100, n)
    delta_x = x[1] - x[0]
    
    # Signal with two frequency components
    signal = (
        np.sin(2 * np.pi * 0.5 * x) +  # 0.5 Hz
        0.5 * np.sin(2 * np.pi * 2.0 * x) +  # 2.0 Hz
        0.25 * np.sin(2 * np.pi * 5.0 * x) +  # 5.0 Hz
        np.random.randn(n) * 0.1  # Noise
    )
    
    # Center the signal
    signal = signal - signal.mean()
    
    # Compute FFT
    freqs, amps = compute_real_fft(signal, delta_x)
    
    # Find peaks
    peak_freqs, peak_amps, _ = find_spectrum_peaks(freqs, amps, percentile=95, distance=5)
    
    print(f"Number of points: {n}")
    print(f"Delta x: {delta_x:.4f}")
    print(f"Frequency range: [{freqs.min():.4f}, {freqs.max():.4f}]")
    print(f"Found {len(peak_freqs)} peaks")
    
    # Print top peaks
    sorted_idx = np.argsort(peak_amps)[::-1][:5]
    print("\nTop 5 peaks:")
    for i in sorted_idx:
        print(f"  Frequency: {peak_freqs[i]:.4f} Hz, Amplitude: {peak_amps[i]:.4f}")
    
    print("\nTest completed successfully!")
