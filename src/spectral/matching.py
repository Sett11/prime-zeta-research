"""
Zeta zero matching module.

This module provides functionality for matching spectrum peaks
with theoretical zeros of the Riemann zeta function.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from loguru import logger


# First zeros of the Riemann zeta function (imaginary parts)
# These are the standard critical line zeros: 0.5 + i*gamma_k
GAMMA_K_FULL = np.array([
    14.134725141734693790457251983562470270785257115279,
    21.022039638771554992261479584198010766351055485913,
    25.010857580145688763213790992562821818659596537252,
    # ... (first 10000 zeros would be loaded from file in practice)
    # For now, we'll generate this programmatically or load from cache
])


def load_gamma_k_values(path: Optional[str] = None) -> np.ndarray:
    """
    Load gamma_k values from a single data file.

    The file should contain one column of γ_k values (path is specified
    in experiment.gamma_k_file config, e.g. data/gamma_k_5000000.txt).

    Args:
        path: Path to file with gamma_k values (optional)

    Returns:
        Array of gamma_k values.
    """
    if path is not None:
        try:
            gamma_path = Path(path)
            gamma_raw = np.loadtxt(gamma_path)
            # Support both 1D and 2D format: take last column
            if gamma_raw.ndim > 1:
                gamma_raw = gamma_raw[:, -1]
            gamma_values = gamma_raw.astype(float)
            logger.info(f"Loaded {gamma_values.size} gamma_k values from {gamma_path}")
            return gamma_values
        except Exception as e:
            logger.warning(f"Could not load gamma_k from {path}: {e}")

    # Fallback: use built-in placeholder if file is not available
    logger.warning(
        "Using placeholder gamma_k values (GAMMA_K_FULL). "
        "For real analysis, specify correct file in configuration."
    )
    return GAMMA_K_FULL


def gamma_to_frequency(gamma_k: float) -> float:
    """
    Convert gamma_k to theoretical frequency.
    
    The relationship is: f_k = gamma_k / (2 * pi)
    
    Args:
        gamma_k: Imaginary part of the k-th zeta zero
        
    Returns:
        Corresponding frequency
    """
    return gamma_k / (2 * np.pi)


def frequency_to_gamma(frequency: float) -> float:
    """
    Convert frequency back to gamma_k.
    
    Args:
        frequency: Frequency value
        
    Returns:
        Corresponding gamma_k
    """
    return frequency * 2 * np.pi


def match_peaks_to_zeros(
    peak_frequencies: np.ndarray,
    peak_amplitudes: np.ndarray,
    gamma_k_values: np.ndarray,
    max_relative_error: float = 0.1
) -> List[Dict]:
    """
    Match spectrum peaks to theoretical zeta zeros.
    
    For each gamma_k, finds the nearest peak and computes the error.
    
    Args:
        peak_frequencies: Array of peak frequencies from spectrum
        peak_amplitudes: Array of peak amplitudes
        gamma_k_values: Array of gamma_k values (theoretical)
        max_relative_error: Maximum error for a "good" match
        
    Returns:
        List of match dictionaries with keys:
        - gamma_index: Index of the zero (1-based)
        - gamma_value: Value of gamma_k
        - f_theory: Theoretical frequency
        - f_peak: Found peak frequency
        - relative_error: Relative error
        - amplitude: Peak amplitude
    """
    # Convert gamma_k to frequencies
    f_theory = gamma_to_frequency(gamma_k_values)
    
    matches = []
    
    for i, f_k in enumerate(f_theory):
        # Find nearest peak
        distances = np.abs(peak_frequencies - f_k)
        
        if len(distances) == 0:
            continue
            
        closest_idx = np.argmin(distances)
        closest_freq = peak_frequencies[closest_idx]
        closest_amp = peak_amplitudes[closest_idx]
        
        relative_error = distances[closest_idx] / f_k if f_k > 0 else np.inf
        
        matches.append({
            "gamma_index": i + 1,  # 1-based indexing
            "gamma_value": gamma_k_values[i],
            "f_theory": f_k,
            "f_peak": closest_freq,
            "relative_error": relative_error,
            "amplitude": float(closest_amp)
        })
    
    # Sort by error
    matches.sort(key=lambda x: x["relative_error"])
    
    return matches


def compute_match_statistics(matches: List[Dict]) -> Dict:
    """
    Compute statistics for the matching results.
    
    Args:
        matches: List of match dictionaries
        
    Returns:
        Dictionary of statistics
    """
    if not matches:
        return {
            "total_matches": 0,
            "message": "No matches found"
        }
    
    errors = [m["relative_error"] for m in matches]
    amplitudes = [m["amplitude"] for m in matches if m["amplitude"] is not None]
    
    # Count good matches at different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    good_counts = {}
    for t in thresholds:
        good_counts[f"< {t*100:.1f}%"] = sum(1 for e in errors if e < t)
    
    return {
        "total_matches": len(matches),
        **good_counts,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "error_percentiles": {
            "1%": float(np.percentile(errors, 1)),
            "5%": float(np.percentile(errors, 5)),
            "10%": float(np.percentile(errors, 10)),
            "25%": float(np.percentile(errors, 25)),
            "50%": float(np.percentile(errors, 50)),
            "75%": float(np.percentile(errors, 75)),
            "90%": float(np.percentile(errors, 90)),
        },
        "mean_amplitude": float(np.mean(amplitudes)) if amplitudes else 0,
        "max_amplitude": float(np.max(amplitudes)) if amplitudes else 0,
    }


def get_best_matches(
    matches: List[Dict],
    n: int = 100,
    error_threshold: float = 0.01
) -> List[Dict]:
    """
    Get the best matching results.
    
    Args:
        matches: List of all matches
        n: Number of best matches to return
        error_threshold: Maximum error for "good" matches
        
    Returns:
        List of best matches
    """
    good_matches = [m for m in matches if m["relative_error"] < error_threshold]
    
    # Sort by amplitude for good matches, otherwise by error
    if len(good_matches) >= n:
        return sorted(good_matches, key=lambda x: -x["amplitude"])[:n]
    else:
        return matches[:n]


def find_specific_zero_match(
    matches: List[Dict],
    gamma_index: int
) -> Optional[Dict]:
    """
    Find match for a specific zero by index.
    
    Args:
        matches: List of all matches
        gamma_index: Index of the zero to find (1-based)
        
    Returns:
        Match dictionary or None if not found
    """
    for m in matches:
        if m["gamma_index"] == gamma_index:
            return m
    return None


def filter_matches_by_frequency_range(
    matches: List[Dict],
    f_min: float,
    f_max: float
) -> List[Dict]:
    """
    Filter matches to a specific frequency range.
    
    Args:
        matches: List of all matches
        f_min: Minimum frequency
        f_max: Maximum frequency
        
    Returns:
        Filtered list of matches
    """
    return [
        m for m in matches 
        if f_min <= m["f_theory"] <= f_max
    ]


def compute_correlation_coefficient(
    matches: List[Dict],
    error_threshold: float = 0.1
) -> float:
    """
    Compute correlation between theoretical and found frequencies.
    
    Args:
        matches: List of all matches
        error_threshold: Maximum error for included matches
        
    Returns:
        Pearson correlation coefficient
    """
    good_matches = [m for m in matches if m["relative_error"] < error_threshold]
    
    if len(good_matches) < 2:
        return 0.0
    
    f_theory = np.array([m["f_theory"] for m in good_matches])
    f_peak = np.array([m["f_peak"] for m in good_matches])
    
    return float(np.corrcoef(f_theory, f_peak)[0, 1])


if __name__ == "__main__":
    # Test matching
    import matplotlib.pyplot as plt
    
    print("Testing zeta zero matching...")
    
    # Create fake peak data
    n_peaks = 1000
    peak_freqs = np.linspace(0.1, 10, n_peaks)
    peak_amps = np.random.rand(n_peaks) * 100 + 50
    
    # Add some "real" peaks at zeta zero frequencies
    f_zeta = gamma_to_frequency(GAMMA_K_FULL[:100])
    for f in f_zeta:
        idx = np.argmin(np.abs(peak_freqs - f))
        peak_amps[idx] += 200  # Add extra amplitude
    
    # Create fake gamma_k values
    gamma_k_test = np.arange(1, 1000) * 10 + np.random.randn(1000) * 0.1
    
    # Match
    matches = match_peaks_to_zeros(peak_freqs, peak_amps, gamma_k_test)
    
    print(f"Matched {len(matches)} peaks to zeros")
    
    # Statistics
    stats = compute_match_statistics(matches)
    print("\nMatch statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")
