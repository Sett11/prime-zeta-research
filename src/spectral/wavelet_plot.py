"""
Wavelet scalogram plotting utilities for Prime Zeta Research.

This module provides functions for building a scalogram (wavelet spectrogram)
based on pre-computed CWT coefficients / power spectrum.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_gamma_k_frequencies(gamma_k_file: str) -> Optional[np.ndarray]:
    """
    Load theoretical frequencies f_k = gamma_k / (2π) from zeta function zeros file.

    Args:
        gamma_k_file: Path to file with imaginary parts of zeta function zeros.

    Returns:
        Array of frequencies f_k or None if file could not be loaded.
    """
    try:
        gamma_k_path = Path(gamma_k_file)
        if not gamma_k_path.exists():
            logger.warning(f"File {gamma_k_file} not found. Skipping zero overlay.")
            return None

        gamma_k = np.loadtxt(gamma_k_path)
        f_theory = gamma_k / (2 * np.pi)
        logger.info(f"Loaded {len(f_theory)} theoretical frequencies from {gamma_k_file}")
        logger.info(f"Frequency range: [{f_theory.min():.2f}, {f_theory.max():.2f}]")
        return f_theory
    except Exception as e:
        logger.warning(f"Could not load gamma_k: {e}. Continuing without zero overlay.")
        return None


def plot_scalogram(
    power: np.ndarray,
    ln_p_grid: np.ndarray,
    frequencies: np.ndarray,
    output_path: Path,
    gamma_k_file: Optional[str] = None,
    max_freq: Optional[float] = None,
    title_suffix: str = "",
) -> None:
    """
    Build scalogram (heatmap) of wavelet transform.

    Args:
        power: Power spectrum of wavelet transform (scales x time).
        ln_p_grid: Grid in ln(p) space.
        frequencies: Frequencies for each scale.
        output_path: Path to save plot.
        gamma_k_file: Optional path to zeta function zeros file.
        max_freq: Maximum frequency for display (None = full range).
        title_suffix: Additional text in title.
    """
    logger.info("Building scalogram...")

    # Time axis should match number of columns in power (wavelet computed on decimated grid)
    n_time = power.shape[1]
    if len(ln_p_grid) != n_time:
        logger.info(
            f"ln_p_grid (len={len(ln_p_grid)}) does not match wavelet time size ({n_time}). "
            "Using uniform grid over the same range."
        )
        ln_p_grid = np.linspace(ln_p_grid.min(), ln_p_grid.max(), n_time, dtype=ln_p_grid.dtype)

    # Limit frequency range if needed
    if max_freq is not None:
        freq_mask = frequencies <= max_freq
        power = power[freq_mask, :]
        frequencies = frequencies[freq_mask]
        logger.info(f"Limited frequency range to {max_freq:.2f}")

    # Convert ln_p_grid to p for more intuitive axis
    p_grid = np.exp(ln_p_grid)

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 10))

    # Main scalogram
    ax1 = plt.subplot(2, 1, 1)

    # Use logarithmic normalization for better visualization
    # Limit minimum value to avoid log(0) issues
    power_plot = power.copy()
    positive_mask = power_plot > 0
    if not np.any(positive_mask):
        logger.warning("All power values are non-positive. Scalogram may be incorrect.")
        power_min = 1e-12
    else:
        power_min = np.percentile(power_plot[positive_mask], 1)  # 1st percentile of positive values
    power_plot[power_plot < power_min] = power_min

    # Build heatmap
    im = ax1.contourf(
        p_grid,
        frequencies,
        power_plot,
        levels=50,
        cmap="viridis",
        norm=LogNorm(vmin=power_min, vmax=power_plot.max()),
    )

    # Add colorbar
    plt.colorbar(im, ax=ax1, label="Wavelet Transform Power")

    # Configure axes
    ax1.set_xlabel("p (prime number)", fontsize=12)
    ax1.set_ylabel("Frequency f = γ/(2π)", fontsize=12)
    ax1.set_title(
        f"Wavelet Transform Scalogram of R(p){title_suffix}\n"
        f"Time-Frequency Localization of Oscillations",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Overlay theoretical zeta function zero frequencies
    if gamma_k_file:
        f_theory = load_gamma_k_frequencies(gamma_k_file)
        if f_theory is not None and len(f_theory) > 0:
            # Limit frequencies for display
            if max_freq is not None:
                f_theory = f_theory[f_theory <= max_freq]

            if len(f_theory) > 0:
                # Mark every 100th frequency for readability
                step = max(1, len(f_theory) // 50)  # Show ~50 zeros
                for f in f_theory[::step]:
                    ax1.axhline(
                        y=f,
                        color="red",
                        linestyle="--",
                        alpha=0.3,
                        linewidth=0.5,
                    )

                # Mark first few zeros more prominently
                for i, f in enumerate(f_theory[:10]):
                    ax1.axhline(
                        y=f,
                        color="red",
                        linestyle="-",
                        alpha=0.6,
                        linewidth=1.0,
                        label="ζ(s) zeros" if i == 0 else None,
                    )

                if len(f_theory) > 10:
                    ax1.legend(loc="upper right", fontsize=10)

    # Second subplot: mean power by frequency
    ax2 = plt.subplot(2, 1, 2)

    # Compute mean power for each frequency
    mean_power_per_freq = np.mean(power_plot, axis=1)

    ax2.plot(frequencies, mean_power_per_freq, "b-", linewidth=1.5, label="Mean Power")
    ax2.set_xlabel("Frequency f = γ/(2π)", fontsize=12)
    ax2.set_ylabel("Mean Power", fontsize=12)
    ax2.set_title("Mean Power by Frequency", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Overlay theoretical frequencies on bottom plot
    if gamma_k_file:
        f_theory = load_gamma_k_frequencies(gamma_k_file)
        if f_theory is not None and len(f_theory) > 0:
            if max_freq is not None:
                f_theory = f_theory[f_theory <= max_freq]

            for f in f_theory[:20]:
                ax2.axvline(
                    x=f,
                    color="red",
                    linestyle="--",
                    alpha=0.4,
                    linewidth=0.8,
                )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Scalogram saved: {output_path}")

    plt.close(fig)
