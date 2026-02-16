#!/usr/bin/env python3
"""
Build scalogram from already computed spectral analysis data.

Uses optimized logic: loads data from existing
spectrum_data.npz (result of stage 2_spectral_analysis.py).
Does not access DB or recompute R(p) or wavelet - only visualization.

Twin analysis removed (already done in past experiments, pattern doesn't change).

Usage:
    python scripts/4_analyze_scalogram.py
    python scripts/4_analyze_scalogram.py --npz path/to/spectrum_data.npz
    python scripts/4_analyze_scalogram.py --npz path/to/spectrum_data.npz --output scalogram.png --max-freq 100
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.config_loader import get_config
from src.spectral.wavelet_plot import plot_scalogram


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    config = get_config()
    level = log_level or config.log_level
    log_file = config.get("logging", "file", "research.log")
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_file, level=level, rotation="10 MB", retention=5)


def find_latest_npz(results_dir: Path) -> Path:
    """Find latest spectrum_data.npz in results directory."""
    npz_files = list(results_dir.glob("**/spectrum_data.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"File spectrum_data.npz not found in {results_dir}. "
            "First run stage 2: python scripts/2_spectral_analysis.py"
        )
    latest = max(npz_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using file: {latest}")
    return latest


def load_wavelet_data(npz_path: Path) -> dict:
    """Load from NPZ data needed for scalogram."""
    logger.info(f"Loading data from: {npz_path}")
    data = np.load(npz_path)

    required = ["wavelet_power", "wavelet_frequencies", "ln_p_grid"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"No wavelet data in file (stage 2 must be run with analysis.wavelet.enabled: true). "
            f"Missing keys: {missing}. Available: {list(data.keys())}"
        )

    power = data["wavelet_power"]
    frequencies = data["wavelet_frequencies"]
    ln_p_grid = data["ln_p_grid"]

    logger.info(
        f"Loaded: power {power.shape}, frequencies {len(frequencies)}, "
        f"ln(p): [{ln_p_grid.min():.2f}, {ln_p_grid.max():.2f}]"
    )
    return {
        "power": power,
        "frequencies": frequencies,
        "ln_p_grid": ln_p_grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build scalogram from data in spectrum_data.npz"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to spectrum_data.npz (default: latest in results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save plot (default: scalogram.png next to NPZ)",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=None,
        help="Max frequency on scalogram (default: from config or all)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = get_config()
    results_dir = Path(config.results_dir)

    logger.info("=" * 60)
    logger.info("SCALOGRAM: building from ready spectrum_data.npz data")
    logger.info("=" * 60)
    logger.info(f"max_n: {config.max_n:,}, c_value: {config.c_value}")

    npz_path = args.npz
    if npz_path is None:
        npz_path = find_latest_npz(results_dir)
    else:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            logger.error(f"File not found: {npz_path}")
            sys.exit(1)

    wavelet_data = load_wavelet_data(npz_path)

    output_path = args.output
    if output_path is None:
        scalogram_filename = config.wavelet_scalogram_filename if hasattr(config, "wavelet_scalogram_filename") else "scalogram.png"
        output_path = npz_path.parent / scalogram_filename
    else:
        output_path = Path(output_path)

    gamma_k_file = config.gamma_k_file if hasattr(config, "gamma_k_file") else config.get_experiment_param("gamma_k_file", None)
    max_freq = args.max_freq
    if max_freq is None and hasattr(config, "wavelet_scalogram_max_freq"):
        max_freq = config.wavelet_scalogram_max_freq

    plot_scalogram(
        power=wavelet_data["power"],
        ln_p_grid=wavelet_data["ln_p_grid"],
        frequencies=wavelet_data["frequencies"],
        output_path=output_path,
        gamma_k_file=gamma_k_file,
        max_freq=max_freq,
        title_suffix=f" (max_n={config.max_n:,}, c={config.c_value})",
    )

    logger.info("=" * 60)
    logger.info(f"Scalogram saved: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
