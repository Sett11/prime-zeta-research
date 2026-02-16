#!/usr/bin/env python3
"""
Save and plot R(p) dynamics from already saved data.

Loads R_resampled and ln_p_grid from spectrum_data.npz (result of stage 2),
saves them to results subdirectory and builds R dynamics plot with clear name.
Full pipeline is not run - only work with already saved data.

Usage:
    python scripts/5_save_R_dynamics.py
    python scripts/5_save_R_dynamics.py --npz data/dataset_sources/spectrum_data.npz --output-dir data/R_dynamics
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config_loader import get_config


def setup_logging(log_level: str = "INFO") -> None:
    config = get_config()
    level = log_level or config.log_level
    log_file = config.get("logging", "file", "research.log")
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_file, level=level, rotation="10 MB", retention=5)


def find_latest_npz(results_dir: Path) -> Path:
    npz_files = list(results_dir.glob("**/spectrum_data.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"File spectrum_data.npz not found in {results_dir}. "
            "First run stage 2: python scripts/2_spectral_analysis.py"
        )
    return max(npz_files, key=lambda f: f.stat().st_mtime)


def load_R_dynamics(npz_path: Path) -> tuple:
    """Load R_resampled and ln_p_grid from NPZ. Checks for R_resampled and ln_p_grid."""
    logger.info(f"Loading data from: {npz_path}")
    data = np.load(npz_path)
    if "R_resampled" not in data:
        raise KeyError(
            "No R_resampled in file. Re-run stage 2 (scripts/2_spectral_analysis.py); "
            "it now saves R to spectrum_data.npz."
        )
    if "ln_p_grid" not in data:
        raise KeyError(
            "No ln_p_grid in file. Re-run stage 2 (scripts/2_spectral_analysis.py); "
            "it now saves ln_p_grid to spectrum_data.npz."
        )
    R_resampled = np.asarray(data["R_resampled"])
    ln_p_grid = np.asarray(data["ln_p_grid"])
    if len(R_resampled) != len(ln_p_grid):
        raise ValueError(
            f"R_resampled ({len(R_resampled)}) and ln_p_grid ({len(ln_p_grid)}) lengths don't match."
        )
    logger.info(f"Loaded: R_resampled and ln_p_grid with {len(R_resampled):,} points")
    return R_resampled, ln_p_grid


def load_metadata(npz_path: Path) -> dict:
    """Load mean, std from spectrum_analysis.json in same directory (if exists)."""
    json_path = npz_path.parent / "spectrum_analysis.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        pre = data.get("preprocessing", {})
        return {"mean": pre.get("mean"), "std": pre.get("std")}
    except Exception as e:
        logger.warning(f"Could not load metadata from {json_path}: {e}")
        return {}


def plot_R_dynamics(
    ln_p_grid: np.ndarray,
    R_resampled: np.ndarray,
    output_path: Path,
    run_label: str = "",
    max_plot_points: int = 400_000,
) -> None:
    """Build R(ln p) plot with decimation for display."""
    n = len(R_resampled)
    step = max(1, n // max_plot_points)
    x = ln_p_grid[::step]
    y = R_resampled[::step]
    logger.info(f"Plot: {len(x):,} points (decimation step {step})")

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(x, y, "b-", alpha=0.7, linewidth=0.4, label="R (centered and norm.)")
    ax.set_xlabel("ln(p)")
    ax.set_ylabel("R (resampled)")
    title = "R(p) dynamics (resampled by ln p)"
    if run_label:
        title += f" — {run_label}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save R dynamics and build plot from data in spectrum_data.npz"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to spectrum_data.npz (default: latest in results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/R_dynamics)",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=400_000,
        help="Max points on plot after decimation (default 400000)",
    )
    parser.add_argument("--log-level", type=str, default=None, help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = get_config()
    results_base = Path(config.results_dir)

    logger.info("=" * 60)
    logger.info("Save and plot R(p) dynamics")
    logger.info("=" * 60)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = results_base / "R_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    if args.npz is not None:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            logger.error(f"File not found: {npz_path}")
            sys.exit(1)
    else:
        npz_path = find_latest_npz(results_base)
        logger.info(f"Using NPZ: {npz_path}")

    try:
        R_resampled, ln_p_grid = load_R_dynamics(npz_path)
    except KeyError as e:
        logger.error(e)
        sys.exit(1)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    meta = load_metadata(npz_path)
    meta["source_npz"] = str(npz_path)
    meta["n_points"] = int(len(R_resampled))
    meta_path = out_dir / "R_dynamics_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")

    npz_out_path = out_dir / "R_dynamics.npz"
    np.savez(npz_out_path, R_resampled=R_resampled, ln_p_grid=ln_p_grid)
    logger.info(f"Data saved: {npz_out_path}")

    run_label = npz_path.parent.name if npz_path.parent.name != "results" else ""
    plot_path = out_dir / "R_dynamics_plot.png"
    plot_R_dynamics(
        ln_p_grid,
        R_resampled,
        plot_path,
        run_label=run_label,
        max_plot_points=args.max_plot_points,
    )

    logger.info("=" * 60)
    logger.info("Done. Results in directory: {}", out_dir)
    logger.info("  R_dynamics.npz, R_dynamics_metadata.json, R_dynamics_plot.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
