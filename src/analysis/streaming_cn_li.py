"""
Streaming CN/Li regression and R(p) construction for file-based spectral analysis.

Provides stream_regression_for_C and build_R_signal_for_fft so that Stage 2
can work without loading primes/need/cn from the database: primes are read
from primes_{max_n}.npy (memmap), and Need/CN/R are computed in streaming passes.
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from loguru import logger

from .li_function import li_vectorized
from .regression import StreamingLinearRegression
from ..primes.need import iter_need_batches


@dataclass
class RegressionSummary:
    """Summary of streaming CN~Li regression for a given C."""
    c_value: float
    k: float
    b: float
    correlation: float
    n_points: int
    mean_need: float
    std_need: float
    mean_cn: float
    std_cn: float


def stream_regression_for_C(
    primes: np.ndarray,
    max_n: int,
    c_value: float,
    li_batch_size: int = 200_000,
    skip_initial: int = 500,
    verbose: bool = True,
) -> RegressionSummary:
    """
    Streaming Need_C(p), CN_C(p) and regression CN_C ~ Li(p) for given C.

    Logic matches main pipeline: iter_need_batches uses window_size_vectorized
    / filter_valid_primes / calculate_window_bounds with min_margin=10.
    """
    if verbose:
        logger.info(
            f"[C={c_value}] Streaming Need/CN/Li for valid primes "
            f"(max_n={max_n:,}), same logic as main pipeline..."
        )

    reg = StreamingLinearRegression(skip_initial=skip_initial)
    running_cn = 0.0
    sum_need = 0.0
    sum_need_sq = 0.0
    sum_cn = 0.0
    sum_cn_sq = 0.0
    total_points = 0
    start_time = time.time()
    batch_index = 0

    for batch_primes, need_batch in iter_need_batches(
        primes,
        c_value,
        max_n,
        batch_size=li_batch_size,
        verbose=verbose,
    ):
        if len(batch_primes) == 0:
            continue

        batch_index += 1
        cn_batch = np.cumsum(need_batch) + running_cn
        running_cn = float(cn_batch[-1])

        sum_need += float(np.sum(need_batch))
        sum_need_sq += float(np.sum(need_batch * need_batch))
        sum_cn += float(np.sum(cn_batch))
        sum_cn_sq += float(np.sum(cn_batch * cn_batch))
        total_points += len(batch_primes)

        primes_arr = np.asarray(batch_primes, dtype=np.float64)
        cn_arr = np.asarray(cn_batch, dtype=np.float64)
        li_arr = li_vectorized(primes_arr, batch_size=len(primes_arr), verbose=False)
        reg.update(li_arr, cn_arr)

        if verbose and batch_index % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"[C={c_value}] Batches={batch_index}, points={total_points:,}, "
                f"elapsed={elapsed/60.0:.1f} min"
            )

    k, b, corr = reg.coefficients()
    mean_need = sum_need / total_points if total_points > 0 else 0.0
    mean_cn = sum_cn / total_points if total_points > 0 else 0.0
    var_need = sum_need_sq / total_points - mean_need * mean_need if total_points > 0 else 0.0
    var_cn = sum_cn_sq / total_points - mean_cn * mean_cn if total_points > 0 else 0.0
    std_need = math.sqrt(max(var_need, 0.0))
    std_cn = math.sqrt(max(var_cn, 0.0))

    elapsed_total = time.time() - start_time
    if verbose:
        logger.info(
            f"[C={c_value}] Regression done: k={k:.6f}, b={b:.6f}, corr={corr:.6f}, "
            f"n={total_points:,}, elapsed={elapsed_total/60.0:.1f} min"
        )

    return RegressionSummary(
        c_value=float(c_value),
        k=float(k),
        b=float(b),
        correlation=float(corr),
        n_points=int(total_points),
        mean_need=float(mean_need),
        std_need=float(std_need),
        mean_cn=float(mean_cn),
        std_cn=float(std_cn),
    )


def build_R_signal_for_fft(
    primes: np.ndarray,
    max_n: int,
    c_value: float,
    k: float,
    b: float,
    li_batch_size: int = 200_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One streaming pass: Need -> CN -> Li -> R for valid primes only.

    Returns (valid_primes, R_values) suitable for prepare_for_fft(valid_primes, R_values, ...).
    Same Need/CN logic as main pipeline (iter_need_batches).
    """
    if verbose:
        logger.info(
            f"[C={c_value}] Building R(p) for FFT (max_n={max_n:,}), valid primes only..."
        )

    valid_primes_list = []
    R_list = []
    running_cn = 0.0
    start_time = time.time()
    batch_index = 0

    for batch_primes, need_batch in iter_need_batches(
        primes,
        c_value,
        max_n,
        batch_size=li_batch_size,
        verbose=verbose,
    ):
        if len(batch_primes) == 0:
            continue

        batch_index += 1
        cn_batch = np.cumsum(need_batch) + running_cn
        running_cn = float(cn_batch[-1])

        primes_arr = np.asarray(batch_primes, dtype=np.float64)
        cn_arr = np.asarray(cn_batch, dtype=np.float64)
        li_arr = li_vectorized(primes_arr, batch_size=len(primes_arr), verbose=False)
        R_batch = cn_arr - (k * li_arr + b)

        valid_primes_list.append(np.asarray(batch_primes, dtype=np.int64))
        R_list.append(R_batch.astype(np.float64))

        if verbose and batch_index % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"[C={c_value}] R(p) batches={batch_index}, elapsed={elapsed/60.0:.1f} min"
            )

    valid_primes = np.concatenate(valid_primes_list, axis=0) if valid_primes_list else np.array([], dtype=np.int64)
    R_values = np.concatenate(R_list, axis=0) if R_list else np.array([], dtype=np.float64)

    elapsed_total = time.time() - start_time
    if verbose:
        logger.info(
            f"[C={c_value}] R(p) built: {len(valid_primes):,} points, "
            f"elapsed={elapsed_total/60.0:.1f} min"
        )

    return valid_primes, R_values
