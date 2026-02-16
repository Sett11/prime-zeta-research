"""
Linear regression module.

This module provides linear regression functionality for fitting
CN(p) to the logarithmic integral Li(p).
"""

from typing import Tuple, Dict, Optional, Iterable, Iterator
import numpy as np
from loguru import logger


def linear_regression(
    x: np.ndarray, 
    y: np.ndarray,
    skip_initial: int = 500
) -> Tuple[float, float, float]:
    """
    Perform linear regression y = k*x + b.
    
    Uses numpy's polyfit for computation.
    
    Args:
        x: Independent variable (e.g., Li(p) values)
        y: Dependent variable (e.g., CN(p) values)
        skip_initial: Number of initial points to skip for stability
        
    Returns:
        Tuple of (k, b, correlation)
        - k: Slope coefficient
        - b: Intercept
        - correlation: Pearson correlation coefficient
    """
    if len(x) < skip_initial + 10:
        logger.warning("Not enough data points for regression")
        return np.nan, np.nan, np.nan
    
    # Skip initial points for stability
    start_idx = min(skip_initial, len(x) // 10)
    x_fit = x[start_idx:]
    y_fit = y[start_idx:]
    
    # Perform linear regression
    k, b = np.polyfit(x_fit, y_fit, 1)
    
    # Compute correlation
    correlation = float(np.corrcoef(x_fit, y_fit)[0, 1])
    
    return k, b, correlation


class StreamingLinearRegression:
    """
    Incremental (streaming) linear regression y = k*x + b.

    Stores only accumulated sums and counters and can accept data in batches,
    without holding the entire array in memory.
    """

    def __init__(self, skip_initial: int = 500) -> None:
        self.skip_initial = max(0, int(skip_initial))
        self._skipped = 0

        # Counters and sums for remaining points
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """
        Add a batch of points to the regression.

        Args:
            x_batch: 1D array of x values
            y_batch: 1D array of y values of the same size
        """
        if x_batch is None or y_batch is None:
            return

        x_batch = np.asarray(x_batch, dtype=np.float64)
        y_batch = np.asarray(y_batch, dtype=np.float64)

        if x_batch.shape != y_batch.shape:
            raise ValueError("x_batch and y_batch must have the same shape")

        m = x_batch.size
        if m == 0:
            return

        # Skip first skip_initial points (possible anomalies at the start of the series)
        if self._skipped < self.skip_initial:
            to_skip = min(self.skip_initial - self._skipped, m)
            self._skipped += to_skip
            x_batch = x_batch[to_skip:]
            y_batch = y_batch[to_skip:]

        if x_batch.size == 0:
            return

        # Accumulate sums
        self.n += x_batch.size
        self.sum_x += float(np.sum(x_batch))
        self.sum_y += float(np.sum(y_batch))
        self.sum_x2 += float(np.sum(x_batch * x_batch))
        self.sum_y2 += float(np.sum(y_batch * y_batch))
        self.sum_xy += float(np.sum(x_batch * y_batch))

    def coefficients(self) -> Tuple[float, float, float]:
        """
        Finish computation and return (k, b, correlation).
        """
        if self.n < 10:
            logger.warning(
                f"Not enough points for streaming regression: n={self.n}"
            )
            return np.nan, np.nan, np.nan

        n = float(self.n)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        # Variance and covariance
        var_x = self.sum_x2 / n - mean_x * mean_x
        var_y = self.sum_y2 / n - mean_y * mean_y
        cov_xy = self.sum_xy / n - mean_x * mean_y

        if var_x <= 0:
            logger.warning("Zero or negative variance of X in streaming regression")
            return np.nan, np.nan, np.nan

        k = cov_xy / var_x
        b = mean_y - k * mean_x

        # Correlation
        if var_x <= 0 or var_y <= 0:
            correlation = np.nan
        else:
            correlation = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))

        return float(k), float(b), float(correlation)


def linear_regression_stream(
    batches: Iterable[Tuple[np.ndarray, np.ndarray]],
    skip_initial: int = 500,
) -> Tuple[float, float, float]:
    """
    Linear regression y = k*x + b by batches (streaming).

    Args:
        batches: Iterable over batches (x_batch, y_batch),
                 where both are 1D numpy arrays of the same length.
        skip_initial: How many initial points to skip globally.

    Returns:
        (k, b, correlation)
    """
    reg = StreamingLinearRegression(skip_initial=skip_initial)

    for x_batch, y_batch in batches:
        reg.update(x_batch, y_batch)

    return reg.coefficients()


def robust_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    skip_initial: int = 500,
    quantile: float = 0.95
) -> Tuple[float, float, float]:
    """
    Perform robust linear regression using quantiles.
    
    This version uses a subset of data points to reduce the influence
    of outliers at the beginning of the series.
    
    Args:
        x: Independent variable
        y: Dependent variable
        skip_initial: Number of initial points to skip
        quantile: Quantile of points to use (0.95 = top 95%)
        
    Returns:
        Tuple of (k, b, correlation)
    """
    if len(x) < skip_initial + 100:
        return linear_regression(x, y, skip_initial)
    
    start_idx = min(skip_initial, len(x) // 10)
    x_fit = x[start_idx:]
    y_fit = y[start_idx:]
    
    # Use last quantile of points for robustness
    n = len(x_fit)
    start_quantile = int(n * (1 - quantile))
    x_quant = x_fit[start_quantile:]
    y_quant = y_fit[start_quantile:]
    
    k, b = np.polyfit(x_quant, y_quant, 1)
    correlation = float(np.corrcoef(x_quant, y_quant)[0, 1])
    
    return k, b, correlation


def fit_cn_to_li(
    prime_N: np.ndarray,
    CN_values: np.ndarray,
    skip_initial: int = 500
) -> Dict[str, float]:
    """
    Fit CN(p) to Li(p) using linear regression.
    
    This is the main function for fitting the cumulative need
    to the logarithmic integral.
    
    Args:
        prime_N: Array of prime numbers
        CN_values: Array of CN(p) values
        skip_initial: Number of initial points to skip
        
    Returns:
        Dictionary with keys:
        - 'k': Slope coefficient
        - 'b': Intercept
        - 'correlation': Pearson correlation
        - 'mean_squared_error': MSE of fit
        - 'mean_absolute_error': MAE of fit
    """
    from .li_function import li_vectorized
    
    # Compute Li(p)
    li_values = li_vectorized(prime_N, verbose=False)
    
    # Perform regression
    k, b, correlation = linear_regression(li_values, CN_values, skip_initial)
    
    # Compute errors
    predicted = k * li_values + b
    residuals = CN_values - predicted
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))
    
    logger.info(
        f"CN vs Li regression: k={k:.6f}, b={b:.6f}, "
        f"correlation={correlation:.6f}, MSE={mse:.4f}"
    )
    
    return {
        "k": float(k),
        "b": float(b),
        "correlation": float(correlation),
        "mean_squared_error": float(mse),
        "mean_absolute_error": float(mae)
    }


def compute_fit_quality(
    x: np.ndarray,
    y: np.ndarray,
    k: float,
    b: float
) -> Dict[str, float]:
    """
    Compute quality metrics for a linear fit.
    
    Args:
        x: Independent variable values
        y: Actual dependent variable values
        k: Fitted slope
        b: Fitted intercept
        
    Returns:
        Dictionary of quality metrics
    """
    predicted = k * x + b
    residuals = y - predicted
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(residuals))
    mae = np.mean(np.abs(residuals))
    
    # Relative errors (normalized by mean of y)
    y_mean = np.mean(y)
    relative_rmse = rmse / y_mean if y_mean != 0 else 0
    relative_mae = mae / y_mean if y_mean != 0 else 0
    
    return {
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "mae": float(mae),
        "relative_rmse": float(relative_rmse),
        "relative_mae": float(relative_mae),
        "max_residual": float(np.max(np.abs(residuals))),
        "residual_std": float(np.std(residuals))
    }


if __name__ == "__main__":
    # Test regression
    import matplotlib.pyplot as plt
    from src.primes.sieve import generate_primes_sieve_optimized
    from src.primes.need import compute_all_need_data
    
    print("Testing linear regression...")
    
    # Generate small dataset
    MAX_N = 100_000
    C = 10.0
    
    primes, _ = generate_primes_sieve_optimized(MAX_N)
    print(f"Generated {len(primes)} primes")
    
    # Compute Need and CN
    need_data = compute_all_need_data(primes, C, MAX_N, verbose=True)
    
    # Fit to Li
    result = fit_cn_to_li(
        need_data["prime_N"],
        need_data["CN"],
        skip_initial=100
    )
    
    print(f"\nRegression results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")
