"""
Configuration parameters for Prime Zeta Research.

This module contains all configurable parameters for the research pipeline,
allowing easy experimentation with different values without code changes.
"""

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class ResearchConfig:
    """
    Main configuration class for the research pipeline.
    
    Attributes:
        max_n: Maximum number for prime generation (default: 10^9)
        c_values: List of C parameters for window calculation (default: [10.0])
        max_resample_points: Maximum points for resampling before FFT (default: 10^7)
        gamma_k_file: Path to file with zeta function zeros (optional)
        db_path: Path to SQLite database (default: research.db in data directory)
        data_dir: Base directory for data storage
        batch_size: Batch size for heavy operations like Li calculation
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    max_n: int = 1_000_000_000
    c_values: List[float] = None
    max_resample_points: int = 10_000_000
    gamma_k_file: Optional[str] = None
    db_path: str = None
    data_dir: str = "data"
    batch_size: int = 1_000_000
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [10.0]
        if self.db_path is None:
            self.db_path = os.path.join(self.data_dir, "research.db")
    
    @property
    def data_dir_abs(self) -> str:
        """Get absolute path to data directory."""
        return os.path.abspath(self.data_dir)
    
    @property
    def db_path_abs(self) -> str:
        """Get absolute path to database file."""
        return os.path.abspath(self.db_path)
    
    def get_primes_file(self) -> str:
        """Get path to binary file with primes."""
        return os.path.join(self.data_dir_abs, "primes", f"primes_{self.max_n}.npy")
    
    def get_residuals_file(self, c_value: float) -> str:
        """Get path to binary file with residuals for given C."""
        return os.path.join(
            self.data_dir_abs, "residuals", 
            f"residuals_{self.max_n}_c{c_value}.npy"
        )
    
    def get_spectrum_file(self, c_value: float) -> str:
        """Get path to binary file with spectrum for given C."""
        return os.path.join(
            self.data_dir_abs, "spectra", 
            f"spectrum_{self.max_n}_c{c_value}.npy"
        )


# Default configuration instance
DEFAULT_CONFIG = ResearchConfig()


@dataclass
class SieveConfig:
    """Configuration for prime sieve algorithm."""
    use_segmented: bool = False
    segment_size: int = 10_000_000  # 10M per segment
    use_odd_only: bool = True
    save_to_disk: bool = True


@dataclass
class FFTConfig:
    """Configuration for FFT analysis."""
    resample_method: str = "log_uniform"  # 'uniform', 'log_uniform'
    percentile_threshold: float = 90.0
    peak_distance: int = 10
    window_function: str = "hann"  # 'hann', 'hamming', 'none'


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    echo_sql: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    check_same_thread: bool = True


# Global configuration instances
SIEVE_CONFIG = SieveConfig()
FFT_CONFIG = FFTConfig()
DB_CONFIG = DatabaseConfig()
