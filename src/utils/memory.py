"""
Memory control utilities.

This module provides memory monitoring and control functions for
handling large datasets during computations.
"""

import gc
import psutil
import os
from typing import Optional, Tuple
import numpy as np
from loguru import logger


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory information
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        "total_mb": psutil.virtual_memory().total / (1024 * 1024),
    }


def check_memory(
    warning_threshold_mb: float = 1000,
    critical_threshold_mb: float = 8000
) -> str:
    """
    Check current memory and return status.
    
    Args:
        warning_threshold_mb: Warning threshold in MB
        critical_threshold_mb: Critical threshold in MB
        
    Returns:
        Status string: "OK", "WARNING", or "CRITICAL"
    """
    mem = get_memory_usage()
    
    if mem["rss_mb"] > critical_threshold_mb:
        return "CRITICAL"
    elif mem["rss_mb"] > warning_threshold_mb:
        return "WARNING"
    else:
        return "OK"


def get_available_memory_mb() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB
    """
    return psutil.virtual_memory().available / (1024 * 1024)


def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Process memory in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def estimate_array_memory(array: np.ndarray) -> int:
    """
    Estimate memory usage of a numpy array.
    
    Args:
        array: Numpy array
        
    Returns:
        Memory usage in bytes
    """
    return array.nbytes


def estimate_array_memory_mb(array: np.ndarray) -> float:
    """
    Estimate memory usage of a numpy array in MB.
    
    Args:
        array: Numpy array
        
    Returns:
        Memory usage in MB
    """
    return array.nbytes / (1024 * 1024)


def suggest_batch_size(
    data_size_mb: float,
    available_memory_mb: float,
    safety_factor: float = 0.8
) -> int:
    """
    Suggest an appropriate batch size based on memory constraints.
    
    Args:
        data_size_mb: Size of full dataset in MB
        available_memory_mb: Available memory in MB
        safety_factor: Fraction of available memory to use
        
    Returns:
        Suggested batch size (number of batches)
    """
    usable_memory = available_memory_mb * safety_factor
    
    if data_size_mb <= usable_memory:
        return 1  # No batching needed
    
    # Number of batches needed
    n_batches = int(np.ceil(data_size_mb / usable_memory))
    
    return n_batches


def bytes_to_human_readable(bytes: int) -> str:
    """
    Convert bytes to human-readable string.
    
    Args:
        bytes: Number of bytes
        
    Returns:
        Human-readable string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def human_readable_to_bytes(s: str) -> int:
    """
    Convert human-readable string to bytes.
    
    Args:
        s: Human-readable string (e.g., "1.5 GB")
        
    Returns:
        Number of bytes
    """
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    
    s = s.strip().upper()
    
    for unit, multiplier in units.items():
        if unit in s:
            value = float(s.replace(unit, "").strip())
            return int(value * multiplier)
    
    # Assume bytes
    return int(s)


class MemoryMonitor:
    """
    Memory monitor for tracking memory usage over time.
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        warning_threshold_mb: float = 4000,
        critical_threshold_mb: float = 8000
    ):
        """
        Initialize memory monitor.
        
        Args:
            log_interval: Log every N measurements
            warning_threshold_mb: Warning threshold in MB
            critical_threshold_mb: Critical threshold in MB
        """
        self.log_interval = log_interval
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.measurements = []
        self.measurement_count = 0
    
    def measure(self) -> dict:
        """Take a memory measurement."""
        mem = get_memory_usage()
        self.measurements.append(mem)
        self.measurement_count += 1
        
        # Log at intervals
        if self.measurement_count % self.log_interval == 0:
            self._log_status(mem)
        
        # Check thresholds
        status = check_memory(
            self.warning_threshold_mb,
            self.critical_threshold_mb
        )
        
        if status == "CRITICAL":
            logger.error(f"CRITICAL memory usage: {mem['rss_mb']:.0f} MB")
        elif status == "WARNING":
            logger.warning(f"High memory usage: {mem['rss_mb']:.0f} MB")
        
        return mem
    
    def _log_status(self, mem: dict):
        """Log current memory status."""
        logger.debug(
            f"Memory: RSS={mem['rss_mb']:.0f}MB, "
            f"VMS={mem['vms_mb']:.0f}MB, "
            f"Available={mem['available_mb']:.0f}MB"
        )
    
    def get_stats(self) -> dict:
        """Get statistics from all measurements."""
        if not self.measurements:
            return {}
        
        rss_values = [m["rss_mb"] for m in self.measurements]
        
        return {
            "n_measurements": len(self.measurements),
            "max_rss_mb": max(rss_values),
            "min_rss_mb": min(rss_values),
            "mean_rss_mb": sum(rss_values) / len(rss_values),
        }
    
    def reset(self):
        """Reset measurements."""
        self.measurements = []
        self.measurement_count = 0


def free_memory(*arrays: np.ndarray):
    """
    Force garbage collection and optionally delete arrays.
    
    Args:
        *arrays: Arrays to delete (optional)
    """
    for arr in arrays:
        if arr is not None and isinstance(arr, np.ndarray):
            del arr
    
    gc.collect()
    logger.debug("Memory freed and garbage collected")


def chunk_large_array(
    array: np.ndarray,
    target_chunk_size_mb: float = 100,
    axis: int = 0
) -> list:
    """
    Split a large array into chunks.
    
    Args:
        array: Array to chunk
        target_chunk_size_mb: Target size of each chunk in MB
        axis: Axis to chunk along
        
    Returns:
        List of array chunks
    """
    array_size_mb = estimate_array_memory_mb(array)
    
    if array_size_mb <= target_chunk_size_mb:
        return [array]
    
    # Calculate number of chunks
    n_chunks = int(np.ceil(array_size_mb / target_chunk_size_mb))
    
    # Split along axis
    chunks = np.array_split(array, n_chunks, axis=axis)
    
    logger.info(
        f"Split array of {array_size_mb:.1f}MB into {len(chunks)} chunks"
    )
    
    return chunks


if __name__ == "__main__":
    # Test memory utilities
    print("Testing memory utilities...")
    
    # Get memory usage
    mem = get_memory_usage()
    print(f"Current memory: RSS={mem['rss_mb']:.1f}MB, Available={mem['available_mb']:.1f}MB")
    
    # Test array memory estimation
    arr = np.zeros(10_000_000, dtype=np.float64)
    print(f"Array of 10M float64: {estimate_array_memory_mb(arr):.1f}MB")
    del arr
    
    # Test batch size suggestion
    batch_size = suggest_batch_size(
        data_size_mb=5000,
        available_memory_mb=get_available_memory_mb(),
        safety_factor=0.8
    )
    print(f"Suggested batch size for 5GB data: {batch_size}")
    
    # Test memory monitor
    monitor = MemoryMonitor(log_interval=5)
    for i in range(20):
        monitor.measure()
        time.sleep(0.1)
    
    stats = monitor.get_stats()
    print(f"Memory stats: {stats}")
    
    print("\nTest completed successfully!")
