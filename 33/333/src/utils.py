"""
Utilities Module for Driving Style ML Project
==============================================

This module provides reusable helper functions and utilities used throughout
the machine learning pipeline.

Contents:
- Logging utilities
- Data utilities
- Visualization helpers
- Performance timing
- File I/O helpers

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
import os
import json
import pickle
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    LOGGING_CONFIG,
    OUTPUT_DIR,
    RANDOM_SEED,
)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logger(
    name: str = "driving_style_ml",
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Set up a configured logger.

    Parameters:
    -----------
    name : str
        Logger name
    log_file : Path, optional
        Path to log file
    level : str
        Logging level

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger(
    log_file=LOGGING_CONFIG.get("log_file"),
    level=LOGGING_CONFIG.get("level", "INFO"),
)


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Parameters:
    -----------
    func : Callable
        Function to time

    Returns:
    --------
    Callable
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")

        return result

    return wrapper


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("Data loading"):
            # code to time
    """

    def __init__(self, description: str = "Operation"):
        """
        Initialize the Timer.

        Parameters:
        -----------
        description : str
            Description of the operation being timed
        """
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Stop the timer and log duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"{self.description} completed in {self.duration:.2f} seconds")


# =============================================================================
# DATA UTILITIES
# =============================================================================

def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility.

    Parameters:
    -----------
    seed : int
        Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug(f"Random seeds set to {seed}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
    --------
    Dict[str, float]
        Memory usage in MB
    """
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by optimizing data types.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool
        Whether to print memory savings

    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


def compute_hash(data: Union[pd.DataFrame, np.ndarray, str]) -> str:
    """
    Compute a hash for data integrity verification.

    Parameters:
    -----------
    data : DataFrame, ndarray, or str
        Data to hash

    Returns:
    --------
    str
        MD5 hash string
    """
    if isinstance(data, pd.DataFrame):
        data_bytes = data.to_json().encode()
    elif isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    else:
        data_bytes = str(data).encode()

    return hashlib.md5(data_bytes).hexdigest()


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to disk using pickle.

    Parameters:
    -----------
    obj : Any
        Object to save
    filepath : str or Path
        Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    logger.info(f"Object saved to {filepath}")


def load_object(filepath: Union[str, Path]) -> Any:
    """
    Load an object from disk.

    Parameters:
    -----------
    filepath : str or Path
        Path to load from

    Returns:
    --------
    Any
        Loaded object
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    logger.info(f"Object loaded from {filepath}")
    return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str or Path
        Path to save file
    indent : int
        JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(data), f, indent=indent)

    logger.info(f"JSON saved to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load data from JSON file.

    Parameters:
    -----------
    filepath : str or Path
        Path to JSON file

    Returns:
    --------
    Dict
        Loaded data
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    logger.info(f"JSON loaded from {filepath}")
    return data


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """
    Simple progress tracker for long-running operations.

    Usage:
        tracker = ProgressTracker(total=100, description="Processing")
        for i in range(100):
            # do work
            tracker.update()
        tracker.close()
    """

    def __init__(
        self,
        total: int,
        description: str = "Progress",
        print_every: int = 10,
    ):
        """
        Initialize the tracker.

        Parameters:
        -----------
        total : int
            Total number of items
        description : str
            Description of the task
        print_every : int
            Print progress every N items
        """
        self.total = total
        self.description = description
        self.print_every = print_every
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n

        if self.current % self.print_every == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0

            pct = 100 * self.current / self.total
            print(
                f"\r{self.description}: {self.current}/{self.total} "
                f"({pct:.1f}%) | {rate:.1f} it/s | ETA: {eta:.1f}s",
                end="",
                flush=True,
            )

    def close(self) -> None:
        """Close the tracker."""
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} complete in {elapsed:.2f}s")


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    formats: List[str] = ["png"],
    dpi: int = 150,
) -> List[Path]:
    """
    Save a matplotlib figure in multiple formats.

    Parameters:
    -----------
    fig : Figure
        Matplotlib figure to save
    filename : str
        Base filename (without extension)
    output_dir : Path, optional
        Output directory
    formats : list
        File formats to save
    dpi : int
        Resolution

    Returns:
    --------
    List[Path]
        Paths to saved files
    """
    output_dir = output_dir or OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", format=fmt)
        saved_paths.append(filepath)
        logger.debug(f"Figure saved: {filepath}")

    return saved_paths


def create_subplot_grid(
    n_plots: int,
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots.

    Parameters:
    -----------
    n_plots : int
        Number of plots needed
    n_cols : int
        Number of columns
    figsize_per_plot : tuple
        Size of each subplot

    Returns:
    --------
    Tuple[Figure, ndarray]
        Figure and axes array
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    return fig, axes


# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

class ExperimentTracker:
    """
    Simple experiment tracking for reproducibility.

    Tracks:
    - Hyperparameters
    - Metrics
    - Timestamps
    - Data hashes
    """

    def __init__(self, experiment_name: str, output_dir: Optional[Path] = None):
        """
        Initialize the tracker.

        Parameters:
        -----------
        experiment_name : str
            Name of the experiment
        output_dir : Path, optional
            Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir or OUTPUT_DIR / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log = {
            "experiment_name": experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "hyperparameters": {},
            "metrics": {},
            "artifacts": [],
            "notes": [],
        }

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.log["hyperparameters"].update(params)
        logger.info(f"Logged parameters: {list(params.keys())}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.log["metrics"]:
            self.log["metrics"][name] = []

        entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step

        self.log["metrics"][name].append(entry)

    def log_artifact(self, filepath: Union[str, Path], description: str = "") -> None:
        """Log an artifact file."""
        self.log["artifacts"].append({
            "path": str(filepath),
            "description": description,
            "timestamp": datetime.now().isoformat(),
        })

    def add_note(self, note: str) -> None:
        """Add a note to the experiment."""
        self.log["notes"].append({
            "note": note,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self) -> Path:
        """Save the experiment log."""
        self.log["end_time"] = datetime.now().isoformat()

        filepath = self.output_dir / f"{self.experiment_name}_{self.experiment_id}.json"
        save_json(self.log, filepath)

        return filepath

    def get_summary(self) -> str:
        """Get a summary of the experiment."""
        summary_lines = [
            f"Experiment: {self.experiment_name}",
            f"ID: {self.experiment_id}",
            f"Start: {self.log['start_time']}",
            "",
            "Hyperparameters:",
        ]

        for param, value in self.log["hyperparameters"].items():
            summary_lines.append(f"  {param}: {value}")

        summary_lines.append("")
        summary_lines.append("Final Metrics:")

        for metric, values in self.log["metrics"].items():
            if values:
                final_value = values[-1]["value"]
                summary_lines.append(f"  {metric}: {final_value:.4f}")

        return "\n".join(summary_lines)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    max_missing_pct: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame for common issues.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        Columns that must be present
    max_missing_pct : float
        Maximum allowed missing percentage per column

    Returns:
    --------
    Tuple[bool, List[str]]
        Validation result and list of issues
    """
    issues = []

    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check missing values
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > max_missing_pct:
            issues.append(f"Column '{col}' has {missing_pct*100:.1f}% missing values")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            issues.append(f"Column '{col}' contains infinite values")

    is_valid = len(issues) == 0

    return is_valid, issues


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("UTILITIES MODULE DEMONSTRATION")
    print("=" * 60)

    # Set random seeds
    set_random_seeds(42)
    print("✓ Random seeds set")

    # Timer demonstration
    with Timer("Sample operation"):
        time.sleep(0.5)

    # Create sample data
    sample_df = pd.DataFrame({
        "feature_1": np.random.randn(1000),
        "feature_2": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })

    # Memory optimization
    print("\nMemory optimization:")
    optimized_df = reduce_memory_usage(sample_df)

    # Data validation
    print("\nData validation:")
    is_valid, issues = validate_dataframe(
        sample_df,
        required_columns=["feature_1", "feature_2"],
    )
    print(f"  Valid: {is_valid}")
    if issues:
        print(f"  Issues: {issues}")

    # Experiment tracking
    print("\nExperiment tracking:")
    tracker = ExperimentTracker("demo_experiment")
    tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
    tracker.log_metric("accuracy", 0.85)
    tracker.log_metric("accuracy", 0.90)
    tracker.add_note("Initial experiment")

    print(tracker.get_summary())

    # Save experiment
    exp_path = tracker.save()
    print(f"\nExperiment saved to: {exp_path}")

    print("\n✓ Utilities module demonstration complete!")
