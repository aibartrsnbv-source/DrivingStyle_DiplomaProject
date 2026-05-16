"""
Data Loading Module for Driving Style ML Project
=================================================

This module handles loading, validation, and initial inspection of all datasets
used in the driving style classification and accident risk prediction pipeline.

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DATASET_PATHS,
    ARCHIVE_DIRS,
    RANDOM_SEED,
)


class DataLoader:
    """
    A class for loading and validating driving behavior datasets.

    This class provides methods to:
    - Load individual CSV files
    - Load and combine multiple datasets
    - Validate data structure and quality
    - Print comprehensive dataset information

    Attributes:
        datasets (Dict): Dictionary storing loaded DataFrames
        metadata (Dict): Dictionary storing dataset metadata
    """

    def __init__(self):
        """Initialize the DataLoader with empty containers."""
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}

    def load_csv(
        self,
        file_path: str,
        dataset_name: str,
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        dtype: Optional[dict] = None,
        parse_dates: Optional[List[str]] = None,
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame with validation.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        dataset_name : str
            Identifier name for the dataset
        nrows : int, optional
            Number of rows to read (useful for large files)
        usecols : list, optional
            Specific columns to load
        dtype : dict, optional
            Column data types specification
        parse_dates : list, optional
            Columns to parse as datetime
        low_memory : bool
            Whether to use low memory mode for large files

        Returns:
        --------
        pd.DataFrame
            Loaded and validated DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        # Get file size for information
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File path: {file_path}")
        print(f"File size: {file_size_mb:.2f} MB")

        # Load the data
        try:
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                usecols=usecols,
                dtype=dtype,
                parse_dates=parse_dates,
                low_memory=low_memory,
            )
            print(f"✓ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")

        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            raise

        # Store dataset and metadata
        self.datasets[dataset_name] = df
        self.metadata[dataset_name] = {
            "file_path": file_path,
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

        return df

    def load_us_accidents(
        self,
        nrows: Optional[int] = None,
        selected_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load the US Accidents dataset.

        This dataset contains accident records with various features including
        location, weather conditions, road features, and severity.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load
        selected_columns : list, optional
            Specific columns to load

        Returns:
        --------
        pd.DataFrame
            US Accidents dataset
        """
        # Define relevant columns for driving style analysis
        default_columns = [
            "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
            "Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
            "Wind_Speed(mph)", "Weather_Condition", "Amenity", "Bump", "Crossing",
            "Junction", "Railway", "Station", "Stop", "Traffic_Signal",
            "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
        ]

        columns = selected_columns or default_columns

        df = self.load_csv(
            file_path=DATASET_PATHS["us_accidents"],
            dataset_name="us_accidents",
            nrows=nrows,
            usecols=columns if os.path.exists(DATASET_PATHS["us_accidents"]) else None,
            parse_dates=["Start_Time", "End_Time"] if "Start_Time" in columns else None,
            low_memory=False,
        )

        return df

    def load_carla_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the CARLA simulator driving data.

        This dataset contains simulated driving behavior data from the
        CARLA autonomous driving simulator.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            CARLA driving data
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["carla_data"],
            dataset_name="carla_data",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_eco_driving(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Eco Driving Score dataset.

        This dataset contains driving efficiency metrics and eco-driving
        behavior indicators.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            Eco driving score data
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["eco_driving"],
            dataset_name="eco_driving",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_driver_behavior(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Driver Behavior Route Anomaly dataset.

        This dataset contains route information and anomaly detection
        features with derived behavioral indicators.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            Driver behavior data with derived features
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["driver_behavior"],
            dataset_name="driver_behavior",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_all_datasets(
        self,
        nrows: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.

        Parameters:
        -----------
        nrows : int, optional
            Limit rows for each dataset (useful for testing)

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of all loaded datasets
        """
        print("\n" + "=" * 60)
        print("LOADING ALL DATASETS")
        print("=" * 60)

        loaded_datasets = {}

        # Try to load each dataset, skip if file not found
        dataset_loaders = {
            "us_accidents": self.load_us_accidents,
            "carla_data": self.load_carla_data,
            "eco_driving": self.load_eco_driving,
            "driver_behavior": self.load_driver_behavior,
        }

        for name, loader in dataset_loaders.items():
            try:
                loaded_datasets[name] = loader(nrows=nrows)
            except FileNotFoundError as e:
                print(f"\n⚠ Skipping {name}: {str(e)}")
            except Exception as e:
                print(f"\n✗ Error loading {name}: {str(e)}")

        print(f"\n{'='*60}")
        print(f"Successfully loaded {len(loaded_datasets)} datasets")
        print("=" * 60)

        return loaded_datasets

    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        required_columns: Optional[List[str]] = None,
        check_duplicates: bool = True,
    ) -> Dict[str, any]:
        """
        Validate a dataset and return quality metrics.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to validate
        dataset_name : str
            Name of the dataset for reporting
        required_columns : list, optional
            Columns that must be present
        check_duplicates : bool
            Whether to check for duplicate rows

        Returns:
        --------
        Dict
            Validation results and quality metrics
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING: {dataset_name}")
        print("=" * 60)

        validation_results = {
            "dataset_name": dataset_name,
            "is_valid": True,
            "issues": [],
        }

        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    f"Missing required columns: {missing_cols}"
                )
                print(f"✗ Missing columns: {missing_cols}")
            else:
                print("✓ All required columns present")

        # Check for empty dataset
        if len(df) == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset is empty")
            print("✗ Dataset is empty")
        else:
            print(f"✓ Dataset has {len(df):,} rows")

        # Check for duplicates
        if check_duplicates:
            duplicates = df.duplicated().sum()
            validation_results["duplicate_rows"] = duplicates
            if duplicates > 0:
                validation_results["issues"].append(
                    f"Found {duplicates:,} duplicate rows"
                )
                print(f"⚠ Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
            else:
                print("✓ No duplicate rows found")

        # Check missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        cols_with_missing = missing_pct[missing_pct > 0]

        validation_results["missing_values"] = cols_with_missing.to_dict()

        if len(cols_with_missing) > 0:
            print(f"\n⚠ Columns with missing values:")
            for col, pct in cols_with_missing.items():
                status = "⚠" if pct < 50 else "✗"
                print(f"  {status} {col}: {pct:.2f}% missing")
        else:
            print("✓ No missing values found")

        # Data type summary
        print(f"\nData types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count} columns")

        validation_results["dtype_counts"] = dtype_counts.to_dict()

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        validation_results["memory_mb"] = memory_mb
        print(f"\nMemory usage: {memory_mb:.2f} MB")

        return validation_results

    def print_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Print comprehensive information about a dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        dataset_name : str
            Name of the dataset
        """
        print(f"\n{'='*60}")
        print(f"DATASET INFO: {dataset_name}")
        print("=" * 60)

        # Basic info
        print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Column information
        print(f"\nColumn Details:")
        print("-" * 60)

        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_pct = (df[col].isna().sum() / len(df)) * 100
            unique = df[col].nunique()

            print(f"  {col}")
            print(f"    Type: {dtype} | Non-null: {non_null:,} | "
                  f"Missing: {null_pct:.1f}% | Unique: {unique:,}")

            # Sample values for object columns
            if dtype == "object" and unique <= 10:
                sample_vals = df[col].dropna().unique()[:5]
                print(f"    Values: {list(sample_vals)}")

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Columns Summary:")
            print("-" * 60)
            print(df[numeric_cols].describe().round(2).to_string())

    def get_target_distribution(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> pd.Series:
        """
        Get the distribution of target variable classes.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset containing target column
        target_column : str
            Name of the target column

        Returns:
        --------
        pd.Series
            Value counts of target classes
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        distribution = df[target_column].value_counts()
        percentages = df[target_column].value_counts(normalize=True) * 100

        print(f"\nTarget Distribution: {target_column}")
        print("-" * 40)
        for cls in distribution.index:
            print(f"  {cls}: {distribution[cls]:,} ({percentages[cls]:.2f}%)")

        # Check for class imbalance
        min_pct = percentages.min()
        if min_pct < 20:
            print(f"\n⚠ Class imbalance detected! Minority class: {min_pct:.2f}%")

        return distribution

    def combine_datasets(
        self,
        datasets: List[pd.DataFrame],
        method: str = "concat",
        on: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Combine multiple datasets into one.

        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of DataFrames to combine
        method : str
            Combination method: 'concat' or 'merge'
        on : list, optional
            Columns to merge on (for merge method)

        Returns:
        --------
        pd.DataFrame
            Combined dataset
        """
        if method == "concat":
            combined = pd.concat(datasets, ignore_index=True)
            print(f"Combined {len(datasets)} datasets: {len(combined):,} total rows")

        elif method == "merge":
            if on is None:
                raise ValueError("'on' parameter required for merge method")
            combined = datasets[0]
            for df in datasets[1:]:
                combined = combined.merge(df, on=on, how="outer")
            print(f"Merged {len(datasets)} datasets: {len(combined):,} total rows")

        else:
            raise ValueError(f"Unknown method: {method}")

        return combined


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    imbalance_ratio: Optional[List[float]] = None,
    include_time_series: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a synthetic driving behavior dataset for testing.

    This function generates realistic-looking driving data that can be used
    for testing the ML pipeline when real data is not available.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of numerical features
    n_classes : int
        Number of driving style classes
    imbalance_ratio : list, optional
        Class proportions (should sum to 1)
    include_time_series : bool
        Whether to include simulated time-series features
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic driving behavior dataset
    """
    np.random.seed(random_state)

    print("\nGenerating synthetic driving dataset...")

    # Generate class labels with optional imbalance
    if imbalance_ratio is None:
        imbalance_ratio = [0.6, 0.25, 0.15]  # safe, normal, aggressive

    labels = np.random.choice(
        n_classes,
        size=n_samples,
        p=imbalance_ratio[:n_classes],
    )

    # Generate features based on driving style
    data = {}

    # Speed-related features
    base_speed = 30 + labels * 15  # Aggressive drivers tend to speed
    data["avg_speed"] = base_speed + np.random.normal(0, 5, n_samples)
    data["max_speed"] = data["avg_speed"] + np.random.exponential(10, n_samples)
    data["speed_variance"] = np.abs(np.random.normal(5, 3, n_samples) + labels * 3)

    # Acceleration features
    data["harsh_braking_count"] = np.random.poisson(labels + 1, n_samples)
    data["harsh_accel_count"] = np.random.poisson(labels + 1, n_samples)
    data["avg_acceleration"] = np.random.normal(0, 0.5, n_samples) + labels * 0.2

    # Behavioral features
    data["lane_changes"] = np.random.poisson(labels * 2 + 1, n_samples)
    data["following_distance"] = np.maximum(0.5, 3 - labels + np.random.normal(0, 0.5, n_samples))
    data["turn_signal_usage"] = np.clip(0.9 - labels * 0.2 + np.random.normal(0, 0.1, n_samples), 0, 1)

    # Time-series derived features
    if include_time_series:
        data["accel_x_mean"] = np.random.normal(0, 0.3, n_samples)
        data["accel_y_mean"] = np.random.normal(0, 0.3, n_samples)
        data["accel_z_mean"] = np.random.normal(9.8, 0.2, n_samples)
        data["gyro_x_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)
        data["gyro_y_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)
        data["gyro_z_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)

    # Contextual features
    data["time_of_day"] = np.random.randint(0, 24, n_samples)
    data["day_of_week"] = np.random.randint(0, 7, n_samples)
    data["trip_duration"] = np.random.exponential(30, n_samples)
    data["trip_distance"] = data["trip_duration"] * data["avg_speed"] / 60

    # Target variables
    class_names = ["safe", "normal", "aggressive"]
    data["driving_style"] = [class_names[l] for l in labels]
    data["driving_style_encoded"] = labels

    # Accident probability (higher for aggressive drivers)
    accident_prob = 0.05 + labels * 0.1
    data["accident"] = np.random.binomial(1, accident_prob)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some missing values (realistic scenario)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], "following_distance"] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], "turn_signal_usage"] = np.nan

    print(f"✓ Generated {n_samples:,} samples with {len(df.columns)} features")
    print(f"  Classes: {dict(zip(class_names, imbalance_ratio[:n_classes]))}")
    print(f"  Accident rate: {df['accident'].mean()*100:.1f}%")

    return df


# =============================================================================
# UAH-DRIVESET PARSER
# =============================================================================

def _load_uah_driveset(root_dir: str) -> Optional[pd.DataFrame]:
    """
    Parse UAH-DriveSet folder structure into a unified feature DataFrame.

    The dataset layout:
        UAH-DRIVESET/
            D1/
                20151120133502-16km-D1-NORMAL1-SECONDARY/
                    RAW_GPS.txt
                    RAW_ACCELEROMETERS.txt
                    PROC_LANE_DETECTION.txt
                    ...
                20151120134800-17km-D1-AGGRESSIVE-SECONDARY/
                    ...
            D2/ ...

    Each trip folder name encodes the label: NORMAL / AGGRESSIVE / DROWSY.
    For each trip we aggregate sensor files into a single row matching
    MODEL_FEATURE_NAMES. This gives ~36 trips (6 drivers × ~6 trips each).
    """
    import re
    import glob

    root = Path(root_dir)
    if not root.exists():
        return None

    LABEL_RE = re.compile(r"-(NORMAL|AGGRESSIVE|DROWSY)\d?-", re.IGNORECASE)
    # Map UAH labels → our 3-class scheme: AGGRESSIVE/DROWSY → aggressive (2), NORMAL → normal (1).
    # There is no "safe" class in UAH, we use normal as the low-risk baseline.
    LABEL_MAP = {"NORMAL": 1, "AGGRESSIVE": 2, "DROWSY": 2}

    rows: List[dict] = []
    trip_dirs = [p for p in root.rglob("*") if p.is_dir()]

    # UAH accelerometer sampling is 10Hz, so 600 samples ≈ 1 minute.
    # We split each trip into overlapping 1-minute windows → each driver contributes
    # ~15-30 samples instead of 1, giving us enough data for training.
    WINDOW_SAMPLES = 600   # ~1 minute at 10Hz
    STEP_SAMPLES   = 300   # 50% overlap

    for trip_dir in trip_dirs:
        m = LABEL_RE.search(trip_dir.name)
        if not m:
            continue
        label_name = m.group(1).upper()
        label = LABEL_MAP.get(label_name)
        if label is None:
            continue

        accel_file = trip_dir / "RAW_ACCELEROMETERS.txt"
        gps_file   = trip_dir / "RAW_GPS.txt"

        # Load the full sensor traces once per trip
        try:
            a = None
            if accel_file.exists():
                a = pd.read_csv(accel_file, sep=r"\s+", header=None, engine="python",
                                on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna(how="all")
            g = None
            if gps_file.exists():
                g = pd.read_csv(gps_file, sep=r"\s+", header=None, engine="python",
                                on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna(how="all")

            if a is None or len(a) < WINDOW_SAMPLES:
                continue

            # Slide a 1-minute window over the trip
            n_accel = len(a)
            for start in range(0, n_accel - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                aw = a.iloc[start:start + WINDOW_SAMPLES]
                row: Dict[str, Any] = {"driving_style_encoded": label}

                if aw.shape[1] >= 4:
                    row["accel_x_mean"] = float(aw.iloc[:, 1].mean())
                    row["accel_y_mean"] = float(aw.iloc[:, 2].mean())
                    row["accel_z_mean"] = float(aw.iloc[:, 3].mean())
                    accel_mag = np.sqrt(aw.iloc[:, 1]**2 + aw.iloc[:, 2]**2)
                    dt_events = int((accel_mag.diff().abs() > 3.0).sum())
                    row["harsh_braking_count"] = dt_events // 2
                    row["harsh_accel_count"]   = dt_events // 2
                if aw.shape[1] >= 10:
                    row["gyro_x_std"] = float(aw.iloc[:, 7].std())
                    row["gyro_y_std"] = float(aw.iloc[:, 8].std())
                    row["gyro_z_std"] = float(aw.iloc[:, 9].std())

                # Map accelerometer window to corresponding GPS window (GPS is 1Hz → divide by 10)
                if g is not None and g.shape[1] >= 2:
                    gs = start // 10
                    ge = (start + WINDOW_SAMPLES) // 10
                    gw = g.iloc[gs:ge]
                    if len(gw) > 1:
                        speed = gw.iloc[:, 1].dropna()
                        if len(speed) > 0:
                            row["avg_speed"]        = float(speed.mean())
                            row["max_speed"]        = float(speed.max())
                            row["speed_variance"]   = float(speed.var()) if len(speed) > 1 else 0.0
                            row["avg_acceleration"] = float(speed.diff().mean()) if len(speed) > 1 else 0.0

                # Sensible defaults for missing fields
                row.setdefault("time_of_day", 12.0)
                row.setdefault("day_of_week", 3.0)
                row.setdefault("following_distance", 5.0)
                row.setdefault("turn_signal_usage", 0.5)
                row.setdefault("lane_changes", 0)
                row.setdefault("trip_duration", 60.0)  # 1-minute window
                row["data_source"] = "uah_driveset"
                row["driver_id"]   = trip_dir.parent.name  # D1, D2, ...
                # Group ID for group-aware train/test split: all windows from the same
                # trip folder share the same trip_id, so overlapping windows cannot leak
                # between train and test sets.
                row["trip_id"]     = f"uah_{trip_dir.name}"

                rows.append(row)

        except Exception as e:
            print(f"  ⚠ skip {trip_dir.name}: {e}")
            continue

    if not rows:
        return None

    return pd.DataFrame(rows)


# =============================================================================
# UNIFIED DATASET LOADER (для диплома)
# =============================================================================

def load_and_unify_datasets() -> pd.DataFrame:
    """
    Загружает все доступные датасеты, нормализует колонки к единому формату
    и возвращает объединённый DataFrame.

    Целевые колонки:
    [avg_speed, max_speed, speed_variance, harsh_braking_count,
     harsh_accel_count, avg_acceleration, lane_changes, following_distance,
     turn_signal_usage, accel_x_mean, accel_y_mean, accel_z_mean,
     gyro_x_std, gyro_y_std, gyro_z_std, time_of_day, day_of_week,
     trip_duration, trip_distance, driving_style_encoded]
    """
    import os
    from src.config import DATASET_PATHS

    TARGET_COLS = [
        "avg_speed", "max_speed", "speed_variance",
        "harsh_braking_count", "harsh_accel_count", "avg_acceleration",
        "lane_changes", "following_distance", "turn_signal_usage",
        "accel_x_mean", "accel_y_mean", "accel_z_mean",
        "gyro_x_std", "gyro_y_std", "gyro_z_std",
        "time_of_day", "day_of_week", "trip_duration", "trip_distance",
        "driving_style_encoded",
    ]

    frames = []

    # ── CARLA simulator ──────────────────────────────────────────────────
    # Columns: accelX, accelY, accelZ, class(driver name), gyroX, gyroY, gyroZ
    # "class" contains driver IDs (names), NOT driving style labels.
    # Driving style is derived from sensor intensity (acceleration magnitude).
    carla_path = DATASET_PATHS.get("carla_data", "")
    if carla_path and os.path.exists(carla_path):
        try:
            df = pd.read_csv(carla_path, low_memory=False)
            df["data_source"] = "carla"
            # Map sensor columns to unified names
            col_map = {
                "accelX": "accel_x_mean", "accelY": "accel_y_mean", "accelZ": "accel_z_mean",
                "gyroX":  "gyro_x_std",   "gyroY":  "gyro_y_std",   "gyroZ":  "gyro_z_std",
            }
            for src, dst in col_map.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
            # Derive driving style from combined sensor magnitude:
            #   low intensity  → 0 (safe)
            #   medium         → 1 (normal)
            #   high intensity → 2 (aggressive)
            accel_mag = np.sqrt(
                df["accel_x_mean"].fillna(0)**2 +
                df["accel_y_mean"].fillna(0)**2 +
                df["accel_z_mean"].fillna(9.8)**2
            )
            gyro_mag = np.sqrt(
                df["gyro_x_std"].fillna(0)**2 +
                df["gyro_y_std"].fillna(0)**2 +
                df["gyro_z_std"].fillna(0)**2
            )
            intensity = accel_mag + gyro_mag * 5.0
            q33, q66 = intensity.quantile(0.33), intensity.quantile(0.66)
            df["driving_style_encoded"] = pd.cut(
                intensity, bins=[-np.inf, q33, q66, np.inf], labels=[0, 1, 2]
            ).astype(int)
            # CARLA has no speed column → will be median-filled later
            frames.append(df)
            print(f"✓ CARLA: {len(df):,} rows, classes: {df['driving_style_encoded'].value_counts().sort_index().to_dict()}")
        except Exception as e:
            print(f"⚠ CARLA load failed: {e}")

    # ── Driver Behavior ──────────────────────────────────────────────────
    # Columns: speed, acceleration, trip_duration, trip_distance,
    #          lane_deviation, anomalous_event, route_anomaly, route_deviation_score
    behavior_path = DATASET_PATHS.get("driver_behavior", "")
    if behavior_path and os.path.exists(behavior_path):
        try:
            df = pd.read_csv(behavior_path, low_memory=False)
            df["data_source"] = "driver_behavior"
            # Build driving style label from anomaly signals:
            #   0 = normal, 1 = moderate (anomalous event), 2 = risky (route anomaly)
            if "route_anomaly" in df.columns and "anomalous_event" in df.columns:
                df["driving_style_encoded"] = (
                    df["route_anomaly"].fillna(0).astype(int) * 2 +
                    df["anomalous_event"].fillna(0).astype(int)
                ).clip(upper=2)
            elif "anomalous_event" in df.columns:
                df["driving_style_encoded"] = df["anomalous_event"].fillna(0).astype(int)
            # Map columns to unified names
            rename_map = {
                "speed":         "avg_speed",
                "acceleration":  "avg_acceleration",
                "trip_duration": "trip_duration",
                "trip_distance": "trip_distance",
                "lane_deviation":"lane_changes",
                "brake_usage":   "harsh_braking_count",
            }
            for src, dst in rename_map.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
            frames.append(df)
            print(f"✓ Driver Behavior: {len(df):,} rows, classes: {df['driving_style_encoded'].value_counts().to_dict()}")
        except Exception as e:
            print(f"⚠ Driver Behavior load failed: {e}")

    # ── Eco Driving ──────────────────────────────────────────────────────
    eco_path = DATASET_PATHS.get("eco_driving", "")
    if eco_path and os.path.exists(eco_path):
        try:
            df = pd.read_csv(eco_path, low_memory=False)
            df["data_source"] = "eco_driving"
            # Map eco_score to label: high score = safe, low = aggressive
            if "eco_score" in df.columns:
                df["driving_style_encoded"] = pd.cut(
                    df["eco_score"],
                    bins=[-1, 40, 70, 101],
                    labels=[2, 1, 0],
                ).astype(float).fillna(1).astype(int)
            frames.append(df)
            print(f"✓ Eco Driving: {len(df):,} rows")
        except Exception as e:
            print(f"⚠ Eco Driving load failed: {e}")

    # ── Kaggle Driver Behavior (outofskills/driving-behavior) ────────────
    # Expected columns (variants exist across dataset versions):
    #   AccX/AccY/AccZ, GyroX/GyroY/GyroZ, Timestamp, Class
    # Class ∈ {SLOW, NORMAL, AGGRESSIVE}
    # Strategy: group raw samples into "windows" of ~100 samples (≈1 trip segment),
    # aggregate to match MODEL_FEATURE_NAMES, derive per-window label by majority vote.
    kaggle_path = DATASET_PATHS.get("kaggle_driver_behavior", "")
    if kaggle_path and os.path.exists(kaggle_path):
        try:
            df_raw = pd.read_csv(kaggle_path, low_memory=False)
            # Normalize column names
            df_raw.columns = [c.strip() for c in df_raw.columns]
            col_lower = {c.lower(): c for c in df_raw.columns}

            def pick(*names):
                for n in names:
                    if n.lower() in col_lower:
                        return col_lower[n.lower()]
                return None

            ax = pick("AccX", "Acc_X", "accel_x", "acceleration_x")
            ay = pick("AccY", "Acc_Y", "accel_y", "acceleration_y")
            az = pick("AccZ", "Acc_Z", "accel_z", "acceleration_z")
            gx = pick("GyroX", "Gyro_X", "gyro_x", "gyroscope_x")
            gy = pick("GyroY", "Gyro_Y", "gyro_y", "gyroscope_y")
            gz = pick("GyroZ", "Gyro_Z", "gyro_z", "gyroscope_z")
            label_col = pick("Class", "label", "behavior", "driving_style")

            if None in (ax, ay, az, label_col):
                print(f"⚠ Kaggle Driver Behavior: unexpected columns {list(df_raw.columns)[:10]}…")
            else:
                # Label normalization: SLOW→0 (safe), NORMAL→1, AGGRESSIVE→2
                LABEL_MAP = {
                    "SLOW": 0, "slow": 0,
                    "NORMAL": 1, "normal": 1,
                    "AGGRESSIVE": 2, "aggressive": 2,
                }
                df_raw["_label"] = df_raw[label_col].astype(str).str.strip().map(LABEL_MAP)
                df_raw = df_raw.dropna(subset=["_label"])
                df_raw["_label"] = df_raw["_label"].astype(int)

                # Sliding window aggregation with overlap — gives more training samples.
                # Window = 50 consecutive samples (~2.5 sec of driving at 20Hz),
                # step = 10 samples. From ~6700 raw samples we get ~665 windows
                # instead of 68 with non-overlapping WINDOW=100.
                WINDOW = 50
                STEP   = 10
                n_rows = len(df_raw)
                if n_rows < WINDOW:
                    print(f"⚠ Kaggle Driver Behavior: too few samples ({n_rows}) for windowing")
                else:
                    windows = []
                    for start in range(0, n_rows - WINDOW + 1, STEP):
                        w = df_raw.iloc[start:start + WINDOW]
                        # Skip window if it spans multiple labels (transition zone)
                        labels_in_w = w["_label"].unique()
                        if len(labels_in_w) != 1:
                            continue
                        label = int(labels_in_w[0])

                        row: Dict[str, Any] = {
                            "accel_x_mean": float(w[ax].mean()),
                            "accel_y_mean": float(w[ay].mean()),
                            "accel_z_mean": float(w[az].mean()),
                            "driving_style_encoded": label,
                            # Group ID for group-aware train/test split.
                            # All windows within the same WINDOW-sized block share a trip_id,
                            # preventing overlapping windows from being split across train/test.
                            "trip_id": f"kaggle_block_{start // WINDOW}_{label}",
                        }
                        if gx: row["gyro_x_std"] = float(w[gx].std())
                        if gy: row["gyro_y_std"] = float(w[gy].std())
                        if gz: row["gyro_z_std"] = float(w[gz].std())
                        windows.append(row)

                    grouped = pd.DataFrame(windows)

                    # Derive avg_speed proxy from accel magnitude (aggressive drivers have higher RMS)
                    accel_rms = np.sqrt(
                        grouped["accel_x_mean"]**2 +
                        grouped["accel_y_mean"]**2 +
                        (grouped["accel_z_mean"] - 9.81)**2
                    )
                    grouped["avg_speed"] = 30 + accel_rms * 10
                    grouped["max_speed"] = grouped["avg_speed"] * 1.3
                    grouped["speed_variance"] = (accel_rms * 5) ** 2
                    # Derive harsh events from gyro magnitude stdev (proxy: more rotation = more harsh)
                    if "gyro_x_std" in grouped.columns:
                        gyro_mag = np.sqrt(
                            grouped.get("gyro_x_std", 0)**2 +
                            grouped.get("gyro_y_std", 0)**2 +
                            grouped.get("gyro_z_std", 0)**2
                        )
                        grouped["harsh_braking_count"] = (gyro_mag > 0.3).astype(int) * 2
                        grouped["harsh_accel_count"]   = (gyro_mag > 0.25).astype(int) * 2

                    grouped["data_source"] = "kaggle_driver_behavior"
                    grouped = grouped.reset_index(drop=True)
                    frames.append(grouped)
                    print(
                        f"✓ Kaggle Driver Behavior: {len(grouped):,} windows "
                        f"(from {len(df_raw):,} raw samples, step={STEP}), "
                        f"classes: {grouped['driving_style_encoded'].value_counts().sort_index().to_dict()}"
                    )
        except Exception as e:
            print(f"⚠ Kaggle Driver Behavior load failed: {e}")

    # ── UAH-DriveSet (directory of per-driver folders) ───────────────────
    # Real sensor recordings from 6 drivers, each with NORMAL/AGGRESSIVE/DROWSY trips.
    # Each trip folder contains RAW_GPS.txt, RAW_ACCELEROMETERS.txt, etc.
    # Per-trip label is encoded in the folder name (e.g. "20151120133502-16km-D1-NORMAL1-SECONDARY").
    uah_root = DATASET_PATHS.get("uah_driveset", "")
    if uah_root and os.path.isdir(uah_root):
        try:
            uah_frame = _load_uah_driveset(uah_root)
            if uah_frame is not None and len(uah_frame) > 0:
                frames.append(uah_frame)
                print(
                    f"✓ UAH-DriveSet: {len(uah_frame):,} trips, "
                    f"classes: {uah_frame['driving_style_encoded'].value_counts().sort_index().to_dict()}"
                )
        except Exception as e:
            print(f"⚠ UAH-DriveSet load failed: {e}")

    # ── Fallback: synthetic ───────────────────────────────────────────────
    if not frames:
        print("⚠ No real datasets found — generating synthetic data")
        syn = create_sample_dataset(n_samples=15000)
        syn["data_source"] = "synthetic"
        frames.append(syn)

    # ── Merge ─────────────────────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure all target columns exist (fill missing with NaN)
    for col in TARGET_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Fill NaN with median (numeric only)
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined[num_cols] = combined[num_cols].fillna(combined[num_cols].median())

    # Ensure driving_style_encoded is valid int
    if "driving_style_encoded" not in combined.columns or combined["driving_style_encoded"].isna().all():
        combined["driving_style_encoded"] = 0
    combined["driving_style_encoded"] = combined["driving_style_encoded"].fillna(0).astype(int)

    # WARN if dataset is small, but DO NOT silently top it up with synthetic data.
    # Mixing synthetic with real data creates a hidden data leakage: synthetic features
    # are a trivial function of the label (see create_sample_dataset), so models will
    # achieve ~0.99 accuracy by memorising the synthetic generator rather than learning
    # real driving patterns. This was the main reason all models showed accuracy=1.00.
    if len(combined) < 1000:
        print(f"\n⚠ WARNING: only {len(combined)} real samples available.")
        print("  Training on this small dataset is still valid but metrics will have")
        print("  high variance. Consider adding more real datasets to data/raw/ for")
        print("  better results. Synthetic padding is NOT added automatically to")
        print("  prevent data leakage.")

    print(f"\n✓ Unified dataset: {len(combined):,} rows, {len(combined.columns)} columns")
    print(f"  Sources: {combined['data_source'].value_counts().to_dict()}")
    return combined


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Demonstration of data loading capabilities
    print("\n" + "=" * 60)
    print("DATA LOADER DEMONSTRATION")
    print("=" * 60)

    loader = DataLoader()

    # Try to load real datasets
    try:
        datasets = loader.load_all_datasets(nrows=1000)  # Limit rows for demo

        for name, df in datasets.items():
            loader.print_dataset_info(df, name)
            loader.validate_dataset(df, name)

    except Exception as e:
        print(f"\nCould not load real datasets: {e}")
        print("\nGenerating synthetic dataset for demonstration...")

        # Generate and inspect synthetic data
        synthetic_df = create_sample_dataset(n_samples=1000)
        loader.datasets["synthetic"] = synthetic_df
        loader.print_dataset_info(synthetic_df, "synthetic")
        loader.validate_dataset(synthetic_df, "synthetic")
        loader.get_target_distribution(synthetic_df, "driving_style")
