"""
Configuration Module for Driving Style ML Project
==================================================

This module contains all global settings, hyperparameters, and file paths
used throughout the machine learning pipeline.

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import os
from pathlib import Path

# Home directory — работает на любой машине
_HOME = os.path.expanduser("~")

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATASET PATHS
# =============================================================================
# Strategy: look in data/raw/ first (project-local, universal),
# fall back to ~/Downloads/ (author's local machine).
# This lets any teammate just drop CSVs into data/raw/ without editing code.


def _find_dataset(filename: str, home_fallback: str) -> str:
    """Return the first existing path: data/raw/<file> then ~/Downloads/<file>."""
    local = RAW_DATA_DIR / filename
    if local.exists():
        return str(local)
    fallback = Path(_HOME) / "Downloads" / home_fallback
    return str(fallback)


DATASET_PATHS = {
    # US Accidents dataset - contains accident records with features
    "us_accidents": _find_dataset(
        "US_Accidents_March23.csv", "US_Accidents_March23.csv",
    ),
    # CARLA simulator driving data - contains simulated driving behavior
    "carla_data": _find_dataset(
        "full_data_carla.csv", "full_data_carla.csv",
    ),
    # Eco driving score dataset - contains driving efficiency metrics
    "eco_driving": _find_dataset(
        "eco_driving_score.csv", "eco_driving_score.csv",
    ),
    # Driver behavior dataset - route and behavior anomalies
    "driver_behavior": _find_dataset(
        "driver_behavior_route_anomaly_dataset_with_derived_features.csv",
        "driver_behavior_route_anomaly_dataset_with_derived_features.csv",
    ),
    # Kaggle Driver Behavior (outofskills) - motion sensor data with labels
    "kaggle_driver_behavior": _find_dataset(
        "kaggle_driver_behavior.csv", "kaggle_driver_behavior.csv",
    ),
    # UAH-DriveSet root directory (sensor traces from real drivers)
    # UAH is a directory, not a CSV — handled separately in loader.
    "uah_driveset": _find_dataset(
        "UAH-DRIVESET", "UAH-DRIVESET",
    ),
}

ARCHIVE_DIRS = {
    "archive_7":          str(Path(_HOME) / "Downloads" / "archive-7"),
    "archive_8":          str(Path(_HOME) / "Downloads" / "archive-8"),
    "drowsiness_dataset": str(Path(_HOME) / "Downloads" / "Driver Drowsiness Dataset (DDD)"),
}

# =============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42
NUMPY_SEED = 42
TORCH_SEED = 42

# =============================================================================
# DATA SPLITTING CONFIGURATION
# =============================================================================

TEST_SIZE = 0.2          # 20% for testing
VALIDATION_SIZE = 0.15   # 15% of training data for validation
STRATIFY = True          # Use stratified splitting for imbalanced classes

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

PREPROCESSING_CONFIG = {
    # Missing value handling
    "missing_value_strategy": "median",  # Options: 'mean', 'median', 'mode', 'drop', 'knn'
    "missing_threshold": 0.5,            # Drop columns with >50% missing values

    # Feature scaling
    "scaling_method": "standard",        # Options: 'standard', 'minmax', 'robust'

    # Outlier handling
    "outlier_method": "iqr",             # Options: 'iqr', 'zscore', 'isolation_forest'
    "outlier_threshold": 1.5,            # IQR multiplier or z-score threshold

    # Time-series feature engineering
    "rolling_window_sizes": [5, 10, 20],  # Window sizes for rolling statistics
    "use_fft_features": True,             # Extract frequency domain features
}

# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================

IMBALANCE_CONFIG = {
    "method": "smote",                   # Options: 'smote', 'adasyn', 'random_oversample',
                                         #          'random_undersample', 'class_weight'
    "sampling_strategy": "auto",         # 'auto' or specific ratio
    "imbalance_threshold": 0.4,          # Trigger resampling if minority/majority < 40%.
                                         # Was 0.3, but our Safe class is 15% (ratio 0.335)
                                         # → it was just above the threshold, skipping SMOTE.
                                         # Raising to 0.4 ensures SMOTE kicks in and
                                         # improves recall on the Safe class.
}

# =============================================================================
# MACHINE LEARNING MODEL HYPERPARAMETERS
# =============================================================================

ML_MODELS_CONFIG = {
    "logistic_regression": {
        "C": 0.5,                           # stronger L2 regularization for small data
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": "balanced",         # compensates 46/39/15 class ratio
        "random_state": RANDOM_SEED,
    },

    "random_forest": {
        "n_estimators": 300,                # more trees for stability on small data
        "max_depth": 10,                    # was 15 — reduced to prevent overfitting on 1.6k rows
        "min_samples_split": 8,             # was 5 — stricter splits
        "min_samples_leaf": 4,              # was 2 — stricter leaf size
        "max_features": "sqrt",             # default for classification, explicit
        "class_weight": "balanced_subsample", # re-weighs each bootstrap sample
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },

    "gradient_boosting": {
        # Optuna v3 (TPE, 50 trials, GroupKFold by trip_id, f1_macro)
        # Honest test F1-macro: 0.5723 (was 0.5652 baseline, +0.7%)
        # Feature engineering v3: 7 interactions, dropped 8 dup features
        "n_estimators": 250,
        "learning_rate": 0.1355,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "subsample": 0.7579,
        "max_features": "log2",
        "random_state": RANDOM_SEED,
    },

    "xgboost": {
        # Optuna v3 (TPE, 50 trials, GroupKFold by trip_id, f1_macro)
        # Honest test F1-macro: 0.6298 (was 0.5858 baseline, +4.4%)
        # Best overall model — XGBoost wins on V3 feature set
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1650,
        "subsample": 0.7725,
        "colsample_bytree": 0.8750,
        "min_child_weight": 3,
        "reg_lambda": 0.6490,
        "random_state": RANDOM_SEED,
        "eval_metric": "mlogloss",
    },
}

# =============================================================================
# DEEP LEARNING MODEL HYPERPARAMETERS (PyTorch)
# =============================================================================

DL_CONFIG = {
    # MLP Configuration for tabular data
    "mlp": {
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rate": 0.3,
        "activation": "relu",
        "batch_norm": True,
    },

    # LSTM Configuration for time-series data
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "sequence_length": 50,
    },

    # Training parameters
    "training": {
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10,
        "lr_scheduler_patience": 5,
        "lr_scheduler_factor": 0.5,
    },
}

# =============================================================================
# FEATURE COLUMNS CONFIGURATION
# =============================================================================

# Expected feature groups (adjust based on your actual data)
FEATURE_GROUPS = {
    # Sensor-based features (from smartphone/vehicle sensors)
    "sensor_features": [
        "acceleration_x", "acceleration_y", "acceleration_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z",
        "speed", "bearing", "accuracy",
    ],

    # Derived driving behavior features
    "behavior_features": [
        "harsh_braking_count", "harsh_acceleration_count",
        "speeding_duration", "lane_change_frequency",
        "following_distance", "turn_signal_usage",
    ],

    # Environmental/contextual features
    "context_features": [
        "time_of_day", "day_of_week", "weather_condition",
        "road_type", "traffic_density",
    ],

    # Target variables
    "target_columns": {
        "driving_style": "driving_style",      # safe/aggressive/risky
        "accident_label": "accident",          # 0/1 binary
        "risk_score": "risk_score",            # continuous 0-1
    },
}

# =============================================================================
# DRIVING STYLE CLASSES
# =============================================================================

DRIVING_STYLE_CLASSES = {
    0: "Safe",
    1: "Normal",
    2: "Aggressive",
    3: "Risky",
}

# Mapping for different label formats
STYLE_LABEL_MAPPING = {
    "safe": 0,
    "normal": 1,
    "aggressive": 2,
    "risky": 3,
    "SAFE": 0,
    "NORMAL": 1,
    "AGGRESSIVE": 2,
    "RISKY": 3,
}

# =============================================================================
# RISK SCORING CONFIGURATION
# =============================================================================

RISK_SCORING_CONFIG = {
    # Risk level thresholds
    "thresholds": {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8,
    },

    # Feature weights for composite risk score
    "feature_weights": {
        "speed_violation": 0.25,
        "harsh_braking": 0.20,
        "harsh_acceleration": 0.15,
        "aggressive_turns": 0.15,
        "following_distance": 0.10,
        "lane_discipline": 0.10,
        "fatigue_indicators": 0.05,
    },
}

# =============================================================================
# EVALUATION METRICS
# =============================================================================

EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "confusion_matrix",
    "classification_report",
]

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": OUTPUT_DIR / "pipeline.log",
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-whitegrid",
    "color_palette": "husl",
    "save_format": "png",
}


def get_config_summary():
    """
    Print a summary of the current configuration.
    Useful for documentation and reproducibility.
    """
    print("=" * 60)
    print("DRIVING STYLE ML - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Validation Size: {VALIDATION_SIZE}")
    print(f"\nDataset Paths:")
    for name, path in DATASET_PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  [{exists}] {name}: {path}")
    print("\nML Models: " + ", ".join(ML_MODELS_CONFIG.keys()))
    print("DL Models: MLP, LSTM")
    print("=" * 60)


if __name__ == "__main__":
    get_config_summary()
