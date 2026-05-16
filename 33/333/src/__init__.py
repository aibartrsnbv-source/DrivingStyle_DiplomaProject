"""
Driving Style ML Source Package
================================

This package contains all modules for the driving style classification
and accident risk prediction machine learning pipeline.

Modules:
    config: Global settings and hyperparameters
    data_loader: Data loading and validation
    preprocessing: Data cleaning and feature engineering
    eda: Exploratory data analysis
    models: Machine learning and deep learning models
    train: Model training logic
    evaluate: Model evaluation and metrics
    risk_scoring: Accident risk scoring system
    utils: Helper utilities

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

from src.config import (
    RANDOM_SEED,
    DATASET_PATHS,
    ML_MODELS_CONFIG,
    DL_CONFIG,
)

from src.data_loader import (
    DataLoader,
    create_sample_dataset,
)

from src.preprocessing import (
    DataPreprocessor,
    preprocess_pipeline,
    handle_missing_values,
    scale_features,
    encode_labels,
)

from src.eda import (
    ExploratoryDataAnalysis,
    quick_eda,
)

from src.models import (
    ModelFactory,
    MLP,
    LSTM,
    PyTorchClassifier,
    create_mlp_classifier,
    create_lstm_classifier,
)

from src.train import (
    ModelTrainer,
    train_and_evaluate_pipeline,
)

from src.evaluate import (
    ModelEvaluator,
    evaluate_all_models,
    quick_evaluate,
)

from src.risk_scoring import (
    RiskScorer,
    RiskLevel,
    RiskAssessment,
    print_risk_interpretation_guide,
)

from src.utils import (
    setup_logger,
    timer,
    Timer,
    set_random_seeds,
    ExperimentTracker,
)

__version__ = "1.0.0"
__author__ = "[Your Name]"

__all__ = [
    # Config
    "RANDOM_SEED",
    "DATASET_PATHS",
    "ML_MODELS_CONFIG",
    "DL_CONFIG",
    # Data Loading
    "DataLoader",
    "create_sample_dataset",
    # Preprocessing
    "DataPreprocessor",
    "preprocess_pipeline",
    "handle_missing_values",
    "scale_features",
    "encode_labels",
    # EDA
    "ExploratoryDataAnalysis",
    "quick_eda",
    # Models
    "ModelFactory",
    "MLP",
    "LSTM",
    "PyTorchClassifier",
    "create_mlp_classifier",
    "create_lstm_classifier",
    # Training
    "ModelTrainer",
    "train_and_evaluate_pipeline",
    # Evaluation
    "ModelEvaluator",
    "evaluate_all_models",
    "quick_evaluate",
    # Risk Scoring
    "RiskScorer",
    "RiskLevel",
    "RiskAssessment",
    "print_risk_interpretation_guide",
    # Utils
    "setup_logger",
    "timer",
    "Timer",
    "set_random_seeds",
    "ExperimentTracker",
]
