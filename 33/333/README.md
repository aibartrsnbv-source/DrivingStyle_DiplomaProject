# Driving Style Classification & Accident Risk Prediction

## Bachelor Diploma Project

**Title:** Developing Machine Learning Algorithms to Assess Driving Style and Predict Accident Risks

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Module Descriptions](#module-descriptions)
6. [Machine Learning Models](#machine-learning-models)
7. [Risk Scoring System](#risk-scoring-system)
8. [Results and Evaluation](#results-and-evaluation)
9. [Academic Context](#academic-context)

---

## Project Overview

This project implements a comprehensive machine learning pipeline for analyzing driving behavior, classifying driving styles, and predicting accident risks. The system processes driving data (sensor readings, behavioral metrics) to provide actionable insights for road safety improvement.

### Key Features

- **Multi-model Approach:** Implements both classical ML (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) and Deep Learning (MLP, LSTM) models
- **Comprehensive Preprocessing:** Handles missing values, class imbalance, feature scaling, and engineering
- **Risk Scoring System:** Converts predictions into interpretable risk scores (0-1 scale)
- **Extensive Evaluation:** Provides accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- **Modular Design:** Clean, maintainable code suitable for academic presentation

### Driving Style Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Safe | Consistent with traffic rules, smooth driving |
| 1 | Normal | Average driving behavior |
| 2 | Aggressive | Frequent harsh braking, speeding |
| 3 | Risky | Dangerous patterns, high accident probability |

---

## Project Structure

```
driving_style_ml/
│
├── data/
│   ├── raw/                    # Raw dataset storage
│   └── processed/              # Preprocessed data
│
├── src/
│   ├── config.py               # Global settings & hyperparameters
│   ├── data_loader.py          # Data loading & validation
│   ├── preprocessing.py        # Cleaning & feature engineering
│   ├── eda.py                  # Exploratory data analysis
│   ├── models.py               # ML & DL model implementations
│   ├── train.py                # Training logic & model persistence
│   ├── evaluate.py             # Metrics & evaluation
│   ├── risk_scoring.py         # Accident risk scoring system
│   └── utils.py                # Helper utilities
│
├── models/                     # Saved trained models
├── outputs/
│   ├── figures/                # Generated visualizations
│   ├── reports/                # Evaluation reports
│   └── experiments/            # Experiment tracking logs
│
├── main.py                     # Pipeline entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project:**
   ```bash
   cd /Users/medetlatip/driving_style_ml
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; import sklearn; print('Installation successful!')"
   ```

---

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py
```

### Command Line Options

```bash
# Use synthetic data (for testing without real datasets)
python main.py --synthetic

# Skip deep learning models (faster execution)
python main.py --no-dl

# Quick mode (reduced data, simplified analysis)
python main.py --quick

# Print configuration summary
python main.py --config

# Set custom random seed
python main.py --seed 123
```

### Running Individual Modules

```python
# Data Loading
from src.data_loader import DataLoader, create_sample_dataset
loader = DataLoader()
df = create_sample_dataset(n_samples=1000)

# Preprocessing
from src.preprocessing import preprocess_pipeline
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_pipeline(
    df, target_column="driving_style_encoded"
)

# Training
from src.train import ModelTrainer
trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train, X_val, y_val)

# Evaluation
from src.evaluate import evaluate_all_models
evaluator, results = evaluate_all_models(trainer.models, X_test, y_test)

# Risk Scoring
from src.risk_scoring import RiskScorer
scorer = RiskScorer()
risk_scores = scorer.compute_risk_score(best_model, X_test)
```

---

## Module Descriptions

### config.py
Central configuration file containing:
- Dataset paths
- Random seeds for reproducibility
- Model hyperparameters
- Preprocessing settings
- Risk scoring thresholds

### data_loader.py
Handles data ingestion:
- CSV file loading with validation
- Dataset inspection and statistics
- Support for multiple data sources
- Synthetic data generation for testing

### preprocessing.py
Data preparation pipeline:
- Missing value imputation (mean, median, KNN)
- Feature scaling (Standard, MinMax, Robust)
- Label encoding
- Feature engineering for time-series data
- Outlier detection and handling

### eda.py
Exploratory data analysis:
- Summary statistics
- Class distribution analysis
- Correlation heatmaps
- Feature distribution plots
- Missing data visualization

### models.py
Model implementations:
- **Classical ML:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Deep Learning:** MLP (tabular), LSTM (time-series)
- Sklearn-compatible PyTorch wrapper

### train.py
Training pipeline:
- Cross-validation
- Class imbalance handling (SMOTE, class weights)
- Model persistence
- Training history tracking

### evaluate.py
Model evaluation:
- Classification metrics (Accuracy, Precision, Recall, F1)
- ROC-AUC curves
- Confusion matrices
- Model comparison visualizations

### risk_scoring.py
Risk assessment system:
- Probability-based risk scores (0-1)
- Risk level classification (Low, Medium, High, Critical)
- Safety recommendations
- Risk distribution visualization

### utils.py
Helper functions:
- Logging setup
- Timing decorators
- Memory optimization
- Experiment tracking

---

## Machine Learning Models

### Classical Models

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| Logistic Regression | Linear classifier with regularization | C=1.0, max_iter=1000 |
| Random Forest | Ensemble of decision trees | n_estimators=200, max_depth=15 |
| Gradient Boosting | Sequential boosting | n_estimators=200, learning_rate=0.1 |
| XGBoost | Optimized gradient boosting | n_estimators=200, max_depth=6 |

### Deep Learning Models

| Model | Architecture | Use Case |
|-------|--------------|----------|
| MLP | 256→128→64→32 hidden layers | Tabular driving data |
| LSTM | 2-layer bidirectional | Time-series sensor data |

### Hyperparameter Configuration

All hyperparameters are centralized in `src/config.py` and can be modified:

```python
ML_MODELS_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "class_weight": "balanced",
    },
    # ... other models
}

DL_CONFIG = {
    "mlp": {
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rate": 0.3,
    },
    "training": {
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
    },
}
```

---

## Risk Scoring System

### Score Interpretation

| Score Range | Risk Level | Description |
|-------------|------------|-------------|
| 0.0 - 0.3 | LOW | Safe driving behavior |
| 0.3 - 0.6 | MEDIUM | Moderate concern |
| 0.6 - 0.8 | HIGH | Significant risk |
| 0.8 - 1.0 | CRITICAL | Immediate attention needed |

### Contributing Factors

The risk score is influenced by:
- Speed violations (25%)
- Harsh braking (20%)
- Harsh acceleration (15%)
- Aggressive turns (15%)
- Following distance (10%)
- Lane discipline (10%)
- Fatigue indicators (5%)

---

## Results and Evaluation

### Output Files

After running the pipeline, the following outputs are generated:

```
outputs/
├── figures/
│   ├── class_distribution_*.png
│   ├── correlation_heatmap_*.png
│   ├── feature_distributions.png
│   ├── confusion_matrix_*.png
│   ├── roc_curves_comparison.png
│   ├── metrics_comparison.png
│   └── risk_score_distribution.png
├── reports/
│   └── evaluation_report.txt
└── pipeline_results.json
```

### Expected Performance

Typical results on driving behavior datasets:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.85-0.92 | 0.84-0.91 | 0.90-0.96 |
| XGBoost | 0.86-0.93 | 0.85-0.92 | 0.91-0.97 |
| MLP (PyTorch) | 0.83-0.90 | 0.82-0.89 | 0.88-0.95 |

---

## Academic Context

### Thesis Chapter: Software Implementation

This codebase is designed to support Chapter 3/4 of your thesis: "Software Implementation and Machine Learning Models."

Key points for documentation:
1. **Modular Architecture:** Each module has a single responsibility
2. **Reproducibility:** Fixed random seeds, experiment tracking
3. **Professional Code Quality:** Docstrings, type hints, logging
4. **Comprehensive Evaluation:** Multiple metrics, visualizations

### Citation

If using this code in academic work:

```
[Your Name]. (2024). Driving Style Classification and Accident Risk Prediction
Using Machine Learning. Bachelor's Thesis, [University Name].
```

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'torch'**
```bash
pip install torch
```

**Memory Error on large datasets**
```bash
python main.py --quick  # Use reduced data mode
```

**No datasets found**
```bash
python main.py --synthetic  # Use synthetic data
```

### Support

For issues or questions:
1. Check the log file: `outputs/pipeline.log`
2. Review configuration: `python main.py --config`
3. Run with synthetic data first to verify installation

---

## License

This project is developed for academic purposes as part of a Bachelor's diploma project.

---

**Author:** [Your Name]
**Supervisor:** [Supervisor Name]
**Institution:** [University Name]
**Date:** 2024
