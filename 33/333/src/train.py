"""
Training Module for Driving Style ML Project
=============================================

This module handles the training of all machine learning and deep learning models,
including cross-validation, hyperparameter handling, and model persistence.

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    IMBALANCE_CONFIG,
    ML_MODELS_CONFIG,
    DL_CONFIG,
)
from src.models import (
    ModelFactory,
    PyTorchClassifier,
    MLP,
    LSTM,
    create_mlp_classifier,
    create_lstm_classifier,
    compute_class_weights,
)


class ModelTrainer:
    """
    A comprehensive class for training and managing ML models.

    This class provides methods for:
    - Training multiple ML models
    - Cross-validation
    - Handling class imbalance
    - Model persistence
    - Training history tracking

    Attributes:
        models (Dict): Dictionary of trained models
        training_results (Dict): Training metrics for each model
        best_model_name (str): Name of the best performing model
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        random_state: int = RANDOM_SEED,
    ):
        """
        Initialize the ModelTrainer.

        Parameters:
        -----------
        models_dir : Path, optional
            Directory to save trained models
        random_state : int
            Random seed for reproducibility
        """
        self.models_dir = models_dir or MODELS_DIR
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.is_fitted = False

    def handle_class_imbalance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        method: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data
        y : Series or ndarray
            Target labels
        method : str, optional
            Resampling method: 'smote', 'adasyn', 'random_oversample',
            'random_undersample'

        Returns:
        --------
        Tuple[ndarray, ndarray]
            Resampled X and y
        """
        method = method or IMBALANCE_CONFIG.get("method", "smote")

        print(f"\nHandling class imbalance using {method}...")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        unique, counts = np.unique(y, return_counts=True)
        min_ratio = counts.min() / counts.max()

        print(f"Original class distribution: {dict(zip(unique, counts))}")
        print(f"Imbalance ratio: {min_ratio:.3f}")

        if min_ratio > IMBALANCE_CONFIG.get("imbalance_threshold", 0.3):
            print("Class imbalance is acceptable. Skipping resampling.")
            return X, y

        if method == "smote":
            sampler = SMOTE(random_state=self.random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=self.random_state)
        elif method == "random_oversample":
            sampler = RandomOverSampler(random_state=self.random_state)
        elif method == "random_undersample":
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            print(f"Unknown method: {method}. Using SMOTE.")
            sampler = SMOTE(random_state=self.random_state)

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"Resampled class distribution: {dict(zip(unique_new, counts_new))}")
            print(f"Samples: {len(y)} -> {len(y_resampled)}")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"⚠ Resampling failed: {e}")
            print("Continuing with original data.")
            return X, y

    def train_classical_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        model_names: Optional[List[str]] = None,
        use_cv: bool = True,
        cv_folds: int = 5,
        handle_imbalance: bool = True,
    ) -> Dict[str, BaseEstimator]:
        """
        Train classical machine learning models.

        Parameters:
        -----------
        X_train : DataFrame or ndarray
            Training features
        y_train : Series or ndarray
            Training labels
        X_val : DataFrame or ndarray, optional
            Validation features
        y_val : Series or ndarray, optional
            Validation labels
        model_names : list, optional
            Names of models to train
        use_cv : bool
            Whether to use cross-validation
        cv_folds : int
            Number of CV folds
        handle_imbalance : bool
            Whether to handle class imbalance

        Returns:
        --------
        Dict[str, BaseEstimator]
            Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("TRAINING CLASSICAL ML MODELS")
        print("=" * 60)

        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
            X_train = X_train.values
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Handle class imbalance
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)

        if model_names is None:
            all_models = ModelFactory.get_all_models()
        else:
            all_models = {name: ModelFactory.get_model(name.lower().replace(" ", "_"))
                         for name in model_names}

        for model_name, model in all_models.items():
            print(f"\n{'─'*60}")
            print(f"Training: {model_name}")
            print("─" * 60)

            if use_cv:
                cv = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )

                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="accuracy", n_jobs=-1
                )

                print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

            # Train on full training data.
            # GradientBoostingClassifier and XGBClassifier do not support the
            # `class_weight` argument in __init__, so we pass per-sample weights
            # computed from the same "balanced" strategy sklearn uses internally.
            # This compensates for our 46/39/15 class imbalance.
            from sklearn.utils.class_weight import compute_sample_weight

            sw = compute_sample_weight(class_weight="balanced", y=y_train)

            model_cls = type(model).__name__
            try:
                if model_cls == "VotingClassifier":
                    # Pass sample_weight to each base estimator that accepts it.
                    # Sklearn routes fit_params like 'rf__sample_weight' to base learner 'rf'.
                    # We only pass it to gb and xgb — LogReg/RF use class_weight instead.
                    fit_params = {}
                    for name, est in model.estimators:
                        ename = type(est).__name__
                        if "GradientBoosting" in ename or ename == "XGBClassifier":
                            fit_params[f"{name}__sample_weight"] = sw
                    model.fit(X_train, y_train, **fit_params)
                elif "GradientBoosting" in model_cls or model_cls == "XGBClassifier":
                    model.fit(X_train, y_train, sample_weight=sw)
                else:
                    # LogReg, RF — use class_weight set in __init__, no sample_weight needed
                    model.fit(X_train, y_train)
            except TypeError as e:
                # Fallback for exotic estimators: fit without sample_weight
                print(f"  ⚠ sample_weight not accepted ({e}), fitting without it")
                model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            print(f"  Training Accuracy: {train_acc:.4f}")

            val_acc = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                val_acc = model.score(X_val, y_val)
                print(f"  Validation Accuracy: {val_acc:.4f}")

            self.models[model_name] = model
            self.training_results[model_name] = {
                "model_type": "classical",
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "cv_scores": cv_scores.tolist() if use_cv else None,
                "cv_mean": cv_scores.mean() if use_cv else None,
                "cv_std": cv_scores.std() if use_cv else None,
            }

            # Store feature importance if available
            if hasattr(model, "feature_importances_"):
                self.training_results[model_name]["feature_importances"] = {
                    feature_names[i]: float(imp)
                    for i, imp in enumerate(model.feature_importances_)
                }

        self.is_fitted = True
        print("\n✓ Classical model training complete")

        return {k: v for k, v in self.models.items()
                if self.training_results[k]["model_type"] == "classical"}

    def train_deep_learning_model(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        model_type: str = "mlp",
        handle_imbalance: bool = True,
        use_class_weights: bool = True,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> PyTorchClassifier:
        """
        Train a deep learning model.

        Parameters:
        -----------
        X_train : DataFrame or ndarray
            Training features
        y_train : Series or ndarray
            Training labels
        X_val : DataFrame or ndarray, optional
            Validation features
        y_val : Series or ndarray, optional
            Validation labels
        model_type : str
            Type of model: 'mlp' or 'lstm'
        handle_imbalance : bool
            Whether to handle class imbalance
        use_class_weights : bool
            Whether to use class weights
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size
        learning_rate : float, optional
            Learning rate

        Returns:
        --------
        PyTorchClassifier
            Trained model
        """
        print("\n" + "=" * 60)
        print(f"TRAINING DEEP LEARNING MODEL: {model_type.upper()}")
        print("=" * 60)

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Handle class imbalance (for non-DL specific handling)
        if handle_imbalance and not use_class_weights:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)

        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        print(f"\nInput features: {n_features}")
        print(f"Number of classes: {n_classes}")

        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_train)
            print(f"Class weights: {class_weights.numpy().round(3)}")

        if model_type == "mlp":
            classifier = create_mlp_classifier(
                input_dim=n_features,
                num_classes=n_classes,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            model_name = "MLP (PyTorch)"

        elif model_type == "lstm":
            # Reshape data for LSTM if needed
            seq_length = DL_CONFIG["lstm"]["sequence_length"]
            if len(X_train.shape) == 2:
                # Create simple sequences (sliding window approach for demo)
                print(f"\nNote: Reshaping data for LSTM (sequence length: {seq_length})")
                # For demonstration, we'll treat features as a single timestep
                # In practice, you'd have proper time-series data
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                if X_val is not None:
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                n_features = X_train.shape[2]

            classifier = create_lstm_classifier(
                input_dim=n_features,
                num_classes=n_classes,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            model_name = "LSTM (PyTorch)"

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        classifier.fit(
            X_train, y_train,
            X_val, y_val,
            class_weights=class_weights,
            verbose=True,
        )

        train_preds = classifier.predict(X_train)
        train_acc = (train_preds == y_train).mean()

        val_acc = None
        if X_val is not None and y_val is not None:
            val_preds = classifier.predict(X_val)
            val_acc = (val_preds == y_val).mean()

        print(f"\nFinal Training Accuracy: {train_acc:.4f}")
        if val_acc is not None:
            print(f"Final Validation Accuracy: {val_acc:.4f}")

        # Store model and results
        self.models[model_name] = classifier
        self.training_results[model_name] = {
            "model_type": "deep_learning",
            "architecture": model_type,
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc) if val_acc else None,
            "training_history": classifier.get_training_history(),
            "n_features": n_features,
            "n_classes": n_classes,
        }

        self.is_fitted = True
        print("\n✓ Deep learning model training complete")

        return classifier

    def train_all_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        include_deep_learning: bool = True,
        dl_model_type: str = "mlp",
    ) -> Dict[str, Any]:
        """
        Train all available models.

        Parameters:
        -----------
        X_train : DataFrame or ndarray
            Training features
        y_train : Series or ndarray
            Training labels
        X_val : DataFrame or ndarray, optional
            Validation features
        y_val : Series or ndarray, optional
            Validation labels
        include_deep_learning : bool
            Whether to include DL models
        dl_model_type : str
            Deep learning model type

        Returns:
        --------
        Dict[str, Any]
            All trained models
        """
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)

        self.train_classical_models(X_train, y_train, X_val, y_val)

        if include_deep_learning:
            self.train_deep_learning_model(
                X_train, y_train, X_val, y_val,
                model_type=dl_model_type,
            )

        # Find best model
        self._find_best_model()

        return self.models

    def _find_best_model(self) -> str:
        """
        Find the best performing model based on validation accuracy.

        Returns:
        --------
        str
            Name of the best model
        """
        best_acc = 0
        best_name = None

        for name, results in self.training_results.items():
            acc = results.get("val_accuracy") or results.get("train_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_name = name

        self.best_model_name = best_name

        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"Accuracy: {best_acc:.4f}")
        print("=" * 60)

        return best_name

    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of training results.

        Returns:
        --------
        pd.DataFrame
            Summary table of all models
        """
        summary_data = []

        for name, results in self.training_results.items():
            summary_data.append({
                "Model": name,
                "Type": results["model_type"],
                "Train Accuracy": results.get("train_accuracy"),
                "Val Accuracy": results.get("val_accuracy"),
                "CV Mean": results.get("cv_mean"),
                "CV Std": results.get("cv_std"),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Val Accuracy", ascending=False)

        return summary_df

    def save_models(
        self,
        save_dir: Optional[Path] = None,
        save_results: bool = True,
    ) -> Dict[str, str]:
        """
        Save all trained models to disk.

        Parameters:
        -----------
        save_dir : Path, optional
            Directory to save models
        save_results : bool
            Whether to save training results

        Returns:
        --------
        Dict[str, str]
            Paths to saved models
        """
        save_dir = save_dir or self.models_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("SAVING MODELS")
        print("=" * 60)

        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            # Create safe filename
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

            if self.training_results[name]["model_type"] == "classical":
                filepath = save_dir / f"{safe_name}_{timestamp}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

            else:
                filepath = save_dir / f"{safe_name}_{timestamp}.pt"
                torch.save({
                    "model_state_dict": model.model.state_dict(),
                    "model_config": {
                        "n_features": self.training_results[name]["n_features"],
                        "n_classes": self.training_results[name]["n_classes"],
                        "architecture": self.training_results[name]["architecture"],
                    },
                }, filepath)

            saved_paths[name] = str(filepath)
            print(f"  ✓ Saved: {filepath.name}")

        # Save training results
        if save_results:
            results_path = save_dir / f"training_results_{timestamp}.json"

            # Convert numpy types for JSON serialization
            serializable_results = {}
            for name, results in self.training_results.items():
                serializable_results[name] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else
                        float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for k, v in results.items()
                }

            with open(results_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"  ✓ Saved results: {results_path.name}")
            saved_paths["results"] = str(results_path)

        print(f"\nAll models saved to: {save_dir}")

        return saved_paths

    def load_model(
        self,
        filepath: Union[str, Path],
        model_name: str,
    ) -> Any:
        """
        Load a saved model.

        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
        model_name : str
            Name to assign to the loaded model

        Returns:
        --------
        Any
            Loaded model
        """
        filepath = Path(filepath)

        if filepath.suffix == ".pkl":
            # Load sklearn model
            with open(filepath, "rb") as f:
                model = pickle.load(f)

        elif filepath.suffix == ".pt":
            # Load PyTorch model
            checkpoint = torch.load(filepath, weights_only=False)
            config = checkpoint["model_config"]

            if config["architecture"] == "mlp":
                model = create_mlp_classifier(
                    input_dim=config["n_features"],
                    num_classes=config["n_classes"],
                )
            elif config["architecture"] == "lstm":
                model = create_lstm_classifier(
                    input_dim=config["n_features"],
                    num_classes=config["n_classes"],
                )
            else:
                raise ValueError(f"Unknown architecture: {config['architecture']}")

            model.model.load_state_dict(checkpoint["model_state_dict"])
            model.is_fitted = True

        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")

        self.models[model_name] = model
        print(f"✓ Loaded model: {model_name} from {filepath}")

        return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_and_evaluate_pipeline(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    include_deep_learning: bool = True,
    save_models: bool = True,
) -> Tuple[ModelTrainer, pd.DataFrame]:
    """
    Complete training and evaluation pipeline.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data
    include_deep_learning : bool
        Whether to train DL models
    save_models : bool
        Whether to save trained models

    Returns:
    --------
    Tuple[ModelTrainer, pd.DataFrame]
        Trainer object and results summary
    """
    trainer = ModelTrainer()

    # Train all models
    trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        include_deep_learning=include_deep_learning,
    )

    # Get summary
    summary = trainer.get_training_summary()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    # Save models
    if save_models:
        trainer.save_models()

    return trainer, summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRAINING MODULE DEMONSTRATION")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    n_samples = 1000
    n_features = 20
    n_classes = 3

    # Create imbalanced dataset
    class_sizes = [600, 300, 100]  # Imbalanced
    X_list, y_list = [], []

    for cls, size in enumerate(class_sizes):
        X_cls = np.random.randn(size, n_features) + cls * 0.5
        y_cls = np.full(size, cls)
        X_list.append(X_cls)
        y_list.append(y_cls)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    print(f"\nClass distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    trainer, summary = train_and_evaluate_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        include_deep_learning=True,
        save_models=True,
    )


# =============================================================================
# CROSS-VALIDATION HELPER (для диплома)
# =============================================================================

def train_with_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    5-fold stratified cross-validation.
    Возвращает словарь со средними и std метриками.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]

    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    result = {}
    for key, vals in scores.items():
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"]  = float(np.std(vals))

    print(f"  CV ({n_splits}-fold) accuracy: {result['test_accuracy_mean']:.4f} ± {result['test_accuracy_std']:.4f}")
    print(f"  CV F1-weighted:  {result['test_f1_weighted_mean']:.4f} ± {result['test_f1_weighted_std']:.4f}")
    print(f"  CV ROC-AUC:      {result['test_roc_auc_ovr_weighted_mean']:.4f} ± {result['test_roc_auc_ovr_weighted_std']:.4f}")

    return result

    print("\n✓ Training module demonstration complete!")
