"""
Evaluation Module for Driving Style ML Project
===============================================

This module provides comprehensive evaluation capabilities for machine learning
models, including metrics calculation, visualization, and model comparison.

Metrics Implemented:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (One-vs-Rest for multiclass)
- Confusion Matrix
- Classification Report
- Learning Curves

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    FIGURES_DIR,
    VISUALIZATION_CONFIG,
    DRIVING_STYLE_CLASSES,
)


class ModelEvaluator:
    """
    A comprehensive class for evaluating machine learning models.

    This class provides methods for:
    - Computing various classification metrics
    - Generating confusion matrices
    - Plotting ROC curves
    - Comparing multiple models
    - Creating evaluation reports

    Attributes:
        results (Dict): Evaluation results for each model
        figures_dir (Path): Directory to save figures
    """

    def __init__(self, figures_dir: Optional[Path] = None):
        """
        Initialize the ModelEvaluator.

        Parameters:
        -----------
        figures_dir : Path, optional
            Directory to save evaluation figures
        """
        self.figures_dir = figures_dir or FIGURES_DIR
        self.results: Dict[str, Dict] = {}

        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(VISUALIZATION_CONFIG.get("color_palette", "husl"))

    def evaluate_model(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_name: str,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single model comprehensively.

        Parameters:
        -----------
        model : Any
            Trained model with predict and predict_proba methods
        X_test : DataFrame or ndarray
            Test features
        y_test : Series or ndarray
            True labels
        model_name : str
            Name of the model for reporting
        class_names : list, optional
            Names of the classes

        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print("=" * 60)

        # Convert to numpy
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        # Get predictions
        y_pred = model.predict(X_test)

        # Get probability predictions if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        # Determine if multiclass
        n_classes = len(np.unique(y_test))
        is_multiclass = n_classes > 2

        # Set averaging method
        average = "weighted" if is_multiclass else "binary"

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
        }

        # Calculate ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                if is_multiclass:
                    # One-vs-Rest ROC-AUC for multiclass
                    metrics["roc_auc"] = roc_auc_score(
                        y_test, y_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            except Exception as e:
                print(f"  ⚠ Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None

        # Print metrics
        print(f"\nMetrics:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name.capitalize():15s}: {value:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print("-" * 40)
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0,
        )
        print(report)

        # Store results
        self.results[model_name] = {
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "n_classes": n_classes,
            "class_names": class_names,
        }

        return metrics

    def plot_confusion_matrix(
        self,
        model_name: str,
        normalize: bool = True,
        save_plot: bool = True,
    ) -> None:
        """
        Plot confusion matrix for a model.

        Parameters:
        -----------
        model_name : str
            Name of the model
        normalize : bool
            Whether to normalize the confusion matrix
        save_plot : bool
            Whether to save the plot
        """
        if model_name not in self.results:
            print(f"⚠ Model '{model_name}' not found in results")
            return

        result = self.results[model_name]
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        class_names = result["class_names"]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = f"Normalized Confusion Matrix: {model_name}"
        else:
            fmt = "d"
            title = f"Confusion Matrix: {model_name}"

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm)),
            ax=ax,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_plot:
            safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = self.figures_dir / f"confusion_matrix_{safe_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_roc_curves(
        self,
        model_names: Optional[List[str]] = None,
        save_plot: bool = True,
    ) -> None:
        """
        Plot ROC curves for one or more models.

        Parameters:
        -----------
        model_names : list, optional
            Models to include (default: all)
        save_plot : bool
            Whether to save the plot
        """
        model_names = model_names or list(self.results.keys())

        # Filter models with probability predictions
        valid_models = [
            name for name in model_names
            if name in self.results and self.results[name]["y_proba"] is not None
        ]

        if not valid_models:
            print("⚠ No models with probability predictions available")
            return

        print(f"\nPlotting ROC curves for: {valid_models}")

        n_classes = self.results[valid_models[0]]["n_classes"]

        if n_classes == 2:
            # Binary classification
            fig, ax = plt.subplots(figsize=(10, 8))

            for model_name in valid_models:
                result = self.results[model_name]
                y_true = result["y_true"]
                y_proba = result["y_proba"][:, 1]

                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, linewidth=2,
                        label=f"{model_name} (AUC = {roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

        else:
            # Multiclass - plot one-vs-rest for each class
            fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))

            class_names = self.results[valid_models[0]]["class_names"] or \
                          [f"Class {i}" for i in range(n_classes)]

            for cls_idx in range(n_classes):
                ax = axes[cls_idx] if n_classes > 1 else axes

                for model_name in valid_models:
                    result = self.results[model_name]
                    y_true = result["y_true"]
                    y_proba = result["y_proba"]

                    # Binarize for one-vs-rest
                    y_true_binary = (y_true == cls_idx).astype(int)
                    y_score = y_proba[:, cls_idx]

                    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                    roc_auc = auc(fpr, tpr)

                    ax.plot(fpr, tpr, linewidth=2,
                            label=f"{model_name} (AUC = {roc_auc:.3f})")

                ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC: {class_names[cls_idx]}", fontweight="bold")
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / "roc_curves_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_precision_recall_curves(
        self,
        model_names: Optional[List[str]] = None,
        save_plot: bool = True,
    ) -> None:
        """
        Plot Precision-Recall curves.

        Parameters:
        -----------
        model_names : list, optional
            Models to include
        save_plot : bool
            Whether to save the plot
        """
        model_names = model_names or list(self.results.keys())

        valid_models = [
            name for name in model_names
            if name in self.results and self.results[name]["y_proba"] is not None
        ]

        if not valid_models:
            print("⚠ No models with probability predictions available")
            return

        n_classes = self.results[valid_models[0]]["n_classes"]

        fig, ax = plt.subplots(figsize=(10, 8))

        if n_classes == 2:
            for model_name in valid_models:
                result = self.results[model_name]
                y_true = result["y_true"]
                y_proba = result["y_proba"][:, 1]

                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                ap = average_precision_score(y_true, y_proba)

                ax.plot(recall, precision, linewidth=2,
                        label=f"{model_name} (AP = {ap:.3f})")

        else:
            # Weighted average for multiclass
            for model_name in valid_models:
                result = self.results[model_name]
                y_true = result["y_true"]
                y_proba = result["y_proba"]

                # Compute average precision per class and weight
                ap_scores = []
                for cls_idx in range(n_classes):
                    y_true_binary = (y_true == cls_idx).astype(int)
                    y_score = y_proba[:, cls_idx]
                    ap = average_precision_score(y_true_binary, y_score)
                    ap_scores.append(ap)

                weighted_ap = np.mean(ap_scores)
                ax.bar(model_name, weighted_ap, alpha=0.7)

            ax.set_ylabel("Weighted Average Precision")
            ax.set_title("Average Precision by Model", fontsize=14, fontweight="bold")

        if n_classes == 2:
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
            ax.legend(loc="lower left")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / "precision_recall_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_metrics_comparison(
        self,
        model_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        save_plot: bool = True,
    ) -> None:
        """
        Create a bar chart comparing metrics across models.

        Parameters:
        -----------
        model_names : list, optional
            Models to compare
        metrics : list, optional
            Metrics to include
        save_plot : bool
            Whether to save the plot
        """
        model_names = model_names or list(self.results.keys())
        metrics = metrics or ["accuracy", "precision", "recall", "f1_score"]

        # Prepare data
        data = []
        for model_name in model_names:
            if model_name in self.results:
                for metric in metrics:
                    value = self.results[model_name]["metrics"].get(metric)
                    if value is not None:
                        data.append({
                            "Model": model_name,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": value,
                        })

        df = pd.DataFrame(data)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(model_names))
        width = 0.8 / len(metrics)
        colors = sns.color_palette("husl", len(metrics))

        for i, metric in enumerate(metrics):
            metric_name = metric.replace("_", " ").title()
            values = [
                self.results[m]["metrics"].get(metric, 0)
                for m in model_names if m in self.results
            ]
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_training_history(
        self,
        model_name: str,
        training_history: Dict[str, List[float]],
        save_plot: bool = True,
    ) -> None:
        """
        Plot training history for a deep learning model.

        Parameters:
        -----------
        model_name : str
            Name of the model
        training_history : dict
            Training history with 'train_loss', 'val_loss', 'val_acc'
        save_plot : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(training_history["train_loss"]) + 1)

        # Loss plot
        axes[0].plot(epochs, training_history["train_loss"], "b-", linewidth=2, label="Training Loss")
        if training_history.get("val_loss"):
            axes[0].plot(epochs, training_history["val_loss"], "r-", linewidth=2, label="Validation Loss")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        if training_history.get("val_acc"):
            axes[1].plot(epochs, training_history["val_acc"], "g-", linewidth=2, label="Validation Accuracy")
            axes[1].set_xlabel("Epoch", fontsize=12)
            axes[1].set_ylabel("Accuracy", fontsize=12)
            axes[1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No validation accuracy recorded",
                         ha="center", va="center", fontsize=12)
            axes[1].set_axis_off()

        plt.suptitle(f"Training History: {model_name}", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_plot:
            safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = self.figures_dir / f"training_history_{safe_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of all evaluated models.

        Returns:
        --------
        pd.DataFrame
            Comparison table with all metrics
        """
        comparison_data = []

        for model_name, result in self.results.items():
            row = {"Model": model_name}
            row.update(result["metrics"])
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by F1 score
        df = df.sort_values("f1_score", ascending=False)

        return df

    def generate_evaluation_report(
        self,
        save_to_file: bool = True,
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Parameters:
        -----------
        save_to_file : bool
            Whether to save the report to a file

        Returns:
        --------
        str
            Report text
        """
        report_lines = [
            "=" * 70,
            "MODEL EVALUATION REPORT",
            "=" * 70,
            "",
        ]

        # Overall comparison
        comparison_df = self.get_comparison_table()
        report_lines.append("OVERALL COMPARISON")
        report_lines.append("-" * 70)
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")

        # Best model
        if len(comparison_df) > 0:
            best_model = comparison_df.iloc[0]["Model"]
            best_f1 = comparison_df.iloc[0]["f1_score"]
            report_lines.append(f"BEST MODEL: {best_model}")
            report_lines.append(f"F1-Score: {best_f1:.4f}")
            report_lines.append("")

        # Detailed results for each model
        report_lines.append("DETAILED RESULTS")
        report_lines.append("=" * 70)

        for model_name, result in self.results.items():
            report_lines.append(f"\n{model_name}")
            report_lines.append("-" * 40)

            for metric, value in result["metrics"].items():
                if value is not None:
                    report_lines.append(f"  {metric:15s}: {value:.4f}")

            # Confusion matrix summary
            y_true = result["y_true"]
            y_pred = result["y_pred"]
            cm = confusion_matrix(y_true, y_pred)
            report_lines.append(f"\n  Confusion Matrix:")
            report_lines.append("  " + str(cm).replace("\n", "\n  "))

        report_lines.append("\n" + "=" * 70)

        report_text = "\n".join(report_lines)

        print(report_text)

        if save_to_file:
            report_path = self.figures_dir / "evaluation_report.txt"
            with open(report_path, "w") as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {report_path}")

        return report_text


# =============================================================================
# STANDALONE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_all_models(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_plots: bool = True,
) -> Tuple[ModelEvaluator, pd.DataFrame]:
    """
    Evaluate all models and generate comparison.

    Parameters:
    -----------
    models : dict
        Dictionary of model name to model object
    X_test : DataFrame or ndarray
        Test features
    y_test : Series or ndarray
        Test labels
    class_names : list, optional
        Names of classes
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    Tuple[ModelEvaluator, pd.DataFrame]
        Evaluator object and comparison table
    """
    evaluator = ModelEvaluator()

    # Evaluate each model
    for model_name, model in models.items():
        evaluator.evaluate_model(model, X_test, y_test, model_name, class_names)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("=" * 60)

    # Comparison chart
    evaluator.plot_metrics_comparison(save_plot=save_plots)

    # ROC curves
    evaluator.plot_roc_curves(save_plot=save_plots)

    # Confusion matrices for each model
    for model_name in models.keys():
        evaluator.plot_confusion_matrix(model_name, save_plot=save_plots)

    # Generate report
    evaluator.generate_evaluation_report(save_to_file=save_plots)

    # Get comparison table
    comparison_df = evaluator.get_comparison_table()

    return evaluator, comparison_df


def quick_evaluate(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Quick evaluation of a single model.

    Parameters:
    -----------
    model : Any
        Trained model
    X_test : DataFrame or ndarray
        Test features
    y_test : Series or ndarray
        Test labels

    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics
    """
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print("\nQuick Evaluation Results:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"  {metric:15s}: {value:.4f}")


# =============================================================================
# SHAP ANALYSIS (для диплома)
# =============================================================================

def generate_shap_analysis(
    models: Dict[str, Any],
    X_test: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    output_dir: Union[str, Path],
    max_samples: int = 200,
) -> None:
    """SHAP-анализ для tree-based моделей (RF, GB, XGB)."""
    try:
        import shap
    except ImportError:
        print("⚠ shap не установлен. Запусти: pip install shap")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(X_test, pd.DataFrame):
        X_arr = X_test.values
    else:
        X_arr = X_test

    X_sample = X_arr[:max_samples]

    tree_model_names = ["random_forest", "gradient_boosting", "xgboost",
                        "Random Forest", "Gradient Boosting", "XGBoost"]

    for model_name, model in models.items():
        if not any(t.lower() in model_name.lower() for t in ["forest", "boost", "xgb"]):
            continue
        try:
            print(f"  SHAP: {model_name}...")
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)

            # Bar plot
            plt.figure(figsize=(10, 6))
            if isinstance(sv, list):
                shap.summary_plot(sv[0], X_sample, feature_names=feature_names,
                                  plot_type="bar", show=False)
            else:
                shap.summary_plot(sv, X_sample, feature_names=feature_names,
                                  plot_type="bar", show=False)
            plt.tight_layout()
            safe = model_name.lower().replace(" ", "_")
            plt.savefig(output_dir / f"shap_bar_{safe}.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Beeswarm plot
            plt.figure(figsize=(10, 8))
            if isinstance(sv, list):
                shap.summary_plot(sv[0], X_sample, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_beeswarm_{safe}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Saved shap_bar_{safe}.png и shap_beeswarm_{safe}.png")
        except Exception as e:
            print(f"    ⚠ SHAP для {model_name}: {e}")


# =============================================================================
# DIPLOMA REPORT GENERATOR
# =============================================================================

def generate_diploma_report(
    all_results: Dict[str, Any],
    output_dir: Union[str, Path],
    dataset_info: Optional[Dict] = None,
) -> str:
    """
    Генерирует финальный JSON и Markdown отчёт для диплома.
    Сохраняет в output_dir/diploma_metrics.json и diploma_report.md.
    """
    import json
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "dataset": dataset_info or {},
        "models": {},
    }

    md_lines = [
        "# Diploma Report — Driving Style Classification",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Dataset",
    ]
    if dataset_info:
        for k, v in dataset_info.items():
            md_lines.append(f"- **{k}**: {v}")

    md_lines.append("\n## Model Comparison\n")
    md_lines.append("| Model | Accuracy | F1 (weighted) | ROC-AUC | CV Mean ± Std |")
    md_lines.append("|-------|----------|---------------|---------|---------------|")

    for model_name, res in all_results.items():
        acc = res.get("val_accuracy") or res.get("test_accuracy") or res.get("train_accuracy", 0)
        f1  = res.get("f1_weighted") or res.get("f1_score") or 0.0
        auc = res.get("roc_auc") or 0.0
        cv_mean = res.get("cv_mean") or 0.0
        cv_std  = res.get("cv_std")  or 0.0

        report["models"][model_name] = {
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_mean": round(float(cv_mean), 4),
            "cv_std":  round(float(cv_std), 4),
            "training_time": res.get("training_time", 0),
        }
        md_lines.append(
            f"| {model_name} | {acc:.4f} | {f1:.4f} | {auc:.4f} | {cv_mean:.4f} ± {cv_std:.4f} |"
        )

    # Save JSON
    json_path = output_dir / "diploma_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save Markdown
    md_path = output_dir / "diploma_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✓ diploma_metrics.json → {json_path}")
    print(f"✓ diploma_report.md   → {md_path}")
    return str(md_path)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EVALUATION MODULE DEMONSTRATION")
    print("=" * 60)

    # Create synthetic data and models for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train some models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Evaluate
    class_names = ["Safe", "Normal", "Aggressive"]
    evaluator, comparison = evaluate_all_models(
        models, X_test, y_test,
        class_names=class_names,
        save_plots=True,
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print("\n✓ Evaluation module demonstration complete!")
