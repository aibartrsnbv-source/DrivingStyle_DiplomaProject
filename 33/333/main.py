#!/usr/bin/env python3
"""
Main Pipeline Entry Point for Driving Style ML Project
========================================================

This script executes the complete machine learning pipeline for driving style
classification and accident risk prediction.

Pipeline Steps:
1. Data Loading and Validation
2. Exploratory Data Analysis
3. Data Preprocessing
4. Model Training (Classical ML + Deep Learning)
5. Model Evaluation
6. Risk Score Computation
7. Results Export

Usage:
    python main.py                    # Run with default settings
    python main.py --synthetic        # Use synthetic data for testing
    python main.py --no-dl            # Skip deep learning models
    python main.py --quick            # Quick run with reduced data

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATHS,
    RANDOM_SEED,
    OUTPUT_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    get_config_summary,
)
from src.data_loader import DataLoader, create_sample_dataset, load_and_unify_datasets
from src.preprocessing import (
    preprocess_pipeline,
    handle_missing_values,
    encode_labels,
    DataPreprocessor,
    FeatureEngineer,
    engineer_features,
)
from src.eda import ExploratoryDataAnalysis, quick_eda
from src.models import ModelFactory, create_mlp_classifier
from src.train import ModelTrainer, train_with_cv
from src.evaluate import ModelEvaluator, evaluate_all_models, generate_shap_analysis, generate_diploma_report
from src.risk_scoring import RiskScorer, print_risk_interpretation_guide
from src.utils import (
    setup_logger,
    timer,
    Timer,
    set_random_seeds,
    ExperimentTracker,
    save_json,
)
import sklearn
sklearn.set_config(enable_metadata_routing=True)

logger = setup_logger(
    name="main_pipeline",
    log_file=OUTPUT_DIR / "pipeline.log",
)


class DrivingStylePipeline:
    """
    Main pipeline class for the driving style ML project.

    This class orchestrates the entire machine learning workflow from
    data loading to model evaluation and risk scoring.

    Attributes:
        config (dict): Pipeline configuration
        loader (DataLoader): Data loading handler
        trainer (ModelTrainer): Model training handler
        evaluator (ModelEvaluator): Model evaluation handler
        risk_scorer (RiskScorer): Risk scoring handler
        experiment (ExperimentTracker): Experiment tracking
    """

    def __init__(
        self,
        use_synthetic: bool = False,
        include_deep_learning: bool = True,
        quick_mode: bool = False,
    ):
        """
        Initialize the pipeline.

        Parameters:
        -----------
        use_synthetic : bool
            Whether to use synthetic data
        include_deep_learning : bool
            Whether to include deep learning models
        quick_mode : bool
            Whether to run in quick mode (reduced data)
        """
        self.use_synthetic = use_synthetic
        self.include_deep_learning = include_deep_learning
        self.quick_mode = quick_mode

        self.loader = DataLoader()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.risk_scorer = RiskScorer()

        self.experiment = ExperimentTracker(
            experiment_name="driving_style_classification",
            output_dir=OUTPUT_DIR / "experiments",
        )

        self.raw_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.preprocessor = None

        self.models = {}
        self.evaluation_results = {}
        self.risk_scores = None

    @timer
    def run(self) -> dict:
        """
        Execute the complete pipeline.

        Returns:
        --------
        dict
            Pipeline results and metrics
        """
        print("\n" + "=" * 70)
        print("  DRIVING STYLE ML PIPELINE")
        print("  Bachelor Diploma Project")
        print("=" * 70)

        self.experiment.log_params(
            {
                "use_synthetic": self.use_synthetic,
                "include_deep_learning": self.include_deep_learning,
                "quick_mode": self.quick_mode,
                "random_seed": RANDOM_SEED,
            }
        )

        try:
            self.step_1_load_data()

            self.step_2_eda()

            self.step_3_preprocess()

            self.step_4_train_models()

            self.step_5_evaluate()

            self.step_6_risk_scoring()

            results = self.step_7_export_results()

            print("\n" + "=" * 70)
            print("  PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.experiment.add_note(f"Pipeline failed: {str(e)}")
            raise

        finally:
            self.experiment.save()

    def step_1_load_data(self) -> None:
        """Step 1: Load and validate data."""
        print("\n" + "=" * 70)
        print("STEP 1: DATA LOADING")
        print("=" * 70)

        with Timer("Data loading"):
            if self.use_synthetic:
                # Generate synthetic data
                print("\nGenerating synthetic driving behavior dataset...")

                n_samples = 1000 if self.quick_mode else 5000
                self.raw_data = create_sample_dataset(
                    n_samples=n_samples,
                    n_features=15,
                    n_classes=3,
                    imbalance_ratio=[0.5, 0.3, 0.2],
                    include_time_series=True,
                    random_state=RANDOM_SEED,
                )
                self.target_column = "driving_style_encoded"

            else:
                # Load and unify all available real datasets (falls back to synthetic)
                self.raw_data = load_and_unify_datasets()
                self.target_column = "driving_style_encoded"

            self.loader.validate_dataset(
                self.raw_data,
                "Loaded Dataset",
                check_duplicates=True,
            )

            self.experiment.log_params(
                {
                    "n_samples": len(self.raw_data),
                    "n_features": len(self.raw_data.columns),
                }
            )

            print(
                f"\n✓ Loaded {len(self.raw_data):,} samples with {len(self.raw_data.columns)} features"
            )

    def _load_real_data(self) -> pd.DataFrame:
        """
        Attempt to load real datasets.

        Returns:
        --------
        pd.DataFrame
            Loaded data or synthetic fallback
        """
        datasets_to_try = [
            ("driver_behavior", DATASET_PATHS.get("driver_behavior")),
            ("eco_driving", DATASET_PATHS.get("eco_driving")),
            ("carla_data", DATASET_PATHS.get("carla_data")),
        ]

        for name, path in datasets_to_try:
            if path and Path(path).exists():
                try:
                    nrows = 5000 if self.quick_mode else None
                    df = pd.read_csv(path, nrows=nrows)
                    print(f"✓ Loaded {name} dataset")

                    # Identify or create target column
                    self.target_column = self._identify_target_column(df)
                    return df

                except Exception as e:
                    print(f"⚠ Could not load {name}: {e}")

        # Fallback to synthetic data
        print("\n⚠ No real datasets found. Using synthetic data.")
        self.use_synthetic = True
        self.target_column = "driving_style_encoded"

        return create_sample_dataset(
            n_samples=5000 if not self.quick_mode else 1000,
            random_state=RANDOM_SEED,
        )

    def _identify_target_column(self, df: pd.DataFrame) -> str:
        """
        Identify or create the target column.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset

        Returns:
        --------
        str
            Name of target column
        """
        target_candidates = [
            "driving_style",
            "label",
            "class",
            "target",
            "accident",
            "risk_level",
            "style",
            "behavior",
        ]

        for candidate in target_candidates:
            if candidate in df.columns:
                return candidate

        # If no target found, create one based on features
        print("⚠ No target column found. Creating synthetic labels...")

        # Simple rule-based labeling
        if "speed" in [c.lower() for c in df.columns]:
            speed_col = [c for c in df.columns if "speed" in c.lower()][0]
            speed_norm = (df[speed_col] - df[speed_col].mean()) / df[speed_col].std()
            df["driving_style_encoded"] = pd.cut(
                speed_norm,
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=[0, 1, 2],
            ).astype(int)
        else:
            # Random labels based on feature patterns
            np.random.seed(RANDOM_SEED)
            df["driving_style_encoded"] = np.random.choice(
                [0, 1, 2], len(df), p=[0.5, 0.3, 0.2]
            )

        return "driving_style_encoded"

    def step_2_eda(self) -> None:
        """Step 2: Exploratory Data Analysis."""
        print("\n" + "=" * 70)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        with Timer("EDA"):
            eda = ExploratoryDataAnalysis(
                self.raw_data,
                target_column=self.target_column,
                figures_dir=FIGURES_DIR,
            )

            quick_eda(self.raw_data, self.target_column)

            if not self.quick_mode:
                # Summary statistics
                eda.summary_statistics(save_to_file=True)

                # Class distribution
                class_info = eda.class_distribution_analysis(save_plot=True)
                self.experiment.log_params(
                    {
                        "class_imbalance_ratio": class_info.get("imbalance_ratio", 1.0),
                    }
                )

                # Correlation analysis
                eda.correlation_analysis(save_plot=True)

                # Feature distributions
                eda.feature_distributions(save_plot=True)

            print("\n✓ EDA complete. Figures saved to:", FIGURES_DIR)

    def step_3_preprocess(self) -> None:
        """Step 3: Data Preprocessing."""
        print("\n" + "=" * 70)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 70)

        with Timer("Preprocessing"):
            # Engineer additional features for diploma analysis
            self.raw_data = engineer_features(self.raw_data)

            self.raw_data = handle_missing_values(
                self.raw_data,
                strategy="median",
                threshold=0.5,
            )

            if self.raw_data[self.target_column].dtype == "object":
                self.raw_data, label_mapping = encode_labels(
                    self.raw_data,
                    self.target_column,
                )
                self.target_column = f"{self.target_column}_encoded"
                self.experiment.log_params({"label_mapping": label_mapping})

            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.preprocessor,
            ) = preprocess_pipeline(
                self.raw_data,
                target_column=self.target_column,
            )

            self.experiment.log_params(
                {
                    "train_samples": len(self.X_train),
                    "val_samples": len(self.X_val),
                    "test_samples": len(self.X_test),
                    "n_features_final": self.X_train.shape[1],
                }
            )

            print(f"\n✓ Preprocessing complete:")
            print(f"  Train: {self.X_train.shape}")
            print(f"  Val: {self.X_val.shape}")
            print(f"  Test: {self.X_test.shape}")

    def step_4_train_models(self) -> None:
        """Step 4: Train Models."""
        print("\n" + "=" * 70)
        print("STEP 4: MODEL TRAINING")
        print("=" * 70)

        with Timer("Model training"):
            self.trainer.train_classical_models(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                use_cv=not self.quick_mode,
                cv_folds=5,
                handle_imbalance=True,
            )

            if self.include_deep_learning:
                self.trainer.train_deep_learning_model(
                    self.X_train,
                    self.y_train,
                    self.X_val,
                    self.y_val,
                    model_type="mlp",
                    use_class_weights=True,
                )

            self.models = self.trainer.models

            for model_name, results in self.trainer.training_results.items():
                self.experiment.log_metric(
                    f"{model_name}_train_acc",
                    results.get("train_accuracy", 0),
                )
                if results.get("val_accuracy"):
                    self.experiment.log_metric(
                        f"{model_name}_val_acc",
                        results["val_accuracy"],
                    )

            saved_paths = self.trainer.save_models()
            for name, path in saved_paths.items():
                self.experiment.log_artifact(path, f"Trained model: {name}")

            print(f"\n✓ Trained {len(self.models)} models")

    def step_5_evaluate(self) -> None:
        """Step 5: Evaluate Models."""
        print("\n" + "=" * 70)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 70)

        with Timer("Model evaluation"):
            # Determine class names dynamically from training labels.
            # This prevents crashes when dataset has 2 or 4 classes but code
            # was hardcoded for 3, and keeps names consistent with DRIVING_STYLE_CLASSES.
            from src.config import DRIVING_STYLE_CLASSES
            unique_classes = sorted(np.unique(np.asarray(self.y_train)))
            class_names = [
                DRIVING_STYLE_CLASSES.get(int(c), f"Class {c}")
                for c in unique_classes
            ]
            print(f"  Detected {len(class_names)} classes: {class_names}")

            # Evaluate all models
            self.evaluator, comparison_df = evaluate_all_models(
                self.models,
                self.X_test,
                self.y_test,
                class_names=class_names,
                save_plots=True,
            )

            self.evaluation_results = self.evaluator.results

            for model_name, results in self.evaluation_results.items():
                for metric, value in results["metrics"].items():
                    if value is not None:
                        self.experiment.log_metric(f"{model_name}_test_{metric}", value)

            # Print comparison
            print("\n" + "=" * 50)
            print("MODEL COMPARISON")
            print("=" * 50)
            print(comparison_df.to_string(index=False))

            # Identify best model
            best_model_name = comparison_df.iloc[0]["Model"]
            best_f1 = comparison_df.iloc[0]["f1_score"]

            print(f"\n✓ Best model: {best_model_name} (F1: {best_f1:.4f})")

            self.experiment.log_params(
                {
                    "best_model": best_model_name,
                    "best_f1_score": best_f1,
                }
            )

            # SHAP analysis for interpretability (diploma requirement)
            if not self.quick_mode:
                feature_names = [f"feature_{i}" for i in range(self.X_test.shape[1])]
                generate_shap_analysis(
                    self.models,
                    self.X_test,
                    feature_names=feature_names,
                    output_dir=FIGURES_DIR,
                )

            # Diploma report
            dataset_info = {
                "n_samples": len(self.raw_data),
                "n_features": self.X_train.shape[1],
                "n_train": len(self.X_train),
                "n_test": len(self.X_test),
            }
            generate_diploma_report(
                self.evaluation_results,
                output_dir=OUTPUT_DIR / "reports",
                dataset_info=dataset_info,
            )

    def step_6_risk_scoring(self) -> None:
        """Step 6: Risk Scoring."""
        print("\n" + "=" * 70)
        print("STEP 6: RISK SCORING")
        print("=" * 70)

        with Timer("Risk scoring"):
            # Print interpretation guide
            print_risk_interpretation_guide()

            best_model_name = self.trainer.best_model_name
            if best_model_name is None or best_model_name not in self.models:
                best_model_name = next(iter(self.models))
                self.trainer.best_model_name = best_model_name
            best_model = self.models[best_model_name]

            self.risk_scores = self.risk_scorer.compute_risk_score(
                best_model,
                self.X_test,
                method="probability",
            )

            risk_levels = self.risk_scorer.get_risk_levels_batch(self.risk_scores)

            risk_stats = {
                "mean": float(self.risk_scores.mean()),
                "std": float(self.risk_scores.std()),
                "min": float(self.risk_scores.min()),
                "max": float(self.risk_scores.max()),
            }

            print(f"\nRisk Score Statistics:")
            for stat, value in risk_stats.items():
                print(f"  {stat.capitalize()}: {value:.3f}")

            # Level distribution
            level_counts = pd.Series([r.value for r in risk_levels]).value_counts()
            print(f"\nRisk Level Distribution:")
            for level, count in level_counts.items():
                pct = count / len(risk_levels) * 100
                print(f"  {level}: {count} ({pct:.1f}%)")

            self.risk_scorer.plot_risk_distribution(
                self.risk_scores,
                save_path=FIGURES_DIR / "risk_score_distribution.png",
            )

            self.experiment.log_params({"risk_score_stats": risk_stats})

            print("\n✓ Risk scoring complete")

    def step_7_export_results(self) -> dict:
        """Step 7: Export Results."""
        print("\n" + "=" * 70)
        print("STEP 7: EXPORTING RESULTS")
        print("=" * 70)

        with Timer("Results export"):
            results = {
                "pipeline_info": {
                    "timestamp": datetime.now().isoformat(),
                    "use_synthetic": self.use_synthetic,
                    "include_deep_learning": self.include_deep_learning,
                    "random_seed": RANDOM_SEED,
                },
                "data_info": {
                    "n_samples_total": len(self.raw_data),
                    "n_train": len(self.X_train),
                    "n_val": len(self.X_val),
                    "n_test": len(self.X_test),
                    "n_features": self.X_train.shape[1],
                },
                "model_results": {},
                "risk_scoring": {
                    "mean": float(self.risk_scores.mean()),
                    "std": float(self.risk_scores.std()),
                },
            }

            for model_name, eval_result in self.evaluation_results.items():
                results["model_results"][model_name] = eval_result["metrics"]

            results_path = OUTPUT_DIR / "pipeline_results.json"
            save_json(results, results_path)

            print(f"\n✓ Results saved to: {results_path}")

            print("\n" + "=" * 50)
            print("PIPELINE SUMMARY")
            print("=" * 50)
            print(
                f"Data: {len(self.raw_data):,} samples, {self.X_train.shape[1]} features"
            )
            print(f"Models trained: {len(self.models)}")
            print(f"Best model: {self.trainer.best_model_name}")
            print(
                f"Best F1-Score: {results['model_results'][self.trainer.best_model_name]['f1_score']:.4f}"
            )
            print(f"Average Risk Score: {results['risk_scoring']['mean']:.3f}")
            print(f"\nOutputs saved to: {OUTPUT_DIR}")

            return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Driving Style ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --synthetic        # Use synthetic data
  python main.py --no-dl            # Skip deep learning models
  python main.py --quick            # Quick run with reduced data
        """,
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real datasets",
    )

    parser.add_argument(
        "--no-dl",
        action="store_true",
        help="Skip deep learning models (faster)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced data and simplified analysis",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Print configuration summary and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.config:
        get_config_summary()
        return

    set_random_seeds(args.seed)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print(
        "║"
        + "  DRIVING STYLE CLASSIFICATION & ACCIDENT RISK PREDICTION".center(68)
        + "║"
    )
    print("║" + "  Bachelor Diploma Project".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    pipeline = DrivingStylePipeline(
        use_synthetic=args.synthetic,
        include_deep_learning=not args.no_dl,
        quick_mode=args.quick,
    )

    results = pipeline.run()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  PIPELINE EXECUTION COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    return results


if __name__ == "__main__":
    main()
