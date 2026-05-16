"""
Risk Scoring Module for Driving Style ML Project
=================================================

This module provides functionality for converting model predictions into
interpretable accident risk scores and risk levels.

Risk Score Interpretation:
- 0.0 - 0.3: LOW RISK     - Safe driving behavior
- 0.3 - 0.6: MEDIUM RISK  - Moderate concern, improvement needed
- 0.6 - 0.8: HIGH RISK    - Significant risk, immediate attention needed
- 0.8 - 1.0: CRITICAL     - Very high accident probability

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    RISK_SCORING_CONFIG,
    FIGURES_DIR,
    DRIVING_STYLE_CLASSES,
)


class RiskLevel(Enum):
    """Enumeration of risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAssessment:
    """
    Data class representing a risk assessment result.

    Attributes:
        risk_score: Numerical risk score between 0 and 1
        risk_level: Categorical risk level
        driving_style: Predicted driving style
        confidence: Model confidence in the prediction
        contributing_factors: Factors contributing to the risk
        recommendations: Safety recommendations
    """
    risk_score: float
    risk_level: RiskLevel
    driving_style: str
    confidence: float
    contributing_factors: Dict[str, float]
    recommendations: List[str]


class RiskScorer:
    """
    A class for computing and interpreting accident risk scores.

    This class provides methods for:
    - Converting model predictions to risk scores
    - Determining risk levels
    - Analyzing contributing factors
    - Generating safety recommendations

    Attributes:
        thresholds (dict): Risk level thresholds
        feature_weights (dict): Weights for composite risk scoring
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the RiskScorer.

        Parameters:
        -----------
        thresholds : dict, optional
            Custom risk level thresholds
        feature_weights : dict, optional
            Custom feature weights for composite scoring
        """
        self.thresholds = thresholds or RISK_SCORING_CONFIG["thresholds"]
        self.feature_weights = feature_weights or RISK_SCORING_CONFIG["feature_weights"]

        # Style to risk mapping
        self.style_risk_mapping = {
            "safe": 0.15,
            "normal": 0.35,
            "aggressive": 0.65,
            "risky": 0.85,
            0: 0.15,  # Numeric labels
            1: 0.35,
            2: 0.65,
            3: 0.85,
        }

    def compute_risk_score(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        method: str = "probability",
    ) -> np.ndarray:
        """
        Compute risk scores from model predictions.

        Parameters:
        -----------
        model : Any
            Trained model with predict_proba method
        X : DataFrame or ndarray
            Feature data
        method : str
            Scoring method: 'probability', 'weighted', or 'style_mapping'

        Returns:
        --------
        ndarray
            Risk scores between 0 and 1
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if method == "probability":
            # Use probability of risky/aggressive classes
            return self._probability_based_score(model, X)

        elif method == "weighted":
            # Weighted combination of class probabilities
            return self._weighted_probability_score(model, X)

        elif method == "style_mapping":
            # Map predicted style to risk score
            return self._style_mapping_score(model, X)

        else:
            raise ValueError(f"Unknown scoring method: {method}")

    def _probability_based_score(
        self,
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute risk score based on probability of risky classes.

        This method uses the model's probability predictions for risky
        driving styles to compute the risk score.
        """
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)

            # Assume higher class indices = higher risk
            n_classes = probas.shape[1]

            # Weight classes by risk level
            weights = np.array([i / (n_classes - 1) for i in range(n_classes)])

            # Compute weighted sum of probabilities
            risk_scores = np.dot(probas, weights)

            return np.clip(risk_scores, 0, 1)

        else:
            # Fallback to style mapping if no probabilities
            return self._style_mapping_score(model, X)

    def _weighted_probability_score(
        self,
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute risk score using weighted class probabilities.
        """
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)

            # Custom weights for each class
            # Assuming classes: safe=0, normal=1, aggressive=2, risky=3
            class_weights = np.array([0.1, 0.3, 0.7, 0.9])[:probas.shape[1]]

            risk_scores = np.dot(probas, class_weights)

            return np.clip(risk_scores, 0, 1)

        else:
            return self._style_mapping_score(model, X)

    def _style_mapping_score(
        self,
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Map predicted driving style to risk score.
        """
        predictions = model.predict(X)

        risk_scores = np.array([
            self.style_risk_mapping.get(pred, 0.5)
            for pred in predictions
        ])

        return risk_scores

    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Determine risk level from risk score.

        Parameters:
        -----------
        risk_score : float
            Risk score between 0 and 1

        Returns:
        --------
        RiskLevel
            Categorical risk level
        """
        if risk_score < self.thresholds["low"]:
            return RiskLevel.LOW
        elif risk_score < self.thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif risk_score < self.thresholds["high"]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def get_risk_levels_batch(
        self,
        risk_scores: np.ndarray,
    ) -> List[RiskLevel]:
        """
        Get risk levels for a batch of scores.

        Parameters:
        -----------
        risk_scores : ndarray
            Array of risk scores

        Returns:
        --------
        List[RiskLevel]
            List of risk levels
        """
        return [self.get_risk_level(score) for score in risk_scores]

    def compute_composite_risk_score(
        self,
        features: pd.DataFrame,
        feature_mapping: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Compute composite risk score from driving behavior features.

        This method uses domain knowledge to weight different behavioral
        indicators and compute an overall risk score.

        Parameters:
        -----------
        features : DataFrame
            Behavioral features
        feature_mapping : dict, optional
            Mapping from standard feature names to actual column names

        Returns:
        --------
        ndarray
            Composite risk scores
        """
        risk_components = []

        # Default feature mapping
        if feature_mapping is None:
            feature_mapping = {
                "speed_violation": ["speeding", "speed_over_limit", "max_speed"],
                "harsh_braking": ["harsh_braking_count", "harsh_braking"],
                "harsh_acceleration": ["harsh_accel_count", "harsh_acceleration"],
                "aggressive_turns": ["sharp_turn", "aggressive_turns", "lane_changes"],
                "following_distance": ["following_distance"],
                "lane_discipline": ["lane_discipline", "lane_change_frequency"],
                "fatigue_indicators": ["fatigue_score", "drowsiness"],
            }

        # Normalize features and compute component scores
        for component, weight in self.feature_weights.items():
            candidate_cols = feature_mapping.get(component, [])

            # Find matching column
            matching_col = None
            for col in candidate_cols:
                if col in features.columns:
                    matching_col = col
                    break

            if matching_col:
                values = features[matching_col].values

                # Normalize to 0-1 range
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)

                # Invert if lower is riskier (e.g., following distance)
                if component == "following_distance":
                    normalized = 1 - normalized

                risk_components.append(normalized * weight)

        if len(risk_components) > 0:
            composite_score = np.sum(risk_components, axis=0)
            # Normalize to 0-1
            composite_score = composite_score / sum(self.feature_weights.values())
            return np.clip(composite_score, 0, 1)

        else:
            # Return neutral score if no features match
            return np.full(len(features), 0.5)

    def assess_risk(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        return_details: bool = True,
    ) -> Union[np.ndarray, List[RiskAssessment]]:
        """
        Perform comprehensive risk assessment.

        Parameters:
        -----------
        model : Any
            Trained model
        X : DataFrame or ndarray
            Feature data
        feature_names : list, optional
            Names of features
        return_details : bool
            Whether to return detailed assessments

        Returns:
        --------
        Union[ndarray, List[RiskAssessment]]
            Risk scores or detailed assessments
        """
        # Compute risk scores
        risk_scores = self.compute_risk_score(model, X)

        if not return_details:
            return risk_scores

        # Get predictions and probabilities
        predictions = model.predict(X if isinstance(X, np.ndarray) else X.values)

        probas = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X if isinstance(X, np.ndarray) else X.values)

        # Generate detailed assessments
        assessments = []

        for i in range(len(risk_scores)):
            # Get risk level
            risk_level = self.get_risk_level(risk_scores[i])

            # Get driving style
            pred = predictions[i]
            if isinstance(pred, (int, np.integer)):
                driving_style = DRIVING_STYLE_CLASSES.get(pred, f"Class {pred}")
            else:
                driving_style = str(pred)

            # Get confidence
            confidence = probas[i].max() if probas is not None else 0.0

            # Analyze contributing factors (if features available)
            contributing_factors = {}
            if isinstance(X, pd.DataFrame) and feature_names:
                # Simple feature importance based on deviation from mean
                for col in feature_names[:5]:  # Top 5 features
                    if col in X.columns:
                        val = X.iloc[i][col]
                        mean_val = X[col].mean()
                        std_val = X[col].std()
                        if std_val > 0:
                            z_score = (val - mean_val) / std_val
                            if abs(z_score) > 1:
                                contributing_factors[col] = float(z_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_level, driving_style, contributing_factors
            )

            assessment = RiskAssessment(
                risk_score=float(risk_scores[i]),
                risk_level=risk_level,
                driving_style=driving_style,
                confidence=float(confidence),
                contributing_factors=contributing_factors,
                recommendations=recommendations,
            )
            assessments.append(assessment)

        return assessments

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        driving_style: str,
        contributing_factors: Dict[str, float],
    ) -> List[str]:
        """
        Generate safety recommendations based on risk assessment.

        Parameters:
        -----------
        risk_level : RiskLevel
            Current risk level
        driving_style : str
            Predicted driving style
        contributing_factors : dict
            Factors contributing to risk

        Returns:
        --------
        List[str]
            Safety recommendations
        """
        recommendations = []

        # General recommendations based on risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "URGENT: Significantly reduce speed and increase caution",
                "Consider taking a break if fatigued",
                "Review and improve overall driving habits",
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Reduce speed, especially in high-traffic areas",
                "Increase following distance",
                "Avoid aggressive maneuvers",
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Be more mindful of speed limits",
                "Practice smoother acceleration and braking",
            ])
        else:  # LOW
            recommendations.append("Continue maintaining safe driving habits")

        # Style-specific recommendations
        style_lower = driving_style.lower()
        if "aggressive" in style_lower:
            recommendations.append("Practice patience and avoid rushing")
            recommendations.append("Use turn signals consistently")
        elif "risky" in style_lower:
            recommendations.append("Enroll in a defensive driving course")
            recommendations.append("Consider using driver assistance features")

        # Factor-specific recommendations
        for factor, value in contributing_factors.items():
            if "speed" in factor.lower() and value > 1.5:
                recommendations.append("Reduce average driving speed")
            elif "braking" in factor.lower() and value > 1.5:
                recommendations.append("Anticipate stops to avoid harsh braking")
            elif "acceleration" in factor.lower() and value > 1.5:
                recommendations.append("Accelerate more gradually from stops")

        return recommendations[:5]  # Limit to 5 recommendations

    def plot_risk_distribution(
        self,
        risk_scores: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the distribution of risk scores.

        Parameters:
        -----------
        risk_scores : ndarray
            Array of risk scores
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        thresholds = [0, self.thresholds["low"], self.thresholds["medium"],
                      self.thresholds["high"], 1.0]

        ax1 = axes[0]
        n, bins, patches = ax1.hist(risk_scores, bins=50, edgecolor="white", alpha=0.7)

        # Color bars by risk level
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge < self.thresholds["low"]:
                patch.set_facecolor(colors[0])
            elif left_edge < self.thresholds["medium"]:
                patch.set_facecolor(colors[1])
            elif left_edge < self.thresholds["high"]:
                patch.set_facecolor(colors[2])
            else:
                patch.set_facecolor(colors[3])

        # Add threshold lines
        for thresh, label in zip(
            [self.thresholds["low"], self.thresholds["medium"], self.thresholds["high"]],
            ["Low/Medium", "Medium/High", "High/Critical"]
        ):
            ax1.axvline(x=thresh, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        ax1.set_xlabel("Risk Score", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Risk Score Distribution", fontsize=14, fontweight="bold")

        # Risk level pie chart
        ax2 = axes[1]
        risk_levels = self.get_risk_levels_batch(risk_scores)
        level_counts = pd.Series([r.value for r in risk_levels]).value_counts()

        # Ensure all levels present
        all_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        counts = [level_counts.get(level, 0) for level in all_levels]

        ax2.pie(
            counts,
            labels=[f"{level}\n({count})" for level, count in zip(all_levels, counts)],
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            explode=[0.02] * 4,
        )
        ax2.set_title("Risk Level Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_risk_factors(
        self,
        feature_importances: Dict[str, float],
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the most important risk factors.

        Parameters:
        -----------
        feature_importances : dict
            Feature importance scores
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save the plot
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#e74c3c" if imp > 0 else "#2ecc71" for imp in importances]

        bars = ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Impact on Risk Score", fontsize=12)
        ax.set_title("Top Risk Contributing Factors", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()


# =============================================================================
# RISK SCORE INTERPRETATION GUIDE
# =============================================================================

def print_risk_interpretation_guide():
    """
    Print a comprehensive guide for interpreting risk scores.
    """
    guide = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              ACCIDENT RISK SCORE INTERPRETATION GUIDE                 ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  RISK SCORE RANGE: 0.0 - 1.0                                         ║
    ║                                                                       ║
    ║  ┌─────────────────────────────────────────────────────────────────┐ ║
    ║  │ SCORE      │ LEVEL    │ INTERPRETATION                         │ ║
    ║  ├─────────────────────────────────────────────────────────────────┤ ║
    ║  │ 0.0 - 0.3  │ LOW      │ Safe driving behavior                  │ ║
    ║  │            │          │ - Consistent with traffic rules        │ ║
    ║  │            │          │ - Smooth acceleration/braking          │ ║
    ║  │            │          │ - Appropriate following distance       │ ║
    ║  ├─────────────────────────────────────────────────────────────────┤ ║
    ║  │ 0.3 - 0.6  │ MEDIUM   │ Moderate risk, room for improvement    │ ║
    ║  │            │          │ - Occasional speeding                  │ ║
    ║  │            │          │ - Some harsh braking events            │ ║
    ║  │            │          │ - Monitor and improve habits           │ ║
    ║  ├─────────────────────────────────────────────────────────────────┤ ║
    ║  │ 0.6 - 0.8  │ HIGH     │ Significant risk, attention needed     │ ║
    ║  │            │          │ - Frequent aggressive maneuvers        │ ║
    ║  │            │          │ - Consistent speed violations          │ ║
    ║  │            │          │ - Recommend intervention               │ ║
    ║  ├─────────────────────────────────────────────────────────────────┤ ║
    ║  │ 0.8 - 1.0  │ CRITICAL │ Very high accident probability         │ ║
    ║  │            │          │ - Dangerous driving patterns           │ ║
    ║  │            │          │ - Immediate action required            │ ║
    ║  │            │          │ - Consider driving restrictions        │ ║
    ║  └─────────────────────────────────────────────────────────────────┘ ║
    ║                                                                       ║
    ║  KEY FACTORS INFLUENCING RISK SCORE:                                 ║
    ║                                                                       ║
    ║  1. Speed Violations (25%)                                           ║
    ║     - Exceeding speed limits                                         ║
    ║     - High speed variance                                            ║
    ║                                                                       ║
    ║  2. Harsh Braking (20%)                                              ║
    ║     - Sudden deceleration events                                     ║
    ║     - Late braking patterns                                          ║
    ║                                                                       ║
    ║  3. Harsh Acceleration (15%)                                         ║
    ║     - Aggressive starts                                              ║
    ║     - Rapid speed increases                                          ║
    ║                                                                       ║
    ║  4. Aggressive Turns (15%)                                           ║
    ║     - Sharp cornering                                                ║
    ║     - Excessive lane changes                                         ║
    ║                                                                       ║
    ║  5. Following Distance (10%)                                         ║
    ║     - Tailgating behavior                                            ║
    ║     - Insufficient safety margin                                     ║
    ║                                                                       ║
    ║  6. Lane Discipline (10%)                                            ║
    ║     - Frequent lane changes                                          ║
    ║     - Lane departure events                                          ║
    ║                                                                       ║
    ║  7. Fatigue Indicators (5%)                                          ║
    ║     - Drowsiness patterns                                            ║
    ║     - Attention lapses                                               ║
    ║                                                                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(guide)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RISK SCORING MODULE DEMONSTRATION")
    print("=" * 60)

    # Print interpretation guide
    print_risk_interpretation_guide()

    # Create synthetic predictions for demonstration
    np.random.seed(42)
    n_samples = 500

    # Simulate model predictions (class probabilities)
    class MockModel:
        def predict(self, X):
            np.random.seed(42)
            return np.random.choice([0, 1, 2, 3], len(X), p=[0.5, 0.25, 0.15, 0.1])

        def predict_proba(self, X):
            np.random.seed(42)
            # Generate random probabilities
            probas = np.random.dirichlet([4, 2, 1, 0.5], len(X))
            return probas

    mock_model = MockModel()
    X_test = np.random.randn(n_samples, 10)

    # Initialize risk scorer
    scorer = RiskScorer()

    # Compute risk scores
    print("\nComputing risk scores...")
    risk_scores = scorer.compute_risk_score(mock_model, X_test)

    print(f"\nRisk Score Statistics:")
    print(f"  Mean: {risk_scores.mean():.3f}")
    print(f"  Std:  {risk_scores.std():.3f}")
    print(f"  Min:  {risk_scores.min():.3f}")
    print(f"  Max:  {risk_scores.max():.3f}")

    # Get risk levels
    risk_levels = scorer.get_risk_levels_batch(risk_scores)
    level_counts = pd.Series([r.value for r in risk_levels]).value_counts()

    print(f"\nRisk Level Distribution:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} ({count/len(risk_levels)*100:.1f}%)")

    # Plot distribution
    scorer.plot_risk_distribution(
        risk_scores,
        save_path=FIGURES_DIR / "risk_score_distribution.png",
    )

    # Detailed assessment example
    print("\n" + "-" * 60)
    print("SAMPLE DETAILED ASSESSMENT")
    print("-" * 60)

    assessments = scorer.assess_risk(mock_model, X_test[:5])

    for i, assessment in enumerate(assessments):
        print(f"\nDriver {i+1}:")
        print(f"  Risk Score: {assessment.risk_score:.3f}")
        print(f"  Risk Level: {assessment.risk_level.value}")
        print(f"  Driving Style: {assessment.driving_style}")
        print(f"  Confidence: {assessment.confidence:.3f}")
        print(f"  Recommendations:")
        for rec in assessment.recommendations[:3]:
            print(f"    - {rec}")

    print("\n✓ Risk scoring module demonstration complete!")
