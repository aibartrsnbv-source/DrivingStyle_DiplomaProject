"""
Exploratory Data Analysis Module for Driving Style ML Project
==============================================================

This module provides comprehensive exploratory data analysis capabilities
including statistical summaries, visualizations, and data quality reports.

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    FIGURES_DIR,
    VISUALIZATION_CONFIG,
    DRIVING_STYLE_CLASSES,
)


class ExploratoryDataAnalysis:
    """
    A comprehensive class for exploratory data analysis of driving behavior data.

    This class provides methods for:
    - Computing summary statistics
    - Generating visualizations
    - Analyzing class distributions
    - Creating correlation analyses
    - Producing data quality reports

    Attributes:
        df (pd.DataFrame): The dataset being analyzed
        target_column (str): Name of the target variable
        figures_dir (Path): Directory to save figures
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        figures_dir: Optional[Path] = None,
    ):
        """
        Initialize the EDA class.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        target_column : str, optional
            Name of the target column
        figures_dir : Path, optional
            Directory to save figures
        """
        self.df = df.copy()
        self.target_column = target_column
        self.figures_dir = figures_dir or FIGURES_DIR

        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(VISUALIZATION_CONFIG.get("color_palette", "husl"))

        # Identify column types
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    def summary_statistics(self, save_to_file: bool = False) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.

        Parameters:
        -----------
        save_to_file : bool
            Whether to save statistics to CSV

        Returns:
        --------
        pd.DataFrame
            Summary statistics table
        """
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        # Basic info
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        # Numeric statistics
        if len(self.numeric_columns) > 0:
            print(f"\n{'─'*60}")
            print("Numeric Features Statistics")
            print("─" * 60)

            numeric_stats = self.df[self.numeric_columns].describe()

            # Add additional statistics
            numeric_stats.loc["skewness"] = self.df[self.numeric_columns].skew()
            numeric_stats.loc["kurtosis"] = self.df[self.numeric_columns].kurtosis()
            numeric_stats.loc["missing"] = self.df[self.numeric_columns].isnull().sum()
            numeric_stats.loc["missing_%"] = (
                self.df[self.numeric_columns].isnull().sum() / len(self.df) * 100
            )

            print(numeric_stats.round(3).to_string())

            if save_to_file:
                numeric_stats.to_csv(self.figures_dir / "numeric_statistics.csv")

        # Categorical statistics
        if len(self.categorical_columns) > 0:
            print(f"\n{'─'*60}")
            print("Categorical Features Statistics")
            print("─" * 60)

            for col in self.categorical_columns:
                unique_count = self.df[col].nunique()
                missing = self.df[col].isnull().sum()
                top_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "N/A"
                top_freq = self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0

                print(f"\n{col}:")
                print(f"  Unique values: {unique_count}")
                print(f"  Missing: {missing} ({missing/len(self.df)*100:.1f}%)")
                print(f"  Most common: {top_value} ({top_freq:,})")

        return numeric_stats if len(self.numeric_columns) > 0 else pd.DataFrame()

    def class_distribution_analysis(
        self,
        target_column: Optional[str] = None,
        save_plot: bool = True,
    ) -> Dict[str, any]:
        """
        Analyze the distribution of target classes.

        Parameters:
        -----------
        target_column : str, optional
            Target column to analyze
        save_plot : bool
            Whether to save the plot

        Returns:
        --------
        Dict
            Distribution statistics and imbalance metrics
        """
        target = target_column or self.target_column

        if target is None or target not in self.df.columns:
            print("⚠ No valid target column specified")
            return {}

        print("\n" + "=" * 60)
        print(f"CLASS DISTRIBUTION ANALYSIS: {target}")
        print("=" * 60)

        # Calculate distribution
        value_counts = self.df[target].value_counts()
        percentages = self.df[target].value_counts(normalize=True) * 100

        # Display distribution
        print("\nClass Distribution:")
        print("-" * 40)
        for cls in value_counts.index:
            print(f"  {cls}: {value_counts[cls]:,} samples ({percentages[cls]:.2f}%)")

        # Calculate imbalance metrics
        max_class = value_counts.max()
        min_class = value_counts.min()
        imbalance_ratio = max_class / min_class

        print(f"\nImbalance Metrics:")
        print(f"  Majority class size: {max_class:,}")
        print(f"  Minority class size: {min_class:,}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 3:
            print("  ⚠ Significant class imbalance detected!")
            print("  Recommendation: Use SMOTE, class weights, or undersampling")

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        colors = sns.color_palette("husl", len(value_counts))
        bars = axes[0].bar(value_counts.index.astype(str), value_counts.values, color=colors)
        axes[0].set_xlabel("Class", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title(f"Class Distribution: {target}", fontsize=14, fontweight="bold")

        # Add value labels on bars
        for bar, val in zip(bars, value_counts.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_class * 0.01,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Pie chart
        axes[1].pie(
            value_counts.values,
            labels=[f"{cls}\n({pct:.1f}%)" for cls, pct in zip(value_counts.index, percentages)],
            colors=colors,
            autopct="",
            startangle=90,
            explode=[0.02] * len(value_counts),
        )
        axes[1].set_title("Class Proportions", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / f"class_distribution_{target}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Plot saved to: {save_path}")

        plt.show()

        return {
            "value_counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio > 3,
        }

    def correlation_analysis(
        self,
        method: str = "pearson",
        threshold: float = 0.7,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Perform correlation analysis on numeric features.

        Parameters:
        -----------
        method : str
            Correlation method: 'pearson', 'spearman', 'kendall'
        threshold : float
            Threshold for highlighting high correlations
        save_plot : bool
            Whether to save the heatmap

        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        if len(self.numeric_columns) == 0:
            print("⚠ No numeric columns for correlation analysis")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print(f"CORRELATION ANALYSIS ({method.upper()})")
        print("=" * 60)

        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_columns].corr(method=method)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": corr_val,
                    })

        if high_corr_pairs:
            print(f"\nHighly Correlated Feature Pairs (|r| >= {threshold}):")
            print("-" * 60)
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True):
                print(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
        else:
            print(f"\nNo feature pairs with |correlation| >= {threshold}")

        # Create correlation heatmap
        n_features = len(self.numeric_columns)
        fig_size = max(10, min(20, n_features * 0.5))

        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Generate heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=n_features <= 15,  # Only show annotations for smaller matrices
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            ax=ax,
        )

        ax.set_title(
            f"Feature Correlation Heatmap ({method.capitalize()})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / f"correlation_heatmap_{method}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Heatmap saved to: {save_path}")

        plt.show()

        return corr_matrix

    def feature_distributions(
        self,
        columns: Optional[List[str]] = None,
        n_cols: int = 3,
        save_plot: bool = True,
    ) -> None:
        """
        Plot distributions of numeric features.

        Parameters:
        -----------
        columns : list, optional
            Specific columns to plot
        n_cols : int
            Number of columns in subplot grid
        save_plot : bool
            Whether to save the plot
        """
        cols_to_plot = columns or self.numeric_columns

        if len(cols_to_plot) == 0:
            print("⚠ No numeric columns to plot")
            return

        print("\n" + "=" * 60)
        print("FEATURE DISTRIBUTIONS")
        print("=" * 60)

        n_features = len(cols_to_plot)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]

            # Plot histogram with KDE
            data = self.df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color="steelblue", alpha=0.7)

            # Add statistics annotation
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()

            stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
            ax.text(
                0.95, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("")

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / "feature_distributions.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Plot saved to: {save_path}")

        plt.show()

    def boxplots_by_class(
        self,
        features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        n_cols: int = 3,
        save_plot: bool = True,
    ) -> None:
        """
        Create boxplots of features grouped by target class.

        Parameters:
        -----------
        features : list, optional
            Features to plot
        target_column : str, optional
            Target column for grouping
        n_cols : int
            Number of columns in subplot grid
        save_plot : bool
            Whether to save the plot
        """
        target = target_column or self.target_column
        cols_to_plot = features or self.numeric_columns[:12]  # Limit for readability

        if target is None or target not in self.df.columns:
            print("⚠ No valid target column for grouped boxplots")
            return

        print("\n" + "=" * 60)
        print(f"BOXPLOTS BY CLASS: {target}")
        print("=" * 60)

        n_features = len(cols_to_plot)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(cols_to_plot):
            if col == target:
                continue

            ax = axes[idx]

            sns.boxplot(
                data=self.df,
                x=target,
                y=col,
                ax=ax,
                palette="husl",
            )

            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("")

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"Feature Distributions by {target}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / f"boxplots_by_{target}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Plot saved to: {save_path}")

        plt.show()

    def pairplot(
        self,
        features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        save_plot: bool = True,
    ) -> None:
        """
        Create pairplot for selected features.

        Parameters:
        -----------
        features : list, optional
            Features to include (max 5-6 for readability)
        target_column : str, optional
            Target column for coloring
        save_plot : bool
            Whether to save the plot
        """
        target = target_column or self.target_column
        cols_to_plot = features or self.numeric_columns[:5]

        print("\n" + "=" * 60)
        print("PAIRPLOT ANALYSIS")
        print("=" * 60)

        plot_data = self.df[cols_to_plot + ([target] if target else [])].dropna()

        if target and target in plot_data.columns:
            g = sns.pairplot(
                plot_data,
                hue=target,
                palette="husl",
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.6, "s": 30},
            )
        else:
            g = sns.pairplot(
                plot_data,
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.6},
            )

        g.fig.suptitle("Feature Pairplot", y=1.02, fontsize=14, fontweight="bold")

        if save_plot:
            save_path = self.figures_dir / "pairplot.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Plot saved to: {save_path}")

        plt.show()

    def missing_data_analysis(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze and visualize missing data patterns.

        Parameters:
        -----------
        save_plot : bool
            Whether to save the plot

        Returns:
        --------
        pd.DataFrame
            Missing data summary
        """
        print("\n" + "=" * 60)
        print("MISSING DATA ANALYSIS")
        print("=" * 60)

        # Calculate missing data statistics
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({
            "Missing Count": missing_counts,
            "Missing %": missing_pct,
        }).sort_values("Missing %", ascending=False)

        # Filter to columns with missing data
        missing_df = missing_df[missing_df["Missing Count"] > 0]

        if len(missing_df) == 0:
            print("\n✓ No missing data found in the dataset!")
            return pd.DataFrame()

        print("\nColumns with Missing Values:")
        print("-" * 40)
        for col in missing_df.index:
            print(f"  {col}: {missing_df.loc[col, 'Missing Count']:,} ({missing_df.loc[col, 'Missing %']:.2f}%)")

        # Create visualization
        if len(missing_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(missing_df) * 0.3)))

            # Bar plot of missing values
            colors = ["#ff6b6b" if pct > 30 else "#ffd93d" if pct > 10 else "#6bcb77"
                      for pct in missing_df["Missing %"]]

            axes[0].barh(missing_df.index, missing_df["Missing %"], color=colors)
            axes[0].set_xlabel("Missing (%)", fontsize=12)
            axes[0].set_title("Missing Data by Column", fontsize=14, fontweight="bold")
            axes[0].axvline(x=10, color="orange", linestyle="--", alpha=0.7, label="10%")
            axes[0].axvline(x=30, color="red", linestyle="--", alpha=0.7, label="30%")
            axes[0].legend()

            # Heatmap of missing pattern (sample)
            sample_size = min(100, len(self.df))
            sample_idx = np.random.choice(len(self.df), sample_size, replace=False)
            missing_matrix = self.df.iloc[sample_idx][missing_df.index].isnull()

            sns.heatmap(
                missing_matrix.T,
                cbar=False,
                cmap=["#e8e8e8", "#ff6b6b"],
                ax=axes[1],
            )
            axes[1].set_title("Missing Data Pattern (Sample)", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Samples")
            axes[1].set_ylabel("Features")

            plt.tight_layout()

            if save_plot:
                save_path = self.figures_dir / "missing_data_analysis.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"\n✓ Plot saved to: {save_path}")

            plt.show()

        return missing_df

    def outlier_analysis(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Detect and visualize outliers in numeric features.

        Parameters:
        -----------
        columns : list, optional
            Columns to analyze
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Detection threshold
        save_plot : bool
            Whether to save the plot

        Returns:
        --------
        pd.DataFrame
            Outlier summary
        """
        cols_to_analyze = columns or self.numeric_columns

        print("\n" + "=" * 60)
        print(f"OUTLIER ANALYSIS ({method.upper()} method)")
        print("=" * 60)

        outlier_summary = []

        for col in cols_to_analyze:
            data = self.df[col].dropna()

            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = data[(data < lower) | (data > upper)]
            else:  # zscore
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold]

            outlier_pct = len(outliers) / len(data) * 100

            outlier_summary.append({
                "Feature": col,
                "Outliers": len(outliers),
                "Outlier %": outlier_pct,
                "Min": data.min(),
                "Max": data.max(),
            })

        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df = outlier_df.sort_values("Outlier %", ascending=False)

        print("\nOutlier Summary:")
        print("-" * 60)
        for _, row in outlier_df.head(10).iterrows():
            status = "⚠" if row["Outlier %"] > 5 else "✓"
            print(f"  {status} {row['Feature']}: {row['Outliers']:,} outliers ({row['Outlier %']:.2f}%)")

        # Visualize top features with outliers
        top_outlier_cols = outlier_df[outlier_df["Outlier %"] > 0].head(6)["Feature"].tolist()

        if len(top_outlier_cols) > 0:
            n_cols = min(3, len(top_outlier_cols))
            n_rows = (len(top_outlier_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten() if len(top_outlier_cols) > 1 else [axes]

            for idx, col in enumerate(top_outlier_cols):
                ax = axes[idx]
                sns.boxplot(data=self.df, y=col, ax=ax, color="steelblue")
                ax.set_title(f"{col}", fontsize=11, fontweight="bold")

            for idx in range(len(top_outlier_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle("Outlier Detection (Boxplots)", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()

            if save_plot:
                save_path = self.figures_dir / "outlier_analysis.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"\n✓ Plot saved to: {save_path}")

            plt.show()

        return outlier_df

    def time_series_analysis(
        self,
        time_column: str,
        value_columns: List[str],
        save_plot: bool = True,
    ) -> None:
        """
        Analyze time-series patterns in the data.

        Parameters:
        -----------
        time_column : str
            Column containing timestamps
        value_columns : list
            Columns to analyze over time
        save_plot : bool
            Whether to save the plot
        """
        if time_column not in self.df.columns:
            print(f"⚠ Time column '{time_column}' not found")
            return

        print("\n" + "=" * 60)
        print("TIME SERIES ANALYSIS")
        print("=" * 60)

        # Convert to datetime if needed
        df_ts = self.df.copy()
        df_ts[time_column] = pd.to_datetime(df_ts[time_column], errors="coerce")
        df_ts = df_ts.sort_values(time_column)

        n_plots = len(value_columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
        axes = [axes] if n_plots == 1 else axes

        for idx, col in enumerate(value_columns):
            if col not in df_ts.columns:
                continue

            ax = axes[idx]

            # Plot time series
            ax.plot(df_ts[time_column], df_ts[col], alpha=0.7, linewidth=0.5)

            # Add rolling mean
            if len(df_ts) > 10:
                rolling_mean = df_ts[col].rolling(window=min(50, len(df_ts) // 10)).mean()
                ax.plot(df_ts[time_column], rolling_mean, color="red", linewidth=2, label="Rolling Mean")

            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f"Time Series: {col}", fontsize=12, fontweight="bold")
            ax.legend()

        axes[-1].set_xlabel("Time", fontsize=11)

        plt.suptitle("Time Series Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_plot:
            save_path = self.figures_dir / "time_series_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n✓ Plot saved to: {save_path}")

        plt.show()

    def generate_full_report(self, save_plots: bool = True) -> Dict[str, any]:
        """
        Generate a complete EDA report.

        Parameters:
        -----------
        save_plots : bool
            Whether to save all plots

        Returns:
        --------
        Dict
            Complete EDA results
        """
        print("\n" + "=" * 60)
        print("GENERATING FULL EDA REPORT")
        print("=" * 60)

        report = {}

        # 1. Summary Statistics
        report["summary_stats"] = self.summary_statistics(save_to_file=save_plots)

        # 2. Missing Data Analysis
        report["missing_data"] = self.missing_data_analysis(save_plot=save_plots)

        # 3. Class Distribution (if target exists)
        if self.target_column:
            report["class_distribution"] = self.class_distribution_analysis(save_plot=save_plots)

        # 4. Feature Distributions
        self.feature_distributions(save_plot=save_plots)

        # 5. Correlation Analysis
        report["correlation"] = self.correlation_analysis(save_plot=save_plots)

        # 6. Outlier Analysis
        report["outliers"] = self.outlier_analysis(save_plot=save_plots)

        # 7. Boxplots by Class (if target exists)
        if self.target_column:
            self.boxplots_by_class(save_plot=save_plots)

        print("\n" + "=" * 60)
        print("EDA REPORT COMPLETE")
        print("=" * 60)
        print(f"Figures saved to: {self.figures_dir}")

        return report


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

def quick_eda(df: pd.DataFrame, target_column: Optional[str] = None) -> None:
    """
    Perform quick exploratory data analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    target_column : str, optional
        Target column name
    """
    print("\n" + "=" * 60)
    print("QUICK EDA SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Data types
    print(f"\nColumn Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count}")

    # Missing values
    missing_total = df.isnull().sum().sum()
    print(f"\nMissing Values: {missing_total:,} ({missing_total/(df.shape[0]*df.shape[1])*100:.2f}%)")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates:,}")

    # Target distribution
    if target_column and target_column in df.columns:
        print(f"\nTarget Distribution ({target_column}):")
        for val, count in df[target_column].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {val}: {count:,} ({pct:.1f}%)")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Features Summary ({len(numeric_cols)} columns):")
        print(df[numeric_cols].describe().round(2).to_string())


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importance scores.

    Parameters:
    -----------
    feature_names : list
        Names of features
    importance_scores : array
        Importance scores
    title : str
        Plot title
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]

    plt.figure(figsize=(10, max(6, top_n * 0.3)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]

    plt.barh(
        range(len(indices)),
        importance_scores[indices][::-1],
        color=colors,
    )

    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices[::-1]],
    )

    plt.xlabel("Importance Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Plot saved to: {save_path}")

    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Demonstration with synthetic data
    print("\n" + "=" * 60)
    print("EDA MODULE DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 500

    sample_df = pd.DataFrame({
        "speed": np.random.exponential(50, n_samples),
        "acceleration": np.random.normal(0, 2, n_samples),
        "harsh_braking": np.random.poisson(2, n_samples),
        "lane_changes": np.random.poisson(3, n_samples),
        "following_distance": np.random.normal(2, 0.5, n_samples),
        "trip_duration": np.random.exponential(30, n_samples),
        "driving_style": np.random.choice(
            ["safe", "normal", "aggressive"],
            n_samples,
            p=[0.6, 0.25, 0.15],
        ),
    })

    # Add some missing values
    sample_df.loc[np.random.choice(n_samples, 30), "following_distance"] = np.nan

    # Run EDA
    eda = ExploratoryDataAnalysis(sample_df, target_column="driving_style")
    report = eda.generate_full_report(save_plots=True)

    print("\n✓ EDA demonstration complete!")
