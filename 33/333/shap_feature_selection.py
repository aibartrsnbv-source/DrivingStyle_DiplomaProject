#!/usr/bin/env python3
"""
SHAP-based feature selection for XGBoost.
Анализ важности признаков через TreeExplainer,
отбор топ-30 и переобучение на сокращённом наборе.
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — не нужен дисплей
import matplotlib.pyplot as plt
from pathlib import Path

import shap
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR, ML_MODELS_CONFIG, FIGURES_DIR
from src.data_loader import load_and_unify_datasets
from src.preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore")

TOP_K = 30  # сколько фичей оставляем


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data():
    df = load_and_unify_datasets()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_pipeline(
        df, target_column="driving_style_encoded"
    )

    assert isinstance(X_train, pd.DataFrame), (
        f"Ожидался DataFrame, получено {type(X_train)}"
    )

    def arr(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    return X_train, X_val, X_test, arr(y_train), arr(y_val), arr(y_test)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, params=None):
    """Обучает XGBoost с balanced sample weights."""
    if params is None:
        params = {k: v for k, v in ML_MODELS_CONFIG["xgboost"].items()}
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(**params)
    X_np = X_train.values if hasattr(X_train, "values") else X_train
    model.fit(X_np, y_train, sample_weight=sw)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, model, X, y):
    X_np = X.values if hasattr(X, "values") else X
    y_pred = model.predict(X_np)
    acc      = accuracy_score(y, y_pred)
    f1_w     = f1_score(y, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro",    zero_division=0)
    print(f"\n{name}:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (weighted)   : {f1_w:.4f}")
    print(f"  F1 (macro)      : {f1_macro:.4f}")
    print("  Classification report:")
    print(classification_report(y, y_pred, digits=3, zero_division=0))
    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_macro}


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_importance(model, X_test, feature_names):
    """
    Вычисляет SHAP values через TreeExplainer.
    Поддерживает старый API (list of arrays) и новый (3-D ndarray).
    Возвращает: importance_df (sorted desc), raw shap_values.
    """
    print("\n  Вычисляем SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    X_np = X_test.values if hasattr(X_test, "values") else X_test
    shap_values = explainer.shap_values(X_np)

    # --- нормализуем к mean |SHAP| по всем классам ---
    if isinstance(shap_values, list):
        # Старый API: list[n_classes] где каждый элемент (n_samples, n_features)
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Новый API: (n_samples, n_features, n_classes)  или  (n_classes, n_samples, n_features)
        n_feat = len(feature_names)
        if shap_values.shape[1] == n_feat:
            # (n_samples, n_features, n_classes)
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        elif shap_values.shape[2] == n_feat:
            # (n_classes, n_samples, n_features)  — редко, но бывает
            mean_abs = np.abs(shap_values).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0).ravel()[:n_feat]
    else:
        # 2-D fallback
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, shap_values


def _shap_values_for_class(shap_values, class_idx, n_features):
    """Вытаскивает SHAP-матрицу (n_samples, n_features) для одного класса."""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    if shap_values.ndim == 3:
        if shap_values.shape[1] == n_features:
            return shap_values[:, :, class_idx]   # (n_samples, n_features, n_classes)
        else:
            return shap_values[class_idx]          # (n_classes, n_samples, n_features)
    return shap_values


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_top_features(importance_df, save_path, top_n=20):
    """Горизонтальный bar chart топ-N признаков."""
    top = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["mean_abs_shap"].values[::-1], color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} features by SHAP importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Bar chart сохранён: {save_path}")


def plot_shap_summary(shap_values, X_test_np, feature_names, save_path):
    """Стандартный shap.summary_plot (beeswarm) для класса 1."""
    try:
        n_feat = len(feature_names)
        sv = _shap_values_for_class(shap_values, 1, n_feat)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X_test_np, feature_names=feature_names,
            max_display=20, show=False, plot_size=None,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ SHAP summary plot сохранён: {save_path}")
    except Exception as e:
        print(f"  ⚠ SHAP summary plot не построился: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SHAP FEATURE SELECTION для XGBoost")
    print("=" * 70)

    # ── Загрузка ─────────────────────────────────────────────────────────
    print("\n── Загрузка данных ──")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    feature_names = list(X_train.columns)
    n_features = len(feature_names)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Признаков: {n_features}")

    # Объединяем train+val для обучения финальной модели
    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val])

    # ── Baseline на всех признаках ────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BASELINE — XGBoost на ВСЕХ {n_features} признаках")
    print("=" * 70)
    model_full = train_xgb(X_full, y_full)
    baseline_metrics = evaluate(
        f"XGBoost (все {n_features} признаков)", model_full, X_test, y_test
    )

    # ── SHAP analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SHAP АНАЛИЗ")
    print("=" * 70)
    importance_df, shap_values = compute_shap_importance(
        model_full, X_test, feature_names
    )
    X_test_np = X_test.values if hasattr(X_test, "values") else X_test

    print(f"\n  Топ-20 самых важных признаков:")
    for i, row in importance_df.head(20).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<45} {row['mean_abs_shap']:.4f}")

    print(f"\n  Топ-10 наименее важных (кандидаты на удаление):")
    for i, row in importance_df.tail(10).iterrows():
        print(f"        {row['feature']:<45} {row['mean_abs_shap']:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_features(
        importance_df, FIGURES_DIR / "shap_top20_features.png", top_n=20
    )
    plot_shap_summary(
        shap_values, X_test_np, feature_names,
        FIGURES_DIR / "shap_summary_plot.png",
    )

    # ── Топ-K отбор и переобучение ────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ПЕРЕОБУЧЕНИЕ на ТОП-{TOP_K} признаков")
    print("=" * 70)

    top_features = importance_df.head(TOP_K)["feature"].tolist()
    print(f"  Оставляем {TOP_K} из {n_features} признаков:")
    for i, f in enumerate(top_features, 1):
        print(f"    {i:2d}. {f}")

    X_train_sel = X_train[top_features]
    X_val_sel   = X_val[top_features]
    X_test_sel  = X_test[top_features]
    X_full_sel  = pd.concat([X_train_sel, X_val_sel], axis=0).reset_index(drop=True)

    model_selected = train_xgb(X_full_sel, y_full)
    selected_metrics = evaluate(
        f"XGBoost (топ-{TOP_K} признаков)", model_selected, X_test_sel, y_test
    )

    # ── Сводка ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"СВОДКА: все {n_features} признаков vs топ-{TOP_K}")
    print("=" * 70)
    print(f"{'Метрика':<20} {'Все ' + str(n_features):<12} {'Топ-' + str(TOP_K):<12} {'Δ':<10}")
    print("─" * 56)
    for key in ["accuracy", "f1_weighted", "f1_macro"]:
        b = baseline_metrics[key]
        s = selected_metrics[key]
        delta = s - b
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {b:<12.4f} {s:<12.4f} {sign}{delta:.4f}")

    if selected_metrics["f1_weighted"] > baseline_metrics["f1_weighted"]:
        verdict = "✓ Отбор признаков улучшил F1. Можно переносить в config."
    elif selected_metrics["f1_weighted"] >= baseline_metrics["f1_weighted"] - 0.005:
        verdict = "⚠ Результат почти одинаковый (±0.5%). Модель проще без потерь — это плюс."
    else:
        verdict = "✗ Результат хуже. Оставляем как было, отбор не помог."
    print(f"\n{verdict}")

    # ── Сохранение ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "n_features_total":    n_features,
        "n_features_selected": TOP_K,
        "baseline_metrics":    baseline_metrics,
        "selected_metrics":    selected_metrics,
        "top_features":        top_features,
        "feature_importance":  importance_df.to_dict(orient="records"),
    }
    out_path = MODELS_DIR / "shap_feature_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ JSON сохранён  : {out_path}")
    print(f"✓ Графики в      : {FIGURES_DIR}")
    print(f"    shap_top20_features.png")
    print(f"    shap_summary_plot.png")


if __name__ == "__main__":
    main()
