#!/usr/bin/env python3
"""
Stacked ensemble: XGBoost + LightGBM + GradientBoosting + RandomForest
с мета-моделью LogisticRegression, обученной на out-of-fold предсказаниях.

Цель: побить одиночный XGBoost (test F1-macro = 0.6298 в честном тесте).
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR
from src.data_loader import load_and_unify_datasets
from src.preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore")

N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_data():
    print("  Загрузка датасета...")
    df = load_and_unify_datasets()
    print(f"  ✓ {len(df)} строк")

    trip_ids = df["trip_id"].copy()

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_pipeline(
        df, target_column="driving_style_encoded"
    )

    assert isinstance(X_train, pd.DataFrame), (
        f"Ожидался DataFrame, получено {type(X_train)}"
    )

    try:
        groups_train = trip_ids.loc[X_train.index].values
        groups_val   = trip_ids.loc[X_val.index].values
    except Exception:
        groups_train = np.arange(len(X_train)) // 5
        groups_val   = np.arange(len(X_val))   // 5

    def arr(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    return (X_train, X_val, X_test,
            arr(y_train), arr(y_val), arr(y_test),
            groups_train, groups_val)


# ---------------------------------------------------------------------------
# Feature engineering v3 (копия из tune_v3.py)
# ---------------------------------------------------------------------------

def feature_engineering_v3(X_train_df, X_val_df, X_test_df):
    """Корреляция>0.97 удалена + 7 interactions по именам."""
    corr  = X_train_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            v = upper.loc[idx, col]
            if pd.notna(v) and v > 0.97:
                mc = corr[col].mean()
                mi = corr[idx].mean()
                to_drop.add(col if mc > mi else idx)

    X_train_df = X_train_df.drop(columns=list(to_drop))
    X_val_df   = X_val_df.drop(columns=list(to_drop))
    X_test_df  = X_test_df.drop(columns=list(to_drop))
    print(f"  ✗ Убрано {len(to_drop)} дубликатов (|r|>0.97)")

    interactions = [
        ("speed_x_brakes",      "avg_speed",           "harsh_braking_count"),
        ("speed_x_accels",      "avg_speed",           "harsh_accel_count"),
        ("aggression_combined", "harsh_braking_count", "harsh_accel_count"),
        ("gyro_speed_ratio",    "gyro_y_std",          "avg_speed"),
        ("gyro_total",          "gyro_x_std",          "gyro_z_std"),
        ("accel_total",         "accel_x_mean",        "accel_y_mean"),
    ]

    added = []
    for new_name, a, b in interactions:
        if a in X_train_df.columns and b in X_train_df.columns:
            X_train_df[new_name] = X_train_df[a] * X_train_df[b]
            X_val_df[new_name]   = X_val_df[a]   * X_val_df[b]
            X_test_df[new_name]  = X_test_df[a]  * X_test_df[b]
            added.append(new_name)
        else:
            missing = a if a not in X_train_df.columns else b
            print(f"  ⚠ Пропуск {new_name}: '{missing}' не найден")

    if "max_speed" in X_train_df.columns and "avg_speed" in X_train_df.columns:
        X_train_df["speed_accel_diff"] = X_train_df["max_speed"] - X_train_df["avg_speed"]
        X_val_df["speed_accel_diff"]   = X_val_df["max_speed"]   - X_val_df["avg_speed"]
        X_test_df["speed_accel_diff"]  = X_test_df["max_speed"]  - X_test_df["avg_speed"]
        added.append("speed_accel_diff")

    print(f"  ✓ Добавлено {len(added)} interactions: {added}")
    print(f"  Итого фич: {X_train_df.shape[1]}")
    return X_train_df, X_val_df, X_test_df


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def load_best_params():
    path = MODELS_DIR / "best_hyperparameters_v3.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала запусти tune_v3.py."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["best_params"]


def make_base_models(best_params):
    # XGBoost
    xgb_p = {k: v for k, v in best_params["xgboost"].items()
              if k != "safe_class_boost"}
    xgb_p.update({"random_state": RANDOM_SEED, "eval_metric": "mlogloss", "verbosity": 0})

    # GradientBoosting
    gb_p = {k: v for k, v in best_params["gradient_boosting"].items()
            if k != "safe_class_boost"}
    gb_p["random_state"] = RANDOM_SEED

    # LightGBM
    lgb_p = {k: v for k, v in best_params["lightgbm"].items()
             if k != "safe_class_boost"}
    lgb_p.update({"random_state": RANDOM_SEED, "verbosity": -1, "n_jobs": -1})

    # RandomForest
    rf_p = {
        "n_estimators":    300,
        "max_depth":       15,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "max_features":    "sqrt",
        "class_weight":    "balanced_subsample",
        "random_state":    RANDOM_SEED,
        "n_jobs":          -1,
    }

    return {
        "xgboost":           xgb.XGBClassifier(**xgb_p),
        "gradient_boosting": GradientBoostingClassifier(**gb_p),
        "lightgbm":          lgb.LGBMClassifier(**lgb_p),
        "random_forest":     RandomForestClassifier(**rf_p),
    }


# ---------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------

def generate_oof_predictions(base_models, X, y, groups, cv):
    """
    GroupKFold OOF: каждая базовая модель обучается на out-of-fold данных
    и предсказывает вероятности на in-fold. Результат — честные OOF-предсказания
    без утечек данных для обучения мета-модели.
    """
    X_np    = X.values if hasattr(X, "values") else np.asarray(X)
    n_samp  = len(X_np)
    n_cls   = len(np.unique(y))

    oof_preds   = {name: np.zeros((n_samp, n_cls)) for name in base_models}
    fold_scores = {name: [] for name in base_models}

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_np, y, groups=groups)):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}:  "
              f"train={len(tr_idx)}, val={len(va_idx)}, "
              f"classes train={dict(zip(*np.unique(y_tr, return_counts=True)))}")

        for name, model_tmpl in base_models.items():
            model = clone(model_tmpl)
            try:
                model.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                model.fit(X_tr, y_tr)

            proba  = model.predict_proba(X_va)
            oof_preds[name][va_idx] = proba

            f1 = f1_score(y_va, np.argmax(proba, axis=1), average="macro", zero_division=0)
            fold_scores[name].append(f1)
            print(f"    {name:<22} F1-macro={f1:.4f}")

    print("\n  ─── Средние OOF F1-macro по базовым моделям ───")
    for name, scores in fold_scores.items():
        print(f"    {name:<22} mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")

    return oof_preds, fold_scores


# ---------------------------------------------------------------------------
# Meta-model
# ---------------------------------------------------------------------------

def build_meta_features(oof_preds):
    """Конкатенирует OOF-предсказания всех моделей (в алфавитном порядке)."""
    return np.hstack([oof_preds[n] for n in sorted(oof_preds)])


def train_meta_model(meta_features, y):
    meta = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=RANDOM_SEED,
    )
    meta.fit(meta_features, y)
    return meta


# ---------------------------------------------------------------------------
# Final models on full train+val
# ---------------------------------------------------------------------------

def train_final_base_models(base_models, X, y):
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    sw   = compute_sample_weight(class_weight="balanced", y=y)

    trained = {}
    for name, model_tmpl in base_models.items():
        model = clone(model_tmpl)
        try:
            model.fit(X_np, y, sample_weight=sw)
        except TypeError:
            model.fit(X_np, y)
        trained[name] = model
        print(f"  ✓ {name}")
    return trained


def predict_test(trained_base, meta_model, X_test):
    X_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    base_probas = []
    base_preds  = {}
    for name in sorted(trained_base):
        proba = trained_base[name].predict_proba(X_np)
        base_probas.append(proba)
        base_preds[name] = np.argmax(proba, axis=1)

    meta_pred = meta_model.predict(np.hstack(base_probas))
    return meta_pred, base_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STACKED ENSEMBLE — XGBoost + LightGBM + GB + RF → LogReg meta")
    print("=" * 70)

    # ── Данные ──────────────────────────────────────────────────────────────
    print("\n── Загрузка данных ──")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     g_train, g_val) = prepare_data()

    # ── Feature engineering v3 ───────────────────────────────────────────────
    print("\n── Feature Engineering v3 ──")
    X_train, X_val, X_test = feature_engineering_v3(X_train, X_val, X_test)

    # Объединяем train+val для OOF
    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val])
    g_full = np.concatenate([g_train, g_val])

    print(f"\n  Full train+val : {X_full.shape}")
    print(f"  Test           : {X_test.shape}")
    print(f"  Уникальных групп (CV): {len(np.unique(g_full))}")
    print(f"  Классы full:    {dict(zip(*np.unique(y_full, return_counts=True)))}")

    # ── Модели ──────────────────────────────────────────────────────────────
    print("\n── Загрузка лучших гиперпараметров (v3) ──")
    best_params = load_best_params()
    base_models = make_base_models(best_params)
    print(f"  Базовые модели: {list(base_models.keys())}")

    # ── OOF predictions ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ — OUT-OF-FOLD PREDICTIONS (GroupKFold, 5 folds)")
    print("=" * 70)

    cv = GroupKFold(n_splits=N_FOLDS)
    oof_preds, fold_scores = generate_oof_predictions(
        base_models, X_full, y_full, g_full, cv
    )

    # ── Мета-модель ──────────────────────────────────────────────────────────
    print("\n── Обучение мета-модели на OOF-предсказаниях ──")
    meta_features_train = build_meta_features(oof_preds)
    print(f"  Размерность meta-фич: {meta_features_train.shape}  "
          f"(4 модели × 3 класса = 12 признаков)")
    meta_model = train_meta_model(meta_features_train, y_full)
    print("  ✓ Meta LogisticRegression обучена")

    meta_oof_pred  = meta_model.predict(meta_features_train)
    stack_oof_f1   = f1_score(y_full, meta_oof_pred, average="macro", zero_division=0)
    stack_oof_acc  = accuracy_score(y_full, meta_oof_pred)
    print(f"\n  STACK OOF F1-macro : {stack_oof_f1:.4f}")
    print(f"  STACK OOF accuracy : {stack_oof_acc:.4f}")
    print("  (OOF — оптимистичная оценка; честный тест — ниже)")

    # ── Финальные базовые модели на полном train+val ──────────────────────────
    print("\n" + "=" * 70)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ — базовые модели на полном train+val")
    print("=" * 70)
    trained_base = train_final_base_models(base_models, X_full, y_full)

    # ── Оценка на test ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ОЦЕНКА НА TEST SET")
    print("=" * 70)

    meta_test_pred, base_test_preds = predict_test(trained_base, meta_model, X_test)

    print("\nБазовые модели по отдельности (test):")
    base_results = {}
    for name, pred in sorted(base_test_preds.items()):
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="macro", zero_division=0)
        base_results[name] = {"accuracy": float(acc), "f1_macro": float(f1)}
        print(f"  {name:<22} F1={f1:.4f}  acc={acc:.4f}")

    stack_test_f1  = f1_score(y_test, meta_test_pred, average="macro", zero_division=0)
    stack_test_acc = accuracy_score(y_test, meta_test_pred)

    print(f"\n  >>> STACKED ENSEMBLE <<<")
    print(f"  Test F1-macro  : {stack_test_f1:.4f}")
    print(f"  Test accuracy  : {stack_test_acc:.4f}")
    print("\n  Stack classification report:")
    print(classification_report(y_test, meta_test_pred, digits=3, zero_division=0))

    # ── Сохранение ────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": ts,
        "method": "stacking (XGB+LGB+GB+RF → LogReg) with GroupKFold OOF",
        "base_oof_f1_mean": {n: float(np.mean(s)) for n, s in fold_scores.items()},
        "base_test_results": base_results,
        "stack_oof_f1":       float(stack_oof_f1),
        "stack_test_f1":      float(stack_test_f1),
        "stack_test_accuracy": float(stack_test_acc),
    }
    out_path = MODELS_DIR / "stacking_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Сохранено: {out_path}")

    # ── Финальная сводка ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("СВОДКА: одиночные модели vs STACKED")
    print("=" * 70)
    print(f"{'Модель':<24} {'Test F1-macro':<16}")
    print("─" * 42)

    sorted_base = sorted(base_results.items(),
                         key=lambda kv: kv[1]["f1_macro"], reverse=True)
    for name, res in sorted_base:
        marker = " ← best single" if name == sorted_base[0][0] else ""
        print(f"{name:<24} {res['f1_macro']:.4f}{marker}")
    print(f"{'STACKED ENSEMBLE':<24} {stack_test_f1:.4f}")

    best_name, best_res = sorted_base[0]
    delta = stack_test_f1 - best_res["f1_macro"]
    sign  = "+" if delta >= 0 else ""
    print(f"\nПрирост стэка над лучшей одиночной ({best_name}): {sign}{delta:.4f}")
    print(f"Baseline V1 XGBoost: 0.5858  |  V3 XGBoost: 0.6298")

    if delta >= 0.02:
        print("✓ Стэк зашёл — переносим в продакшен")
    elif delta >= 0:
        print("⚠ Стэк чуть лучше, но прирост маленький — на грани")
    else:
        print("✗ Стэк хуже одиночной модели — нужен другой подход")


if __name__ == "__main__":
    main()
