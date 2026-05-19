#!/usr/bin/env python3
"""
Feature engineering v2 + class weight tuning + Optuna.
Не модифицирует исходный pipeline — все улучшения в этом скрипте.
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR
from src.data_loader import load_and_unify_datasets
from src.preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_data_v2():
    df = load_and_unify_datasets()
    print(f"  Загружено: {len(df)} строк")

    trip_ids = df["trip_id"].copy() if "trip_id" in df.columns else None

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_pipeline(
        df, target_column="driving_style_encoded"
    )

    # feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        n_feat = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train.columns)
        feature_names = [f"f_{i}" for i in range(n_feat)]

    # groups — X_train/X_val are still DataFrames with original indices here
    groups_train = groups_val = None
    if trip_ids is not None:
        try:
            idx_train = X_train.index if hasattr(X_train, "index") else y_train.index
            idx_val   = X_val.index   if hasattr(X_val,   "index") else y_val.index
            groups_train = trip_ids.loc[idx_train].values
            groups_val   = trip_ids.loc[idx_val].values
            print(f"  ✓ trip_id: train={len(np.unique(groups_train))} групп, "
                  f"val={len(np.unique(groups_val))} групп")
        except Exception as e:
            print(f"  ⚠ trip_id не восстановлен ({e}), pseudo-groups")

    if groups_train is None:
        n_tr = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
        n_va = X_val.shape[0]   if hasattr(X_val,   "shape") else len(X_val)
        groups_train = np.arange(n_tr) // 5
        groups_val   = np.arange(n_va) // 5
        print("  ⚠ Используем pseudo-groups (по 5 строк)")

    # to numpy
    def to_arr(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    return (
        to_arr(X_train), to_arr(X_val), to_arr(X_test),
        to_arr(y_train), to_arr(y_val), to_arr(y_test),
        groups_train, groups_val, feature_names,
    )


# ---------------------------------------------------------------------------
# Feature engineering v2
# ---------------------------------------------------------------------------

def feature_engineering_v2(X_train, X_val, X_test, feature_names):
    """
    1) Убирает константные фичи (std < 1e-5)
    2) Убирает одну из каждой высоко коррелированной пары (|r| > 0.92)
    3) Добавляет interaction features

    Возвращает: (X_train_new, X_val_new, X_test_new, new_feature_names, summary)
    """
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_val   = pd.DataFrame(X_val,   columns=feature_names)
    df_test  = pd.DataFrame(X_test,  columns=feature_names)

    summary = {}

    # ── 1. Константные фичи ─────────────────────────────────────────────────
    stds = df_train.std()
    constant_cols = stds[stds < 1e-5].index.tolist()
    summary["removed_constant"] = constant_cols
    df_train = df_train.drop(columns=constant_cols)
    df_val   = df_val.drop(columns=constant_cols)
    df_test  = df_test.drop(columns=constant_cols)
    print(f"  ✗ Убрано {len(constant_cols)} константных: "
          f"{constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}")

    # ── 2. Высоко коррелированные пары ──────────────────────────────────────
    corr  = df_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            if pd.notna(upper.loc[idx, col]) and upper.loc[idx, col] > 0.92:
                victim = col if col > idx else idx
                to_drop.add(victim)

    summary["removed_correlated"] = list(to_drop)
    df_train = df_train.drop(columns=list(to_drop))
    df_val   = df_val.drop(columns=list(to_drop))
    df_test  = df_test.drop(columns=list(to_drop))
    print(f"  ✗ Убрано {len(to_drop)} сильно коррелированных (|r|>0.92)")

    # ── 3. Interaction features ──────────────────────────────────────────────
    def safe_get(df, name):
        if name in df.columns:
            return df[name]
        matches = [c for c in df.columns if name in c]
        return df[matches[0]] if matches else None

    pairs = [
        ("speed_x_brakes",      "avg_speed",          "harsh_braking_count"),
        ("speed_x_accels",      "avg_speed",          "harsh_accel_count"),
        ("gyro_speed_ratio",    "gyro_y_std",         "avg_speed"),
        ("accel_x_speed_var",   "accel_x_mean",       "speed_variance"),
        ("aggression_combined", "harsh_braking_count", "harsh_accel_count"),
        ("gyro_combined",       "gyro_x_std",         "gyro_z_std"),
    ]

    added = []
    for new_name, col_a, col_b in pairs:
        a_tr = safe_get(df_train, col_a)
        b_tr = safe_get(df_train, col_b)
        if a_tr is not None and b_tr is not None:
            a_val = safe_get(df_val,  col_a)
            b_val = safe_get(df_val,  col_b)
            a_te  = safe_get(df_test, col_a)
            b_te  = safe_get(df_test, col_b)
            df_train[new_name] = (a_tr * b_tr).astype(np.float32)
            df_val[new_name]   = (a_val * b_val).astype(np.float32)
            df_test[new_name]  = (a_te  * b_te).astype(np.float32)
            added.append(new_name)

    summary["added_interactions"] = added
    print(f"  ✓ Добавлено {len(added)} interaction features: {added}")
    print(f"  Итого фич: {len(feature_names)} → {len(df_train.columns)}")

    new_feature_names = list(df_train.columns)
    return (df_train.values, df_val.values, df_test.values, new_feature_names, summary)


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def make_sample_weights(y, safe_class_boost=1.8):
    """
    Класс 0 (Safe) недопредставлен и плохо предсказывается.
    Дополнительно усиливаем его вес поверх 'balanced'.
    """
    base_weights = compute_sample_weight(class_weight="balanced", y=y)
    boosted = base_weights.copy()
    boosted[y == 0] *= safe_class_boost
    return boosted


# ---------------------------------------------------------------------------
# Optuna objectives (manual CV для передачи sample_weight)
# ---------------------------------------------------------------------------

def objective_xgboost_v2(trial, X, y, groups, cv):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "random_state":     RANDOM_SEED,
        "eval_metric":      "mlogloss",
        "verbosity":        0,
    }
    safe_boost = trial.suggest_float("safe_class_boost", 1.0, 2.5)

    scores = []
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        sw_tr = make_sample_weights(y_tr, safe_class_boost=safe_boost)

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        y_pred = model.predict(X_va)
        scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))

    return np.mean(scores)


def objective_gb_v2(trial, X, y, groups, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 350, step=50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 8),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state":      RANDOM_SEED,
    }
    safe_boost = trial.suggest_float("safe_class_boost", 1.0, 2.5)

    scores = []
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        sw_tr = make_sample_weights(y_tr, safe_class_boost=safe_boost)

        model = GradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
        y_pred = model.predict(X_va)
        scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))

    return np.mean(scores)


# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------

def evaluate_final(name, model, X_train_val, y_train_val, X_test, y_test, sample_weight):
    print(f"\n{name}:")
    model.fit(X_train_val, y_train_val, sample_weight=sample_weight)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Test F1-macro : {f1:.4f}")
    print("  Classification report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    return {"accuracy": acc, "f1_macro": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TUNE V2 — Feature engineering + class weight boost")
    print("=" * 70)

    # ── Загрузка ─────────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     g_train, g_val, fnames) = prepare_data_v2()

    print(f"\nИсходные данные: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── Feature engineering v2 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING V2")
    print("=" * 70)
    X_train, X_val, X_test, fnames_new, fe_summary = feature_engineering_v2(
        X_train, X_val, X_test, fnames
    )
    print(f"\nПосле FE: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── Объединяем train+val для CV ───────────────────────────────────────────
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    g_full = np.concatenate([g_train, g_val])
    n_groups = len(np.unique(g_full))
    print(f"\nCV data: {X_full.shape}, {n_groups} уникальных групп, {CV_FOLDS} folds")

    cv = GroupKFold(n_splits=CV_FOLDS)

    # ── Optuna tuning ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTUNA TUNING V2 (with sample weights + Safe class boost)")
    print("=" * 70)

    studies = {}

    for model_name, obj_fn in [("xgboost", objective_xgboost_v2),
                                ("gradient_boosting", objective_gb_v2)]:
        print(f"\n─── Tuning: {model_name} ───")
        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _make_cb(total):
            def cb(study, trial):
                if trial.number % 10 == 0 or trial.number == total - 1:
                    best = study.best_value if study.best_trial else 0
                    print(f"  Trial {trial.number + 1:>3}/{total} | "
                          f"current={trial.value:.4f} | best={best:.4f}")
            return cb

        study.optimize(
            lambda t, fn=obj_fn: fn(t, X_full, y_full, g_full, cv),
            n_trials=N_TRIALS,
            callbacks=[_make_cb(N_TRIALS)],
            show_progress_bar=False,
        )
        studies[model_name] = study
        print(f"  Best CV F1-macro: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ОЦЕНКА НА TEST SET (V2)")
    print("=" * 70)

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    test_results = {}

    # XGBoost
    xgb_best_params = {k: v for k, v in studies["xgboost"].best_params.items()
                       if k != "safe_class_boost"}
    xgb_best_params.update({
        "random_state": RANDOM_SEED,
        "eval_metric":  "mlogloss",
        "verbosity":    0,
    })
    safe_boost_xgb = studies["xgboost"].best_params["safe_class_boost"]
    sw_xgb = make_sample_weights(y_train_val, safe_class_boost=safe_boost_xgb)
    test_results["xgboost"] = evaluate_final(
        "XGBoost (v2)",
        xgb.XGBClassifier(**xgb_best_params),
        X_train_val, y_train_val, X_test, y_test, sw_xgb,
    )

    # GradientBoosting
    gb_best_params = {k: v for k, v in studies["gradient_boosting"].best_params.items()
                      if k != "safe_class_boost"}
    gb_best_params["random_state"] = RANDOM_SEED
    safe_boost_gb = studies["gradient_boosting"].best_params["safe_class_boost"]
    sw_gb = make_sample_weights(y_train_val, safe_class_boost=safe_boost_gb)
    test_results["gradient_boosting"] = evaluate_final(
        "GradientBoosting (v2)",
        GradientBoostingClassifier(**gb_best_params),
        X_train_val, y_train_val, X_test, y_test, sw_gb,
    )

    # ── Сохранение ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp":   timestamp,
        "method":      "feature_engineering_v2 + class_weight_boost",
        "fe_summary":  fe_summary,
        "optuna_cv":   {n: s.best_value for n, s in studies.items()},
        "best_params": {n: s.best_params for n, s in studies.items()},
        "test_results": test_results,
    }
    out_path = MODELS_DIR / "best_hyperparameters_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Сохранено: {out_path}")

    # ── Итоговая сводка ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("СВОДКА: V1 vs V2")
    print("=" * 70)
    print(f"{'Модель':<22} {'V1 Test F1':<14} {'V2 Test F1':<14} {'Δ':<10}")
    print("─" * 60)

    v1_baseline = {"xgboost": 0.5858, "gradient_boosting": 0.5652}
    for name in ["xgboost", "gradient_boosting"]:
        v1    = v1_baseline[name]
        v2    = test_results[name]["f1_macro"]
        delta = v2 - v1
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<22} {v1:<14.4f} {v2:<14.4f} {sign}{delta:.4f}")

    print("\nИнтерпретация Safe class (0):")
    print("  Цель: recall(0) > 0.50 при сохранении F1-macro > 0.58")


if __name__ == "__main__":
    main()
