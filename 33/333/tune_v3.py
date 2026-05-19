#!/usr/bin/env python3
"""
TUNE V3 — корректный feature engineering на DataFrame + LightGBM.

Отличия от V2:
- DataFrame сохраняется до Optuna (имена колонок не теряются)
- Не удаляем константные фичи
- Корреляция > 0.97 (только реальные дубликаты)
- Interactions добавляются по нормальным именам
- safe_class_boost ограничен 1.0-1.5
- Добавлен LightGBM
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
from sklearn.model_selection import GroupKFold
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

LGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print(f"✓ LightGBM {lgb.__version__} доступен")
except ImportError:
    print("⚠ lightgbm не установлен. Будет пропущен.")

N_TRIALS = 50
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading — возвращает DataFrame, не numpy
# ---------------------------------------------------------------------------

def prepare_data_v3():
    """
    Возвращает X как PANDAS DataFrame (имена колонок сохранены).
    y и groups — numpy.
    """
    print("Загрузка...")
    df = load_and_unify_datasets()
    print(f"  ✓ {len(df)} строк")

    trip_ids = df["trip_id"].copy() if "trip_id" in df.columns else None

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_pipeline(
        df, target_column="driving_style_encoded"
    )

    assert isinstance(X_train, pd.DataFrame), \
        f"Ожидался DataFrame, получено {type(X_train)}"

    print(f"  ✓ Колонки X: {list(X_train.columns[:5])}... ({X_train.shape[1]} всего)")

    # groups через pandas-индексы (preprocess_pipeline их сохраняет)
    if trip_ids is not None:
        try:
            groups_train = trip_ids.loc[X_train.index].values
            groups_val   = trip_ids.loc[X_val.index].values
            print(f"  ✓ trip_id: train={len(np.unique(groups_train))} групп, "
                  f"val={len(np.unique(groups_val))} групп")
        except Exception as e:
            print(f"  ⚠ trip_id не восстановлен ({e}), pseudo-groups")
            groups_train = np.arange(len(X_train)) // 5
            groups_val   = np.arange(len(X_val))   // 5
    else:
        groups_train = np.arange(len(X_train)) // 5
        groups_val   = np.arange(len(X_val))   // 5

    def to_arr(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    return (X_train, X_val, X_test,
            to_arr(y_train), to_arr(y_val), to_arr(y_test),
            groups_train, groups_val)


# ---------------------------------------------------------------------------
# Feature engineering v3 — работает на DataFrame
# ---------------------------------------------------------------------------

def feature_engineering_v3(X_train_df, X_val_df, X_test_df):
    """
    - НЕ удаляет константы (tree-based их и так игнорят)
    - Удаляет только пары с |r| > 0.97 (реальные дубликаты)
    - Добавляет interaction features по нормальным именам

    Возвращает: (X_train_df, X_val_df, X_test_df, summary)
    """
    summary = {}

    # ── Шаг 1: Убрать почти-дубликаты (|r| > 0.97) ─────────────────────────
    corr  = X_train_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            v = upper.loc[idx, col]
            if pd.notna(v) and v > 0.97:
                # Оставляем более "уникальную" колонку (меньшее среднее corr со всеми)
                mean_col = corr[col].mean()
                mean_idx = corr[idx].mean()
                victim = col if mean_col > mean_idx else idx
                to_drop.add(victim)

    summary["removed_correlated"] = sorted(to_drop)
    X_train_df = X_train_df.drop(columns=list(to_drop))
    X_val_df   = X_val_df.drop(columns=list(to_drop))
    X_test_df  = X_test_df.drop(columns=list(to_drop))
    print(f"  ✗ Убрано {len(to_drop)} дубликатов (|r|>0.97): "
          f"{sorted(to_drop)[:6]}{'...' if len(to_drop) > 6 else ''}")

    # ── Шаг 2: Interaction features по именам ───────────────────────────────
    interactions = [
        ("speed_x_brakes",      "avg_speed",           "harsh_braking_count"),
        ("speed_x_accels",      "avg_speed",           "harsh_accel_count"),
        ("aggression_combined", "harsh_braking_count", "harsh_accel_count"),
        ("gyro_speed_ratio",    "gyro_y_std",          "avg_speed"),
        ("accel_speed_var",     "accel_x_mean",        "speed_variance"),
        ("gyro_total",          "gyro_x_std",          "gyro_z_std"),
        ("accel_total",         "accel_x_mean",        "accel_y_mean"),
    ]

    added = []
    for new_name, col_a, col_b in interactions:
        if col_a in X_train_df.columns and col_b in X_train_df.columns:
            X_train_df[new_name] = X_train_df[col_a] * X_train_df[col_b]
            X_val_df[new_name]   = X_val_df[col_a]   * X_val_df[col_b]
            X_test_df[new_name]  = X_test_df[col_a]  * X_test_df[col_b]
            added.append(new_name)
        else:
            missing = col_a if col_a not in X_train_df.columns else col_b
            print(f"  ⚠ Пропуск {new_name}: '{missing}' не найден")

    # speed_accel_diff — разность (полезнее произведения для max-avg)
    if "max_speed" in X_train_df.columns and "avg_speed" in X_train_df.columns:
        X_train_df["speed_accel_diff"] = X_train_df["max_speed"] - X_train_df["avg_speed"]
        X_val_df["speed_accel_diff"]   = X_val_df["max_speed"]   - X_val_df["avg_speed"]
        X_test_df["speed_accel_diff"]  = X_test_df["max_speed"]  - X_test_df["avg_speed"]
        added.append("speed_accel_diff")
    else:
        print("  ⚠ Пропуск speed_accel_diff: max_speed или avg_speed не найден")

    summary["added_interactions"] = added
    print(f"  ✓ Добавлено {len(added)} interactions: {added}")
    print(f"  Итого фич: {X_train_df.shape[1]}")

    return X_train_df, X_val_df, X_test_df, summary


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def make_sample_weights(y, safe_class_boost=1.0):
    base    = compute_sample_weight(class_weight="balanced", y=y)
    boosted = base.copy()
    boosted[y == 0] *= safe_class_boost
    return boosted


# ---------------------------------------------------------------------------
# Optuna objectives (safe_boost ограничен до 1.5)
# ---------------------------------------------------------------------------

def objective_xgb_v3(trial, X_df, y, groups, cv):
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
    safe_boost = trial.suggest_float("safe_class_boost", 1.0, 1.5)

    X_np = X_df.values
    scores = []
    for tr_idx, va_idx in cv.split(X_np, y, groups=groups):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=make_sample_weights(y_tr, safe_boost))
        scores.append(f1_score(y_va, model.predict(X_va), average="macro", zero_division=0))
    return np.mean(scores)


def objective_gb_v3(trial, X_df, y, groups, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 8),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state":      RANDOM_SEED,
    }
    safe_boost = trial.suggest_float("safe_class_boost", 1.0, 1.5)

    X_np = X_df.values
    scores = []
    for tr_idx, va_idx in cv.split(X_np, y, groups=groups):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = GradientBoostingClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=make_sample_weights(y_tr, safe_boost))
        scores.append(f1_score(y_va, model.predict(X_va), average="macro", zero_division=0))
    return np.mean(scores)


def objective_lgb_v3(trial, X_df, y, groups, cv):
    if not LGB_AVAILABLE:
        return 0.0
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 80),
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state":      RANDOM_SEED,
        "verbosity":         -1,
        "n_jobs":            -1,
    }
    safe_boost = trial.suggest_float("safe_class_boost", 1.0, 1.5)

    X_np = X_df.values
    scores = []
    for tr_idx, va_idx in cv.split(X_np, y, groups=groups):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=make_sample_weights(y_tr, safe_boost))
        scores.append(f1_score(y_va, model.predict(X_va), average="macro", zero_division=0))
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

def evaluate_final(name, model, X_train_val_df, y_train_val, X_test_df, y_test, safe_boost):
    print(f"\n{name}:")
    sw = make_sample_weights(y_train_val, safe_boost)
    model.fit(X_train_val_df.values, y_train_val, sample_weight=sw)
    y_pred = model.predict(X_test_df.values)

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
    print("TUNE V3 — FE on DataFrame + LightGBM + careful class boost")
    print("=" * 70)

    # ── Загрузка ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val = prepare_data_v3()
    print(f"\nИсходные: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Тип данных: {type(X_train).__name__}")

    # ── Feature engineering v3 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING V3")
    print("=" * 70)
    X_train, X_val, X_test, fe_summary = feature_engineering_v3(X_train, X_val, X_test)
    print(f"После FE: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── Объединяем train+val для CV ───────────────────────────────────────────
    X_full_df = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full    = np.concatenate([y_train, y_val])
    g_full    = np.concatenate([g_train, g_val])
    n_groups  = len(np.unique(g_full))
    print(f"\nCV data: {X_full_df.shape}, {n_groups} уникальных групп, {CV_FOLDS} folds")

    cv = GroupKFold(n_splits=CV_FOLDS)

    # ── Optuna ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTUNA TUNING V3")
    print("=" * 70)

    objectives = {
        "xgboost":           objective_xgb_v3,
        "gradient_boosting": objective_gb_v3,
    }
    if LGB_AVAILABLE:
        objectives["lightgbm"] = objective_lgb_v3

    studies = {}
    for model_name, obj_fn in objectives.items():
        print(f"\n─── Tuning: {model_name} ───")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_SEED),
        )

        def _make_cb(total):
            def cb(study, trial):
                if trial.number % 10 == 0 or trial.number == total - 1:
                    best = study.best_value if study.best_trial else 0
                    print(f"  Trial {trial.number + 1:>3}/{total} | "
                          f"current={trial.value:.4f} | best={best:.4f}")
            return cb

        study.optimize(
            lambda t, fn=obj_fn: fn(t, X_full_df, y_full, g_full, cv),
            n_trials=N_TRIALS,
            callbacks=[_make_cb(N_TRIALS)],
            show_progress_bar=False,
        )
        studies[model_name] = study
        print(f"  Best CV F1-macro: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    # ── Оценка на test set ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ОЦЕНКА НА TEST SET (V3)")
    print("=" * 70)

    X_train_val_df = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val    = np.concatenate([y_train, y_val])

    test_results = {}

    # XGBoost
    p = {k: v for k, v in studies["xgboost"].best_params.items()
         if k != "safe_class_boost"}
    p.update({"random_state": RANDOM_SEED, "eval_metric": "mlogloss", "verbosity": 0})
    test_results["xgboost"] = evaluate_final(
        "XGBoost (v3)", xgb.XGBClassifier(**p),
        X_train_val_df, y_train_val, X_test, y_test,
        studies["xgboost"].best_params["safe_class_boost"],
    )

    # GradientBoosting
    p = {k: v for k, v in studies["gradient_boosting"].best_params.items()
         if k != "safe_class_boost"}
    p["random_state"] = RANDOM_SEED
    test_results["gradient_boosting"] = evaluate_final(
        "GradientBoosting (v3)", GradientBoostingClassifier(**p),
        X_train_val_df, y_train_val, X_test, y_test,
        studies["gradient_boosting"].best_params["safe_class_boost"],
    )

    # LightGBM
    if LGB_AVAILABLE:
        p = {k: v for k, v in studies["lightgbm"].best_params.items()
             if k != "safe_class_boost"}
        p.update({"random_state": RANDOM_SEED, "verbosity": -1, "n_jobs": -1})
        test_results["lightgbm"] = evaluate_final(
            "LightGBM (v3)", lgb.LGBMClassifier(**p),
            X_train_val_df, y_train_val, X_test, y_test,
            studies["lightgbm"].best_params["safe_class_boost"],
        )

    # ── Сохранение ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp":   ts,
        "method":      "FE on DataFrame, |r|>0.97 only, safe_boost 1.0-1.5, +LightGBM",
        "fe_summary":  {k: list(v) if isinstance(v, set) else v
                        for k, v in fe_summary.items()},
        "optuna_cv":   {n: s.best_value for n, s in studies.items()},
        "best_params": {n: s.best_params for n, s in studies.items()},
        "test_results": test_results,
    }
    out = MODELS_DIR / "best_hyperparameters_v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Сохранено: {out}")

    # ── Сводка V1 → V2 → V3 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("СВОДКА: V1 → V2 → V3")
    print("=" * 70)
    print(f"{'Модель':<22} {'V1 Test F1':<12} {'V2 Test F1':<12} "
          f"{'V3 Test F1':<12} {'Δ vs V1':<10}")
    print("─" * 70)

    v1 = {"xgboost": 0.5858, "gradient_boosting": 0.5652, "lightgbm": None}
    v2 = {"xgboost": 0.5240, "gradient_boosting": 0.5355, "lightgbm": None}

    for name, res in test_results.items():
        v1f = v1.get(name)
        v2f = v2.get(name)
        v3f = res["f1_macro"]
        v1s = f"{v1f:.4f}" if v1f is not None else "—"
        v2s = f"{v2f:.4f}" if v2f is not None else "—"
        if v1f is not None:
            delta = v3f - v1f
            sign  = "+" if delta >= 0 else ""
            delta_s = f"{sign}{delta:.4f}"
        else:
            delta_s = "new"
        print(f"{name:<22} {v1s:<12} {v2s:<12} {v3f:<12.4f} {delta_s:<10}")

    print("\nИнтерпретация Safe class (0):")
    print("  Цель: recall(0) > 0.50 при сохранении F1-macro > 0.58")


if __name__ == "__main__":
    main()
