#!/usr/bin/env python3
"""
Hyperparameter tuning via Optuna with HONEST GroupKFold cross-validation.

Использует GroupKFold по trip_id, чтобы избежать утечки между фолдами
через временную близость rolling features внутри одной поездки.
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ML_MODELS_CONFIG, MODELS_DIR, RANDOM_SEED
from src.data_loader import create_sample_dataset, load_and_unify_datasets
from src.preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_data():
    """
    Загружает данные, делает preprocessing, возвращает:
        X_train, X_val, X_test, y_train, y_val, y_test,
        groups_train, groups_val  (для GroupKFold)
    """
    print("Загрузка данных...")
    try:
        df = load_and_unify_datasets()
        print(f"  ✓ Реальные данные: {len(df)} строк")
    except Exception as e:
        print(f"  ⚠ Реальные данные не загрузились ({e}), используем синтетику")
        df = create_sample_dataset(n_samples=5000, random_state=RANDOM_SEED)

    # Сохраняем trip_id ДО препроцессинга — он будет удалён из X внутри pipeline
    trip_ids = df["trip_id"].copy() if "trip_id" in df.columns else None

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_pipeline(
        df, target_column="driving_style_encoded"
    )

    # Восстанавливаем group-метки: preprocess_pipeline сохраняет оригинальные
    # pandas-индексы через df.copy(), поэтому y_train.index → строки исходного df
    groups_train = groups_val = None
    if trip_ids is not None:
        try:
            groups_train = trip_ids.loc[y_train.index].values
            groups_val   = trip_ids.loc[y_val.index].values
            print(f"  ✓ Восстановили trip_id: "
                  f"train={len(np.unique(groups_train))} групп, "
                  f"val={len(np.unique(groups_val))} групп")
        except Exception as e:
            print(f"  ⚠ Не удалось восстановить trip_id ({e}), используем pseudo-groups")

    if groups_train is None:
        # Каждые 5 соседних строк ≈ одна поездка (грубая аппроксимация)
        n_train = X_train.shape[0]
        n_val   = X_val.shape[0]
        groups_train = np.arange(n_train) // 5
        groups_val   = np.arange(n_val) // 5
        print(f"  ⚠ Используем pseudo-groups (по 5 строк) — менее надёжно")

    # Конвертируем в numpy
    def to_arr(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    return (
        to_arr(X_train), to_arr(X_val), to_arr(X_test),
        to_arr(y_train), to_arr(y_val), to_arr(y_test),
        groups_train, groups_val,
    )


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def make_cv_data(X_train, X_val, y_train, y_val, groups_train, groups_val):
    """Объединить train+val для CV-тюнинга, сохранив группы."""
    X_full      = np.vstack([X_train, X_val])
    y_full      = np.concatenate([y_train, y_val])
    groups_full = np.concatenate([groups_train, groups_val])
    return X_full, y_full, groups_full


def get_cv_splitter(groups, n_unique_groups, cv_folds=CV_FOLDS):
    """Возвращает (cv_object, kwargs_for_cross_val_score)."""
    if n_unique_groups >= cv_folds * 2:
        return GroupKFold(n_splits=cv_folds), {"groups": groups}
    print(f"  ⚠ Слишком мало уникальных групп ({n_unique_groups}) для GroupKFold, "
          f"fallback на StratifiedKFold")
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED), {}


# ---------------------------------------------------------------------------
# Optuna objectives  (metric: f1_macro — честнее accuracy при дисбалансе)
# ---------------------------------------------------------------------------

def objective_xgboost(trial, X, y, cv, cv_kwargs):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
        "random_state":     RANDOM_SEED,
        "eval_metric":      "mlogloss",
        "verbosity":        0,
    }
    scores = cross_val_score(
        xgb.XGBClassifier(**params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    return scores.mean()


def objective_random_forest(trial, X, y, cv, cv_kwargs):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":       trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":    "balanced_subsample",
        "random_state":    RANDOM_SEED,
        "n_jobs":          -1,
    }
    scores = cross_val_score(
        RandomForestClassifier(**params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    return scores.mean()


def objective_gradient_boosting(trial, X, y, cv, cv_kwargs):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state":    RANDOM_SEED,
    }
    scores = cross_val_score(
        GradientBoostingClassifier(**params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    return scores.mean()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def compute_baseline_cv(X, y, cv, cv_kwargs):
    """CV с ручными параметрами из src/config.py для сравнения с Optuna."""
    print("\nBaseline: текущие гиперпараметры из src/config.py")
    print("─" * 60)

    results = {}

    # RandomForest — убираем n_jobs, чтобы не конфликтовал с cross_val_score n_jobs
    rf_params = {k: v for k, v in ML_MODELS_CONFIG["random_forest"].items()}
    rf_scores = cross_val_score(
        RandomForestClassifier(**rf_params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    results["random_forest"] = rf_scores.mean()
    print(f"  random_forest      F1-macro: {rf_scores.mean():.4f}  "
          f"(+/- {rf_scores.std() * 2:.4f})")

    # GradientBoosting — параметры берём как есть
    gb_params = {k: v for k, v in ML_MODELS_CONFIG["gradient_boosting"].items()}
    gb_scores = cross_val_score(
        GradientBoostingClassifier(**gb_params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    results["gradient_boosting"] = gb_scores.mean()
    print(f"  gradient_boosting  F1-macro: {gb_scores.mean():.4f}  "
          f"(+/- {gb_scores.std() * 2:.4f})")

    # XGBoost — убираем ключи, которые XGBClassifier не принимает через sklearn API
    xgb_skip = set()
    xgb_params = {k: v for k, v in ML_MODELS_CONFIG["xgboost"].items()
                  if k not in xgb_skip}
    xgb_params["verbosity"] = 0
    xgb_scores = cross_val_score(
        xgb.XGBClassifier(**xgb_params), X, y,
        cv=cv, scoring="f1_macro", n_jobs=-1, **cv_kwargs,
    )
    results["xgboost"] = xgb_scores.mean()
    print(f"  xgboost            F1-macro: {xgb_scores.mean():.4f}  "
          f"(+/- {xgb_scores.std() * 2:.4f})")

    return results


# ---------------------------------------------------------------------------
# Final test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(name, model_class, params, X_train_val, y_train_val, X_test, y_test):
    """Обучить лучшую модель на train+val, оценить на test."""
    print(f"\n{name}:")
    model = model_class(**params)
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="macro")

    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Test F1-macro : {test_f1:.4f}")
    print("  Classification report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    return {"test_accuracy": test_acc, "test_f1_macro": test_f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("HYPERPARAMETER TUNING — HONEST GROUP-AWARE CV")
    print("=" * 70)

    # ── Загрузка ─────────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     groups_train, groups_val) = prepare_data()

    print(f"\n  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Классы: {len(np.unique(y_train))}")

    # ── Объединяем train+val для CV ───────────────────────────────────────────
    X_full, y_full, groups_full = make_cv_data(
        X_train, X_val, y_train, y_val, groups_train, groups_val,
    )
    n_groups = len(np.unique(groups_full))
    print(f"  CV data: {X_full.shape}, {n_groups} уникальных групп")

    cv, cv_kwargs = get_cv_splitter(groups_full, n_groups, CV_FOLDS)
    cv_type = type(cv).__name__
    print(f"  CV: {cv_type}, {CV_FOLDS} folds")
    print(f"  Metric: f1_macro (честнее accuracy при несбалансированных классах)")
    print(f"  Trials per model: {N_TRIALS}")

    # ── ЭТАП 1: BASELINE ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ЭТАП 1: BASELINE (ручные гиперпараметры из src/config.py)")
    print("=" * 70)
    baselines = compute_baseline_cv(X_full, y_full, cv, cv_kwargs)

    # ── ЭТАП 2: OPTUNA ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ЭТАП 2: OPTUNA TUNING")
    print("=" * 70)

    objectives_map = {
        "xgboost":           objective_xgboost,
        "random_forest":     objective_random_forest,
        "gradient_boosting": objective_gradient_boosting,
    }
    studies = {}

    for model_name, objective_fn in objectives_map.items():
        print(f"\n─── Tuning: {model_name} ───")
        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{model_name}_tuning",
        )

        def _make_cb(total):
            def cb(study, trial):
                if trial.number % 10 == 0 or trial.number == total - 1:
                    best = study.best_value if study.best_trial else 0.0
                    print(f"  Trial {trial.number + 1:>3}/{total} | "
                          f"current={trial.value:.4f} | best={best:.4f}")
            return cb

        study.optimize(
            lambda trial, fn=objective_fn: fn(trial, X_full, y_full, cv, cv_kwargs),
            n_trials=N_TRIALS,
            callbacks=[_make_cb(N_TRIALS)],
            show_progress_bar=False,
        )
        studies[model_name] = study
        print(f"  Best CV F1-macro: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

    # ── ЭТАП 3: ОЦЕНКА НА TEST SET ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ЭТАП 3: ОЦЕНКА НА TEST SET (честные финальные числа)")
    print("=" * 70)

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    xgb_best = {
        **studies["xgboost"].best_params,
        "random_state": RANDOM_SEED,
        "eval_metric":  "mlogloss",
        "verbosity":    0,
    }
    test_results = {}
    test_results["xgboost"] = evaluate_on_test(
        "XGBoost", xgb.XGBClassifier, xgb_best,
        X_train_val, y_train_val, X_test, y_test,
    )

    rf_best = {
        **studies["random_forest"].best_params,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced_subsample",
        "n_jobs":       -1,
    }
    test_results["random_forest"] = evaluate_on_test(
        "RandomForest", RandomForestClassifier, rf_best,
        X_train_val, y_train_val, X_test, y_test,
    )

    gb_best = {
        **studies["gradient_boosting"].best_params,
        "random_state": RANDOM_SEED,
    }
    test_results["gradient_boosting"] = evaluate_on_test(
        "GradientBoosting", GradientBoostingClassifier, gb_best,
        X_train_val, y_train_val, X_test, y_test,
    )

    # ── Сохранение ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp":    timestamp,
        "cv_method":    cv_type,
        "metric":       "f1_macro",
        "n_trials":     N_TRIALS,
        "cv_folds":     CV_FOLDS,
        "baselines_cv": baselines,
        "optuna_cv":    {name: s.best_value for name, s in studies.items()},
        "optuna_params": {name: s.best_params for name, s in studies.items()},
        "test_results": test_results,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    output_path = MODELS_DIR / "best_hyperparameters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Сохранено: {output_path}")

    # ── Итоговая сводка ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("СВОДКА: BASELINE vs OPTUNA vs TEST")
    print("=" * 70)
    print(f"{'Модель':<22} {'Baseline CV':<14} {'Optuna CV':<14} "
          f"{'Test F1':<10} {'Прирост CV':<12}")
    print("─" * 75)
    for name in ["xgboost", "random_forest", "gradient_boosting"]:
        bcv   = baselines.get(name, 0)
        ocv   = studies[name].best_value
        tf1   = test_results[name]["test_f1_macro"]
        delta = ocv - bcv
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<22} {bcv:<14.4f} {ocv:<14.4f} {tf1:<10.4f} {sign}{delta:.4f}")

    print("\nИнтерпретация:")
    print("- Если Optuna CV ≈ Baseline CV — ручные параметры уже хороши")
    print("- Если Test F1 << Optuna CV    — даже честный CV переоценивает (overfit)")
    print("- Если Test F1 ≈ Optuna CV     — модель честно генерализуется")


if __name__ == "__main__":
    main()
