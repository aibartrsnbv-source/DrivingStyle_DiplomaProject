#!/usr/bin/env python3
"""
Diagnostic script: проверка на data leakage после Optuna-тюнинга.

Выполняет 4 проверки:
  1. Classification report + confusion matrix на held-out test set
  2. Source-aware analysis: точность отдельно по каждому источнику данных
  3. Top-15 feature importances (доминирует ли одна фича?)
  4. Permutation test: accuracy на перемешанных метках (должно быть ~1/n_classes)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR
from src.data_loader import load_and_unify_datasets
from src.preprocessing import preprocess_pipeline


def main():
    print("=" * 70)
    print("DATA LEAKAGE DIAGNOSIS")
    print("=" * 70)

    # ── 1. Загрузить best params ─────────────────────────────────────────────
    params_path = MODELS_DIR / "best_hyperparameters.json"
    if not params_path.exists():
        print(f"\n✗ Файл не найден: {params_path}")
        print("  Сначала запустите: python tune_hyperparameters.py")
        sys.exit(1)

    with open(params_path, "r", encoding="utf-8") as f:
        best_params_json = json.load(f)

    gb_params = dict(best_params_json["results"]["gradient_boosting"]["best_params"])
    gb_params["random_state"] = RANDOM_SEED
    print(f"\nИспользуем gradient_boosting с параметрами:")
    for k, v in gb_params.items():
        print(f"  {k}: {v}")

    # ── 2. Загрузить данные и сохранить source до препроцессинга ─────────────
    print("\nЗагрузка данных...")
    df_raw = load_and_unify_datasets()
    print(f"  Всего строк: {len(df_raw)}")

    # Сохраняем mapping: оригинальный индекс (0..N-1) → data_source
    # ВАЖНО: делаем это ДО вызова preprocess_pipeline, пока индексы совпадают.
    # preprocess_pipeline не меняет строки (только clip/impute), поэтому
    # y_test.index после препроцессинга по-прежнему указывает на строки df_raw.
    if "data_source" in df_raw.columns:
        source_by_index = df_raw["data_source"].copy()
        print(f"\n  Источники данных (до препроцессинга):")
        print(df_raw.groupby("data_source")["driving_style_encoded"]
              .value_counts().unstack(fill_value=0).to_string())
    else:
        source_by_index = None
        print("  ⚠ Колонка 'data_source' не найдена")

    # ── 3. Препроцессинг ──────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_pipeline(
        df_raw, target_column="driving_style_encoded"
    )
    print(f"\n  После препроцессинга:")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"    Признаков: {X_train.shape[1]}")

    # ── 4. Обучение модели ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ GRADIENT BOOSTING С BEST PARAMS")
    print("=" * 70)

    model = GradientBoostingClassifier(**gb_params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # ────────────────────────────────────────────────────────────────────────
    # ПРОВЕРКА 1: Classification report + Confusion matrix
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ПРОВЕРКА 1: CLASSIFICATION REPORT")
    print("=" * 70)

    classes = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
    class_labels = {0: "Safe", 1: "Normal", 2: "Aggressive", 3: "Risky"}
    target_names = [class_labels.get(c, str(c)) for c in classes]

    print(classification_report(
        y_test, y_pred_test,
        labels=classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (строки = true, столбцы = predicted):")
    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{class_labels.get(c, c)}" for c in classes],
        columns=[f"pred_{class_labels.get(c, c)}" for c in classes],
    )
    print(cm_df.to_string())

    print("\nРаспределение классов в test set:")
    dist = pd.Series(y_test).value_counts().sort_index()
    for cls, cnt in dist.items():
        print(f"  Класс {cls} ({class_labels.get(cls, '?')}): {cnt} ({cnt/len(y_test)*100:.1f}%)")

    # ────────────────────────────────────────────────────────────────────────
    # ПРОВЕРКА 2: Source-aware analysis
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ПРОВЕРКА 2: SOURCE-AWARE ANALYSIS (главная проверка)")
    print("=" * 70)

    if source_by_index is not None:
        # y_test.index содержит оригинальные позиции строк из df_raw.
        # preprocess_pipeline использует clip/impute (без удаления строк),
        # поэтому индексы после препроцессинга совпадают с df_raw.
        valid_mask = y_test.index.isin(source_by_index.index)
        n_valid = valid_mask.sum()
        n_total = len(y_test)

        if n_valid < n_total:
            print(f"  ⚠ Найдено только {n_valid}/{n_total} совпадающих индексов")
            print("    (возможно, некоторые строки были удалены при препроцессинге)")

        if n_valid > 0:
            valid_positions = np.where(valid_mask)[0]

            analysis_df = pd.DataFrame({
                "true_class":  y_test.values[valid_positions],
                "pred_class":  y_pred_test[valid_positions],
                "data_source": source_by_index.loc[y_test.index[valid_positions]].values,
            })

            print("\nТочность модели по источникам данных:")
            print("-" * 50)
            source_acc = {}
            for src in sorted(analysis_df["data_source"].unique()):
                mask = analysis_df["data_source"] == src
                subset = analysis_df[mask]
                acc = accuracy_score(subset["true_class"], subset["pred_class"])
                source_acc[src] = acc
                n_src = len(subset)
                cls_dist = subset["true_class"].value_counts().sort_index().to_dict()
                print(f"  {src}:")
                print(f"    N={n_src}, accuracy={acc:.4f}, классы={cls_dist}")

            print("\nМатрица: data_source × true_class (кол-во строк):")
            pivot = analysis_df.groupby(["data_source", "true_class"]).size().unstack(fill_value=0)
            print(pivot.to_string())

            print("\nМатрица: data_source × predicted_class (кол-во строк):")
            pivot_pred = analysis_df.groupby(["data_source", "pred_class"]).size().unstack(fill_value=0)
            print(pivot_pred.to_string())

            # Флаг тревоги: если один source имеет accuracy 1.0, а другой нет
            acc_values = list(source_acc.values())
            max_acc = max(acc_values)
            min_acc = min(acc_values)
            print(f"\nMax accuracy по источнику: {max_acc:.4f}")
            print(f"Min accuracy по источнику: {min_acc:.4f}")
            if max_acc > 0.98 and min_acc < 0.80:
                print("⚠ СИГНАЛ: большой разброс точности между источниками — вероятен source leakage!")
                print("  Модель может различать ИСТОЧНИК данных, а не стиль вождения.")
            elif max_acc > 0.95 and (max_acc - min_acc) > 0.15:
                print("⚠ Умеренный сигнал: разница accuracy между источниками > 15%.")
            else:
                print("✓ Точность примерно одинакова по всем источникам — явного source leakage нет.")
    else:
        print("  ⚠ Невозможно выполнить source-aware анализ (нет колонки data_source)")

    # ────────────────────────────────────────────────────────────────────────
    # ПРОВЕРКА 3: Top-15 feature importances
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ПРОВЕРКА 3: TOP-15 FEATURE IMPORTANCES")
    print("=" * 70)

    importances = model.feature_importances_

    # Имена признаков из X_train (pandas DataFrame с сохранёнными именами колонок)
    if hasattr(X_train, "columns"):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    n_feat = min(len(feature_names), len(importances))
    imp_df = (
        pd.DataFrame({
            "feature":    feature_names[:n_feat],
            "importance": importances[:n_feat],
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(imp_df.head(15).to_string(index=True))

    top_imp = importances.max()
    top_feat = imp_df.iloc[0]["feature"]
    print(f"\nМаксимальная importance одного признака: {top_imp:.4f} ({top_feat})")

    if top_imp > 0.50:
        print("⚠ КРИТИЧНО: один признак отвечает за >50% важности — очень вероятен leakage!")
    elif top_imp > 0.30:
        print("⚠ ВНИМАНИЕ: один признак доминирует (>30%) — подозрительно, проверь этот признак.")
    else:
        print("✓ Importances распределены без явного доминирования одной фичи.")

    top3_imp = imp_df["importance"].head(3).sum()
    print(f"Суммарная importance топ-3 признаков: {top3_imp:.4f}")
    if top3_imp > 0.80:
        print("⚠ Топ-3 признака объясняют >80% — модель почти не использует остальные фичи.")

    # ────────────────────────────────────────────────────────────────────────
    # ПРОВЕРКА 4: Permutation test
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ПРОВЕРКА 4: PERMUTATION TEST (контрольный)")
    print("=" * 70)
    print("Перемешиваем y_train случайно и обучаем модель заново.")
    print("Ожидание: если признаки не несут информации о лейбле сами по себе,")
    print(f"accuracy должна упасть до ~1/n_classes = {1/len(classes):.2f}")

    y_train_shuffled = shuffle(y_train, random_state=RANDOM_SEED)
    model_perm = GradientBoostingClassifier(**gb_params)
    model_perm.fit(X_train, y_train_shuffled)

    perm_train_acc = accuracy_score(y_train_shuffled, model_perm.predict(X_train))
    perm_test_acc  = accuracy_score(y_test,           model_perm.predict(X_test))

    print(f"\nAccuracy на ПЕРЕМЕШАННЫХ labels (fit на train, evaluate на train): {perm_train_acc:.4f}")
    print(f"Accuracy на ПЕРЕМЕШАННЫХ labels (fit на train, evaluate на test):  {perm_test_acc:.4f}")

    random_baseline = 1.0 / len(classes)
    print(f"Случайный baseline (1/{len(classes)} классов): {random_baseline:.4f}")

    if perm_test_acc > 0.50:
        print("⚠ ВНИМАНИЕ: на перемешанных labels accuracy > 50% — признаки несут")
        print("  информацию о SOURCE/СТРУКТУРЕ данных, независимо от лейбла. Leakage!")
    elif perm_test_acc < random_baseline + 0.05:
        print("✓ OK: accuracy около случайного baseline — leakage через признаки не обнаружен.")
    else:
        print("⚠ Промежуточный результат (между random и 50%) — стоит проверить дополнительно.")

    # ────────────────────────────────────────────────────────────────────────
    # ИТОГИ
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ИТОГИ ДИАГНОСТИКИ")
    print("=" * 70)
    print(f"Test accuracy (реальная):            {test_acc:.4f}")
    print(f"Accuracy при shuffled labels (test): {perm_test_acc:.4f}")
    print(f"Random baseline (1/{len(classes)} классов):      {random_baseline:.4f}")
    print(f"Max feature importance:              {top_imp:.4f}  ({top_feat})")
    print(f"Top-3 features importance sum:       {top3_imp:.4f}")
    print()

    leakage_signals = []

    if test_acc > 0.97:
        leakage_signals.append(f"Очень высокая test accuracy ({test_acc:.4f}) — подозрительно для реальных данных")

    if perm_test_acc > 0.50:
        leakage_signals.append(f"Perm test accuracy {perm_test_acc:.4f} >> baseline {random_baseline:.4f}")

    if top_imp > 0.30:
        leakage_signals.append(f"Одна фича доминирует ({top_feat}: {top_imp:.4f})")

    if source_by_index is not None and "source_acc" in dir():
        if max(source_acc.values()) > 0.98 and min(source_acc.values()) < 0.80:
            leakage_signals.append("Большой разрыв accuracy между источниками данных")

    if leakage_signals:
        print("ВЕРДИКТ: ⚠ ОБНАРУЖЕНЫ СИГНАЛЫ ВОЗМОЖНОГО LEAKAGE:")
        for sig in leakage_signals:
            print(f"  • {sig}")
        print()
        print("Рекомендации:")
        print("  1. Проверь, не попала ли колонка-источник в признаки (data_source, trip_id)")
        print("  2. Проверь, не коррелирует ли top-1 фича с источником данных напрямую")
        print("  3. Попробуй обучить модель ТОЛЬКО на данных одного источника и проверить")
        print("     accuracy на другом источнике — она должна упасть")
        print("  4. Убедись, что group-aware split правильно разделяет trip_id по наборам")
    else:
        print("ВЕРДИКТ: ✓ Явных сигналов leakage не обнаружено.")
        print("Высокая accuracy (0.98+) может быть объяснена:")
        print("  • Чёткими различиями в сигналах ускорениями/скоростью между классами")
        print("  • Хорошо настроенными гиперпараметрами")
        print("  • Небольшим размером датасета (модель может переобучиться на нём)")


if __name__ == "__main__":
    main()
