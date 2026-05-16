# Дипломный проект: Анализ стиля вождения и прогнозирование рисков ДТП

## Информация о проекте

**Тема:** Разработка алгоритмов машинного обучения для оценки стиля вождения и прогнозирования рисков аварий

**Тип:** Бакалаврская дипломная работа

**Автор:** Медет

---

## Структура проекта

```
driving_style_ml/
├── data/                    # Данные
├── src/
│   ├── config.py           # Настройки, пути к датасетам, гиперпараметры
│   ├── data_loader.py      # Загрузка данных
│   ├── preprocessing.py    # Предобработка, SMOTE, scaling
│   ├── eda.py              # Разведочный анализ данных
│   ├── models.py           # ML модели + PyTorch (MLP, LSTM)
│   ├── train.py            # Обучение моделей
│   ├── evaluate.py         # Метрики, ROC-AUC, confusion matrix
│   ├── risk_scoring.py     # Система оценки риска (0-1)
│   └── utils.py            # Вспомогательные функции
├── models/                  # Сохранённые модели
├── outputs/                 # Графики, отчёты
├── main.py                  # Главный pipeline
├── requirements.txt         # Зависимости
└── README.md               # Документация
```

---

## Датасеты

Расположены в `/Users/medetlatip/Downloads/`:

| Файл | Описание |
|------|----------|
| `US_Accidents_March23.csv` | Данные о ДТП в США |
| `full_data_carla.csv` | Симулятор CARLA |
| `eco_driving_score.csv` | Эко-вождение |
| `driver_behavior_route_anomaly_dataset_with_derived_features.csv` | Поведение водителей |

Дополнительные папки:
- `/Users/medetlatip/Downloads/archive-7`
- `/Users/medetlatip/Downloads/archive-8`
- `/Users/medetlatip/Downloads/Driver Drowsiness Dataset (DDD)`

---

## Модели

### Классические ML:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Deep Learning (PyTorch):
- MLP — для табличных данных
- LSTM — для временных рядов (сенсоры)

---

## Классы стиля вождения

| Код | Класс | Описание |
|-----|-------|----------|
| 0 | Safe | Безопасное вождение |
| 1 | Normal | Нормальное |
| 2 | Aggressive | Агрессивное |
| 3 | Risky | Рискованное |

---

## Система оценки риска

| Диапазон | Уровень | Описание |
|----------|---------|----------|
| 0.0 - 0.3 | LOW | Низкий риск |
| 0.3 - 0.6 | MEDIUM | Средний риск |
| 0.6 - 0.8 | HIGH | Высокий риск |
| 0.8 - 1.0 | CRITICAL | Критический |

---

## Команды запуска

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск с синтетическими данными (для теста)
python main.py --synthetic

# Запуск с реальными данными
python main.py

# Быстрый режим (меньше данных)
python main.py --quick

# Без deep learning (быстрее)
python main.py --no-dl

# Показать конфигурацию
python main.py --config
```

---

## Текущий статус

- [x] Создана структура проекта
- [x] Реализованы все модули
- [x] Настроены пути к датасетам
- [ ] Тестирование на реальных данных
- [ ] Тонкая настройка гиперпараметров
- [ ] Финальные эксперименты для диплома

---

## Следующие шаги

1. Установить зависимости: `pip install -r requirements.txt`
2. Протестировать pipeline: `python main.py --synthetic`
3. Запустить на реальных данных
4. Проанализировать результаты в `outputs/`
5. Настроить гиперпараметры в `src/config.py`

---

## Заметки

- Все гиперпараметры в `src/config.py`
- Графики сохраняются в `outputs/figures/`
- Модели сохраняются в `models/`
- Логи в `outputs/pipeline.log`

---

*Последнее обновление: Январь 2026*
