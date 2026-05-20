# DriveGuard AI — Driving Style Classification & Risk Prediction

Дипломный проект: real-time анализ стиля вождения и риска ДТП с дашкам-видео через ML-модели (XGBoost) + computer vision (YOLOv8, optical flow).

## Демо

Локальный веб-интерфейс на localhost:8000. Загружаете видеофайл (или подключаете USB-камеру), система в реальном времени показывает:
- Скорость, счётчики резких маневров, дистанцию до впереди едущего автомобиля
- Уровень риска (LOW / MEDIUM / HIGH / CRITICAL)
- Визуализацию детекции автомобилей (YOLOv8) на видеопотоке

## Структура репозитория

**Важно:** весь код лежит в подпапке `33/333/` от корня репозитория. После клонирования нужно зайти именно туда:

```bash
git clone https://github.com/aibartrsnbv-source/DrivingStyle_DiplomaProject.git
cd DrivingStyle_DiplomaProject/33/333
```

Дальше в этом README все пути и команды даются относительно `33/333/`.

## Стек

- Python 3.10+
- ML: scikit-learn, XGBoost, LightGBM, PyTorch (MLP)
- Tuning: Optuna
- Interpretability: SHAP
- CV: OpenCV, Ultralytics (YOLOv8)
- Web: FastAPI, uvicorn (wsproto backend), WebSockets

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/aibartrsnbv-source/DrivingStyle_DiplomaProject.git
cd DrivingStyle_DiplomaProject/33/333
```

### 2. Виртуальное окружение

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Зависимости

```bash
pip install -r requirements.txt
pip install ultralytics wsproto lightgbm shap
```

Часть пакетов не в requirements.txt — поставить отдельно. YOLOv8-веса `yolov8n.pt` скачаются автоматически при первом запуске сервера.

## Датасеты

В репозитории датасетов нет (большой размер). Нужно скачать вручную.

### Обязательные:

**UAH-DriveSet** — реальные сенсорные данные с поездок.
- Источник: https://robesafe.uah.es/personal/eduardo.romera/uah-driveset/
- Распаковать в: `data/raw/UAH-DRIVESET/`

**Kaggle Driver Behavior** — данные с смартфонов (acc/gyro + labels).
- Источник: https://www.kaggle.com/datasets/outofskills/driving-behavior
- Файл `kaggle_driver_behavior.csv` (или `train_motion_data.csv`/`test_motion_data.csv` после переименования) положить в: `data/raw/`

После скачивания структура должна быть:

```
data/raw/
├── UAH-DRIVESET/
│   └── (папки с поездками)
└── kaggle_driver_behavior.csv
```

## Обучение моделей

ML-модели не в репозитории (тоже большой размер). Нужно обучить локально.

```bash
python main.py
```

Это запустит полный pipeline: загрузка → EDA → preprocessing → обучение 6 моделей (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, Voting Ensemble, MLP) → оценка на test → risk scoring. Время: ~3-5 минут.

После обучения в `models/` появятся `.pkl` и `.pt` файлы, в `outputs/figures/` — графики метрик и confusion matrices.

### Лучшая модель

XGBoost: **F1=0.7265**, Accuracy=0.7023 на test (group-aware split по trip_id).

Гиперпараметры подобраны через Optuna (Bayesian optimization, 50 trials, GroupKFold) — см. `tune_v3.py` и `models/best_hyperparameters_v3.json`.

## Запуск веб-интерфейса

```bash
python server.py
```

Открыть в браузере: http://localhost:8000

### Использование:
1. В выпадающем списке выбрать модель (обычно последний `xgboost_*.pkl`)
2. Выбрать источник: **USB Камера** или **Загрузить видео**
3. Если видео — drag-and-drop файл (.mp4, .avi, .mov, .mkv)
4. Нажать **Начать анализ**

UI обновляется в реальном времени: видео + метрики + риск-скор + алерты при HIGH/CRITICAL.

## Структура кода

```
33/333/
├── src/
│   ├── config.py              # Гиперпараметры моделей, пути, settings
│   ├── data_loader.py         # Загрузка Kaggle + UAH датасетов
│   ├── preprocessing.py       # Cleaning, scaling, feature engineering, GroupKFold split
│   ├── eda.py                 # Exploratory data analysis
│   ├── models.py              # ML модели (включая PyTorch MLP)
│   ├── train.py               # Training pipeline с SMOTE и class weights
│   ├── evaluate.py            # Метрики, ROC-AUC, confusion matrices
│   ├── risk_scoring.py        # Конвертация ML-предсказаний в risk score 0-1
│   ├── camera_inference.py    # Real-time inference: YOLO + optical flow + risk
│   └── utils.py
│
├── web/
│   └── index.html             # Веб-интерфейс
│
├── models/                    # Обученные модели (gitignored)
├── outputs/                   # Графики, отчёты (gitignored)
├── data/raw/                  # Датасеты (gitignored)
│
├── main.py                    # Главный pipeline обучения
├── server.py                  # FastAPI веб-сервер + WebSocket inference
│
├── tune_hyperparameters.py    # Optuna tuning с GroupKFold (honest CV)
├── tune_v2.py                 # Feature engineering v2 + class weight boost
├── tune_v3.py                 # Feature engineering v3 + LightGBM (лучший результат)
├── tune_stacking.py           # Stacking ensemble (не дал прирост)
├── shap_feature_selection.py  # SHAP-анализ важности признаков
├── diagnose_leakage.py        # Диагностика data leakage в CV
│
└── requirements.txt
```

## Командная работа

Каждый разработчик работает в своей ветке от master:

```bash
git checkout -b feature/имя-задачи
# ... работаешь ...
git add .
git commit -m "Описание"
git push -u origin feature/имя-задачи
```

В master сливаемся через Pull Request на GitHub после ревью.

## Troubleshooting

**`ultralytics not installed`** при запуске сервера → `pip install ultralytics`

**WebSocket падает с AssertionError** → сервер запущен без `wsproto`. Проверить что в `server.py` есть `uvicorn.run(..., ws="wsproto")` и что `pip install wsproto` выполнено.

**Камера не открывается** → закрыть Zoom/Teams/Discord/Skype, которые могут держать камеру.

**Скорость в UI заниженная или завышенная** → калибровочная константа в `src/camera_inference.py`, переменная `FLOW_TO_KMH`. Текущее значение 1.2 откалибровано на реальном дашкам-видео (24fps, 4K).

**`yolov8n.pt` не скачался** → нужен интернет при первом запуске сервера. Если нет — скачать вручную: https://github.com/ultralytics/assets/releases/

## Авторы

[список команды]

## Лицензия

Учебный проект, не для коммерческого использования.
