# DriveGuard — Driving Style Classification & Accident Risk Prediction

Real-time analysis of driving style and accident risk from dashcam video,
combining machine learning models (XGBoost and others) with computer vision
(YOLOv8 object detection and optical flow).

> Bachelor diploma project — Astana IT University, School of Intelligent
> Systems, Smart Technology, Group ST-2303.

---

## Overview

DriveGuard AI classifies driving behaviour into three styles — **Safe**,
**Normal** and **Aggressive** — and converts the prediction into an
interpretable accident-risk score on a four-level scale
(**LOW / MEDIUM / HIGH / CRITICAL**).

The system has two parts:

- An **offline ML pipeline** that unifies telematic and inertial-sensor data,
  trains and compares six classifiers, and evaluates them with a
  leakage-aware, group-based methodology.
- A **real-time web application** that processes a dashcam video stream (or a
  USB camera feed), detects surrounding vehicles, estimates motion, and
  displays live driving metrics and risk alerts.

### Demo

A local web interface runs on `http://localhost:8000`. You upload a video file
or connect a USB camera, and the system shows in real time:

- Speed, harsh-manoeuvre counters, and the distance to the leading vehicle
- The current risk level (LOW / MEDIUM / HIGH / CRITICAL)
- A live YOLOv8 visualisation of detected vehicles on the video stream
- A trip report with an overall safety score when the session ends

---

## Tech Stack

| Area              | Tools                                          |
|-------------------|------------------------------------------------|
| Language          | Python 3.11                                    |
| Machine learning  | scikit-learn, XGBoost, LightGBM, PyTorch (MLP) |
| Hyperparameter tuning | Optuna                                         |
| Interpretability  | SHAP                                           |
| Computer vision   | OpenCV, Ultralytics (YOLOv8)                   |
| Web               | FastAPI, uvicorn (wsproto backend), WebSockets |

---

## Repository Structure

> **Important:** all source code lives in the `33/333/` subfolder of the
> repository root. After cloning, change into that directory — every path and
> command below is relative to `33/333/`.

```
33/333/
├── src/
│   ├── config.py            # Model hyperparameters, paths, settings
│   ├── data_loader.py       # Loading Kaggle + UAH datasets
│   ├── preprocessing.py     # Cleaning, scaling, feature engineering, GroupKFold split
│   ├── eda.py               # Exploratory data analysis
│   ├── models.py            # ML models (including the PyTorch MLP)
│   ├── train.py             # Training pipeline with SMOTE and class weights
│   ├── evaluate.py          # Metrics, ROC-AUC, confusion matrices
│   ├── risk_scoring.py      # Converting ML predictions into a 0–1 risk score
│   ├── camera_inference.py  # Real-time inference: YOLO + optical flow + risk
│   └── utils.py
│
├── web/
│   └── index.html           # Web interface (HTML + CSS + JS in one file)
│
├── models/                  # Trained models (gitignored)
├── outputs/                 # Figures and reports (gitignored)
├── data/raw/                # Datasets (gitignored)
├── temp/                    # Temporary uploaded video files
│
├── main.py                  # Main training pipeline
├── server.py                # FastAPI web server + WebSocket inference
│
├── tune_v3.py               # Optuna tuning, feature engineering v3 (best result)
├── shap_feature_selection.py# SHAP-based feature importance analysis
├── diagnose_leakage.py      # Data-leakage diagnostics for cross-validation
│
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aibartrsnbv-source/DrivingStyle_DiplomaProject.git
cd DrivingStyle_DiplomaProject/33/333
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install ultralytics wsproto lightgbm shap
```

Some packages are not listed in `requirements.txt` and must be installed
separately. The YOLOv8 weights file `yolov8n.pt` is downloaded automatically
the first time the server starts.

---

## Datasets

Datasets are **not** included in the repository because of their size and must
be downloaded manually.

### UAH-DriveSet — real sensor data from driving sessions

- Source: https://robesafe.uah.es/personal/eduardo.romera/uah-driveset/
- Extract into: `data/raw/UAH-DRIVESET/`

### Kaggle Driver Behavior — smartphone data (accelerometer/gyroscope + labels)

- Source: https://www.kaggle.com/datasets/outofskills/driving-behavior
- Place the file `kaggle_driver_behavior.csv` into: `data/raw/`

After downloading, the structure should look like this:

```
data/raw/
├── UAH-DRIVESET/
│   └── (trip folders)
└── kaggle_driver_behavior.csv
```

---

## Training the Models

Trained models are not stored in the repository either — they must be trained
locally:

```bash
python main.py
```

This runs the full pipeline: data loading → EDA → preprocessing → training of
six models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
Voting Ensemble, MLP) → evaluation on the test set → risk scoring. It takes
about 3–5 minutes.

After training, `.pkl` and `.pt` model files appear in `models/`, and metric
plots and confusion matrices appear in `outputs/figures/`.

### Best Model

**XGBoost** — F1 = 0.7265, Accuracy = 0.7023 on the test set (group-aware
split by `trip_id`).

Hyperparameters were tuned with Optuna (Bayesian optimization, 50 trials,
GroupKFold) — see `tune_v3.py` and `models/best_hyperparameters_v3.json`.

---

## Running the Web Interface

```bash
python server.py
```

Then open `http://localhost:8000` in your browser.

### How to use

1. Select a model from the dropdown list (usually the latest `xgboost_*.pkl`).
2. Choose a video source: **USB Camera** or **Upload Video**.
3. For a video file, drag and drop it (`.mp4`, `.avi`, `.mov`, `.mkv`).
4. Click **Start Analysis**.

The UI updates in real time with the video, the driving metrics, the risk
score, and alerts when the risk level reaches HIGH or CRITICAL.

### API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/`                   | Main page |
| GET  | `/api/models`         | List of available models |
| POST | `/api/upload-video`   | Upload a video file |
| WS   | `/ws/{session_id}`    | WebSocket for real-time inference |

---

## Team Workflow

Each developer works on their own branch created from `master`:

```bash
git checkout -b feature/your-task
# ... work ...
git add .
git commit -m "Description"
git push -u origin feature/your-task
```

Changes are merged into `master` through a Pull Request on GitHub after review.

---

## Troubleshooting

**`ultralytics not installed` when starting the server**
Run `pip install ultralytics`.

**WebSocket fails with an `AssertionError`**
The server was started without `wsproto`. Make sure `server.py` calls
`uvicorn.run(..., ws="wsproto")` and that `pip install wsproto` has been run.

**Camera does not open**
Close other applications that may be holding the camera (Zoom, Teams, Discord,
Skype).

**Speed in the UI is too low or too high**
Adjust the calibration constant `FLOW_TO_KMH` in `src/camera_inference.py`.
The current value of `1.2` is calibrated on real dashcam video (24 fps, 4K).

**`yolov8n.pt` did not download**
An internet connection is required on the first server start. If you have no
connection, download the weights manually from
https://github.com/ultralytics/assets/releases/.

---

## Authors

- Latip Medet
- Kadiraly Miras
- Tursynbayev Aibar

Supervisor: Sadvakasova Assemgul — Astana IT University, School of Intelligent
Systems.

## License

Academic project. Not for commercial use.
