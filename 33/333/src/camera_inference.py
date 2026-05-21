"""
Camera Inference Module for Driving Style ML Project
=====================================================

Real-time driving style assessment and accident risk prediction
using a camera feed. Integrates YOLOv8 object detection with
the trained ML/DL models for risk scoring.

Pipeline:
  Camera → YOLOv8 Detection → Feature Extraction → FeatureAdapter → ML Model → Risk Score

Feature mismatch solution
--------------------------
Models are trained on 19 telematic features (avg_speed, max_speed, …, trip_distance).
The camera can only observe optical-flow and bounding-box signals.
``FeatureAdapter`` bridges the gap with physically motivated mappings:

  Camera signal              → Model feature          Conversion
  ─────────────────────────────────────────────────────────────────
  flow_magnitude  (px/frame) → avg_speed   (km/h)     * FLOW_TO_KMH
  flow_magnitude max         → max_speed   (km/h)     * FLOW_TO_KMH
  flow_magnitude std²        → speed_variance         (km/h)²
  harsh_brake_count          → harsh_braking_count    direct
  harsh_accel_count          → harsh_accel_count      direct
  flow_delta_mean            → avg_acceleration       * DELTA_TO_ACCEL
  flow_lateral_mean          → lane_changes           * LAT_TO_LANES
  1/(leading_area+ε)         → following_distance     * AREA_TO_DIST
  —  (unobservable)          → turn_signal_usage      default 0.5
  flow_delta_mean            → accel_x_mean           * DELTA_TO_ACCEL
  flow_lateral_mean          → accel_y_mean           * LAT_TO_ACCEL
  constant                   → accel_z_mean           9.81 m/s²
  flow_delta_std             → gyro_x_std             * FLOW_TO_GYRO
  flow_lateral_std           → gyro_y_std             * FLOW_TO_GYRO
  flow_lateral_std           → gyro_z_std             * FLOW_TO_GYRO
  datetime.now()             → time_of_day            hour (0-23)
  datetime.now()             → day_of_week            weekday (0-6)
  elapsed session time       → trip_duration          seconds
  avg_speed * duration / 3600→ trip_distance          km

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
import time
import pickle
import logging
import argparse
import collections
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path so src.* imports work regardless of CWD
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    MODELS_DIR,
    DRIVING_STYLE_CLASSES,
    RISK_SCORING_CONFIG,
)
from src.risk_scoring import RiskScorer, RiskLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YOLO classes that represent vehicles (COCO dataset indices)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Colour palette for risk levels (BGR)
RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    RiskLevel.LOW.value:      (0, 200, 0),    # green
    RiskLevel.MEDIUM.value:   (0, 200, 255),  # yellow
    RiskLevel.HIGH.value:     (0, 100, 255),  # orange
    RiskLevel.CRITICAL.value: (0, 0, 220),    # red
}

# Window of frames used to build one feature vector for the ML model
DEFAULT_WINDOW = 30   # ~1 second at 30 fps

# Minimum frames in buffer before running the model (warm-up period)
MIN_WINDOW = 10

# Optimization constants
FLOW_RESIZE = (320, 240)   # compute optical flow on reduced frame
YOLO_SKIP_FRAMES = 3       # run YOLO every N frames
FLOW_TO_KMH = 1.2          # recalibrated on real dashcam footage (24fps, 4K).
                            # Previous 0.44 underestimated speed by factor ~2.77.
                            # Verified empirically: UI shows ~23 km/h while real speed is ~65 km/h.

# ---------------------------------------------------------------------------
# Feature name lists — single source of truth
# ---------------------------------------------------------------------------

# 19 features the telematic models were trained on (from data_loader.py)
MODEL_FEATURE_NAMES: List[str] = [
    "avg_speed",
    "max_speed",
    "speed_variance",
    "harsh_braking_count",
    "harsh_accel_count",
    "avg_acceleration",
    "lane_changes",
    "following_distance",
    "turn_signal_usage",
    "accel_x_mean",
    "accel_y_mean",
    "accel_z_mean",
    "gyro_x_std",
    "gyro_y_std",
    "gyro_z_std",
    "time_of_day",
    "day_of_week",
    "trip_duration",
    "trip_distance",
]

# 18 features extracted directly from the camera pipeline
CAMERA_FEATURE_NAMES: List[str] = [
    "speed_mean", "speed_std", "speed_max",
    "flow_lateral_mean", "flow_lateral_std",
    "flow_delta_mean", "flow_delta_std",
    "harsh_brake_count", "harsh_accel_count",
    "close_follow_count",
    "vehicle_count_mean", "vehicle_count_max",
    "leading_area_mean", "leading_area_std", "leading_area_max",
    "leading_delta_mean",
    "accel_magnitude_proxy",
    "speeding_ratio",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """Raw features extracted from a single video frame."""

    flow_magnitude: float = 0.0    # mean dense-optical-flow magnitude
    flow_lateral:   float = 0.0    # mean |horizontal| flow (lane-drift proxy)
    flow_delta:     float = 0.0    # change in magnitude (accel / brake proxy)

    vehicle_count:     int   = 0   # YOLO vehicle detections
    leading_box_area:  float = 0.0 # normalised bbox area of nearest vehicle
    leading_box_delta: float = 0.0 # change in bbox area (closing-speed proxy)

    harsh_brake:  int = 0
    harsh_accel:  int = 0
    close_follow: int = 0          # leading vehicle too close (bbox width > 15 %)


@dataclass
class InferenceResult:
    """Output of one inference step."""

    risk_score:      float
    risk_level:      str
    driving_style:   str
    confidence:      float
    fps:             float
    vehicle_count:   int
    speed_kmh:       float
    brake_count:  int   = 0
    accel_count:  int   = 0
    lane_changes: int   = 0
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optical Flow Estimator
# ---------------------------------------------------------------------------

class OpticalFlowEstimator:
    """
    Estimates camera motion via Farneback dense optical flow.
    Returns (magnitude, lateral, delta) per frame.
    """

    def __init__(self) -> None:
        self._prev_gray:      Optional[np.ndarray] = None
        self._prev_magnitude: float = 0.0
        self.last_fx:         Optional[np.ndarray] = None

    def compute(self, frame: np.ndarray) -> Tuple[float, float, float]:
        small = cv2.resize(frame, FLOW_RESIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )

        fx, fy   = flow[..., 0], flow[..., 1]
        magnitude = float(np.mean(np.sqrt(fx ** 2 + fy ** 2)))
        lateral   = float(np.mean(np.abs(fx)))
        delta     = magnitude - self._prev_magnitude

        self._prev_gray      = gray
        self._prev_magnitude = magnitude
        self.last_fx         = fx

        return magnitude, lateral, delta

    def reset(self) -> None:
        self._prev_gray      = None
        self._prev_magnitude = 0.0
        self.last_fx         = None


# ---------------------------------------------------------------------------
# Lane Change Detector
# ---------------------------------------------------------------------------

class LaneChangeDetector:
    LATERAL_THRESHOLD = 1.2
    SUSTAINED_FRAMES = 8
    COOLDOWN_FRAMES = 45

    def __init__(self):
        self._history = collections.deque(maxlen=self.SUSTAINED_FRAMES)
        self._cooldown = 0
        self._total_changes = 0

    def update(self, flow_fx: np.ndarray, frame_h: int, frame_w: int) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        roi_y1, roi_y2 = int(frame_h * 0.3), int(frame_h * 0.7)
        roi_x1, roi_x2 = int(frame_w * 0.2), int(frame_w * 0.8)
        # flow_fx may be computed on FLOW_RESIZE — scale ROI accordingly
        fh, fw = flow_fx.shape[:2]
        ry1 = int(fh * 0.3); ry2 = int(fh * 0.7)
        rx1 = int(fw * 0.2); rx2 = int(fw * 0.8)
        roi_flow = flow_fx[ry1:ry2, rx1:rx2]
        lateral_mean = float(np.mean(roi_flow))
        self._history.append(lateral_mean)
        if len(self._history) == self.SUSTAINED_FRAMES:
            all_pos = all(v > self.LATERAL_THRESHOLD for v in self._history)
            all_neg = all(v < -self.LATERAL_THRESHOLD for v in self._history)
            if all_pos or all_neg:
                self._total_changes += 1
                self._cooldown = self.COOLDOWN_FRAMES
                return True
        return False

    @property
    def total_changes(self) -> int:
        return self._total_changes


# ---------------------------------------------------------------------------
# Harsh Event Detector
# ---------------------------------------------------------------------------

class HarshEventDetector:
    WINDOW = 60
    K_SIGMA = 2.5
    COOLDOWN = 15

    def __init__(self):
        self._history = collections.deque(maxlen=self.WINDOW)
        self._brake_cooldown = 0
        self._accel_cooldown = 0
        self.brake_count = 0
        self.accel_count = 0

    def update(self, flow_delta: float) -> tuple[bool, bool]:
        self._history.append(flow_delta)
        if len(self._history) < 20:
            return False, False
        arr = np.array(self._history)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr)) + 1e-6
        is_brake = False
        is_accel = False
        if self._brake_cooldown > 0:
            self._brake_cooldown -= 1
        elif flow_delta < mean_val - self.K_SIGMA * std_val:
            is_brake = True
            self.brake_count += 1
            self._brake_cooldown = self.COOLDOWN
        if self._accel_cooldown > 0:
            self._accel_cooldown -= 1
        elif flow_delta > mean_val + self.K_SIGMA * std_val:
            is_accel = True
            self.accel_count += 1
            self._accel_cooldown = self.COOLDOWN
        return is_brake, is_accel


# ---------------------------------------------------------------------------
# YOLOv8 Detector wrapper
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    Thin wrapper around an Ultralytics YOLOv8 model for vehicle detection.
    Gracefully falls back to no-op if ``ultralytics`` is not installed.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self._model = None

        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(model_path)
            self._model.to(device)
            logger.info("YOLOv8 loaded: %s on %s", model_path, device)
        except ImportError:
            logger.warning(
                "ultralytics not installed – running without object detection. "
                "Install with: pip install ultralytics"
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Return list of vehicle detections with class_id, class_name, confidence, bbox."""
        if self._model is None:
            return []

        results = self._model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id":   cls_id,
                "class_name": VEHICLE_CLASSES[cls_id],
                "confidence": conf,
                "bbox":       (x1, y1, x2, y2),
            })

        return detections


# ---------------------------------------------------------------------------
# Camera Feature Extractor
# ---------------------------------------------------------------------------

class CameraFeatureExtractor:
    """
    Maintains a rolling buffer of FrameFeatures and aggregates them into
    an 18-element camera feature vector (``CAMERA_FEATURE_NAMES``).
    """

    FEATURE_NAMES = CAMERA_FEATURE_NAMES
    N_FEATURES    = len(CAMERA_FEATURE_NAMES)

    # Event-detection thresholds — lowered to match real optical-flow magnitudes
    # at 320×240 resolution: typical flow_delta ±0.1-0.5 px/frame
    HARSH_BRAKE_THRESH   = -0.25   # negative delta in flow magnitude
    HARSH_ACCEL_THRESH   =  0.25   # positive delta
    CLOSE_FOLLOW_THRESH  =  0.12   # leading bbox normalised width > 12 %
    SPEEDING_FLOW_THRESH =  1.2    # flow magnitude proxy for "over speed limit"

    def __init__(self, window_size: int = DEFAULT_WINDOW) -> None:
        self.window_size = window_size
        self._buffer: Deque[FrameFeatures] = collections.deque(maxlen=window_size)
        self.lane_detector  = LaneChangeDetector()
        self.harsh_detector = HarshEventDetector()

    def update(self, ff: FrameFeatures) -> None:
        self._buffer.append(ff)

    def update_with_flow(self, ff: FrameFeatures, flow_fx: np.ndarray) -> None:
        """Update buffer and also run lane-change and harsh-event detectors."""
        self.update(ff)
        fh, fw = flow_fx.shape[:2]
        self.lane_detector.update(flow_fx, fh, fw)
        self.harsh_detector.update(ff.flow_delta)

    @property
    def ready(self) -> bool:
        return len(self._buffer) >= MIN_WINDOW

    @property
    def lane_changes(self) -> int:
        return self.lane_detector.total_changes

    @property
    def brake_count(self) -> int:
        return self.harsh_detector.brake_count

    @property
    def accel_count(self) -> int:
        return self.harsh_detector.accel_count

    def extract(self) -> np.ndarray:
        """Return the 18-element camera feature vector for the current window."""
        if not self._buffer:
            return np.zeros(self.N_FEATURES, dtype=np.float32)

        speeds   = np.array([f.flow_magnitude   for f in self._buffer])
        laterals = np.array([f.flow_lateral      for f in self._buffer])
        deltas   = np.array([f.flow_delta        for f in self._buffer])
        areas    = np.array([f.leading_box_area  for f in self._buffer])
        area_d   = np.array([f.leading_box_delta for f in self._buffer])
        counts   = np.array([f.vehicle_count     for f in self._buffer], dtype=float)

        harsh_brake_cnt  = int(np.sum([f.harsh_brake  for f in self._buffer]))
        harsh_accel_cnt  = int(np.sum([f.harsh_accel  for f in self._buffer]))
        close_follow_cnt = int(np.sum([f.close_follow for f in self._buffer]))
        speeding_ratio   = float(np.mean(speeds > self.SPEEDING_FLOW_THRESH))
        accel_proxy      = float(np.mean(np.sqrt(speeds ** 2 + laterals ** 2)))

        return np.array([
            np.mean(speeds),    np.std(speeds),    np.max(speeds),
            np.mean(laterals),  np.std(laterals),
            np.mean(deltas),    np.std(deltas),
            harsh_brake_cnt,    harsh_accel_cnt,
            close_follow_cnt,
            np.mean(counts),    np.max(counts),
            np.mean(areas),     np.std(areas),     np.max(areas),
            np.mean(area_d),
            accel_proxy,
            speeding_ratio,
        ], dtype=np.float32)

    def rule_based_risk(self, cam_vec: np.ndarray, speed_kmh: float) -> Tuple[float, str]:
        """
        Compute risk score 0-1 directly from camera features without the ML model.
        Uses the same weights as RISK_SCORING_CONFIG.
        This always works regardless of model availability or feature-count mismatches.
        """
        idx = {name: i for i, name in enumerate(CAMERA_FEATURE_NAMES)}

        # 1. Speed violation (25%) — реальное превышение
        # До 60 км/ч риск 0, к 100 км/ч полный вес
        speed_excess = max(0.0, speed_kmh - 60.0)
        speed_risk = min(speed_excess / 40.0, 1.0) * 0.25

        # 2. Harsh braking (15%) — нужно много событий для значимого риска,
        # учитывая что детектор фолсит на качке камеры
        brakes = float(cam_vec[idx["harsh_brake_count"]])
        brake_risk = min(max(0.0, brakes - 2.0) / 4.0, 1.0) * 0.15

        # 3. Harsh acceleration (10%)
        accels = float(cam_vec[idx["harsh_accel_count"]])
        accel_risk = min(max(0.0, accels - 2.0) / 4.0, 1.0) * 0.10

        # 4. Lateral / lane (10%) — порог сильно поднят: на прямой трассе
        # из-за перспективы камеры lat_flow стабильно 1.5-1.8, это норма.
        # Реальные перестроения дают всплески 2.5+
        lat_flow = float(cam_vec[idx["flow_lateral_mean"]])
        lane_risk = min(max(0.0, lat_flow - 2.0) / 1.5, 1.0) * 0.10

        # 5. Following distance (25%) — главный реальный сигнал опасности.
        # Tailgating: leading_area_mean > 8% (порог поднят с 5%, т.к. на
        # FullHD-видео даже далёкая машина даёт 2-3%). К 15% — полный риск.
        lead_area = float(cam_vec[idx["leading_area_mean"]])
        follow_risk = min(max(0.0, lead_area - 0.08) / 0.07, 1.0) * 0.25

        # 6. Speed variance (10%) — реальная эрратичная езда
        speed_std = float(cam_vec[idx["speed_std"]])
        variance_risk = min(max(0.0, speed_std - 0.3) / 0.4, 1.0) * 0.10

        # 7. Speeding ratio (5%) — компонент сильно ослаблен, т.к. в текущем виде
        # (speeds > SPEEDING_FLOW_THRESH=1.2) он всегда 1.0 при любой езде.
        # Считаем долю кадров с РЕАЛЬНО высоким flow (proxy для серьёзного превышения)
        speeds = np.array([cam_vec[idx["speed_mean"]], cam_vec[idx["speed_max"]]])
        real_speeding = 1.0 if cam_vec[idx["speed_max"]] > 4.5 else 0.0
        speeding_risk = real_speeding * 0.05

        risk_score = float(np.clip(
            speed_risk + brake_risk + accel_risk + lane_risk +
            follow_risk + variance_risk + speeding_risk,
            0.0, 1.0
        ))

        # Пороги уровней — без изменений
        if risk_score >= 0.75:
            level = "CRITICAL"
        elif risk_score >= 0.55:
            level = "HIGH"
        elif risk_score >= 0.30:
            level = "MEDIUM"
        else:
            level = "LOW"

        breakdown = {
            "speed": round(speed_risk, 3),
            "brake": round(brake_risk, 3),
            "accel": round(accel_risk, 3),
            "lane": round(lane_risk, 3),
            "follow": round(follow_risk, 3),
            "variance": round(variance_risk, 3),
            "speeding": round(speeding_risk, 3),
            "lead_area_raw": round(float(cam_vec[idx["leading_area_mean"]]), 4),
            "lat_flow_raw": round(float(cam_vec[idx["flow_lateral_mean"]]), 4),
            "speed_kmh_raw": round(speed_kmh, 1),
        }
        return risk_score, level, breakdown

    def reset(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Feature Adapter  ← THE KEY CLASS THAT FIXES FEATURE MISMATCH
# ---------------------------------------------------------------------------

class FeatureAdapter:
    """
    Maps 18 camera-extracted features → 19 model-trained features.

    The trained models expect the feature vector produced by
    ``data_loader.create_sample_dataset`` (telematic data).
    This adapter converts camera signals to physically plausible
    approximations of those features so the same model can be used
    without retraining.

    Calibration constants
    ---------------------
    All constants have a physical interpretation and can be tuned
    once the camera's field-of-view and mounting angle are known.

    Parameters
    ----------
    fps : float
        Estimated frame rate of the camera (used for speed conversion).
    trip_start : float
        Unix timestamp of session start (from ``time.time()``).
    flow_to_kmh : float
        Pixels/frame → km/h multiplier.  Default 1.0 means 1 px/frame ≈ 1 km/h
        at 30 fps.  Calibrate for your camera.
    delta_to_accel : float
        flow_delta_mean → m/s² multiplier.
    lat_to_lanes : float
        flow_lateral_mean → lane_changes (count per window) multiplier.
    area_to_dist : float
        Scale factor for following-distance inversion from bbox area.
    flow_to_gyro : float
        Flow std → gyroscope std multiplier.
    default_turn_signal : float
        Default value for turn_signal_usage (0–1, unobservable from camera).
    """

    # Index map into the 18-element camera feature vector
    _IDX = {name: i for i, name in enumerate(CAMERA_FEATURE_NAMES)}

    def __init__(
        self,
        fps:                  float = 30.0,
        trip_start:           float = 0.0,
        flow_to_kmh:          float = FLOW_TO_KMH,
        delta_to_accel:       float = 0.1,
        lat_to_lanes:         float = 2.0,
        area_to_dist:         float = 0.05,
        flow_to_gyro:         float = 0.05,
        default_turn_signal:  float = 0.5,
    ) -> None:
        self.fps                 = fps
        self.trip_start          = trip_start or time.time()
        self.flow_to_kmh         = flow_to_kmh
        self.delta_to_accel      = delta_to_accel
        self.lat_to_lanes        = lat_to_lanes
        self.area_to_dist        = area_to_dist
        self.flow_to_gyro        = flow_to_gyro
        self.default_turn_signal = default_turn_signal

        self._cum_distance_km:   float = 0.0
        self._last_adapt_time:   float = time.time()

    # ------------------------------------------------------------------

    def adapt(self, cam_vec: np.ndarray) -> np.ndarray:
        """
        Convert the 18-element camera feature vector into the 19-element
        model feature vector expected by the trained sklearn / XGBoost models.

        Parameters
        ----------
        cam_vec : ndarray, shape (18,)
            Output of ``CameraFeatureExtractor.extract()``.

        Returns
        -------
        ndarray, shape (19,)
            Feature vector aligned with ``MODEL_FEATURE_NAMES``.
        """
        idx = self._IDX

        # --- speed signals ---
        speed_mean = float(cam_vec[idx["speed_mean"]]) * self.flow_to_kmh * self.fps
        speed_mean = np.clip(speed_mean, 0, 200)
        speed_max  = float(cam_vec[idx["speed_max"]])  * self.flow_to_kmh * self.fps
        speed_std  = float(cam_vec[idx["speed_std"]])  * self.flow_to_kmh * self.fps
        speed_var  = speed_std ** 2

        # --- acceleration / braking events ---
        harsh_brake_count = float(cam_vec[idx["harsh_brake_count"]])
        harsh_accel_count = float(cam_vec[idx["harsh_accel_count"]])

        # avg_acceleration: flow_delta_mean scaled to m/s²
        avg_acceleration = float(cam_vec[idx["flow_delta_mean"]]) * self.delta_to_accel

        # --- lane changes: lateral flow proxy ---
        lane_changes = float(cam_vec[idx["flow_lateral_mean"]]) * self.lat_to_lanes

        # --- following distance: invert normalised bbox area ---
        # larger bbox area → vehicle is closer → shorter following distance
        leading_area = float(cam_vec[idx["leading_area_mean"]])
        following_distance = self.area_to_dist / (leading_area + 1e-6)
        following_distance = float(np.clip(following_distance, 0.3, 10.0))

        # --- turn signal: unobservable, use default ---
        turn_signal_usage = self.default_turn_signal

        # --- IMU proxies ---
        # accel_x (longitudinal): flow_delta as proxy
        accel_x_mean = float(cam_vec[idx["flow_delta_mean"]]) * self.delta_to_accel
        # accel_y (lateral): lateral flow as proxy
        accel_y_mean = float(cam_vec[idx["flow_lateral_mean"]]) * self.delta_to_accel
        # accel_z: gravity constant
        accel_z_mean = 9.81

        # gyro std proxies
        gyro_x_std = float(cam_vec[idx["flow_delta_std"]])    * self.flow_to_gyro
        gyro_y_std = float(cam_vec[idx["flow_lateral_std"]])  * self.flow_to_gyro
        gyro_z_std = float(cam_vec[idx["flow_lateral_std"]])  * self.flow_to_gyro

        # --- contextual: real-time clock ---
        now          = datetime.now()
        time_of_day  = float(now.hour)
        day_of_week  = float(now.weekday())

        # --- trip metrics: session elapsed time and distance ---
        trip_duration = float(time.time() - self.trip_start)

        # accumulate distance estimate
        dt = time.time() - self._last_adapt_time
        self._cum_distance_km += speed_mean * dt / 3600.0
        self._last_adapt_time  = time.time()
        trip_distance = self._cum_distance_km

        # --- assemble output in MODEL_FEATURE_NAMES order ---
        model_vec = np.array([
            avg_speed          := speed_mean,       # avg_speed
            speed_max,                              # max_speed
            speed_var,                              # speed_variance
            harsh_brake_count,                      # harsh_braking_count
            harsh_accel_count,                      # harsh_accel_count
            avg_acceleration,                       # avg_acceleration
            lane_changes,                           # lane_changes
            following_distance,                     # following_distance
            turn_signal_usage,                      # turn_signal_usage
            accel_x_mean,                           # accel_x_mean
            accel_y_mean,                           # accel_y_mean
            accel_z_mean,                           # accel_z_mean
            gyro_x_std,                             # gyro_x_std
            gyro_y_std,                             # gyro_y_std
            gyro_z_std,                             # gyro_z_std
            time_of_day,                            # time_of_day
            day_of_week,                            # day_of_week
            trip_duration,                          # trip_duration
            trip_distance,                          # trip_distance
        ], dtype=np.float32)

        return model_vec

    def reset_trip(self) -> None:
        """Reset trip timer and odometer (call when starting a new session)."""
        self.trip_start        = time.time()
        self._cum_distance_km  = 0.0
        self._last_adapt_time  = time.time()


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Loads a trained model from disk.

    Supported formats
    -----------------
    - ``.pkl`` / ``.pickle``  — scikit-learn / XGBoost via pickle
    - ``.joblib``             — scikit-learn via joblib
    - ``.pt`` / ``.pth``      — full PyTorch model (torch.save(model, path))
    """

    @staticmethod
    def load(model_path: str) -> Any:
        path   = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".pkl", ".pickle"):
            with open(path, "rb") as fh:
                model = pickle.load(fh)
            logger.info("Loaded pickle model from %s", path)
            return model

        if suffix == ".joblib":
            import joblib  # type: ignore
            model = joblib.load(path)
            logger.info("Loaded joblib model from %s", path)
            return model

        if suffix in (".pt", ".pth"):
            return ModelLoader._load_pytorch(path)

        raise ValueError(f"Unsupported model format: {suffix}")

    @staticmethod
    def _load_pytorch(path: Path) -> Any:
        import torch  # type: ignore
        obj = torch.load(path, map_location="cpu")
        if hasattr(obj, "forward"):
            obj.eval()
            logger.info("Loaded full PyTorch model from %s", path)
            return _TorchModelWrapper(obj)
        raise ValueError(
            f"{path} appears to be a PyTorch state-dict. "
            "Save the full model with torch.save(model, path)."
        )


class _TorchModelWrapper:
    """Sklearn-compatible wrapper for a PyTorch nn.Module."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.FloatTensor(X))
            return torch.argmax(out, dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch, torch.nn.functional as F
        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.FloatTensor(X))
            return F.softmax(out, dim=1).numpy()


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class RealTimeVisualizer:
    """Draws bounding boxes, HUD overlay, and risk gauge on each frame."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.fw = frame_w
        self.fh = frame_h

    def draw(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        detections: List[Dict],
    ) -> np.ndarray:
        frame = frame.copy()
        frame = self._draw_detections(frame, detections)
        frame = self._draw_hud(frame, result)
        frame = self._draw_risk_bar(frame, result.risk_score)
        frame = self._draw_speedometer(frame, result.speed_kmh)
        return frame

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                frame, f"{d['class_name']} {d['confidence']:.2f}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1,
            )
        return frame

    def _draw_hud(self, frame: np.ndarray, result: InferenceResult) -> np.ndarray:
        color   = RISK_COLORS.get(result.risk_level, (200, 200, 200))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 230), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        speed_str = f"~{result.speed_kmh:.0f} km/h" if result.vehicle_count > 0 else "N/A"
        style_str = result.driving_style if result.driving_style else "---"
        lines = [
            (f"Style:  {style_str}",                          color),
            (f"Risk:   {result.risk_level}  ({result.risk_score:.2f})", color),
            (f"Speed:  {speed_str}",                          (200, 200, 200)),
            (f"Brake(total): {result.brake_count}",           (200, 200, 200)),
            (f"Accel(total): {result.accel_count}",           (200, 200, 200)),
            (f"Lane changes: {result.lane_changes}",          (200, 200, 200)),
            (f"Vehicles: {result.vehicle_count}",             (200, 200, 200)),
            (f"FPS:    {result.fps:.1f}",                     (180, 180, 180)),
        ]
        y0 = 24
        for text, clr in lines:
            cv2.putText(frame, text, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, clr, 1, cv2.LINE_AA)
            y0 += 26

        if result.recommendations:
            rec_y = self.fh - 20 * len(result.recommendations) - 10
            for rec in result.recommendations[:3]:
                cv2.putText(frame, f"! {rec}", (10, rec_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 255), 1, cv2.LINE_AA)
                rec_y += 20
        return frame

    def _draw_speedometer(self, frame: np.ndarray, speed_kmh: float) -> np.ndarray:
        center = (self.fw - 70, self.fh - 70)
        radius = 55
        max_speed = 120
        angle = int(240 * min(speed_kmh / max_speed, 1.0))
        cv2.ellipse(frame, center, (radius, radius), 150, 0, 240, (60, 60, 60), 8)
        color = (0, 200, 0) if speed_kmh < 60 else (0, 165, 255) if speed_kmh < 100 else (0, 0, 220)
        if angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 150, 0, angle, color, 8)
        cv2.putText(frame, f"{int(speed_kmh)}", (center[0] - 18, center[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "km/h", (center[0] - 20, center[1] + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        return frame

    def _draw_risk_bar(self, frame: np.ndarray, risk_score: float) -> np.ndarray:
        bar_h  = 12
        bar_y  = self.fh - bar_h - 4
        bar_w  = self.fw - 8
        filled = int(bar_w * np.clip(risk_score, 0.0, 1.0))

        cv2.rectangle(frame, (4, bar_y), (4 + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        clr = (0, 200, 0) if risk_score < 0.3 else \
              (0, 200, 255) if risk_score < 0.6 else \
              (0, 100, 255) if risk_score < 0.8 else (0, 0, 220)
        cv2.rectangle(frame, (4, bar_y), (4 + filled, bar_y + bar_h), clr, -1)
        cv2.rectangle(frame, (4, bar_y), (4 + bar_w, bar_y + bar_h), (120, 120, 120), 1)
        cv2.putText(frame, f"RISK {risk_score:.2f}", (6, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        return frame


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class CameraInference:
    """
    Orchestrates real-time driving-style assessment from a camera feed.

    Parameters
    ----------
    model_path : str
        Path to trained sklearn / XGBoost / PyTorch model (.pkl / .joblib / .pt).
    yolo_model_path : str
        YOLOv8 weight file. Downloaded automatically if not found locally.
    source : int or str
        OpenCV video source – 0 for webcam, or path to a video file.
    window_size : int
        Rolling frame buffer size.
    conf_threshold : float
        YOLO detection confidence threshold.
    device : str
        Compute device ('cpu', 'cuda', 'mps').
    show : bool
        Display annotated video in a window.
    save_path : str, optional
        Path to save annotated output video.
    num_classes : int
        Number of driving-style classes the model was trained on.
    use_adapter : bool
        If True (default), apply ``FeatureAdapter`` so telematic-trained
        models receive the correct 19-feature vector.
        Set to False only when the model was trained directly on
        ``CAMERA_FEATURE_NAMES`` (18 features).
    flow_to_kmh : float
        Camera-specific calibration constant passed to ``FeatureAdapter``.
    """

    def __init__(
        self,
        model_path:       str,
        yolo_model_path:  str  = "yolov8n.pt",
        source:           Any  = 0,
        window_size:      int  = DEFAULT_WINDOW,
        conf_threshold:   float = 0.40,
        device:           str  = "cpu",
        show:             bool = True,
        save_path:        Optional[str] = None,
        num_classes:      int  = 4,
        use_adapter:      bool = True,
        flow_to_kmh:      float = 1.0,
    ) -> None:
        logger.info("Initialising CameraInference …")

        self.source      = source
        self.show        = show
        self.save_path   = save_path
        self.num_classes = num_classes
        self.use_adapter = use_adapter

        self.model     = ModelLoader.load(model_path)
        self.detector  = VehicleDetector(yolo_model_path, conf_threshold, device)
        self.flow_est  = OpticalFlowEstimator()
        self.extractor = CameraFeatureExtractor(window_size)
        self.scorer    = RiskScorer()
        self.adapter   = FeatureAdapter(trip_start=time.time(), flow_to_kmh=flow_to_kmh)

        self._prev_area: float = 0.0
        self._cap:    Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter]  = None
        self._frame_counter:   int       = 0
        self._last_detections: List[Dict] = []

        n_expected = len(MODEL_FEATURE_NAMES) if use_adapter else len(CAMERA_FEATURE_NAMES)
        logger.info(
            "Feature adapter: %s  |  model expects %d features",
            "ON" if use_adapter else "OFF", n_expected,
        )

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        """Start the inference loop. Press Q to quit."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        fw      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Keep adapter's fps estimate up-to-date with the actual source fps
        self.adapter.fps = fps_src

        visualiser = RealTimeVisualizer(fw, fh)

        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps_src, (fw, fh))
            logger.info("Writing output to %s", self.save_path)

        result = InferenceResult(
            risk_score=0.0, risk_level=RiskLevel.LOW.value,
            driving_style="Initialising…", confidence=0.0,
            fps=0.0, vehicle_count=0, speed_kmh=0.0,
        )

        t_prev = time.perf_counter()
        logger.info("Capture loop started. Press Q to quit.")

        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    logger.info("End of stream.")
                    break

                t_now  = time.perf_counter()
                fps    = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                self._frame_counter += 1
                if self._frame_counter % YOLO_SKIP_FRAMES == 0:
                    self._last_detections = self.detector.detect(frame)
                detections = self._last_detections

                ff, flow_fx = self._extract_frame_features(frame, detections)
                self.extractor.update_with_flow(ff, flow_fx)

                if self.extractor.ready:
                    result = self._infer(ff, fps)

                result.fps           = fps
                result.vehicle_count = ff.vehicle_count

                annotated = visualiser.draw(frame, result, detections)

                if self.show:
                    cv2.imshow("Driving Style – Risk Assessment", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed Q.")
                        break

                if self._writer:
                    self._writer.write(annotated)

        finally:
            self._cleanup()

    # --------------------------------------------------------------- helpers

    def _extract_frame_features(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> Tuple[FrameFeatures, np.ndarray]:
        """Build a FrameFeatures from raw frame data and YOLO detections.

        Returns
        -------
        (FrameFeatures, flow_fx)
            flow_fx is the raw horizontal flow array (may be None on first frame).
        """
        magnitude, lateral, delta = self.flow_est.compute(frame)
        flow_fx = self.flow_est.last_fx if self.flow_est.last_fx is not None \
                  else np.zeros((FLOW_RESIZE[1], FLOW_RESIZE[0]), dtype=np.float32)

        leading_area = 0.0
        area_delta   = 0.0
        close_follow = 0

        if detections:
            frame_area = frame.shape[0] * frame.shape[1]
            areas = [
                (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / frame_area
                for d in detections
            ]
            leading_area = float(max(areas))
            area_delta   = leading_area - self._prev_area
            self._prev_area = leading_area

            frame_w = frame.shape[1]
            for d in detections:
                x1, _, x2, _ = d["bbox"]
                if (x2 - x1) / frame_w > CameraFeatureExtractor.CLOSE_FOLLOW_THRESH:
                    close_follow = 1
                    break
        else:
            self._prev_area = 0.0

        ff = FrameFeatures(
            flow_magnitude    = magnitude,
            flow_lateral      = lateral,
            flow_delta        = delta,
            vehicle_count     = len(detections),
            leading_box_area  = leading_area,
            leading_box_delta = area_delta,
            harsh_brake  = int(delta < CameraFeatureExtractor.HARSH_BRAKE_THRESH),
            harsh_accel  = int(delta > CameraFeatureExtractor.HARSH_ACCEL_THRESH),
            close_follow = close_follow,
        )
        return ff, flow_fx

    def _infer(self, ff: FrameFeatures, fps: float) -> InferenceResult:
        """
        Run the ML model on the current feature window.

        If ``use_adapter`` is True the 18-element camera vector is first
        converted to the 19-element model vector via ``FeatureAdapter.adapt``.
        """
        cam_vec = self.extractor.extract()   # shape (18,)

        if self.use_adapter:
            model_vec = self.adapter.adapt(cam_vec)   # shape (19,)
        else:
            model_vec = cam_vec                        # shape (18,) – camera-trained model

        X = model_vec.reshape(1, -1)

        prediction    = int(self.model.predict(X)[0])
        driving_style = DRIVING_STYLE_CLASSES.get(prediction, f"Class {prediction}")

        if hasattr(self.model, "predict_proba"):
            probas     = self.model.predict_proba(X)[0]
            n_cls      = len(probas)
            weights    = np.array([i / max(n_cls - 1, 1) for i in range(n_cls)])
            risk_score = float(np.clip(np.dot(probas, weights), 0.0, 1.0))
            confidence = float(probas.max())
        else:
            style_map  = {0: 0.15, 1: 0.35, 2: 0.65, 3: 0.85}
            risk_score = style_map.get(prediction, 0.5)
            confidence = 0.0

        risk_level = self.scorer.get_risk_level(risk_score).value
        speed_kmh  = float(cam_vec[CameraFeatureExtractor.FEATURE_NAMES.index("speed_mean")]) \
                     * self.adapter.flow_to_kmh * fps
        recs = self.scorer._generate_recommendations(
            self.scorer.get_risk_level(risk_score), driving_style, {}
        )

        return InferenceResult(
            risk_score    = risk_score,
            risk_level    = risk_level,
            driving_style = driving_style,
            confidence    = confidence,
            fps           = fps,
            vehicle_count = ff.vehicle_count,
            speed_kmh     = speed_kmh,
            brake_count   = self.extractor.brake_count,
            accel_count   = self.extractor.accel_count,
            lane_changes  = self.extractor.lane_changes,
            recommendations = recs,
        )

    def _cleanup(self) -> None:
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        logger.info("Resources released.")


# ---------------------------------------------------------------------------
# Public helper — maps camera vector → model vector (standalone utility)
# ---------------------------------------------------------------------------

def map_camera_features_to_model_features(
    cam_vec: np.ndarray,
    fps:           float = 30.0,
    trip_start:    float = 0.0,
    flow_to_kmh:   float = 1.0,
    delta_to_accel: float = 0.1,
    lat_to_lanes:  float = 2.0,
    area_to_dist:  float = 0.05,
    flow_to_gyro:  float = 0.05,
) -> np.ndarray:
    """
    Convert a single camera feature vector (18 elements) to a model feature
    vector (19 elements) matching ``MODEL_FEATURE_NAMES``.

    This is the standalone version of ``FeatureAdapter.adapt`` for use in
    notebooks, tests, or offline evaluation.

    Parameters
    ----------
    cam_vec : ndarray, shape (18,)
        Camera feature vector from ``CameraFeatureExtractor.extract()``.
    fps : float
        Camera frame rate for speed conversion.
    trip_start : float
        Unix timestamp of trip start (``time.time()``). 0 → use current time.
    flow_to_kmh : float
        Optical-flow to km/h conversion factor.
    delta_to_accel : float
        Flow-delta to m/s² scale factor.
    lat_to_lanes : float
        Lateral-flow to lane-changes scale factor.
    area_to_dist : float
        Bounding-box area to following-distance scale factor.
    flow_to_gyro : float
        Flow std to gyroscope std scale factor.

    Returns
    -------
    ndarray, shape (19,)
        Feature vector ready to pass to a model trained on telematic data.

    Example
    -------
    >>> cam_vec = extractor.extract()   # shape (18,)
    >>> model_vec = map_camera_features_to_model_features(cam_vec, fps=30)
    >>> model_vec.shape
    (19,)
    >>> prediction = model.predict(model_vec.reshape(1, -1))
    """
    adapter = FeatureAdapter(
        fps            = fps,
        trip_start     = trip_start or time.time(),
        flow_to_kmh    = flow_to_kmh,
        delta_to_accel = delta_to_accel,
        lat_to_lanes   = lat_to_lanes,
        area_to_dist   = area_to_dist,
        flow_to_gyro   = flow_to_gyro,
    )
    return adapter.adapt(cam_vec)


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------

def run_camera_inference(
    model_path:     str,
    source:         Any  = 0,
    yolo_model_path: str = "yolov8n.pt",
    window_size:    int  = DEFAULT_WINDOW,
    conf_threshold: float = 0.40,
    device:         str  = "cpu",
    show:           bool = True,
    save_path:      Optional[str] = None,
    num_classes:    int  = 4,
    use_adapter:    bool = True,
    flow_to_kmh:    float = 1.0,
) -> None:
    """
    Launch real-time driving-style inference from a camera or video file.

    Parameters
    ----------
    model_path : str
        Path to trained model (.pkl / .joblib / .pt).
    source : int or str
        0 = default webcam; or path to a .mp4 / .avi file.
    yolo_model_path : str
        YOLOv8 weights. ``yolov8n.pt`` is downloaded automatically (~6 MB).
    window_size : int
        Rolling window of frames for feature aggregation.
    conf_threshold : float
        YOLO detection confidence threshold.
    device : str
        'cpu', 'cuda', or 'mps'.
    show : bool
        Display annotated video.
    save_path : str, optional
        Save annotated video to this path.
    num_classes : int
        Number of driving-style classes.
    use_adapter : bool
        True  → use FeatureAdapter (telematic-trained models, default).
        False → pass raw camera features (camera-trained models only).
    flow_to_kmh : float
        Optical-flow calibration constant.

    Example
    -------
    >>> from src.camera_inference import run_camera_inference
    >>> run_camera_inference(model_path="models/random_forest.pkl", source=0)
    """
    CameraInference(
        model_path      = model_path,
        yolo_model_path = yolo_model_path,
        source          = source,
        window_size     = window_size,
        conf_threshold  = conf_threshold,
        device          = device,
        show            = show,
        save_path       = save_path,
        num_classes     = num_classes,
        use_adapter     = use_adapter,
        flow_to_kmh     = flow_to_kmh,
    ).run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time driving style inference from camera / video."
    )
    p.add_argument("--model",   required=True,
                   help="Path to trained model (.pkl / .joblib / .pt)")
    p.add_argument("--source",  default="0",
                   help="Video source: '0' for webcam or path to video file (default: 0)")
    p.add_argument("--yolo",    default="yolov8n.pt",
                   help="YOLOv8 weights (default: yolov8n.pt, downloaded automatically)")
    p.add_argument("--window",  type=int,   default=DEFAULT_WINDOW,
                   help=f"Rolling window size in frames (default: {DEFAULT_WINDOW})")
    p.add_argument("--conf",    type=float, default=0.40,
                   help="YOLO detection confidence threshold (default: 0.40)")
    p.add_argument("--device",  default="cpu",
                   help="Compute device: cpu / cuda / mps (default: cpu)")
    p.add_argument("--no-show", action="store_true",
                   help="Disable OpenCV display window (headless mode)")
    p.add_argument("--save",    default=None,
                   help="Save annotated output video to this path (e.g. output.mp4)")
    p.add_argument("--num-classes", type=int, default=4,
                   help="Number of driving-style classes (default: 4)")
    p.add_argument("--no-adapter", action="store_true",
                   help=(
                       "Disable FeatureAdapter. Use only when the model was trained "
                       "on CAMERA_FEATURE_NAMES (18 features) instead of the default "
                       "telematic MODEL_FEATURE_NAMES (19 features)."
                   ))
    p.add_argument("--flow-to-kmh", type=float, default=1.0,
                   help="Optical-flow to km/h calibration factor (default: 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    args = _parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    run_camera_inference(
        model_path      = args.model,
        source          = source,
        yolo_model_path = args.yolo,
        window_size     = args.window,
        conf_threshold  = args.conf,
        device          = args.device,
        show            = not args.no_show,
        save_path       = args.save,
        num_classes     = args.num_classes,
        use_adapter     = not args.no_adapter,
        flow_to_kmh     = args.flow_to_kmh,
    )
