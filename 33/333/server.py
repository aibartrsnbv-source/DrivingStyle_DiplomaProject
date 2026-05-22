import matplotlib
matplotlib.use("Agg")  # ОБЯЗАТЕЛЬНО ПЕРВЫМ — фикс для работы без GUI-дисплея

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

import asyncio
import base64
import json
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import sys
sys.path.append(str(Path(__file__).parent))

# Импорт существующего модуля с graceful fallback если не существует
CAMERA_INFERENCE_AVAILABLE = False
try:
    from src.camera_inference import (
        CameraInference,
        CameraFeatureExtractor,
        OpticalFlowEstimator,
        VehicleDetector,
        FeatureAdapter,
        ModelLoader,
        RealTimeVisualizer,
        InferenceResult,
        FrameFeatures,
        RISK_COLORS,
        DEFAULT_WINDOW,
        MIN_WINDOW,
    )
    from src.risk_scoring import RiskLevel
    CAMERA_INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] camera_inference не найден: {e}")
    print("[WARN] Сервер запустится, но анализ видео недоступен.")
    # Заглушка для RiskLevel
    from enum import Enum
    class RiskLevel(Enum):
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"

app = FastAPI(title="DriveGuard AI", version="1.0.0")

# Папки
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")

WEB_DIR = Path("web")

# Монтируем статику
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

executor = ThreadPoolExecutor(max_workers=4)

# Хранилище активных сессий (session_id -> stop_event)
active_sessions: dict[str, threading.Event] = {}


# ─────────────────────────── HTTP эндпоинты ────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>DriveGuard AI</h1><p>web/index.html не найден</p>")


@app.get("/api/models")
async def get_models():
    if not MODELS_DIR.exists():
        return JSONResponse({"models": []})
    models = [f.name for f in MODELS_DIR.iterdir() if f.suffix == ".pkl"]
    return JSONResponse({"models": models})


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = TEMP_DIR / f"{video_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)
    return JSONResponse({
        "video_id": video_id,
        "filename": file.filename,
        "status": "ready",
        "_path": str(save_path),
    })


# ─────────────────────────── WebSocket ─────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    stop_event = threading.Event()
    active_sessions[session_id] = stop_event
    loop = asyncio.get_event_loop()

    async def send(data: dict):
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                continue

            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "message": "Неверный формат команды"})
                continue

            action = cmd.get("action")

            if action == "start":
                # Останавливаем предыдущую сессию если есть
                stop_event.set()
                stop_event = threading.Event()
                active_sessions[session_id] = stop_event

                source = cmd.get("source", "camera")
                model_name = cmd.get("model", "")
                video_id = cmd.get("video_id")
                conditions = cmd.get("conditions", [])  # список строк: ["rain", "night"] и т.п.

                if not CAMERA_INFERENCE_AVAILABLE:
                    await send({"type": "error", "message": "Модуль camera_inference недоступен. Убедитесь что он существует в src/"})
                    continue

                model_path = MODELS_DIR / model_name if model_name else None
                if model_path and not model_path.exists():
                    await send({"type": "error", "message": f"Модель не найдена: {model_name}"})
                    continue

                # Определяем путь к видео
                video_path = None
                if source == "video":
                    if not video_id:
                        await send({"type": "error", "message": "video_id не указан"})
                        continue
                    # Ищем файл по video_id
                    matches = list(TEMP_DIR.glob(f"{video_id}*"))
                    if not matches:
                        await send({"type": "error", "message": "Видеофайл не найден"})
                        continue
                    video_path = str(matches[0])

                await send({"type": "status", "message": "Инициализация..."})

                current_stop = stop_event

                def run_inference():
                    asyncio.run_coroutine_threadsafe(
                        _inference_loop(websocket, send, source, model_path, video_path, current_stop, loop, conditions),
                        loop
                    )

                loop.run_in_executor(executor, run_inference)

            elif action == "stop":
                stop_event.set()
                # trip report + stopped отправляет _inference_loop после выхода из цикла

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        active_sessions.pop(session_id, None)


def _build_trip_report(risk_scores, speeds, high_risk_frames, total_frames,
                       brakes, accels, lane_changes, conditions, start_time):
    """Собирает итоговый отчёт о поездке из накопленной статистики."""
    import numpy as np

    if total_frames == 0 or not risk_scores:
        return {
            "type": "report",
            "available": False,
            "message": "Недостаточно данных для отчёта (анализ был слишком коротким).",
        }

    avg_risk = float(np.mean(risk_scores))
    safety_score = int(round(max(0.0, min(100.0, 100.0 * (1.0 - avg_risk)))))

    if safety_score >= 85:
        verdict = "Отличный водитель"
    elif safety_score >= 70:
        verdict = "Хороший водитель"
    elif safety_score >= 50:
        verdict = "Средний уровень"
    else:
        verdict = "Опасный стиль вождения"

    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    high_risk_pct = round(100.0 * high_risk_frames / total_frames, 1)
    duration_sec = round(time.time() - start_time, 1)

    recommendations = []
    if brakes >= 5:
        recommendations.append("Вы часто резко тормозите. Старайтесь держать большую дистанцию и тормозить плавно.")
    if accels >= 5:
        recommendations.append("Замечены частые резкие ускорения. Плавный разгон снижает риск и расход топлива.")
    if high_risk_pct >= 30:
        recommendations.append("Значительную часть поездки уровень риска был высоким. Обратите внимание на дистанцию и скорость.")
    if max_speed > 100:
        recommendations.append("Зафиксирована высокая скорость. Соблюдение скоростного режима критично для безопасности.")
    if lane_changes >= 8:
        recommendations.append("Частые перестроения. Убедитесь, что используете поворотники и проверяете слепые зоны.")
    if avg_risk < 0.25 and brakes < 3:
        recommendations.append("Вы вели машину спокойно и безопасно. Так держать.")
    if not recommendations:
        recommendations.append("Поездка прошла в штатном режиме без существенных замечаний.")

    return {
        "type": "report",
        "available": True,
        "safety_score": safety_score,
        "verdict": verdict,
        "avg_risk": round(avg_risk, 3),
        "avg_speed": round(avg_speed, 1),
        "max_speed": round(max_speed, 1),
        "total_brakes": int(brakes),
        "total_accels": int(accels),
        "total_lane_changes": int(lane_changes),
        "high_risk_pct": high_risk_pct,
        "duration_sec": duration_sec,
        "conditions": conditions or [],
        "recommendations": recommendations,
    }


async def _inference_loop(
    websocket: WebSocket,
    send,
    source: str,
    model_path,
    video_path: str | None,
    stop_event: threading.Event,
    loop,
    conditions: list = None,
):
    """Основной цикл инференса (выполняется в потоке через ThreadPoolExecutor)."""
    cap = None
    try:
        # ── Загрузка модели ──
        await send({"type": "status", "message": "Загрузка модели..."})
        if model_path:
            model = ModelLoader.load(str(model_path))
        else:
            model = None

        # ── Открываем видеоисточник ──
        await send({"type": "status", "message": "Инициализация камеры..." if source == "camera" else "Открытие видеофайла..."})

        if source == "camera":
            # Пытаемся найти iPhone через iVCam (обычно камера #1 если есть встроенная вебка).
            # Перебираем индексы 1, 2, 0 — сначала внешние, потом встроенная как fallback.
            cap = None
            for cam_idx in [1, 2, 0]:
                test_cap = cv2.VideoCapture(cam_idx)
                if test_cap.isOpened():
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        await send({"type": "status", "message": f"Камера открыта (index={cam_idx})"})
                        break
                    test_cap.release()
                else:
                    test_cap.release()

            if cap is None:
                await send({"type": "error", "message": "Не удалось найти ни одну рабочую камеру"})
                return
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                await send({"type": "error", "message": "Не удалось открыть видеофайл"})
                return

        await send({"type": "status", "message": "Анализ запущен..."})

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Инициализация компонентов ──
        flow_estimator = OpticalFlowEstimator()
        vehicle_detector = VehicleDetector()
        extractor = CameraFeatureExtractor()
        adapter = FeatureAdapter(trip_start=time.time())
        visualizer = RealTimeVisualizer(fw, fh)

        frame_interval = 0.033  # ~30 FPS
        last_frame_time = 0.0
        prev_leading_area = 0.0
        # Session-level accumulators (only increment, never reset per-frame)
        no_vehicle_frames = 0   # consecutive frames without a vehicle
        NO_VEHICLE_THRESHOLD = 15  # if no vehicle for this many frames → not a driving scene
        # EMA-сглаживание риска во времени (α=0.15 → лаг ~6-7 кадров)
        ema_risk_score = 0.0
        EMA_ALPHA = 0.15
        ema_initialized = False

        # EMA-сглаживание скорости (alpha=0.10 → лаг ~10 кадров)
        ema_speed_kmh = 0.0
        EMA_ALPHA_SPEED = 0.05
        ema_speed_initialized = False

        # Контекстный риск: множитель чувствительности по дорожным условиям
        CONDITION_MULTIPLIERS = {
            "dry": 1.0,
            "rain": 1.4,
            "snow": 1.6,
            "night": 1.25,
        }
        _conditions = conditions or []
        context_multiplier = 1.0
        for cond in _conditions:
            context_multiplier *= CONDITION_MULTIPLIERS.get(cond, 1.0)
        context_multiplier = min(context_multiplier, 2.0)  # потолок
        if _conditions and context_multiplier > 1.0:
            await send({"type": "status", "message": f"Условия: {', '.join(_conditions)} (×{context_multiplier:.2f})"})

        # --- Trip report accumulators ---
        report_risk_scores = []      # все risk_score за сессию
        report_speeds = []           # все speed_kmh где scene активна
        report_high_risk_frames = 0  # кадры с HIGH/CRITICAL
        report_total_frames = 0      # всего кадров с extractor.ready
        report_start_time = time.time()

        harsh_brakes = 0
        harsh_accels = 0
        lane_changes = 0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if source == "video":
                    await send({"type": "status", "message": "Видео завершено"})
                else:
                    await send({"type": "error", "message": "Потеря сигнала с камеры"})
                break

            now = time.time()
            if now - last_frame_time < frame_interval:
                await asyncio.sleep(0.005)
                continue
            last_frame_time = now

            # ── Optical flow ──
            flow_mag, flow_lat, flow_delta = flow_estimator.compute(frame)

            # ── Детекция машин ──
            detections = vehicle_detector.detect(frame)

            # ── Вычисляем bbox ведущего автомобиля ──
            leading_area = 0.0
            close_follow = 0
            if detections:
                frame_area = frame.shape[0] * frame.shape[1]
                areas = [
                    (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / frame_area
                    for d in detections
                ]
                leading_area = float(max(areas))
                frame_w = frame.shape[1]
                for d in detections:
                    x1, _, x2, _ = d["bbox"]
                    if (x2 - x1) / frame_w > CameraFeatureExtractor.CLOSE_FOLLOW_THRESH:
                        close_follow = 1
                        break
            area_delta = leading_area - prev_leading_area
            prev_leading_area = leading_area

            # ── Обновление экстрактора признаков ──
            frame_feat = FrameFeatures(
                flow_magnitude=flow_mag,
                flow_lateral=flow_lat,
                flow_delta=flow_delta,
                vehicle_count=len(detections),
                leading_box_area=leading_area,
                leading_box_delta=area_delta,
                harsh_brake=int(flow_delta < CameraFeatureExtractor.HARSH_BRAKE_THRESH),
                harsh_accel=int(flow_delta > CameraFeatureExtractor.HARSH_ACCEL_THRESH),
                close_follow=close_follow,
            )
            if flow_estimator.last_fx is not None:
                extractor.update_with_flow(frame_feat, flow_estimator.last_fx)
            else:
                extractor.update(frame_feat)

            risk_score = 0.0
            risk_level = RiskLevel.LOW
            result = None
            speed_kmh = 0.0
            following_distance = -1.0   # -1 = no vehicle detected

            # Track consecutive frames without vehicle
            if len(detections) == 0:
                no_vehicle_frames += 1
            else:
                no_vehicle_frames = 0

            is_driving_scene = (no_vehicle_frames < NO_VEHICLE_THRESHOLD)

            if extractor.ready:
                cam_vec = extractor.extract()

                # --- Session-cumulative event counters (never decrement) ---
                # extractor.brake_count / accel_count are session totals from HarshEventDetector
                # extractor.lane_changes is session total from LaneChangeDetector
                harsh_brakes = extractor.brake_count
                harsh_accels = extractor.accel_count
                lane_changes = extractor.lane_changes

                # --- Speed: only meaningful in a driving scene ---
                if is_driving_scene:
                    raw_speed_kmh = float(cam_vec[0]) * adapter.flow_to_kmh * adapter.fps
                    # EMA-сглаживание чтобы скорость не дёргалась кадр-в-кадр
                    if not ema_speed_initialized:
                        ema_speed_kmh = raw_speed_kmh
                        ema_speed_initialized = True
                    else:
                        ema_speed_kmh = EMA_ALPHA_SPEED * raw_speed_kmh + (1 - EMA_ALPHA_SPEED) * ema_speed_kmh
                    speed_kmh = ema_speed_kmh
                else:
                    speed_kmh = 0.0
                    # Сброс сглаживания, чтобы при возврате на дорогу не было артефакта
                    ema_speed_initialized = False

                # --- Following distance: only when a vehicle is visible ---
                la = float(cam_vec[12])  # leading_area_mean
                if la > 0.001 and len(detections) > 0:
                    following_distance = float(np.clip(adapter.area_to_dist / (la + 1e-6), 0.3, 9.9))
                else:
                    following_distance = -1.0  # sentinel: "no vehicle ahead"

                # --- Rule-based risk (always computed from camera features) ---
                if is_driving_scene:
                    risk_score, risk_level_str, _risk_breakdown = extractor.rule_based_risk(cam_vec, speed_kmh)
                else:
                    # Non-driving context: show 0 risk, don't mislead
                    risk_score, risk_level_str, _risk_breakdown = 0.0, "LOW", {}

                level_map = {
                    "LOW": RiskLevel.LOW, "MEDIUM": RiskLevel.MEDIUM,
                    "HIGH": RiskLevel.HIGH, "CRITICAL": RiskLevel.CRITICAL,
                }
                risk_level = level_map.get(risk_level_str, RiskLevel.LOW)

                # --- Optional ML model refinement ---
                confidence = 0.0
                if model is not None and is_driving_scene:
                    try:
                        model_vec = adapter.adapt(cam_vec)
                        proba = model.predict_proba([model_vec])[0]
                        n_cls = len(proba)
                        weights = np.array([i / max(n_cls - 1, 1) for i in range(n_cls)])
                        model_score = float(np.clip(np.dot(proba, weights), 0.0, 1.0))
                        confidence = float(proba.max())

                        # Подмешиваем ML только если модель достаточно уверена
                        if confidence >= 0.60:
                            ml_weight = 0.3 + 0.3 * (confidence - 0.6) / 0.4
                            risk_score = float(np.clip((1 - ml_weight) * risk_score + ml_weight * model_score, 0.0, 1.0))
                    except Exception:
                        pass

                # Контекстный риск: усиливаем по дорожным условиям
                risk_score = float(np.clip(risk_score * context_multiplier, 0.0, 1.0))

                # EMA smoothing
                if not ema_initialized:
                    ema_risk_score = risk_score
                    ema_initialized = True
                else:
                    ema_risk_score = EMA_ALPHA * risk_score + (1 - EMA_ALPHA) * ema_risk_score
                risk_score = ema_risk_score

                # Единый threshold-блок по сглаженному скору
                if risk_score >= 0.75:
                    risk_level = RiskLevel.CRITICAL
                elif risk_score >= 0.55:
                    risk_level = RiskLevel.HIGH
                elif risk_score >= 0.30:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW

                # Накопление статистики для итогового отчёта
                report_total_frames += 1
                report_risk_scores.append(risk_score)
                if is_driving_scene and speed_kmh > 0:
                    report_speeds.append(speed_kmh)
                if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                    report_high_risk_frames += 1

                driving_style_str = "Non-driving" if not is_driving_scene else "Unknown"
                result = InferenceResult(
                    risk_score=risk_score,
                    risk_level=risk_level.value if hasattr(risk_level, "value") else str(risk_level),
                    driving_style=driving_style_str,
                    confidence=confidence,
                    fps=0.0,
                    vehicle_count=len(detections),
                    speed_kmh=speed_kmh,
                    brake_count=harsh_brakes,
                    accel_count=harsh_accels,
                    lane_changes=lane_changes,
                )

            # ── Отрисовка HUD ──
            if result is not None:
                frame = visualizer.draw(frame, result, detections)

            # ── Кодирование кадра ──
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            image_b64 = base64.b64encode(buffer).decode("utf-8")

            level_val = risk_level.value if hasattr(risk_level, "value") else str(risk_level)

            await send({
                "type": "frame",
                "image": image_b64,
                "risk_score": round(risk_score, 3),
                "risk_level": level_val,
                "speed_kmh": round(speed_kmh, 1),
                "harsh_brakes": harsh_brakes,
                "harsh_accels": harsh_accels,
                "following_distance": round(following_distance, 1) if following_distance >= 0 else -1,
                "lane_changes": lane_changes,
                "timestamp": now,
                "context_multiplier": round(context_multiplier, 2),
                "active_conditions": _conditions,
            })

            if level_val in ("HIGH", "CRITICAL"):
                msg = "Критическая угроза! Немедленно снизьте скорость!" if level_val == "CRITICAL" else "Высокий риск ДТП! Соблюдайте дистанцию!"
                await send({"type": "alert", "risk_level": level_val, "message": msg})

            await asyncio.sleep(0)  # отдаём управление event loop

        # Завершение сессии: отправляем итоговый отчёт, затем stopped
        report = _build_trip_report(
            report_risk_scores, report_speeds, report_high_risk_frames,
            report_total_frames, harsh_brakes, harsh_accels, lane_changes,
            _conditions, report_start_time,
        )
        await send(report)
        await send({"type": "stopped"})

    except Exception as e:
        await send({"type": "error", "message": f"Ошибка анализа: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()


# ───────────────────────────── Запуск ──────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="wsproto", reload=False)
