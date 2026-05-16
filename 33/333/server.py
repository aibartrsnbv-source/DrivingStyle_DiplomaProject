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
                        _inference_loop(websocket, send, source, model_path, video_path, current_stop, loop),
                        loop
                    )

                loop.run_in_executor(executor, run_inference)

            elif action == "stop":
                stop_event.set()
                await send({"type": "stopped"})

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        active_sessions.pop(session_id, None)


async def _inference_loop(
    websocket: WebSocket,
    send,
    source: str,
    model_path,
    video_path: str | None,
    stop_event: threading.Event,
    loop,
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
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            await send({"type": "error", "message": "Не удалось открыть источник видео"})
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

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if source == "video":
                    await send({"type": "status", "message": "Видео завершено"})
                    await send({"type": "stopped"})
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
            harsh_brakes = 0
            harsh_accels = 0
            following_distance = -1.0   # -1 = no vehicle detected
            lane_changes = 0

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
                    speed_kmh = float(cam_vec[0]) * adapter.flow_to_kmh * adapter.fps
                else:
                    speed_kmh = 0.0

                # --- Following distance: only when a vehicle is visible ---
                la = float(cam_vec[12])  # leading_area_mean
                if la > 0.001 and len(detections) > 0:
                    following_distance = float(np.clip(adapter.area_to_dist / (la + 1e-6), 0.3, 9.9))
                else:
                    following_distance = -1.0  # sentinel: "no vehicle ahead"

                # --- Rule-based risk (always computed from camera features) ---
                if is_driving_scene:
                    risk_score, risk_level_str = extractor.rule_based_risk(cam_vec, speed_kmh)
                else:
                    # Non-driving context: show 0 risk, don't mislead
                    risk_score, risk_level_str = 0.0, "LOW"

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
                        risk_score = float(np.clip(0.5 * risk_score + 0.5 * model_score, 0.0, 1.0))
                        confidence = float(proba.max())
                        if risk_score >= 0.65:
                            risk_level = RiskLevel.CRITICAL
                        elif risk_score >= 0.45:
                            risk_level = RiskLevel.HIGH
                        elif risk_score >= 0.22:
                            risk_level = RiskLevel.MEDIUM
                        else:
                            risk_level = RiskLevel.LOW
                    except Exception:
                        pass

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
            })

            if level_val in ("HIGH", "CRITICAL"):
                msg = "Критическая угроза! Немедленно снизьте скорость!" if level_val == "CRITICAL" else "Высокий риск ДТП! Соблюдайте дистанцию!"
                await send({"type": "alert", "risk_level": level_val, "message": msg})

            await asyncio.sleep(0)  # отдаём управление event loop

    except Exception as e:
        await send({"type": "error", "message": f"Ошибка анализа: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()


# ───────────────────────────── Запуск ──────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="wsproto", reload=False)
