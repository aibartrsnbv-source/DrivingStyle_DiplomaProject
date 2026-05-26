import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent.parent / "driveguard.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trips (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    safety_score        INTEGER,
    risk_level          TEXT,
    duration_sec        REAL,
    avg_speed           REAL,
    max_speed           REAL,
    harsh_braking       INTEGER,
    harsh_accel         INTEGER,
    lane_changes        INTEGER,
    high_risk_time_pct  REAL,
    weather             TEXT,
    model_name          TEXT,
    recommendation      TEXT
)
"""


def init_db() -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()
        logger.info("Database initialised at %s", DB_PATH)
    except Exception:
        logger.exception("Failed to initialise database")


def _derive_risk_level(avg_risk: float) -> str:
    if avg_risk >= 0.75:
        return "CRITICAL"
    if avg_risk >= 0.55:
        return "HIGH"
    if avg_risk >= 0.30:
        return "MEDIUM"
    return "LOW"


def save_trip(data: dict) -> int:
    sql = """
    INSERT INTO trips (
        created_at, safety_score, risk_level, duration_sec,
        avg_speed, max_speed, harsh_braking, harsh_accel,
        lane_changes, high_risk_time_pct, weather, model_name, recommendation
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        now = datetime.now(timezone.utc).isoformat()
        avg_risk = float(data.get("avg_risk") or 0.0)
        risk_level = _derive_risk_level(avg_risk)

        conditions = data.get("conditions") or []
        weather = ", ".join(conditions) if conditions else None

        recommendations = data.get("recommendations") or []
        recommendation = "\n".join(recommendations) if recommendations else None

        params = (
            now,
            data.get("safety_score"),
            risk_level,
            data.get("duration_sec"),
            data.get("avg_speed"),
            data.get("max_speed"),
            data.get("total_brakes"),
            data.get("total_accels"),
            data.get("total_lane_changes"),
            data.get("high_risk_pct"),
            weather,
            data.get("model_name"),
            recommendation,
        )
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            row_id = cursor.lastrowid
        logger.info("Trip saved with id=%s", row_id)
        return row_id
    except Exception:
        logger.exception("Failed to save trip to database")
        return -1


def get_trips(limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM trips ORDER BY id DESC LIMIT ?"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        logger.exception("Failed to retrieve trips")
        return []


def get_trip(trip_id: int) -> dict | None:
    sql = "SELECT * FROM trips WHERE id = ?"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, (trip_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception:
        logger.exception("Failed to retrieve trip %d", trip_id)
        return None
