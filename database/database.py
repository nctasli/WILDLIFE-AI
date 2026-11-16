"""SQLite persistence for detections, events, and target snapshots."""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from detection.animal_detector import Detection
from deterrent.trigger import TriggerResult

logger = logging.getLogger("database.sqlite")


class WildlifeDatabase:
    """Thread-safe SQLite storage for detection history and event summaries."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(config.DATABASE_PATH if path is None else path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                timeout=10,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()
        except sqlite3.Error:
            logger.exception("Could not initialize SQLite database at %s", self.path)
            raise

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    species TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detector_confidence REAL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    target_confirmed INTEGER NOT NULL DEFAULT 0,
                    snapshot_path TEXT,
                    audio_triggered INTEGER NOT NULL DEFAULT 0,
                    trigger_reason TEXT NOT NULL DEFAULT 'not_target',
                    cooldown_remaining_seconds REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    track_id INTEGER,
                    species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detector_confidence REAL,
                    detected_at TEXT NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    target_progress INTEGER NOT NULL DEFAULT 0,
                    target_confirmed INTEGER NOT NULL DEFAULT 0,
                    track_state TEXT NOT NULL DEFAULT 'untracked',
                    snapshot_path TEXT,
                    FOREIGN KEY (event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_detections_detected_at
                    ON detections(detected_at DESC);
                CREATE INDEX IF NOT EXISTS idx_detections_event_id
                    ON detections(event_id);
                CREATE INDEX IF NOT EXISTS idx_events_last_seen_at
                    ON events(last_seen_at DESC);
                """
            )
        logger.info("SQLite database ready: %s", self.path)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def save_snapshot(self, detection: Detection, frame: np.ndarray) -> Path:
        """Store only a newly confirmed target animal, never a person image."""
        if (
            not detection.target_confirmed
            or detection.class_name.casefold() == "person"
            or detection.class_name.casefold().startswith("animal (uncertain)")
        ):
            raise ValueError("Snapshots are only allowed for confirmed, classified targets.")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        snapshot_dir = Path(config.SNAPSHOTS_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        snapshot_path = snapshot_dir / filename
        try:
            written = cv2.imwrite(str(snapshot_path), frame)
        except cv2.error as exc:
            logger.exception("OpenCV failed to save target snapshot for %s", detection.event_id)
            raise RuntimeError(f"Could not save snapshot for event {detection.event_id}") from exc
        if not written:
            error = RuntimeError(f"Could not save snapshot to {snapshot_path}")
            logger.error("%s", error)
            raise error
        logger.info("Saved target snapshot for event %s: %s", detection.event_id, snapshot_path)
        return snapshot_path

    def record_detection(
        self,
        detection: Detection,
        *,
        snapshot_path: str | Path | None = None,
        trigger_result: TriggerResult | None = None,
    ) -> int:
        """Persist a detection and update its event summary; return its row ID."""
        x1, y1, x2, y2 = detection.coordinates
        detected_at = self._timestamp(detection.detected_at)
        image_path = str(snapshot_path) if snapshot_path is not None else None
        if trigger_result is not None:
            trigger_reason = trigger_result.reason
            audio_triggered = int(trigger_result.triggered)
            cooldown_remaining = trigger_result.cooldown_remaining_seconds
        elif detection.target_confirmed:
            trigger_reason = "target_confirmed"
            audio_triggered = 0
            cooldown_remaining = 0.0
        elif detection.target_progress > 0:
            trigger_reason = "awaiting_confirmation"
            audio_triggered = 0
            cooldown_remaining = 0.0
        else:
            trigger_reason = "not_target"
            audio_triggered = 0
            cooldown_remaining = 0.0

        try:
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    INSERT INTO events (
                        event_id, species, first_seen_at, last_seen_at, confidence,
                        detector_confidence, x1, y1, x2, y2, target_confirmed,
                        snapshot_path, audio_triggered, trigger_reason,
                        cooldown_remaining_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        species = excluded.species,
                        last_seen_at = excluded.last_seen_at,
                        confidence = excluded.confidence,
                        detector_confidence = excluded.detector_confidence,
                        x1 = excluded.x1,
                        y1 = excluded.y1,
                        x2 = excluded.x2,
                        y2 = excluded.y2,
                        target_confirmed = MAX(events.target_confirmed, excluded.target_confirmed),
                        snapshot_path = COALESCE(excluded.snapshot_path, events.snapshot_path),
                        audio_triggered = MAX(events.audio_triggered, excluded.audio_triggered),
                        trigger_reason = CASE
                            WHEN excluded.trigger_reason IN (
                                'played', 'cooldown', 'deterrence_disabled',
                                'event_already_triggered', 'audio_error'
                            ) THEN excluded.trigger_reason
                            WHEN events.trigger_reason IN (
                                'played', 'cooldown', 'deterrence_disabled',
                                'event_already_triggered', 'audio_error'
                            ) THEN events.trigger_reason
                            ELSE excluded.trigger_reason
                        END,
                        cooldown_remaining_seconds = MAX(
                            events.cooldown_remaining_seconds,
                            excluded.cooldown_remaining_seconds
                        )
                    """,
                    (
                        detection.event_id,
                        detection.class_name,
                        detected_at,
                        detected_at,
                        detection.confidence,
                        detection.detector_confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        int(detection.target_confirmed),
                        image_path,
                        audio_triggered,
                        trigger_reason,
                        cooldown_remaining,
                    ),
                )
                cursor = self._connection.execute(
                    """
                    INSERT INTO detections (
                        event_id, track_id, species, confidence, detector_confidence,
                        detected_at, x1, y1, x2, y2, target_progress,
                        target_confirmed, track_state, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        detection.event_id,
                        detection.track_id,
                        detection.class_name,
                        detection.confidence,
                        detection.detector_confidence,
                        detected_at,
                        x1,
                        y1,
                        x2,
                        y2,
                        detection.target_progress,
                        int(detection.target_confirmed),
                        detection.track_state,
                        image_path,
                    ),
                )
                row_id = int(cursor.lastrowid)
        except sqlite3.Error:
            logger.exception(
                "Could not persist detection event %s in %s",
                detection.event_id,
                self.path,
            )
            raise
        return row_id

    def record_trigger_result(self, result: TriggerResult) -> None:
        """Update an event's sound/cooldown outcome."""
        try:
            with self._lock, self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE events
                    SET audio_triggered = MAX(audio_triggered, ?),
                        trigger_reason = ?,
                        cooldown_remaining_seconds = ?
                    WHERE event_id = ?
                    """,
                    (
                        int(result.triggered),
                        result.reason,
                        result.cooldown_remaining_seconds,
                        result.event_id,
                    ),
                )
                if cursor.rowcount == 0:
                    logger.warning(
                        "Trigger result references unknown event %s",
                        result.event_id,
                    )
        except sqlite3.Error:
            logger.exception("Could not persist trigger result for %s", result.event_id)
            raise

    def list_recent_events(self, limit: int = 100) -> list[dict[str, object]]:
        """Return recent event summaries for dashboard and tests."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            with self._lock:
                rows = self._connection.execute(
                    """
                    SELECT * FROM events
                    ORDER BY last_seen_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Could not read recent events")
            raise

    def list_recent_detections(self, limit: int = 100) -> list[dict[str, object]]:
        """Return individual detection observations, newest first."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            with self._lock:
                rows = self._connection.execute(
                    """
                    SELECT * FROM detections
                    ORDER BY detected_at DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Could not read detection history")
            raise

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()
            logger.info("Closed SQLite database: %s", self.path)

    def __enter__(self) -> WildlifeDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["WildlifeDatabase"]
