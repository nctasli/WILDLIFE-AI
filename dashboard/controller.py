"""Background camera and detection lifecycle for the web dashboard."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

import config
from camera import create_camera
from database import WildlifeDatabase
from detection import AnimalDetector
from detection.base_detector import FrameDetector
from detection.tracker import AnimalTracker
from deterrent import AudioPlayer, DeterrentTrigger
from deterrent.trigger import AudioOutput
from main import PreviewCamera, run_preview

logger = logging.getLogger("dashboard.controller")


class DashboardAudioOutput(AudioOutput, Protocol):
    @property
    def is_playing(self) -> bool: ...


class DashboardCamera(PreviewCamera, Protocol):
    @property
    def is_opened(self) -> bool: ...


class DashboardController:
    """Own the dashboard's single camera, inference, and audio worker."""

    def __init__(
        self,
        database: WildlifeDatabase | None = None,
        *,
        camera_factory: Callable[[], DashboardCamera] | None = None,
        detector_factory: Callable[[], FrameDetector] | None = None,
        audio_player: DashboardAudioOutput | None = None,
    ) -> None:
        if config.DASHBOARD_ANALYSIS_EVERY <= 0:
            raise ValueError("DASHBOARD_ANALYSIS_EVERY must be greater than zero.")
        if config.DASHBOARD_STREAM_FPS <= 0:
            raise ValueError("DASHBOARD_STREAM_FPS must be greater than zero.")

        self.database = database or WildlifeDatabase()
        self._camera_factory = camera_factory or self._create_camera
        self._detector_factory = detector_factory or self._create_detector
        self._audio_player = audio_player or AudioPlayer()
        self._lock = threading.RLock()
        self._state = "stopped"
        self._detector_state = "stopped"
        self._last_error: str | None = None
        self._camera: DashboardCamera | None = None
        self._trigger: DeterrentTrigger | None = None
        self._latest_frame: bytes | None = None
        self._last_encoded_at = 0.0
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @staticmethod
    def _create_camera() -> DashboardCamera:
        return create_camera(
            config.CAMERA_SOURCE,
            video_path=config.VIDEO_PATH,
            webcam_index=config.WEBCAM_INDEX,
            esp32_stream_url=config.ESP32_STREAM_URL,
        )

    @staticmethod
    def _create_detector() -> FrameDetector:
        if config.DASHBOARD_DETECTOR == "yolo":
            return AnimalDetector()
        if config.DASHBOARD_DETECTOR == "speciesnet":
            from detection.speciesnet_detector import SpeciesNetAnimalDetector

            return SpeciesNetAnimalDetector()
        raise ValueError(
            "DASHBOARD_DETECTOR must be either 'yolo' or 'speciesnet'."
        )

    @property
    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def start(self) -> bool:
        """Start the worker once; return False if it is already active."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Dashboard controller is closed.")
            if self._thread is not None and self._thread.is_alive():
                return False

            self._audio_player.stop()
            self._stop_event = threading.Event()
            self._state = "starting"
            self._detector_state = "loading"
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                args=(self._stop_event,),
                name="wildlife-dashboard",
                daemon=True,
            )
            self._thread.start()
            logger.info("Dashboard monitoring worker started.")
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Request shutdown and wait up to ``timeout`` seconds."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                if self._state != "error":
                    self._state = "stopped"
                    self._detector_state = "stopped"
                trigger = self._trigger
                thread = None
            else:
                if self._stop_event is not None:
                    self._stop_event.set()
                self._state = "stopping"
                trigger = self._trigger

        if thread is None:
            if trigger is not None:
                trigger.emergency_stop()
            else:
                self._audio_player.stop()
            return True
        if trigger is not None:
            trigger.emergency_stop()
        else:
            self._audio_player.stop()
        thread.join(timeout=max(0.0, timeout))
        stopped = not thread.is_alive()
        if not stopped:
            logger.warning("Dashboard worker did not stop within %.1f seconds.", timeout)
        return stopped

    def set_deterrence_enabled(self, enabled: bool) -> None:
        """Change the automatic deterrence gate while monitoring is active."""
        with self._lock:
            if self._state != "running" or self._trigger is None:
                raise RuntimeError("Start monitoring before changing deterrence.")
            self._trigger.set_enabled(enabled)

    def emergency_stop(self) -> None:
        """Disable automatic deterrence and stop any current sound immediately."""
        with self._lock:
            trigger = self._trigger
        if trigger is not None:
            trigger.emergency_stop()
        else:
            self._audio_player.stop()

    def test_sound(self) -> None:
        """Play the configured test sound only when monitoring is stopped."""
        with self._lock:
            if self._state in {"starting", "running", "stopping"}:
                raise RuntimeError("Stop monitoring before testing the sound.")
            if self._audio_player.is_playing:
                raise RuntimeError("Audio playback is already in progress.")
            self._audio_player.play(
                config.DEFAULT_SOUND_PATH,
                volume=config.MAX_AUDIO_VOLUME,
            )

    def status(self) -> dict[str, object]:
        """Return a JSON-ready status snapshot and the latest detection."""
        with self._lock:
            camera_connected = bool(self._camera and self._camera.is_opened)
            trigger = self._trigger
            status: dict[str, object] = {
                "state": self._state,
                "camera_connected": camera_connected,
                "stream_available": self._latest_frame is not None,
                "detector_state": self._detector_state,
                "deterrence_enabled": bool(trigger and trigger.enabled),
                "cooldown_remaining_seconds": (
                    trigger.cooldown_remaining_seconds if trigger else 0.0
                ),
                "last_error": self._last_error,
            }

        recent = self.database.list_recent_detections(1)
        if recent:
            detection = recent[0]
            status["last_detection"] = {
                "species": detection["species"],
                "confidence": detection["confidence"],
                "detected_at": detection["detected_at"],
                "event_id": detection["event_id"],
                "coordinates": [
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
                ],
            }
        else:
            status["last_detection"] = None
        return status

    def close(self) -> None:
        """Stop the worker and close its SQLite connection."""
        if not self.stop():
            raise RuntimeError("Cannot close dashboard while its worker is still running.")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.database.close()

    def _run(self, stop_event: threading.Event) -> None:
        camera: DashboardCamera | None = None
        trigger: DeterrentTrigger | None = None
        try:
            camera = self._camera_factory()
            detector = self._detector_factory()
            tracker = AnimalTracker()
            trigger = DeterrentTrigger(
                self._audio_player,
                enabled=config.AUTOMATIC_DETERRENCE_ENABLED,
            )
            with self._lock:
                self._camera = camera
                self._trigger = trigger
                self._detector_state = "ready"
                self._state = "running"

            run_preview(
                camera,
                display=False,
                detector=detector,
                analysis_every=config.DASHBOARD_ANALYSIS_EVERY,
                tracker=tracker,
                trigger=trigger,
                database=self.database,
                stop_event=stop_event,
                frame_callback=self._publish_frame,
            )
        except Exception as exc:
            logger.exception("Dashboard monitoring worker failed.")
            with self._lock:
                self._state = "error"
                self._detector_state = "error"
                self._last_error = str(exc)
        finally:
            if camera is not None:
                camera.release()
            self._audio_player.stop()
            with self._lock:
                self._camera = None
                self._trigger = None
                if self._state != "error":
                    self._state = "stopped"
                    self._detector_state = "stopped"
                logger.info("Dashboard monitoring worker stopped.")

    def _publish_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        minimum_interval = 1.0 / config.DASHBOARD_STREAM_FPS
        with self._lock:
            if now - self._last_encoded_at < minimum_interval:
                return
            self._last_encoded_at = now

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Could not encode the dashboard camera frame.")
        with self._lock:
            self._latest_frame = encoded.tobytes()


__all__ = ["DashboardController"]
