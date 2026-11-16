from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import config
from dashboard.app import create_app
from dashboard.controller import DashboardController
from database import WildlifeDatabase
from detection.animal_detector import Detection


class FakeAudioPlayer:
    def __init__(self) -> None:
        self.play_count = 0
        self.stop_count = 0
        self.is_playing = False

    def play(self, sound_path: str | Path, *, volume: float | None = None) -> None:
        del sound_path, volume
        self.play_count += 1
        self.is_playing = True

    def stop(self) -> None:
        self.stop_count += 1
        self.is_playing = False


class FakeCamera:
    def __init__(self) -> None:
        self.released = False
        self.frames = [np.full((24, 32, 3), 80, dtype=np.uint8)]

    def open(self) -> bool:
        return True

    def read(self) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None

    def release(self) -> None:
        self.released = True

    @property
    def is_opened(self) -> bool:
        return not self.released


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return []

    def annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        del detections
        return frame


class FakeDatabase:
    def __init__(self, events: list[dict[str, object]] | None = None) -> None:
        self._events = events or []

    def list_recent_events(self, limit: int = 100) -> list[dict[str, object]]:
        return self._events[:limit]


class FakeController:
    def __init__(self, events: list[dict[str, object]] | None = None) -> None:
        self.database = FakeDatabase(events)
        self.latest_frame: bytes | None = None

    def status(self) -> dict[str, object]:
        return {
            "state": "stopped",
            "camera_connected": False,
            "stream_available": False,
            "detector_state": "stopped",
            "deterrence_enabled": False,
            "cooldown_remaining_seconds": 0.0,
            "last_error": None,
            "last_detection": None,
        }

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def emergency_stop(self) -> None:
        pass

    def set_deterrence_enabled(self, enabled: bool) -> None:
        del enabled

    def test_sound(self) -> None:
        pass


def test_dashboard_status_history_and_snapshot_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    image = snapshots / "animal.jpg"
    image.write_bytes(b"jpeg fixture")
    monkeypatch.setattr(config, "SNAPSHOTS_DIR", snapshots)
    controller = FakeController(
        [
            {
                "event_id": "event-1",
                "species": "fox",
                "confidence": 0.91,
                "last_seen_at": "2026-11-16T10:00:00+00:00",
                "target_confirmed": 1,
                "audio_triggered": 0,
                "trigger_reason": "deterrence_disabled",
                "snapshot_path": str(image),
            }
        ]
    )
    client = create_app(controller).test_client()

    assert client.get("/").status_code == 200
    status = client.get("/api/status")
    assert status.status_code == 200
    status_data = status.json
    assert status_data is not None
    assert status_data["status"]["state"] == "stopped"

    events = client.get("/api/events")
    assert events.status_code == 200
    events_data = events.json
    assert events_data is not None
    snapshot_url = events_data["events"][0]["snapshot_url"]
    assert client.get(snapshot_url).data == b"jpeg fixture"
    assert client.get("/api/events?limit=0").status_code == 400
    assert client.get("/snapshots/../secret.jpg").status_code == 404


def test_remote_dashboard_requires_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WILDLIFE_DASHBOARD_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="WILDLIFE_DASHBOARD_PASSWORD"):
        create_app(FakeController(), host="0.0.0.0")

    monkeypatch.setenv("WILDLIFE_DASHBOARD_PASSWORD", "test-password")
    client = create_app(FakeController(), host="0.0.0.0").test_client()
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("operator", "test-password")).status_code == 200


def test_dashboard_worker_stream_control_and_emergency_stop(
    tmp_path: Path,
) -> None:
    camera = FakeCamera()
    audio = FakeAudioPlayer()
    database = WildlifeDatabase(tmp_path / "wildlife.sqlite3")
    controller = DashboardController(
        database,
        camera_factory=lambda: camera,
        detector_factory=FakeDetector,
        audio_player=audio,
    )
    client = create_app(controller).test_client()

    try:
        assert client.post("/api/system/start").status_code == 202
        deadline = time.monotonic() + 2.0
        while controller.latest_frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert controller.latest_frame is not None
        assert controller.latest_frame.startswith(b"\xff\xd8")
        assert controller.status()["camera_connected"] is True
        assert controller.status()["stream_available"] is True

        enabled = client.post("/api/deterrence", json={"enabled": True})
        assert enabled.status_code == 200
        enabled_data = enabled.json
        assert enabled_data is not None
        assert enabled_data["status"]["deterrence_enabled"] is True

        emergency = client.post("/api/emergency-stop")
        assert emergency.status_code == 200
        emergency_data = emergency.json
        assert emergency_data is not None
        assert emergency_data["status"]["deterrence_enabled"] is False
        assert audio.stop_count > 0

        stopped = client.post("/api/system/stop")
        assert stopped.status_code == 200
        stopped_data = stopped.json
        assert stopped_data is not None
        assert stopped_data["stopped"] is True
        assert camera.released
        assert client.post("/api/deterrence", json={"enabled": True}).status_code == 409
    finally:
        controller.close()


def test_sound_test_is_only_available_while_monitoring_is_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "DEFAULT_SOUND_PATH", tmp_path / "warning.wav")
    audio = FakeAudioPlayer()
    controller = DashboardController(
        WildlifeDatabase(tmp_path / "wildlife.sqlite3"),
        camera_factory=FakeCamera,
        detector_factory=FakeDetector,
        audio_player=audio,
    )
    client = create_app(controller).test_client()

    try:
        assert client.post("/api/test-sound").status_code == 200
        assert audio.play_count == 1
        assert client.post("/api/system/start").status_code == 202
        assert client.post("/api/test-sound").status_code == 409
        assert audio.play_count == 1
        controller.stop()
    finally:
        controller.close()
