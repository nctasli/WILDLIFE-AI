from __future__ import annotations

import threading

import numpy as np
import pytest

from camera import ESP32Camera, RTSPCamera, VideoFileCamera, WebcamCamera, create_camera
from camera import base_camera
from detection.animal_detector import Detection
from main import run_preview


class FakeCapture:
    def __init__(self, source: int | str, outcomes: list[tuple[bool, np.ndarray | None]]) -> None:
        self.source = source
        self.outcomes = outcomes
        self.released = False
        self.opened = True
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.outcomes:
            return self.outcomes.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True

    def set(self, property_id: int, value: float) -> bool:
        self.properties[property_id] = value
        return True


def test_factory_creates_configured_source_types() -> None:
    assert isinstance(create_camera("webcam", webcam_index=2), WebcamCamera)
    assert isinstance(create_camera("video", video_path="clip.mp4"), VideoFileCamera)
    assert isinstance(create_camera("esp32", esp32_stream_url="http://camera/stream"), ESP32Camera)
    assert isinstance(create_camera("rtsp", rtsp_stream_url="rtsp://camera/stream2"), RTSPCamera)
    assert isinstance(create_camera("tapo", rtsp_stream_url="rtsp://camera/stream2"), RTSPCamera)


def test_factory_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unsupported camera source"):
        create_camera("unknown")


def test_esp32_rejects_non_http_stream_url() -> None:
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        ESP32Camera("rtsp://camera/stream")


def test_video_file_read_and_end_of_file(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FakeCapture("clip.mp4", [(True, frame), (False, None)])
    monkeypatch.setattr(base_camera.cv2, "VideoCapture", lambda _source: capture)
    camera = VideoFileCamera("clip.mp4")

    assert camera.open()
    assert camera.read() is frame
    assert camera.read() is None
    assert camera.read() is None
    assert capture.released


def test_webcam_retries_after_reconnect_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    captures = [
        FakeCapture(0, [(False, None)]),
        FakeCapture(0, [(True, frame)]),
    ]
    monkeypatch.setattr(base_camera.cv2, "VideoCapture", lambda _source: captures.pop(0))
    now = [10.0]
    monkeypatch.setattr(base_camera.time, "monotonic", lambda: now[0])
    camera = WebcamCamera(reconnect_interval_seconds=2.0)

    assert camera.open()
    assert camera.read() is None
    assert camera.read() is None
    now[0] += 2.0
    assert camera.read() is frame


def test_failed_initial_connection_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    captures = [
        FakeCapture(0, []),
        FakeCapture(0, [(True, frame)]),
    ]
    captures[0].opened = False
    monkeypatch.setattr(base_camera.cv2, "VideoCapture", lambda _source: captures.pop(0))
    now = [10.0]
    monkeypatch.setattr(base_camera.time, "monotonic", lambda: now[0])
    camera = WebcamCamera(reconnect_interval_seconds=1.0)

    assert not camera.open()
    assert camera.read() is None
    now[0] += 1.0
    assert camera.read() is frame


def test_headless_preview_reads_until_frame_limit() -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class CameraStub:
        def __init__(self) -> None:
            self.released = False

        def open(self) -> bool:
            return True

        def read(self) -> np.ndarray | None:
            return frames.pop(0) if frames else None

        def release(self) -> None:
            self.released = True

    camera = CameraStub()
    assert run_preview(camera, display=False, max_frames=2) == 2
    assert camera.released


def test_preview_can_analyze_every_nth_frame() -> None:
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(5)]

    class CameraStub:
        def open(self) -> bool:
            return True

        def read(self) -> np.ndarray | None:
            return frames.pop(0) if frames else None

        def release(self) -> None:
            pass

    class DetectorStub:
        def __init__(self) -> None:
            self.analyzed: list[int] = []

        def detect(self, frame: np.ndarray) -> list[Detection]:
            self.analyzed.append(int(frame[0, 0, 0]))
            return []

        def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
            return frame

    detector = DetectorStub()
    assert run_preview(
        CameraStub(),
        display=False,
        max_frames=5,
        detector=detector,
        analysis_every=2,
    ) == 5
    assert detector.analyzed == [0, 2, 4]


def test_preview_stops_on_event_and_publishes_processed_frame() -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    class CameraStub:
        def __init__(self) -> None:
            self.released = False

        def open(self) -> bool:
            return True

        def read(self) -> np.ndarray | None:
            return frames.pop(0) if frames else None

        def release(self) -> None:
            self.released = True

    stop_event = threading.Event()
    published: list[np.ndarray] = []

    def publish(frame: np.ndarray) -> None:
        published.append(frame)
        stop_event.set()

    camera = CameraStub()
    assert run_preview(
        camera,
        display=False,
        stop_event=stop_event,
        frame_callback=publish,
    ) == 1
    assert camera.released
    assert len(published) == 1
