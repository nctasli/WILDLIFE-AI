"""Factory for configured camera sources."""

from __future__ import annotations

from pathlib import Path

import config
from camera.base_camera import BaseCamera
from camera.esp32_camera import ESP32Camera
from camera.video_source import VideoFileCamera
from camera.webcam import WebcamCamera


def create_camera(
    source: str | None = None,
    *,
    video_path: str | Path | None = None,
    webcam_index: int | None = None,
    esp32_stream_url: str | None = None,
) -> BaseCamera:
    """Create the selected source, defaulting to ``config.CAMERA_SOURCE``."""
    selected_source = config.CAMERA_SOURCE if source is None else source
    if selected_source == "webcam":
        return WebcamCamera(device_index=webcam_index)
    if selected_source == "video":
        return VideoFileCamera(video_path=video_path)
    if selected_source == "esp32":
        return ESP32Camera(stream_url=esp32_stream_url)
    raise ValueError(
        f"Unsupported camera source {selected_source!r}; "
        "choose 'webcam', 'video', or 'esp32'."
    )
