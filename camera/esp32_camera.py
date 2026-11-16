"""ESP32-CAM HTTP/MJPEG stream source."""

from __future__ import annotations

import config
from camera.base_camera import BaseCamera


class ESP32Camera(BaseCamera):
    """Read an ESP32-CAM stream using its HTTP/MJPEG URL."""

    def __init__(
        self,
        stream_url: str | None = None,
        reconnect_interval_seconds: float | None = None,
    ) -> None:
        super().__init__(reconnect_interval_seconds)
        self.stream_url = config.ESP32_STREAM_URL if stream_url is None else stream_url
        if not self.stream_url.startswith(("http://", "https://")):
            raise ValueError("ESP32 stream URL must use HTTP or HTTPS")

    @property
    def source(self) -> str:
        return self.stream_url
