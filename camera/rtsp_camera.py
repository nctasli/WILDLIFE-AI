"""RTSP / IP network camera stream source (e.g. TP-Link Tapo, Hikvision, Dahua)."""

from __future__ import annotations

import config
from camera.base_camera import BaseCamera


class RTSPCamera(BaseCamera):
    """Read an RTSP / IP camera stream (e.g. Tapo rtsp://user:pass@ip:554/stream2)."""

    def __init__(
        self,
        stream_url: str | None = None,
        reconnect_interval_seconds: float | None = None,
    ) -> None:
        super().__init__(reconnect_interval_seconds)
        self.stream_url = config.RTSP_STREAM_URL if stream_url is None else stream_url
        if not self.stream_url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise ValueError(
                "RTSP stream URL must use rtsp://, rtsps://, http://, or https://"
            )

    @property
    def source(self) -> str:
        return self.stream_url


__all__ = ["RTSPCamera"]
