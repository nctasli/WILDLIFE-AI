"""Local video file source."""

from __future__ import annotations

from pathlib import Path

import config
from camera.base_camera import BaseCamera


class VideoFileCamera(BaseCamera):
    """Read a local video file; end-of-file is treated as normal completion."""

    reconnect_on_failure = False

    def __init__(
        self,
        video_path: str | Path | None = None,
        reconnect_interval_seconds: float | None = None,
    ) -> None:
        super().__init__(reconnect_interval_seconds)
        self.video_path = Path(config.VIDEO_PATH if video_path is None else video_path)

    @property
    def source(self) -> str:
        return str(self.video_path)
