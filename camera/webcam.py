"""Local webcam source."""

from __future__ import annotations

import cv2

import config
from camera.base_camera import BaseCamera


class WebcamCamera(BaseCamera):
    """Capture frames from a local webcam by device index."""

    def __init__(
        self,
        device_index: int | None = None,
        reconnect_interval_seconds: float | None = None,
    ) -> None:
        super().__init__(reconnect_interval_seconds)
        self.device_index = config.WEBCAM_INDEX if device_index is None else device_index

    @property
    def source(self) -> int:
        return self.device_index

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
