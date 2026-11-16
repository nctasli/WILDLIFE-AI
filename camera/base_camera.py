"""Shared OpenCV capture lifecycle and reconnection behavior."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TypeAlias

import cv2
import numpy as np

import config

Frame: TypeAlias = np.ndarray
logger = logging.getLogger("camera.connection")


class BaseCamera(ABC):
    """Base class for OpenCV-backed camera sources."""

    reconnect_on_failure = True

    def __init__(self, reconnect_interval_seconds: float | None = None) -> None:
        interval = (
            config.CAMERA_RECONNECT_SECONDS
            if reconnect_interval_seconds is None
            else reconnect_interval_seconds
        )
        if interval <= 0:
            raise ValueError("reconnect_interval_seconds must be greater than zero")

        self._reconnect_interval = interval
        self._capture: cv2.VideoCapture | None = None
        self._last_open_attempt = float("-inf")
        self._ended = False

    @property
    @abstractmethod
    def source(self) -> int | str:
        """Return the value passed to ``cv2.VideoCapture``."""

    def open(self) -> bool:
        """Open the source, logging failures instead of crashing the app."""
        self._release_capture()
        self._ended = False
        self._last_open_attempt = time.monotonic()

        try:
            capture = cv2.VideoCapture(self.source)
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                logger.warning("Camera source could not be opened: %s", self.source)
                return False

            self._capture = capture
            self._configure_capture(capture)
            logger.info("Camera connected: %s", self.source)
            return True
        except cv2.error:
            logger.exception("OpenCV failed to open camera source: %s", self.source)
            self._release_capture()
            return False

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        """Allow source-specific capture configuration."""

    def read(self) -> Frame | None:
        """Read a frame; return ``None`` while disconnected or at end-of-file."""
        if self._ended:
            return None

        if self._capture is None or not self._capture.isOpened():
            self._release_capture()
            if not self.reconnect_on_failure or not self._reconnect_due():
                return None
            if not self.open():
                return None

        try:
            success, frame = self._capture.read()
        except cv2.error:
            logger.exception("OpenCV failed while reading camera source: %s", self.source)
            self._release_capture()
            return None

        if success and frame is not None:
            return frame

        if self.reconnect_on_failure:
            logger.warning("Camera read failed; will retry connection: %s", self.source)
        else:
            logger.info("Video source reached its end or could not provide a frame: %s", self.source)
            self._ended = True
        self._release_capture()
        return None

    def _reconnect_due(self) -> bool:
        now = time.monotonic()
        if now - self._last_open_attempt < self._reconnect_interval:
            return False
        return True

    @property
    def is_opened(self) -> bool:
        """Whether the underlying capture currently reports an open connection."""
        return self._capture is not None and self._capture.isOpened()

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def release(self) -> None:
        """Release camera resources."""
        self._release_capture()

    def __enter__(self) -> BaseCamera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
