"""Shared interface for image-frame detectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from detection.animal_detector import Detection


class FrameDetector(Protocol):
    """Detector interface consumed by the camera preview."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray: ...
