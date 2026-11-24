"""YOLO-backed object detection and frame annotation."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

logger = logging.getLogger("detection.model")
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One detected object in a frame."""

    class_name: str
    confidence: float
    coordinates: Box
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detector_confidence: float | None = None
    taxonomy_path: str | None = None
    track_id: int | None = None
    target_progress: int = 0
    target_confirmed: bool = False
    newly_confirmed: bool = False
    track_state: str = "untracked"


class AnimalDetector:
    """Run an Ultralytics YOLO model and expose its actual class names."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(config.MODEL_PATH if model_path is None else model_path)
        self.confidence_threshold = (
            config.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self._model = model if model is not None else self._load_model(self.model_path)
        self._class_names = self._normalize_names(self._model.names)
        logger.info(
            "YOLO model loaded from %s with %d class(es)",
            self.model_path,
            len(self._class_names),
        )
        if self.unsupported_targets:
            logger.warning(
                "Configured target classes are not supported by this model: %s",
                ", ".join(self.unsupported_targets),
            )

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                f"YOLO dependency is missing or failed to import ({exc}). "
                "Ensure your virtual environment is active (source .venv/bin/activate) "
                "and install dependencies with 'pip install -r requirements-rpi.txt'."
            ) from exc

        try:
            if not model_path.exists() and model_path.name == "yolo11n.pt":
                model = YOLO("yolo11n.pt")
            else:
                model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load YOLO model from {model_path}: {exc}") from exc
        return model

    @staticmethod
    def _normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    @property
    def supported_classes(self) -> tuple[str, ...]:
        """Return names supplied by the loaded model; no unsupported classes are implied."""
        return tuple(self._class_names.values())

    @property
    def unsupported_targets(self) -> tuple[str, ...]:
        """Return configured target names that this model cannot explicitly output."""
        supported = {name.casefold() for name in self.supported_classes}
        return tuple(sorted(target for target in config.TARGET_ANIMALS if target.casefold() not in supported))

    def supports_class(self, class_name: str) -> bool:
        """Whether the loaded model has this exact class name, case-insensitively."""
        return class_name.casefold() in {name.casefold() for name in self.supported_classes}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR OpenCV frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO inference failed")
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        detections: list[Detection] = []
        for result in results:
            names = self._normalize_names(result.names)
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(self._to_scalar(box.cls))
                class_name = names.get(class_id, self._class_names.get(class_id))
                if class_name is None:
                    logger.warning("Model returned unknown class id %d; skipping detection", class_id)
                    continue
                confidence = self._to_scalar(box.conf)
                if confidence < self.confidence_threshold:
                    continue
                coordinates = self._to_coordinates(box.xyxy)
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        coordinates=coordinates,
                    )
                )
        return detections

    @staticmethod
    def _to_scalar(value: Any) -> int | float:
        if hasattr(value, "item"):
            value = value.item()
        return int(value) if isinstance(value, (int, np.integer)) else float(value)

    @staticmethod
    def _to_coordinates(value: Any) -> Box:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        coordinates = np.asarray(value).reshape(-1)
        if coordinates.size != 4:
            raise ValueError("YOLO bounding box must contain four coordinates")
        return (
            int(round(float(coordinates[0]))),
            int(round(float(coordinates[1]))),
            int(round(float(coordinates[2]))),
            int(round(float(coordinates[3]))),
        )

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw labeled bounding boxes on a copy of the input frame."""
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.coordinates
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.track_id is not None:
                label += f" #{detection.track_id}"
                if detection.target_progress:
                    label += f" {detection.target_progress}/{config.CONSECUTIVE_CONFIRMATION_FRAMES}"
                if detection.target_confirmed:
                    label += " TARGET"
                if detection.track_state != "unknown":
                    label += f" {detection.track_state}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 100), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 220, 100),
                2,
                cv2.LINE_AA,
            )
        return annotated
