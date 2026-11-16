"""Adapter for Google's camera-trap SpeciesNet detector and classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import config
from detection.animal_detector import Detection

logger = logging.getLogger("detection.speciesnet")


@dataclass(frozen=True)
class _NormalizedBox:
    xmin: float
    ymin: float
    width: float
    height: float


class SpeciesNetAnimalDetector:
    """Detect animals, then classify the detected crops to SpeciesNet taxa."""

    def __init__(
        self,
        confidence_threshold: float | None = None,
        *,
        model_name: str | None = None,
        animal_detector: Any | None = None,
        species_classifier: Any | None = None,
    ) -> None:
        self.confidence_threshold = (
            config.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model_name = config.SPECIESNET_MODEL if model_name is None else model_name

        if animal_detector is None or species_classifier is None:
            try:
                from speciesnet.classifier import SpeciesNetClassifier
                from speciesnet.detector import SpeciesNetDetector
            except ImportError as exc:
                raise RuntimeError(
                    "SpeciesNet is not installed in this Python environment. "
                    "Create the separate environment using requirements-speciesnet.txt."
                ) from exc

            animal_detector = animal_detector or SpeciesNetDetector(self.model_name)
            species_classifier = species_classifier or SpeciesNetClassifier(self.model_name)

        self._animal_detector = animal_detector
        self._species_classifier = species_classifier

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return species labels and bounding boxes for animals in a BGR frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        try:
            from PIL import Image

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            detector_input = self._animal_detector.preprocess(image)
            detector_result = self._animal_detector.predict("camera-frame", detector_input)
            detector_detections = [
                detection
                for detection in detector_result.get("detections", [])
                if float(detection.get("conf", 0.0))
                >= config.SPECIESNET_DETECTION_THRESHOLD
            ]
            animal_boxes = [
                detection
                for detection in detector_detections
                if detection.get("label") == "animal"
            ]
            human_boxes = [
                detection
                for detection in detector_detections
                if detection.get("label") == "human"
            ]
            bboxes = [
                _NormalizedBox(
                    xmin=float(detection["bbox"][0]),
                    ymin=float(detection["bbox"][1]),
                    width=float(detection["bbox"][2]),
                    height=float(detection["bbox"][3]),
                )
                for detection in animal_boxes
            ]
            classifier_input = self._species_classifier.preprocess(image, bboxes=bboxes)
            classifier_result = (
                self._species_classifier.predict(
                    "camera-frame",
                    classifier_input,
                    num_classes=5,
                )
                if animal_boxes and classifier_input
                else {"classifications_list": []}
            )
        except Exception as exc:
            logger.exception("SpeciesNet inference failed")
            raise RuntimeError(f"SpeciesNet inference failed: {exc}") from exc

        classified_crops = classifier_result.get("classifications_list", [])
        detections: list[Detection] = []
        height, width = frame.shape[:2]
        for animal_box, classification in zip(animal_boxes, classified_crops):
            classes = classification.get("classes", [])
            scores = classification.get("scores", [])
            if not classes or not scores:
                continue

            class_name = self._taxon_name(str(classes[0]))
            confidence = float(scores[0])
            if confidence < self.confidence_threshold or class_name == "blank":
                class_name = "animal (uncertain)"

            x, y, box_width, box_height = map(float, animal_box["bbox"])
            x1 = max(0, min(width, round(x * width)))
            y1 = max(0, min(height, round(y * height)))
            x2 = max(x1, min(width, round((x + box_width) * width)))
            y2 = max(y1, min(height, round((y + box_height) * height)))
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    coordinates=(x1, y1, x2, y2),
                    detector_confidence=float(animal_box["conf"]),
                    taxonomy_path=str(classes[0]),
                )
            )
        for human_box in human_boxes:
            confidence = float(human_box["conf"])
            coordinates = self._pixel_box(human_box["bbox"], width, height)
            detections.append(
                Detection(
                    class_name="person",
                    confidence=confidence,
                    coordinates=coordinates,
                    detector_confidence=confidence,
                    taxonomy_path="human",
                )
            )
        return detections

    @staticmethod
    def _pixel_box(
        normalized_box: list[float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x, y, box_width, box_height = map(float, normalized_box)
        x1 = max(0, min(width, round(x * width)))
        y1 = max(0, min(height, round(y * height)))
        x2 = max(x1, min(width, round((x + box_width) * width)))
        y2 = max(y1, min(height, round((y + box_height) * height)))
        return x1, y1, x2, y2

    @staticmethod
    def _taxon_name(taxon: str) -> str:
        """Extract the display name from SpeciesNet's semicolon-delimited taxonomy."""
        parts = taxon.split(";")
        return parts[-1].strip() if parts else taxon

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw SpeciesNet predictions and both confidence scores on a copy."""
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.coordinates
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.detector_confidence is not None:
                label += f" / det {detection.detector_confidence:.2f}"
            if detection.track_id is not None:
                label += f" #{detection.track_id}"
                if detection.target_progress:
                    label += (
                        f" {detection.target_progress}/"
                        f"{config.CONSECUTIVE_CONFIRMATION_FRAMES}"
                    )
                if detection.target_confirmed:
                    label += " TARGET"
                if detection.track_state != "unknown":
                    label += f" {detection.track_state}"
            color = (
                (40, 170, 255)
                if detection.class_name == "person"
                else (60, 220, 100)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated


__all__ = ["SpeciesNetAnimalDetector"]
