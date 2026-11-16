from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import config
from detection import AnimalDetector, Detection


class FakeModel:
    names = {0: "person", 15: "cat"}

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[object]:
        self.kwargs = kwargs
        boxes = [
            SimpleNamespace(
                cls=np.array(15),
                conf=np.array(0.91),
                xyxy=np.array([[10.2, 20.4, 80.6, 90.8]]),
            ),
            SimpleNamespace(
                cls=np.array(0),
                conf=np.array(0.40),
                xyxy=np.array([[1, 2, 3, 4]]),
            ),
        ]
        return [SimpleNamespace(names=self.names, boxes=boxes)]


def test_detection_uses_model_classes_and_confidence_threshold() -> None:
    model = FakeModel()
    detector = AnimalDetector(confidence_threshold=0.65, model=model)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    result = detections[0]
    assert result.class_name == "cat"
    assert result.confidence == pytest.approx(0.91)
    assert result.coordinates == (10, 20, 81, 91)
    assert result.event_id
    assert result.detected_at.tzinfo is not None
    assert model.kwargs["conf"] == 0.65
    assert detector.supports_class("CAT")
    assert not detector.supports_class("fox")


def test_configured_targets_absent_from_model_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TARGET_ANIMALS", frozenset({"fox", "cat"}))
    detector = AnimalDetector(model=FakeModel())
    assert detector.unsupported_targets == ("fox",)


def test_default_priority_species_are_explicit() -> None:
    assert config.TARGET_ANIMALS == frozenset(
        {"wolf", "fox", "coyote", "wild boar"}
    )


def test_invalid_frame_is_rejected() -> None:
    detector = AnimalDetector(model=FakeModel())
    with pytest.raises(ValueError, match="non-empty NumPy array"):
        detector.detect(np.empty((0, 0, 3), dtype=np.uint8))


def test_detection_annotation_draws_on_copy() -> None:
    detector = AnimalDetector(model=FakeModel())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection("cat", 0.9, (10, 20, 80, 90))

    annotated = detector.annotate(frame, [detection])

    assert annotated is not frame
    assert np.any(annotated != frame)
    assert not np.any(frame)
