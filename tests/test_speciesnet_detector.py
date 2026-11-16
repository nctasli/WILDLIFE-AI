from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from detection.speciesnet_detector import SpeciesNetAnimalDetector


class FakeSpeciesDetector:
    def preprocess(self, image: Any) -> Any:
        return image

    def predict(self, filepath: str, image: Any) -> dict[str, Any]:
        assert filepath == "camera-frame"
        return {
            "detections": [
                {"label": "animal", "conf": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]},
                {"label": "animal", "conf": 0.1, "bbox": [0.5, 0.5, 0.1, 0.1]},
                {"label": "human", "conf": 0.99, "bbox": [0.0, 0.0, 0.1, 0.1]},
            ]
        }


class FakeSpeciesClassifier:
    def preprocess(self, image: Any, *, bboxes: list[Any]) -> list[Any]:
        self.bboxes = bboxes
        return bboxes

    def predict(
        self, filepath: str, images: list[Any], *, num_classes: int
    ) -> dict[str, Any]:
        assert filepath == "camera-frame"
        assert num_classes == 5
        return {
            "classifications_list": [
                {
                    "classes": [
                        "taxon-id;mammalia;carnivora;canidae;canis;latrans;coyote"
                    ],
                    "scores": [0.91],
                }
            ]
        }


def test_speciesnet_converts_taxonomy_and_normalized_box() -> None:
    detector = SpeciesNetAnimalDetector(
        confidence_threshold=0.65,
        animal_detector=FakeSpeciesDetector(),
        species_classifier=FakeSpeciesClassifier(),
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 2
    coyote, person = detections
    assert coyote.class_name == "coyote"
    assert coyote.confidence == pytest.approx(0.91)
    assert coyote.coordinates == (20, 20, 80, 60)
    assert coyote.detector_confidence == pytest.approx(0.95)
    assert coyote.taxonomy_path.endswith(";coyote")
    assert person.class_name == "person"
    assert person.confidence == pytest.approx(0.99)
    assert person.coordinates == (0, 0, 20, 10)
    assert person.taxonomy_path == "human"


def test_person_detection_does_not_require_species_classification() -> None:
    class PersonOnlyDetector(FakeSpeciesDetector):
        def predict(self, filepath: str, image: Any) -> dict[str, Any]:
            return {
                "detections": [
                    {"label": "human", "conf": 0.8, "bbox": [0.2, 0.3, 0.4, 0.5]}
                ]
            }

    class EmptyClassifier:
        def preprocess(self, image: Any, *, bboxes: list[Any]) -> list[Any]:
            assert bboxes == []
            return []

        def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            pytest.fail("Species classifier should not run when only a human is detected")

    detector = SpeciesNetAnimalDetector(
        animal_detector=PersonOnlyDetector(),
        species_classifier=EmptyClassifier(),
    )
    results = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(results) == 1
    assert results[0].class_name == "person"
    assert results[0].coordinates == (40, 30, 120, 80)


def test_low_confidence_species_is_labeled_uncertain() -> None:
    class LowConfidenceClassifier(FakeSpeciesClassifier):
        def predict(
            self, filepath: str, images: list[Any], *, num_classes: int
        ) -> dict[str, Any]:
            return {
                "classifications_list": [
                    {"classes": ["taxon;mammalia;carnivora;wolf"], "scores": [0.3]}
                ]
            }

    detector = SpeciesNetAnimalDetector(
        confidence_threshold=0.65,
        animal_detector=FakeSpeciesDetector(),
        species_classifier=LowConfidenceClassifier(),
    )
    result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert result[0].class_name == "animal (uncertain)"


def test_speciesnet_annotation_does_not_mutate_input() -> None:
    detector = SpeciesNetAnimalDetector(
        animal_detector=FakeSpeciesDetector(),
        species_classifier=FakeSpeciesClassifier(),
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = detector.detect(frame)

    annotated = detector.annotate(frame, result)

    assert annotated is not frame
    assert np.any(annotated != frame)
    assert not np.any(frame)
