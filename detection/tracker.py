"""Short-lived object tracking and consecutive target validation."""

from __future__ import annotations

import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass, field, replace

import config
from detection.animal_detector import Detection

logger = logging.getLogger("detection.tracker")


@dataclass
class _Track:
    track_id: int
    event_id: str
    class_name: str
    coordinates: tuple[int, int, int, int]
    target_species: str | None
    confidence_streak: int = 0
    missed_frames: int = 0
    target_confirmed: bool = False
    areas: deque[float] = field(default_factory=deque)


class AnimalTracker:
    """Associate detections across analyzed frames and validate configured targets."""

    def __init__(
        self,
        *,
        target_species: frozenset[str] | set[str] | None = None,
        confidence_threshold: float | None = None,
        confirmation_frames: int | None = None,
        match_iou_threshold: float | None = None,
        max_missed_frames: int | None = None,
    ) -> None:
        self.target_species = (
            frozenset(config.TARGET_ANIMALS)
            if target_species is None
            else frozenset(target_species)
        )
        self.confidence_threshold = (
            config.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.confirmation_frames = (
            config.CONSECUTIVE_CONFIRMATION_FRAMES
            if confirmation_frames is None
            else confirmation_frames
        )
        self.match_iou_threshold = (
            config.TRACK_MATCH_IOU_THRESHOLD
            if match_iou_threshold is None
            else match_iou_threshold
        )
        self.max_missed_frames = (
            config.TRACK_MAX_MISSED_ANALYSIS_FRAMES
            if max_missed_frames is None
            else max_missed_frames
        )
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.confirmation_frames <= 0:
            raise ValueError("confirmation_frames must be greater than zero")
        if not 0.0 <= self.match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must be between 0 and 1")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Track the current analyzed frame and return detections with track metadata."""
        tracks = list(self._tracks.values())
        matches = self._match(tracks, detections)
        matched_tracks = {track_id for track_id, _ in matches.values()}

        for track in tracks:
            if track.track_id not in matched_tracks:
                track.missed_frames += 1
                track.confidence_streak = 0
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track.track_id]

        enriched: list[Detection] = []
        for detection_index, detection in enumerate(detections):
            target_species = self._target_species_for(detection)
            previous_confirmed = False

            if detection_index in matches:
                track = self._tracks[matches[detection_index][0]]
                previous_confirmed = track.target_confirmed
                previous_target_species = track.target_species
                if target_species is not None and track.target_species == target_species:
                    track.confidence_streak += 1
                else:
                    track.confidence_streak = 1 if target_species is not None else 0
                    track.target_confirmed = False
                track.target_species = target_species
                track.class_name = detection.class_name
                track.coordinates = detection.coordinates
                track.missed_frames = 0
            else:
                track = self._create_track(detection, target_species)

            self._append_area(track, detection.coordinates)
            newly_confirmed = (
                target_species is not None
                and track.confidence_streak >= self.confirmation_frames
                and not track.target_confirmed
            )
            if newly_confirmed:
                track.target_confirmed = True
                logger.info(
                    "Target confirmed: %s, track %d, event %s",
                    target_species,
                    track.track_id,
                    track.event_id,
                )
            elif (
                previous_confirmed
                and target_species is not None
                and target_species == previous_target_species
            ):
                track.target_confirmed = True

            enriched.append(
                replace(
                    detection,
                    event_id=track.event_id,
                    track_id=track.track_id,
                    target_progress=(
                        self.confirmation_frames
                        if track.target_confirmed
                        else min(track.confidence_streak, self.confirmation_frames)
                        if target_species is not None
                        else 0
                    ),
                    target_confirmed=track.target_confirmed,
                    newly_confirmed=newly_confirmed,
                    track_state=self._movement_state(track),
                )
            )
        return enriched

    def _create_track(self, detection: Detection, target_species: str | None) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            event_id=str(uuid.uuid4()),
            class_name=detection.class_name,
            coordinates=detection.coordinates,
            target_species=target_species,
            confidence_streak=1 if target_species is not None else 0,
            areas=deque(maxlen=max(2, config.TRACK_AREA_TREND_WINDOW)),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _target_species_for(self, detection: Detection) -> str | None:
        if detection.confidence < self.confidence_threshold:
            return None
        label = self._normalize_name(detection.class_name)
        if label == "animal uncertain" or label == "blank":
            return None

        for configured_target in self.target_species:
            target = self._normalize_name(configured_target)
            aliases = {
                self._normalize_name(alias)
                for alias in config.TARGET_SPECIES_ALIASES.get(configured_target, frozenset())
            }
            if label == target or label in aliases:
                return target
        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.casefold().replace("_", " ")).strip()

    def _match(
        self,
        tracks: list[_Track],
        detections: list[Detection],
    ) -> dict[int, tuple[int, float]]:
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            detection_class = self._normalize_name(detection.class_name)
            for track in tracks:
                if track.missed_frames > self.max_missed_frames:
                    continue
                if self._normalize_name(track.class_name) != detection_class:
                    continue
                overlap = self._iou(track.coordinates, detection.coordinates)
                if overlap >= self.match_iou_threshold:
                    candidates.append((overlap, detection_index, track.track_id))

        matches: dict[int, tuple[int, float]] = {}
        used_tracks: set[int] = set()
        for overlap, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in matches or track_id in used_tracks:
                continue
            matches[detection_index] = (track_id, overlap)
            used_tracks.add(track_id)
        return matches

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _append_area(track: _Track, coordinates: tuple[int, int, int, int]) -> None:
        width = max(0, coordinates[2] - coordinates[0])
        height = max(0, coordinates[3] - coordinates[1])
        track.areas.append(float(width * height))

    @staticmethod
    def _movement_state(track: _Track) -> str:
        window = max(2, config.TRACK_AREA_TREND_WINDOW)
        if len(track.areas) < window:
            return "unknown"
        initial = track.areas[0]
        if initial <= 0:
            return "unknown"
        ratio = track.areas[-1] / initial
        if ratio >= config.TRACK_APPROACHING_AREA_RATIO:
            return "approaching"
        if ratio <= config.TRACK_MOVING_AWAY_AREA_RATIO:
            return "moving_away"
        return "stable"


__all__ = ["AnimalTracker"]
