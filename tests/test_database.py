from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from database import WildlifeDatabase
from detection.animal_detector import Detection
from detection.tracker import AnimalTracker
from deterrent.trigger import TriggerResult
from main import run_preview


def _detection(
    *,
    event_id: str = "event-1",
    class_name: str = "coyote",
    target_confirmed: bool = False,
    newly_confirmed: bool = False,
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=0.91,
        coordinates=(10, 20, 80, 90),
        detected_at=datetime(2026, 11, 13, 10, 0, tzinfo=timezone.utc),
        event_id=event_id,
        detector_confidence=0.95,
        track_id=7,
        target_progress=3 if target_confirmed else 1,
        target_confirmed=target_confirmed,
        newly_confirmed=newly_confirmed,
        track_state="approaching",
    )


def test_detection_history_and_event_summary_are_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "wildlife.sqlite3"
    with WildlifeDatabase(db_path) as database:
        first_id = database.record_detection(_detection())
        second = _detection(target_confirmed=True, newly_confirmed=True)
        second_id = database.record_detection(
            second,
            trigger_result=TriggerResult(
                event_id="event-1",
                species="coyote",
                triggered=False,
                reason="deterrence_disabled",
                cooldown_remaining_seconds=0,
            ),
        )

        assert first_id == 1
        assert second_id == 2
        history = database.list_recent_detections()
        assert len(history) == 2
        assert history[0]["event_id"] == "event-1"
        assert history[0]["x1"] == 10
        assert history[0]["x2"] == 80
        assert history[0]["target_confirmed"] == 1

        events = database.list_recent_events()
        assert len(events) == 1
        assert events[0]["species"] == "coyote"
        assert events[0]["first_seen_at"] == "2026-11-13T10:00:00+00:00"
        assert events[0]["trigger_reason"] == "deterrence_disabled"
        assert events[0]["audio_triggered"] == 0


def test_played_trigger_outcome_is_recorded(tmp_path: Path) -> None:
    with WildlifeDatabase(tmp_path / "wildlife.sqlite3") as database:
        database.record_detection(_detection(target_confirmed=True, newly_confirmed=True))
        database.record_trigger_result(
            TriggerResult(
                event_id="event-1",
                species="coyote",
                triggered=True,
                reason="played",
                cooldown_remaining_seconds=300,
            )
        )

        event = database.list_recent_events()[0]
        assert event["audio_triggered"] == 1
        assert event["trigger_reason"] == "played"
        assert event["cooldown_remaining_seconds"] == 300


def test_snapshot_saved_only_for_confirmed_nonhuman_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = tmp_path / "snapshots"
    monkeypatch.setattr(config, "SNAPSHOTS_DIR", snapshots)
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)
    with WildlifeDatabase(tmp_path / "wildlife.sqlite3") as database:
        snapshot = database.save_snapshot(
            _detection(target_confirmed=True, newly_confirmed=True),
            frame,
        )
        assert snapshot.exists()
        assert cv2.imread(str(snapshot)) is not None
        with pytest.raises(ValueError, match="confirmed, classified targets"):
            database.save_snapshot(
                _detection(class_name="person", target_confirmed=True),
                frame,
            )


def test_recent_query_limit_must_be_positive(tmp_path: Path) -> None:
    with WildlifeDatabase(tmp_path / "wildlife.sqlite3") as database:
        with pytest.raises(ValueError, match="greater than zero"):
            database.list_recent_events(0)
        with pytest.raises(ValueError, match="greater than zero"):
            database.list_recent_detections(-1)


def test_preview_persists_every_detection_and_target_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "SNAPSHOTS_DIR", tmp_path / "snapshots")
    frames = [np.full((64, 64, 3), 80, dtype=np.uint8) for _ in range(3)]

    class CameraStub:
        def open(self) -> bool:
            return True

        def read(self) -> np.ndarray | None:
            return frames.pop(0) if frames else None

        def release(self) -> None:
            pass

    class DetectorStub:
        def detect(self, frame: np.ndarray) -> list[Detection]:
            del frame
            return [Detection("coyote", 0.9, (8, 8, 48, 56))]

        def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
            del detections
            return frame.copy()

    class AudioStub:
        def __init__(self) -> None:
            self.play_count = 0

        def play(self, *_args: object, **_kwargs: object) -> None:
            self.play_count += 1

        def stop(self) -> None:
            pass

    audio = AudioStub()
    from deterrent.trigger import DeterrentTrigger

    trigger = DeterrentTrigger(
        audio,
        enabled=False,
        target_species={"coyote"},
        default_sound_path=tmp_path / "unused.wav",
    )
    with WildlifeDatabase(tmp_path / "wildlife.sqlite3") as database:
        assert run_preview(
            CameraStub(),
            display=False,
            max_frames=3,
            detector=DetectorStub(),
            tracker=AnimalTracker(
                target_species={"coyote"},
                confidence_threshold=0.65,
                confirmation_frames=3,
            ),
            trigger=trigger,
            database=database,
        ) == 3

        history = database.list_recent_detections()
        event = database.list_recent_events()[0]

    assert len(history) == 3
    assert event["target_confirmed"] == 1
    assert event["trigger_reason"] == "deterrence_disabled"
    assert event["snapshot_path"] is not None
    assert Path(str(event["snapshot_path"])).exists()
    assert audio.play_count == 0
