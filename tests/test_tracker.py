from __future__ import annotations

from detection.animal_detector import Detection
from detection.tracker import AnimalTracker


def _detection(
    class_name: str,
    confidence: float = 0.9,
    box: tuple[int, int, int, int] = (10, 10, 50, 50),
) -> Detection:
    return Detection(class_name=class_name, confidence=confidence, coordinates=box)


def test_target_requires_consecutive_analyzed_frames_and_confirms_once() -> None:
    tracker = AnimalTracker(
        target_species={"coyote"},
        confidence_threshold=0.65,
        confirmation_frames=3,
        match_iou_threshold=0.2,
    )

    first, second, third, fourth = (
        tracker.update([_detection("coyote")])[0],
        tracker.update([_detection("coyote", box=(11, 10, 51, 50))])[0],
        tracker.update([_detection("coyote", box=(12, 10, 52, 50))])[0],
        tracker.update([_detection("coyote", box=(13, 10, 53, 50))])[0],
    )

    assert [first.target_progress, second.target_progress] == [1, 2]
    assert not first.target_confirmed and not second.target_confirmed
    assert third.target_confirmed and third.newly_confirmed
    assert fourth.target_confirmed and not fourth.newly_confirmed
    assert first.event_id == second.event_id == third.event_id == fourth.event_id
    assert first.track_id == fourth.track_id


def test_low_confidence_and_non_target_do_not_advance_confirmation() -> None:
    tracker = AnimalTracker(
        target_species={"coyote"},
        confidence_threshold=0.65,
        confirmation_frames=2,
    )

    low_confidence = tracker.update([_detection("coyote", confidence=0.64)])[0]
    non_target = tracker.update([_detection("deer")])[0]
    first_target = tracker.update([_detection("coyote")])[0]

    assert not low_confidence.target_confirmed
    assert low_confidence.target_progress == 0
    assert not non_target.target_confirmed
    assert non_target.target_progress == 0
    assert first_target.target_progress == 1
    assert not first_target.target_confirmed


def test_missed_analysis_frame_breaks_consecutive_streak() -> None:
    tracker = AnimalTracker(
        target_species={"dog"},
        confidence_threshold=0.65,
        confirmation_frames=3,
        max_missed_frames=1,
    )

    first = tracker.update([_detection("dog")])[0]
    tracker.update([])
    after_gap = tracker.update([_detection("dog", box=(11, 10, 51, 50))])[0]

    assert first.target_progress == 1
    assert after_gap.target_progress == 1
    assert not after_gap.target_confirmed
    assert after_gap.track_id == first.track_id


def test_confirmed_event_remains_confirmed_after_brief_missed_frame() -> None:
    tracker = AnimalTracker(
        target_species={"dog"},
        confidence_threshold=0.65,
        confirmation_frames=2,
        max_missed_frames=1,
    )
    tracker.update([_detection("dog")])
    confirmed = tracker.update([_detection("dog", box=(11, 10, 51, 50))])[0]
    tracker.update([])
    seen_again = tracker.update([_detection("dog", box=(12, 10, 52, 50))])[0]

    assert confirmed.target_confirmed
    assert seen_again.target_confirmed
    assert seen_again.target_progress == 2
    assert not seen_again.newly_confirmed
    assert seen_again.event_id == confirmed.event_id


def test_configured_species_alias_is_targeted() -> None:
    tracker = AnimalTracker(
        target_species={"wolf"},
        confidence_threshold=0.65,
        confirmation_frames=1,
    )

    result = tracker.update([_detection("grey wolf")])[0]

    assert result.target_confirmed
    assert result.newly_confirmed


def test_uncertain_taxonomy_is_never_a_target() -> None:
    tracker = AnimalTracker(
        target_species={"wolf"},
        confidence_threshold=0.2,
        confirmation_frames=1,
    )

    result = tracker.update([_detection("animal (uncertain)", confidence=0.99)])[0]

    assert not result.target_confirmed
    assert result.target_progress == 0


def test_person_can_be_tracked_but_is_never_a_target() -> None:
    tracker = AnimalTracker(
        target_species={"wolf", "fox", "coyote", "wild boar"},
        confidence_threshold=0.2,
        confirmation_frames=1,
    )

    result = tracker.update([_detection("person", confidence=0.99)])[0]

    assert result.track_id is not None
    assert not result.target_confirmed
    assert result.target_progress == 0


def test_bbox_area_trend_reports_approach_after_observation_window() -> None:
    tracker = AnimalTracker(
        target_species={"coyote"},
        confidence_threshold=0.65,
        confirmation_frames=3,
    )
    boxes = [
        (20, 20, 60, 60),
        (18, 18, 62, 62),
        (16, 16, 64, 64),
        (14, 14, 66, 66),
        (12, 12, 68, 68),
    ]

    results = [tracker.update([_detection("coyote", box=box)])[0] for box in boxes]

    assert results[-1].track_state == "approaching"
    assert results[-1].target_confirmed
