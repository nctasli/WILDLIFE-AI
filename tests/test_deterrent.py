from __future__ import annotations

import threading
from pathlib import Path

import pytest

import config
from detection.animal_detector import Detection
from deterrent.trigger import DeterrentTrigger


class FakeAudioPlayer:
    def __init__(self) -> None:
        self.plays: list[tuple[Path, float | None]] = []
        self.stop_count = 0
        self.playing = False

    def play(self, sound_path: str | Path, *, volume: float | None = None) -> None:
        self.plays.append((Path(sound_path), volume))
        self.playing = True

    def stop(self) -> None:
        self.stop_count += 1
        self.playing = False


def _confirmed(
    species: str = "coyote",
    *,
    event_id: str = "event-1",
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        class_name=species,
        confidence=confidence,
        coordinates=(10, 10, 50, 50),
        event_id=event_id,
        target_confirmed=True,
        newly_confirmed=True,
    )


def test_deterrence_is_disabled_by_default(tmp_path: Path) -> None:
    audio = FakeAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        default_sound_path=tmp_path / "warning.wav",
    )

    result = trigger.evaluate(_confirmed())

    assert result is not None
    assert not result.triggered
    assert result.reason == "deterrence_disabled"
    assert audio.plays == []


def test_confirmed_target_plays_once_and_deduplicates_event(tmp_path: Path) -> None:
    sound = tmp_path / "coyote.wav"
    audio = FakeAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        cooldown_seconds=300,
        target_species={"coyote"},
        species_sound_files={"coyote": sound},
        clock=lambda: 100.0,
    )

    first = trigger.evaluate(_confirmed())
    duplicate = trigger.evaluate(_confirmed())

    assert first is not None and first.triggered and first.reason == "played"
    assert duplicate is not None and not duplicate.triggered
    assert duplicate.reason == "event_already_triggered"
    assert audio.plays == [(sound, config.MAX_AUDIO_VOLUME)]


def test_global_cooldown_blocks_a_different_event(tmp_path: Path) -> None:
    audio = FakeAudioPlayer()
    now = [100.0]
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        cooldown_seconds=300,
        target_species={"coyote"},
        default_sound_path=tmp_path / "warning.wav",
        clock=lambda: now[0],
    )

    played = trigger.evaluate(_confirmed(event_id="one"))
    blocked = trigger.evaluate(_confirmed(event_id="two"))

    assert played is not None and played.triggered
    assert blocked is not None and blocked.reason == "cooldown"
    assert blocked.cooldown_remaining_seconds == pytest.approx(300)
    assert len(audio.plays) == 1

    now[0] += 300
    allowed = trigger.evaluate(_confirmed(event_id="two"))
    assert allowed is not None and allowed.triggered
    assert len(audio.plays) == 2


def test_human_pet_or_low_confidence_cannot_trigger(tmp_path: Path) -> None:
    audio = FakeAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        target_species={"coyote"},
        default_sound_path=tmp_path / "warning.wav",
    )

    human = trigger.evaluate(_confirmed("person", event_id="human"))
    dog = trigger.evaluate(_confirmed("dog", event_id="dog"))
    low_confidence = trigger.evaluate(_confirmed(confidence=0.4, event_id="weak"))

    assert human is not None and human.reason == "not_configured_target"
    assert dog is not None and dog.reason == "not_configured_target"
    assert low_confidence is not None and low_confidence.reason == "not_configured_target"
    assert audio.plays == []


def test_species_specific_sound_and_wolf_alias(tmp_path: Path) -> None:
    wolf_sound = tmp_path / "wolf.wav"
    audio = FakeAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        cooldown_seconds=0,
        target_species={"wolf"},
        species_sound_files={"wolf": wolf_sound},
    )

    result = trigger.evaluate(_confirmed("grey wolf"))

    assert result is not None and result.triggered
    assert audio.plays[0][0] == wolf_sound


def test_audio_error_disables_automatic_deterrence(tmp_path: Path) -> None:
    class BrokenAudioPlayer(FakeAudioPlayer):
        def play(self, sound_path: str | Path, *, volume: float | None = None) -> None:
            raise FileNotFoundError(f"Missing file: {sound_path}")

    trigger = DeterrentTrigger(
        BrokenAudioPlayer(),
        enabled=True,
        target_species={"coyote"},
        default_sound_path=tmp_path / "missing.wav",
    )

    result = trigger.evaluate(_confirmed())

    assert result is not None and result.reason == "audio_error"
    assert not trigger.enabled


def test_emergency_stop_disables_trigger_and_stops_audio(tmp_path: Path) -> None:
    audio = FakeAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        default_sound_path=tmp_path / "warning.wav",
    )

    trigger.emergency_stop()
    result = trigger.evaluate(_confirmed())

    assert not trigger.enabled
    assert audio.stop_count == 1
    assert result is not None and result.reason == "deterrence_disabled"


def test_emergency_stop_waits_for_in_progress_audio_trigger(tmp_path: Path) -> None:
    class BlockingAudioPlayer(FakeAudioPlayer):
        def __init__(self) -> None:
            super().__init__()
            self.play_started = threading.Event()
            self.allow_play_to_finish = threading.Event()

        def play(self, sound_path: str | Path, *, volume: float | None = None) -> None:
            self.play_started.set()
            assert self.allow_play_to_finish.wait(timeout=2)
            super().play(sound_path, volume=volume)

    audio = BlockingAudioPlayer()
    trigger = DeterrentTrigger(
        audio,
        enabled=True,
        cooldown_seconds=0,
        target_species={"coyote"},
        default_sound_path=tmp_path / "warning.wav",
    )
    evaluation = threading.Thread(target=trigger.evaluate, args=(_confirmed(),))
    evaluation.start()
    assert audio.play_started.wait(timeout=2)

    stop_started = threading.Event()
    stop_finished = threading.Event()

    def stop_trigger() -> None:
        stop_started.set()
        trigger.emergency_stop()
        stop_finished.set()

    emergency = threading.Thread(target=stop_trigger)
    emergency.start()
    assert stop_started.wait(timeout=2)
    assert not stop_finished.wait(timeout=0.05)

    audio.allow_play_to_finish.set()
    evaluation.join(timeout=2)
    emergency.join(timeout=2)
    assert not evaluation.is_alive()
    assert not emergency.is_alive()
    assert not trigger.enabled
    assert audio.stop_count == 1
    assert not audio.playing
