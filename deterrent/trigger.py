"""Target-gated deterrent trigger with event de-duplication and cooldown."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import config
from detection.animal_detector import Detection
from deterrent.audio_player import AudioPlayer

logger = logging.getLogger("deterrent.trigger")
_MAX_REMEMBERED_EVENTS = 4096


class AudioOutput(Protocol):
    def play(self, sound_path: str | Path, *, volume: float | None = None) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class TriggerResult:
    event_id: str
    species: str
    triggered: bool
    reason: str
    cooldown_remaining_seconds: float


class DeterrentTrigger:
    """Play at most once per confirmed event, with a global cooldown."""

    def __init__(
        self,
        audio_player: AudioOutput | None = None,
        *,
        enabled: bool | None = None,
        cooldown_seconds: float | None = None,
        target_species: frozenset[str] | set[str] | None = None,
        confidence_threshold: float | None = None,
        default_sound_path: str | Path | None = None,
        species_sound_files: dict[str, str | Path] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.audio_player = audio_player or AudioPlayer()
        self.enabled = (
            config.AUTOMATIC_DETERRENCE_ENABLED if enabled is None else enabled
        )
        self.cooldown_seconds = (
            config.DETERRENCE_COOLDOWN_SECONDS
            if cooldown_seconds is None
            else cooldown_seconds
        )
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
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
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.default_sound_path = Path(
            config.DEFAULT_SOUND_PATH
            if default_sound_path is None
            else default_sound_path
        )
        configured_sounds = (
            config.SPECIES_SOUND_FILES
            if species_sound_files is None
            else species_sound_files
        )
        self.species_sound_files = {
            self._normalize_name(name): Path(path)
            for name, path in configured_sounds.items()
        }
        self._clock = time.monotonic if clock is None else clock
        self._lock = threading.RLock()
        self._last_played_at: float | None = None
        self._triggered_events: OrderedDict[str, None] = OrderedDict()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.casefold().replace("_", " ").split())

    @property
    def cooldown_remaining_seconds(self) -> float:
        with self._lock:
            if self._last_played_at is None:
                return 0.0
            elapsed = max(0.0, self._clock() - self._last_played_at)
            return max(0.0, self.cooldown_seconds - elapsed)

    def set_enabled(self, enabled: bool) -> None:
        """Set the automatic deterrence gate."""
        with self._lock:
            self.enabled = enabled
        logger.warning("Automatic deterrence %s.", "enabled" if enabled else "disabled")

    def emergency_stop(self) -> None:
        """Disable future automatic triggers and stop current audio immediately."""
        with self._lock:
            self.enabled = False
            self.audio_player.stop()
        logger.critical("Emergency stop: deterrence disabled and audio stopped.")

    def _configured_species_for(self, class_name: str) -> str | None:
        label = self._normalize_name(class_name)
        for configured_target in self.target_species:
            target = self._normalize_name(configured_target)
            aliases = {
                self._normalize_name(alias)
                for alias in config.TARGET_SPECIES_ALIASES.get(configured_target, frozenset())
            }
            if label == target or label in aliases:
                return target
        return None

    def evaluate(self, detection: Detection) -> TriggerResult | None:
        """Evaluate a detection; only a newly confirmed configured target can play."""
        with self._lock:
            return self._evaluate_locked(detection)

    def _evaluate_locked(self, detection: Detection) -> TriggerResult | None:
        if not detection.target_confirmed or not detection.newly_confirmed:
            return None

        event_id = detection.event_id
        species = detection.class_name
        remaining = self.cooldown_remaining_seconds
        configured_species = self._configured_species_for(species)
        if (
            detection.confidence < self.confidence_threshold
            or configured_species is None
        ):
            logger.warning(
                "Rejected non-target or low-confidence deterrence request: %s (%s, %.2f).",
                species,
                event_id,
                detection.confidence,
            )
            return TriggerResult(event_id, species, False, "not_configured_target", remaining)
        if event_id in self._triggered_events:
            return TriggerResult(event_id, species, False, "event_already_triggered", remaining)
        if not self.enabled:
            logger.info("Deterrence is disabled; event %s did not trigger audio.", event_id)
            return TriggerResult(event_id, species, False, "deterrence_disabled", remaining)
        if remaining > 0:
            logger.info(
                "Deterrence cooldown active; event %s blocked for %.1f seconds.",
                event_id,
                remaining,
            )
            return TriggerResult(event_id, species, False, "cooldown", remaining)

        sound_path = self.species_sound_files.get(configured_species, self.default_sound_path)
        try:
            self.audio_player.play(sound_path, volume=config.MAX_AUDIO_VOLUME)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            self.enabled = False
            logger.exception(
                "Audio trigger failed for event %s; automatic deterrence disabled.",
                event_id,
            )
            return TriggerResult(event_id, species, False, "audio_error", 0.0)

        self._last_played_at = self._clock()
        self._triggered_events[event_id] = None
        self._triggered_events.move_to_end(event_id)
        if len(self._triggered_events) > _MAX_REMEMBERED_EVENTS:
            self._triggered_events.popitem(last=False)

        logger.warning(
            "Audio triggered for confirmed target %s (event %s); cooldown %.1f seconds.",
            species,
            event_id,
            self.cooldown_seconds,
        )
        return TriggerResult(
            event_id,
            species,
            True,
            "played",
            self.cooldown_remaining_seconds,
        )


__all__ = ["DeterrentTrigger", "TriggerResult"]
