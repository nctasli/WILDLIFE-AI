"""Cross-platform local audio playback with a conservative volume ceiling."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import config

logger = logging.getLogger("deterrent.audio")
_HARD_VOLUME_CEILING = 0.25


class AudioPlayer:
    """Play WAV/MP3 assets in a helper process to isolate SDL from OpenCV."""

    def __init__(
        self,
        *,
        max_volume: float | None = None,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        configured_limit = config.MAX_AUDIO_VOLUME if max_volume is None else max_volume
        if not 0.0 < configured_limit <= _HARD_VOLUME_CEILING:
            raise ValueError(
                f"max_volume must be greater than 0 and no greater than "
                f"{_HARD_VOLUME_CEILING}."
            )
        self.max_volume = configured_limit
        self._process_factory = subprocess.Popen if process_factory is None else process_factory
        self._process: Any | None = None

    def play(
        self,
        sound_path: str | Path,
        *,
        volume: float | None = None,
    ) -> None:
        """Start playback asynchronously; missing/invalid media raises."""
        path = Path(sound_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        selected_volume = self.max_volume if volume is None else volume
        if not 0.0 < selected_volume <= self.max_volume:
            raise ValueError(
                f"volume must be greater than 0 and no greater than {self.max_volume}."
            )

        if self.is_playing:
            raise RuntimeError("Audio playback is already in progress.")

        worker_path = Path(__file__).with_name("audio_worker.py")
        environment = os.environ.copy()
        environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        process = self._process_factory(
            [
                sys.executable,
                str(worker_path),
                str(path),
                f"{selected_volume:.4f}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            env=environment,
        )
        ready = process.stdout.readline() if process.stdout is not None else ""
        if process.stdout is not None:
            process.stdout.close()
        if ready.strip() != "READY":
            return_code = process.wait()
            raise RuntimeError(
                f"Audio worker failed to start playback for {path} "
                f"(exit code {return_code})."
            )

        self._process = process
        logger.info("Started audio playback: %s at volume %.2f", path, selected_volume)

    @property
    def is_playing(self) -> bool:
        if self._process is None:
            return False
        return_code = self._process.poll()
        if return_code is None:
            return True
        if return_code != 0:
            raise RuntimeError(f"Audio worker exited with code {return_code}.")
        return False

    def stop(self) -> None:
        """Immediately terminate current playback, if any."""
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        logger.info("Audio playback stopped.")

    def wait_until_finished(self, poll_interval_seconds: float = 0.05) -> None:
        """Block until test-mode playback finishes or is stopped."""
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")
        while self.is_playing:
            time.sleep(poll_interval_seconds)

    def close(self) -> None:
        """Stop current playback."""
        self.stop()


__all__ = ["AudioPlayer"]
