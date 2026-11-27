"""Cross-platform audio playback worker.

On Linux (Raspberry Pi headless): uses ``aplay`` (ALSA) – no SDL/X11 needed.
On Windows / macOS: uses pygame, which requires a sound device but no display.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _play_linux_alsa(sound_path: Path, volume: float) -> int:
    """Play audio via ALSA aplay – works headless on Raspberry Pi."""
    # aplay does not support volume natively; use sox `play` if available,
    # otherwise fall back to aplay at system volume.
    try:
        # Try sox `play` first (supports -v gain factor 0.0-1.0)
        result = subprocess.run(
            ["which", "play"], capture_output=True, text=True
        )
        if result.returncode == 0:
            subprocess.run(
                ["play", "-q", "-v", str(volume), str(sound_path)],
                check=False,
            )
        else:
            # Fall back to plain aplay (ignores software volume, uses system volume)
            subprocess.run(["aplay", "-q", str(sound_path)], check=False)
    except FileNotFoundError:
        # aplay not available – try paplay (PulseAudio)
        try:
            subprocess.run(["paplay", str(sound_path)], check=False)
        except FileNotFoundError:
            print("No audio backend found (aplay/paplay/play). Install alsa-utils.", file=sys.stderr)
            return 1
    print("READY", flush=True)
    return 0


def _play_pygame(sound_path: Path, volume: float) -> int:
    """Play audio via pygame (Windows / macOS)."""
    import pygame

    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound(str(sound_path))
        sound.set_volume(volume)
        channel = pygame.mixer.find_channel(force=False)
        if channel is None:
            raise RuntimeError("No free audio channel is available.")
        channel.play(sound)
        print("READY", flush=True)
        while channel.get_busy():
            time.sleep(0.05)
    except (pygame.error, RuntimeError):
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
    return 0


def main(arguments: list[str] | None = None) -> int:
    args = sys.argv[1:] if arguments is None else arguments
    if len(args) != 2:
        print("Usage: audio_worker.py <sound-path> <volume>", file=sys.stderr)
        return 2

    sound_path = Path(args[0])
    volume = float(args[1])

    if not sound_path.is_file():
        print(f"Audio file does not exist: {sound_path}", file=sys.stderr)
        return 2
    if not 0.0 < volume <= 0.25:
        print("Worker volume must be greater than 0 and at most 0.25.", file=sys.stderr)
        return 2

    if platform.system() == "Linux":
        return _play_linux_alsa(sound_path, volume)
    else:
        return _play_pygame(sound_path, volume)


if __name__ == "__main__":
    raise SystemExit(main())
