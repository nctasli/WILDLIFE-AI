"""Private pygame playback worker; keep pygame/SDL out of the OpenCV process."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


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


if __name__ == "__main__":
    raise SystemExit(main())
