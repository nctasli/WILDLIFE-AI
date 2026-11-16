from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from deterrent.audio_player import AudioPlayer


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("READY\n")
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return 0 if self.returncode is None else self.returncode


def test_player_validates_file_volume_and_can_stop(tmp_path: Path) -> None:
    sound_file = tmp_path / "test.wav"
    sound_file.write_bytes(b"test")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    process = FakeProcess()

    def start_process(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    player = AudioPlayer(max_volume=0.2, process_factory=start_process)
    player.play(sound_file, volume=0.2)

    assert calls[0][0][-2:] == [str(sound_file), "0.2000"]
    assert calls[0][1]["stdin"] is not None
    assert calls[0][1]["stdout"] is not None
    assert calls[0][1]["env"]["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"
    assert player.is_playing
    player.stop()
    assert not player.is_playing
    assert process.terminated


def test_missing_file_is_reported(tmp_path: Path) -> None:
    player = AudioPlayer(process_factory=lambda *_args, **_kwargs: pytest.fail("worker must not start"))
    with pytest.raises(FileNotFoundError, match="Audio file does not exist"):
        player.play(tmp_path / "missing.wav")


def test_volume_cannot_exceed_configured_or_hard_limit(tmp_path: Path) -> None:
    sound_file = tmp_path / "test.wav"
    sound_file.write_bytes(b"test")
    with pytest.raises(ValueError, match="max_volume"):
        AudioPlayer(max_volume=0.3)

    player = AudioPlayer(max_volume=0.2, process_factory=lambda *_args, **_kwargs: pytest.fail("worker must not start"))
    with pytest.raises(ValueError, match="volume"):
        player.play(sound_file, volume=0.21)


def test_worker_start_failure_is_reported(tmp_path: Path) -> None:
    sound_file = tmp_path / "test.wav"
    sound_file.write_bytes(b"test")
    process = FakeProcess()
    process.stdout = io.StringIO("worker error\n")
    process.returncode = 1
    player = AudioPlayer(process_factory=lambda *_args, **_kwargs: process)

    with pytest.raises(RuntimeError, match="Audio worker failed"):
        player.play(sound_file)
    assert not player.is_playing


def test_worker_runtime_failure_is_reported(tmp_path: Path) -> None:
    sound_file = tmp_path / "test.wav"
    sound_file.write_bytes(b"test")
    process = FakeProcess()
    player = AudioPlayer(process_factory=lambda *_args, **_kwargs: process)
    player.play(sound_file)
    process.returncode = 1

    with pytest.raises(RuntimeError, match="Audio worker exited"):
        player.wait_until_finished(poll_interval_seconds=0.001)


def test_close_stops_current_process() -> None:
    process = FakeProcess()
    player = AudioPlayer(process_factory=lambda *_args, **_kwargs: process)
    sound_file = Path(__file__)
    player.play(sound_file)

    player.close()

    assert process.terminated
    assert not player.is_playing
