"""Camera preview entry point with optional YOLO inference."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

import config
from camera import NightModeEnhancer, create_camera
from database import WildlifeDatabase
from detection import AnimalDetector
from detection.base_detector import FrameDetector
from detection.tracker import AnimalTracker
from deterrent import AudioPlayer, DeterrentTrigger

logger = logging.getLogger("wildlife_ai")


class PreviewCamera(Protocol):
    def open(self) -> bool: ...

    def read(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


def configure_logging() -> None:
    """Configure console output and independent category log files."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for category in (
        "wildlife_ai",
        "camera",
        "detection",
        "deterrent",
        "database",
        "dashboard",
    ):
        category_logger = logging.getLogger(category)
        handler_name = f"wildlife_ai_{category}"
        if any(handler.get_name() == handler_name for handler in category_logger.handlers):
            continue
        handler = logging.FileHandler(config.LOGS_DIR / f"{category}.log", encoding="utf-8")
        handler.set_name(handler_name)
        handler.setFormatter(formatter)
        category_logger.addHandler(handler)


def run_preview(
    camera: PreviewCamera,
    *,
    display: bool = True,
    max_frames: int | None = None,
    detector: FrameDetector | None = None,
    analysis_every: int = 1,
    tracker: AnimalTracker | None = None,
    trigger: DeterrentTrigger | None = None,
    database: WildlifeDatabase | None = None,
    stop_event: threading.Event | None = None,
    frame_callback: Callable[[np.ndarray], None] | None = None,
    enhancer: NightModeEnhancer | None = None,
) -> int:
    """Preview camera frames, optionally detecting and drawing model results."""
    if analysis_every <= 0:
        raise ValueError("analysis_every must be greater than zero.")

    frame_count = 0
    if not camera.open():
        logger.warning("Camera is not available yet; waiting for connection.")

    try:
        while (
            (stop_event is None or not stop_event.is_set())
            and (max_frames is None or frame_count < max_frames)
        ):
            frame = camera.read()
            if frame is None:
                if display:
                    if cv2.waitKey(config.CAMERA_PREVIEW_WAIT_MS) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(min(config.CAMERA_PREVIEW_WAIT_MS / 1000, 0.1))
                continue

            if enhancer is not None:
                frame, _, _ = enhancer.enhance(frame)

            frame_count += 1
            if detector is not None and (frame_count - 1) % analysis_every == 0:
                detections = detector.detect(frame)
                if tracker is not None:
                    detections = tracker.update(detections)
                frame = detector.annotate(frame, detections)
                for detection in detections:
                    snapshot_path: Path | None = None
                    if (
                        database is not None
                        and detection.newly_confirmed
                        and detection.target_confirmed
                    ):
                        try:
                            snapshot_path = database.save_snapshot(detection, frame)
                        except (OSError, RuntimeError, ValueError):
                            logging.getLogger("database.snapshots").exception(
                                "Snapshot capture failed for event %s",
                                detection.event_id,
                            )

                    trigger_result = None
                    logger.info(
                        "Detected %s (%.2f) at %s, track=%s, target=%s (%d/%d), "
                        "movement=%s%s",
                        detection.class_name,
                        detection.confidence,
                        detection.coordinates,
                        detection.track_id,
                        detection.target_confirmed,
                        detection.target_progress,
                        config.CONSECUTIVE_CONFIRMATION_FRAMES,
                        detection.track_state,
                        " [NEWLY CONFIRMED]" if detection.newly_confirmed else "",
                    )
                    if trigger is not None:
                        trigger_result = trigger.evaluate(detection)
                        if trigger_result is not None:
                            logger.info(
                                "Deterrence result: event=%s species=%s triggered=%s "
                                "reason=%s cooldown=%.1fs",
                                trigger_result.event_id,
                                trigger_result.species,
                                trigger_result.triggered,
                                trigger_result.reason,
                                trigger_result.cooldown_remaining_seconds,
                            )
                    if database is not None:
                        database.record_detection(
                            detection,
                            snapshot_path=snapshot_path,
                            trigger_result=trigger_result,
                        )
            if frame_callback is not None:
                frame_callback(frame)
            if display:
                cv2.imshow("WILDLIFE-AI Camera Preview", frame)
                key = cv2.waitKey(config.CAMERA_PREVIEW_WAIT_MS) & 0xFF
                if key == ord("x") and trigger is not None:
                    trigger.emergency_stop()
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Preview stopped by user.")
    finally:
        camera.release()
        if trigger is not None:
            trigger.audio_player.stop()
        if display:
            cv2.destroyAllWindows()

    return frame_count


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a WILDLIFE-AI camera source.")
    parser.add_argument(
        "--source",
        choices=("webcam", "video", "esp32"),
        default=config.CAMERA_SOURCE,
        help="Camera source (default: config.CAMERA_SOURCE).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file path when --source video is selected.",
    )
    parser.add_argument("--webcam-index", type=int, default=None)
    parser.add_argument("--esp32-url", default=None, help="ESP32 HTTP/MJPEG stream URL.")
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run the selected detector and draw bounding boxes.",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo", "speciesnet"),
        default="yolo",
        help="Detection engine (SpeciesNet activates detection automatically).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODEL_PATH,
        help="YOLO .pt model path (default: config.MODEL_PATH).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Read frames without opening a preview window (useful for headless tests).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames; omit to run until stopped.",
    )
    parser.add_argument(
        "--analysis-every",
        type=int,
        default=1,
        help="Run detection every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--enable-deterrence",
        action="store_true",
        help="Enable automatic audio only after target confirmation (disabled by default).",
    )
    parser.add_argument(
        "--test-sound",
        action="store_true",
        help="Play the configured/test audio file once and exit.",
    )
    parser.add_argument(
        "--sound",
        type=Path,
        default=config.DEFAULT_SOUND_PATH,
        help="Audio file for --test-sound (default: config.DEFAULT_SOUND_PATH).",
    )
    parser.add_argument(
        "--night-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="Night mode / low-light contrast enhancement with CLAHE (default: auto).",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(arguments)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero.")
    if args.analysis_every <= 0:
        raise ValueError("--analysis-every must be greater than zero.")

    audio_player = AudioPlayer()
    if args.test_sound:
        try:
            audio_player.play(args.sound, volume=config.MAX_AUDIO_VOLUME)
            audio_player.wait_until_finished()
        finally:
            audio_player.close()
        return 0

    camera = create_camera(
        args.source,
        video_path=args.video,
        webcam_index=args.webcam_index,
        esp32_stream_url=args.esp32_url,
    )
    detector: FrameDetector | None = None
    if args.detect and args.detector == "yolo":
        detector = AnimalDetector(model_path=args.model)
    elif args.detector == "speciesnet":
        from detection.speciesnet_detector import SpeciesNetAnimalDetector

        detector = SpeciesNetAnimalDetector()
    tracker = AnimalTracker() if detector is not None else None
    trigger = (
        DeterrentTrigger(
            audio_player,
            enabled=config.AUTOMATIC_DETERRENCE_ENABLED or args.enable_deterrence,
        )
        if detector is not None
        else None
    )
    database = WildlifeDatabase()
    enhancer: NightModeEnhancer | None = None
    if args.night_mode == "on":
        enhancer = NightModeEnhancer(enabled=True)
    elif args.night_mode == "auto":
        enhancer = NightModeEnhancer(auto=True)
    elif args.night_mode == "off":
        enhancer = NightModeEnhancer(enabled=False)

    try:
        frames_read = run_preview(
            camera,
            display=not args.no_display,
            max_frames=args.max_frames,
            detector=detector,
            analysis_every=args.analysis_every,
            tracker=tracker,
            trigger=trigger,
            database=database,
            enhancer=enhancer,
        )
    finally:
        database.close()
    logger.info("Preview finished after reading %d frame(s).", frames_read)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
