"""Central configuration for WILDLIFE-AI."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Select "webcam", "video", or "esp32".
CAMERA_SOURCE: str = "webcam"
WEBCAM_INDEX: int = 0
VIDEO_PATH: Path = BASE_DIR / "samples" / "wildlife.mp4"
ESP32_CAMERA_IP: str = "192.168.1.50"
ESP32_STREAM_URL: str = f"http://{ESP32_CAMERA_IP}:81/stream"
CAMERA_RECONNECT_SECONDS: float = 3.0
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_PREVIEW_WAIT_MS: int = 20

# The default COCO model does not include fox, coyote, or wild boar classes.
# Keep automatic deterrence disabled until the selected model and target
# classes have been verified for the deployment.
MODEL_PATH: Path = BASE_DIR / "models" / "yolo11n.pt"
SPECIESNET_MODEL: str = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
SPECIESNET_DETECTION_THRESHOLD: float = 0.5
# English labels must match model class names. The default COCO model supports
# "dog" but not wolf, fox, coyote, or wild boar.
TARGET_ANIMALS: frozenset[str] = frozenset(
    {"wolf", "fox", "coyote", "wild boar"}
)
TARGET_SPECIES_ALIASES: dict[str, frozenset[str]] = {
    "wolf": frozenset({"grey wolf", "gray wolf", "timber wolf"}),
    "wild boar": frozenset({"wild pig", "feral pig"}),
    "dog": frozenset({"domestic dog"}),
}
CONFIDENCE_THRESHOLD: float = 0.65
CONSECUTIVE_CONFIRMATION_FRAMES: int = 3
ANALYSIS_INTERVAL_SECONDS: float = 0.25
TRACK_MATCH_IOU_THRESHOLD: float = 0.2
TRACK_MAX_MISSED_ANALYSIS_FRAMES: int = 1
TRACK_AREA_TREND_WINDOW: int = 5
TRACK_APPROACHING_AREA_RATIO: float = 1.25
TRACK_MOVING_AWAY_AREA_RATIO: float = 0.8

AUTOMATIC_DETERRENCE_ENABLED: bool = False
DETERRENCE_COOLDOWN_SECONDS: int = 300
DEFAULT_SOUND_PATH: Path = BASE_DIR / "sounds" / "warning.wav"
SPECIES_SOUND_FILES: dict[str, Path] = {}
MAX_AUDIO_VOLUME: float = 0.25

DATABASE_PATH: Path = BASE_DIR / "wildlife_ai.sqlite3"
SNAPSHOTS_DIR: Path = BASE_DIR / "snapshots"
LOGS_DIR: Path = BASE_DIR / "logs"
LOG_LEVEL: str = "INFO"

DASHBOARD_HOST: str = "127.0.0.1"
DASHBOARD_PORT: int = 5000
DASHBOARD_DETECTOR: str = "yolo"
DASHBOARD_ANALYSIS_EVERY: int = 1
DASHBOARD_STREAM_FPS: int = 5
