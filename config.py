"""Central configuration for WILDLIFE-AI."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Select "webcam", "video", "esp32", or "rtsp" (Tapo / IP camera).
CAMERA_SOURCE: str = "webcam"
WEBCAM_INDEX: int = 0
VIDEO_PATH: Path = BASE_DIR / "samples" / "wildlife.mp4"
ESP32_CAMERA_IP: str = "192.168.1.50"
ESP32_STREAM_URL: str = f"http://{ESP32_CAMERA_IP}:81/stream"

# TP-Link Tapo / RTSP IP kamera yapılandırması:
# Tapo Uygulaması -> Kamera Ayarları -> Gelişmiş Ayarlar -> Kamera Hesabı (kullanıcı adı ve şifre)
# Yüksek Çözünürlük: stream1, Düşük Çözünürlük/Hızlı Akış: stream2 (RPi için stream2 önerilir)
RTSP_CAMERA_USER: str = "admin"
RTSP_CAMERA_PASSWORD: str = "admin123"
RTSP_CAMERA_IP: str = "192.168.1.100"
RTSP_STREAM_URL: str = f"rtsp://{RTSP_CAMERA_USER}:{RTSP_CAMERA_PASSWORD}@{RTSP_CAMERA_IP}:554/stream2"
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
# Hedef hayvan listesi: YOLO11n (COCO) modelinin tanıdığı ve alarmı tetikleyecek sınıflar.
# Test amaçlı "dog", "cat", "bird", "bear", "horse", "cow", "sheep" eklendi.
TARGET_ANIMALS: frozenset[str] = frozenset(
    {
        # Gerçek hedefler
        "wolf", "fox", "coyote", "wild boar",
        # Test amaçlı eklenen YOLO sınıfları (telefondan/ekrandan gösterip test edebilirsiniz)
        "dog", "cat", "bird", "bear",  "cow", "sheep"
    }
)
TARGET_SPECIES_ALIASES: dict[str, frozenset[str]] = {
    "wolf": frozenset({"grey wolf", "gray wolf", "timber wolf"}),
    "wild boar": frozenset({"wild pig", "feral pig"}),
    "dog": frozenset({"domestic dog"}),
}
CONFIDENCE_THRESHOLD: float = 0.50
CONSECUTIVE_CONFIRMATION_FRAMES: int = 2
ANALYSIS_INTERVAL_SECONDS: float = 0.25
TRACK_MATCH_IOU_THRESHOLD: float = 0.2
TRACK_MAX_MISSED_ANALYSIS_FRAMES: int = 1
TRACK_AREA_TREND_WINDOW: int = 5
TRACK_APPROACHING_AREA_RATIO: float = 1.25
TRACK_MOVING_AWAY_AREA_RATIO: float = 0.8

AUTOMATIC_DETERRENCE_ENABLED: bool = True
DETERRENCE_COOLDOWN_SECONDS: int = 5
DEFAULT_SOUND_PATH: Path = BASE_DIR / "sounds" / "warning.wav"
SPECIES_SOUND_FILES: dict[str, Path] = {}
MAX_AUDIO_VOLUME: float = 0.25

DATABASE_PATH: Path = BASE_DIR / "wildlife_ai.sqlite3"
SNAPSHOTS_DIR: Path = BASE_DIR / "snapshots"
LOGS_DIR: Path = BASE_DIR / "logs"
LOG_LEVEL: str = "INFO"

# Gece Modu & Kontrast Artırma (CLAHE)
NIGHT_MODE_AUTO: bool = True
NIGHT_MODE_THRESHOLD: float = 65.0
CLAHE_CLIP_LIMIT: float = 2.5
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
NIGHT_MODE_DRAW_INDICATOR: bool = True

DASHBOARD_HOST: str = "127.0.0.1"
DASHBOARD_PORT: int = 5000
DASHBOARD_DETECTOR: str = "yolo"
DASHBOARD_ANALYSIS_EVERY: int = 3   # Akıcı yayın için her 3 karede 1 analiz (RPi CPU'sunu rahatlatır)
DASHBOARD_STREAM_FPS: int = 15      # Canlı yayın saniyedeki kare hızı (5'ten 15'e çıkarıldı)
DASHBOARD_JPEG_QUALITY: int = 70    # Hızlı aktarım ve düşük gecikme için optimize JPEG kalitesi
