"""Export and benchmark YOLO models for Raspberry Pi 4 (ONNX and NCNN).

Converts PyTorch (.pt) weights into lightweight, high-FPS deployment formats
for ARM64 Cortex-A72 CPU on Raspberry Pi 4.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize YOLO models for Raspberry Pi 4 (ONNX / NCNN)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "yolo11n.pt",
        help="Path to source .pt model (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--format",
        choices=("onnx", "ncnn", "both"),
        default="both",
        help="Export format (onnx, ncnn, or both. default: both).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input resolution (default: 640. Use 320 or 480 for 2x-3x higher FPS on RPi 4).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run speed benchmark after export (default: True).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of benchmark frames (default: 30).",
    )
    return parser.parse_args()


def benchmark_model(model_path: Path | str, imgsz: int, frames: int = 30) -> tuple[float, float]:
    """Measure average inference latency (ms) and FPS."""
    model = YOLO(str(model_path))
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(source=dummy_frame, verbose=False)

    times: list[float] = []
    for _ in range(frames):
        start = time.perf_counter()
        model.predict(source=dummy_frame, verbose=False)
        times.append(time.perf_counter() - start)

    avg_latency_ms = (sum(times) / len(times)) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return avg_latency_ms, fps


def main() -> int:
    args = parse_args()
    source_model = args.model.resolve()

    if not source_model.is_file():
        # Check if in models directory
        alt_path = MODELS_DIR / args.model.name
        if alt_path.is_file():
            source_model = alt_path
        else:
            print(f"❌ Kaynak model bulunamadı: {source_model}")
            return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    formats = ["onnx", "ncnn"] if args.format == "both" else [args.format]

    print("=" * 65)
    print("🍓 RASPBERRY PI 4 MODEL OPTİMİZASYON ARACI")
    print(f"📦 Kaynak Model : {source_model.name}")
    print(f"🎯 Format(lar)  : {', '.join(formats).upper()}")
    print(f"📏 Çözünürlük   : {args.imgsz}x{args.imgsz}")
    print("=" * 65)

    exported_paths: dict[str, Path] = {}
    model = YOLO(str(source_model))

    for fmt in formats:
        print(f"\n⚙️  '{fmt.upper()}' formatına dönüştürülüyor (imgsz={args.imgsz})...")
        try:
            exported_file = model.export(
                format=fmt,
                imgsz=args.imgsz,
                dynamic=False,
            )
            exported_path = Path(exported_file).resolve()

            # Move to models/ directory if not already there
            target_path = MODELS_DIR / exported_path.name
            if exported_path != target_path:
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()
                shutil.move(exported_path, target_path)
                exported_path = target_path

            exported_paths[fmt] = exported_path
            print(f"✅ {fmt.upper()} modeli hazır: {exported_path}")
        except Exception as exc:
            print(f"⚠️ {fmt.upper()} export başarısız oldu: {exc}")

    if args.benchmark:
        print("\n" + "=" * 65)
        print("⚡ PERFORMANS KARŞILAŞTIRMA TESTİ (BENCHMARK)")
        print("=" * 65)

        # Baseline PyTorch
        print(f"⏳ Ölçülüyor: PyTorch (.pt)...", end="", flush=True)
        pt_ms, pt_fps = benchmark_model(source_model, args.imgsz, args.frames)
        print(f"  -> {pt_ms:.1f} ms/kare ({pt_fps:.1f} FPS)")

        for fmt, path in exported_paths.items():
            print(f"⏳ Ölçülüyor: {fmt.upper()}...", end="", flush=True)
            try:
                fmt_ms, fmt_fps = benchmark_model(path, args.imgsz, args.frames)
                speedup = (pt_ms / fmt_ms) if fmt_ms > 0 else 1.0
                print(f"  -> {fmt_ms:.1f} ms/kare ({fmt_fps:.1f} FPS) [{speedup:.2f}x hızlanma]")
            except Exception as e:
                print(f"  -> Hata: {e}")

    print("\n" + "=" * 65)
    print("📋 RASPBERRY PI 4 DAĞITIM KILAVUZU:")
    print("1. 'models/' klasöründeki .onnx veya _ncnn_model klasörünü RPi'a kopyalayın.")
    print("2. Raspberry Pi 4'te çalıştırmak için:")
    if "onnx" in exported_paths:
        print(f"   python main.py --model models/{exported_paths['onnx'].name} --detect")
    if "ncnn" in exported_paths:
        print(f"   python main.py --model models/{exported_paths['ncnn'].name} --detect")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
