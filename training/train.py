"""Wildlife-AI Transfer Learning & Fine-Tuning Pipeline.

Train YOLO11n on Turkish wildlife (wild boar, wolf, jackal, fox, bear).
Automatically saves the best weights and exports to ONNX for Raspberry Pi 4.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = BASE_DIR / "datasets" / "wildlife_tr" / "data.yaml"
OUTPUT_MODEL_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on Turkish wildlife dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Path to data.yaml file from Roboflow.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO base model to fine-tune (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce to 8 if out of memory).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution for training (default: 640).",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of backbone layers to freeze for fast transfer learning (default: 10).",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Automatically export best.pt to ONNX for Raspberry Pi 4 (default: True).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_yaml_path = args.data.resolve()
    if not data_yaml_path.is_file():
        print(
            f"\n❌ HATA: Veri seti konfigürasyonu bulunamadı: {data_yaml_path}\n"
            f"Lütfen Roboflow veri setinizi 'datasets/wildlife_tr/' klasörüne çıkartın\n"
            f"veya '--data /path/to/data.yaml' parametresiyle dosya yolunu belirtin.\n",
            file=sys.stderr,
        )
        return 1

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    print("=" * 60)
    print("🐾 WILDLIFE-AI EĞİTİM (TRANSFER LEARNING) BAŞLATILIYOR")
    print(f"📦 Temel Model : {args.base_model}")
    print(f"📂 Veri Seti   : {data_yaml_path}")
    print(f"⚙️  Cihaz       : {device_name} ({device})")
    print(f"🔄 Epoch       : {args.epochs}")
    print(f"📏 Resim Boyutu: {args.imgsz}")
    print(f"🧊 Dondurulan Katman: {args.freeze}")
    print("=" * 60)

    # Load base model
    model = YOLO(args.base_model)

    # Train
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        device=device,
        project="training/runs",
        name="wildlife_tr",
        exist_ok=True,
        verbose=True,
    )

    # Locate best.pt
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("training/runs/wildlife_tr")
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.is_file():
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        final_pt_path = OUTPUT_MODEL_DIR / "wildlife_tr_yolo11n.pt"
        shutil.copy2(best_weights, final_pt_path)
        print(f"\n✅ Başarılı! En iyi model kaydedildi: {final_pt_path.resolve()}")

        # Export to ONNX for Raspberry Pi 4
        if args.export_onnx:
            print("\n🍓 Raspberry Pi 4 için ONNX formatına dönüştürülüyor...")
            trained_model = YOLO(str(final_pt_path))
            onnx_path = trained_model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
            print(f"✅ ONNX Modeli hazır: {onnx_path}")
            print("Bu dosyayı Raspberry Pi 4'e kopyalayarak yüksek FPS ile çalıştırabilirsiniz!")
    else:
        print(f"⚠️ Uyarı: 'best.pt' dosyası {best_weights} konumunda bulunamadı.")

    print("\n🎉 Eğitim tamamlandı!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
