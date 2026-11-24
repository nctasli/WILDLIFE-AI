"""Helper script to download Turkish wildlife datasets from Roboflow Universe."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "wildlife_tr"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow Universe.")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", ""), help="Roboflow API Key")
    parser.add_argument("--workspace", default="", help="Roboflow workspace name")
    parser.add_argument("--project", default="", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version (default: 1)")
    args = parser.parse_args()

    if not args.api_key or not args.workspace or not args.project:
        print(
            "\n💡 Roboflow'dan Veri Seti İndirme Kılavuzu:\n\n"
            "1. Roboflow Universe (https://universe.roboflow.com/) adresine gidin.\n"
            "2. Türkiye veya hedef vahşi hayvan veri setlerinden birini seçin:\n"
            "   - Örn: 'wild-boar-detection', 'wildlife-animals', 'wolf-coyote-bear'\n"
            "3. 'Download Dataset' butonuna tıklayın.\n"
            "4. Format olarak 'YOLOv11' veya 'YOLOv8' seçin.\n"
            "5. İki yöntemle indirebilirsiniz:\n"
            "   a) 'Terminal / Python' kodunu kopyalayıp API anahtarınızla şu komutu çalıştırın:\n"
            "      python training/download_dataset.py --api-key YOUR_KEY --workspace WORKSPACE --project PROJECT --version 1\n\n"
            "   b) Doğrudan 'Download .zip' seçeneğiyle indirip ZIP içindekileri şu klasöre çıkartın:\n"
            f"      {DATASET_DIR.resolve()}\n\n"
            f"Klasör yapısı şöyle olmalıdır:\n"
            f"  {DATASET_DIR}/data.yaml\n"
            f"  {DATASET_DIR}/train/images\n"
            f"  {DATASET_DIR}/train/labels\n"
            f"  {DATASET_DIR}/valid/images\n"
            f"  {DATASET_DIR}/valid/labels\n"
        )
        return 0

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow kütüphanesi eksik. Yükleniyor: pip install roboflow")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    print(f"İndiriliyor: {args.project} v{args.version} -> {DATASET_DIR} ...")
    version.download("yolov11", location=str(DATASET_DIR.resolve()))
    print(f"\n✅ Veri seti başarıyla indirildi: {DATASET_DIR.resolve()}")
    print("Şimdi eğitimi başlatabilirsiniz:")
    print("  python training/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
