#!/usr/bin/env python3
"""Faz 4 uyarlama: yolo11_drone.yaml (w0.20, nc=1) modelini 128^2 kesit verisinde egit.

Sifirdan egitim (pretrained yok: genislik 0.20 icin hazir agirlik bulunmuyor).
Augment notu: degradasyon (blur/jpeg/dusuk kontrast) plani Faz 5 fine-tune'da;
burada standart YOLO augment + mosaic kapali (kesitler zaten hedef-merkezli,
mosaic 128^2'de hedefleri asiri kucultur).

Kullanim: python3 train_crops.py [--epochs 150] [--batch 1024]
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "drone-tracker-ml/crops128/data.yaml"))
    ap.add_argument("--model", default=str(Path.home() / "drone-tracker-ml/yolo11_drone.yaml"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data,
        imgsz=128,
        epochs=a.epochs,
        batch=a.batch,
        device="0,1",
        workers=16,
        mosaic=0.0,        # kesit verisinde mosaic zararli (hedefler 32px'e iner)
        scale=0.3,         # olcek augmenti sinirli: servis dagilimi sabit 128 kesit
        patience=30,
        project=str(Path.home() / "drone-tracker-ml/runs"),
        name="yolo11_w020_crops128",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
