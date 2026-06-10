#!/usr/bin/env python3
"""Faz 2: Liftoff domain fine-tune.

best.pt (public airborne, mAP50=0.953) -> Liftoff kesitleriyle kisa fine-tune.

CATASTROPHIC FORGETTING onlemi: yalniz Liftoff ile egitirsek model genel drone
bilgisini unutur. Karisim: Liftoff train + public crops'tan ~8k rastgele altkume
(symlink). Oran ~1:2 — Liftoff azinlikta ama lr dusuk, genel bilgi korunur.
Val = YALNIZ Liftoff val (domain metrigi: sim'e uyum olcuyoruz).

Kullanim (kutuda): python3 finetune_liftoff.py
"""
import argparse, random
from pathlib import Path
from ultralytics import YOLO

HOME = Path.home() / "drone-tracker-ml"

def make_public_subset(n, seed=42):
    """public crops train'den n goruntuluk symlink altkumesi olustur."""
    src_img = HOME / "crops128/images/train"
    src_lbl = HOME / "crops128/labels/train"
    dst = HOME / "public_subset"
    di, dl = dst / "images/train", dst / "labels/train"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    if len(list(di.iterdir())) >= n:   # idempotent
        return dst
    imgs = sorted(src_img.glob("*.jpg"))
    for p in random.Random(seed).sample(imgs, min(n, len(imgs))):
        (di / p.name).symlink_to(p)
        lbl = src_lbl / (p.stem + ".txt")
        if lbl.exists():
            (dl / lbl.name).symlink_to(lbl)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-n", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    subset = make_public_subset(a.public_n)
    data_yaml = HOME / "liftoff_ft.yaml"
    data_yaml.write_text(f"""\
train:
  - {HOME}/liftoff128/images/train
  - {subset}/images/train
val: {HOME}/liftoff128/images/val
nc: 1
names: ['drone']
""")

    model = YOLO(str(HOME / "runs/yolo11_w020_crops128/weights/best.pt"))
    model.train(
        data=str(data_yaml),
        imgsz=128,
        epochs=a.epochs,
        batch=512,
        device="0,1",
        workers=16,
        lr0=0.002,         # fine-tune: tam egitimin ~1/5'i — ogrenileni ezme
        warmup_epochs=1,
        mosaic=0.0,
        scale=0.3,
        patience=15,
        project=str(HOME / "runs"),
        name="yolo11_w020_liftoff_ft",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
