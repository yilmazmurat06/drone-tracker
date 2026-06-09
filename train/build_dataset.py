#!/usr/bin/env python3
"""
build_dataset.py — Temiz trajectory CSV'lerinden SiamFC eğitim crop'ları çıkar.

NEDEN ÖNCEDEN CROP? Uzak makinede (RTX 6000) eğitim yaparken video'dan rastgele
kare okumak ÇOK yavaş. Bunun yerine her hedef için tek seferlik bir "instance"
crop'u (255×255, hedef ortalı) çıkarıp JPG kaydederiz. Eğitimde:
  - template (127) = instance crop'un merkez 127'si
  - search   (255) = instance crop (rastgele kaydırma ile augment)
Böylece uzak makineye sadece küçük JPG'ler + index gider (video gerekmez).

SiamFC crop konvansiyonu (context_amount=0.5):
  wc = w + 0.5(w+h);  hc = h + 0.5(w+h);  s_z = sqrt(wc·hc)   # exemplar yan (orijinal px)
  scale = 127 / s_z
  s_x = s_z · (255/127)                                       # instance yan (orijinal px)
  → hedef merkezli s_x×s_x kare kırp, 255×255'e ölçekle, kenar taşarsa ortalama-değerle doldur.

Çıktı:
  <out>/crops/<flight>_<seq>/<frame>.jpg      (255×255 instance crop'lar)
  <out>/index.csv : flight,seq,frame,path,scale   (scale = orijinal→crop ölçeği)

Kullanım:
    python3 train/build_dataset.py data/pseudo/flight_0*_clean.csv \
        --videos data/ --out data/dataset
"""
import argparse, csv, os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXEMPLAR = 127
INSTANCE = 255
CONTEXT  = 0.5


def instance_crop(frame, box):
    """SiamFC instance crop (255×255, hedef ortalı) + orijinal→crop ölçeği döndür."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    wc = w + CONTEXT * (w + h)
    hc = h + CONTEXT * (w + h)
    s_z = (wc * hc) ** 0.5                  # exemplar yan (orijinal px)
    s_x = s_z * (INSTANCE / EXEMPLAR)       # instance yan (orijinal px)

    # ortalama renkle doldurulmuş kare bağlam kırp
    avg = frame.mean(axis=(0, 1))
    half = s_x / 2.0
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half

    H, W = frame.shape[:2]
    # taşan kenarlar için padding miktarı
    left   = int(max(0, -x0)); top    = int(max(0, -y0))
    right  = int(max(0, x1 - W)); bottom = int(max(0, y1 - H))
    if left or top or right or bottom:
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=avg.tolist())
        x0 += left; x1 += left; y0 += top; y1 += top

    crop = frame[int(round(y0)):int(round(y0)) + int(round(s_x)),
                 int(round(x0)):int(round(x0)) + int(round(s_x))]
    if crop.size == 0:
        return None, None
    crop = cv2.resize(crop, (INSTANCE, INSTANCE), interpolation=cv2.INTER_LINEAR)
    scale = INSTANCE / s_x                   # orijinal px → crop px
    return crop, scale


def resolve_video(flight, videos_dir):
    """flight adından .mp4 yolunu bul."""
    p = Path(videos_dir) / f"{flight}.mp4"
    return str(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="temiz trajectory CSV'leri")
    ap.add_argument("--videos", default="data", help="video klasörü")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--stride", type=int, default=1, help="her N karede bir crop (seyreltme)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    # tüm CSV'leri (flight, seq) → satırlar olarak grupla
    traj = defaultdict(list)
    for csv_path in args.csvs:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                key = (r["flight"], int(r["seq_id"]))
                traj[key].append((int(r["frame_idx"]), int(r["x"]), int(r["y"]),
                                  int(r["w"]), int(r["h"])))

    # video başına aç (tekrar açmamak için cache)
    caps = {}
    index_rows = []
    for (flight, seq), rows in sorted(traj.items()):
        rows.sort()
        if flight not in caps:
            vp = resolve_video(flight, args.videos)
            if not vp:
                print(f"UYARI: {flight} videosu bulunamadı, atlanıyor"); continue
            caps[flight] = cv2.VideoCapture(vp)
        cap = caps[flight]
        seq_dir = out / "crops" / f"{flight}_{seq}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for k, (fi, x, y, w, h) in enumerate(rows):
            if k % args.stride != 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            crop, scale = instance_crop(frame, (x, y, w, h))
            if crop is None:
                continue
            path = seq_dir / f"{fi}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            index_rows.append((flight, seq, fi, str(path.relative_to(out)), round(scale, 5)))
            n += 1
        print(f"{flight} seq {seq}: {n} crop")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flight", "seq", "frame", "path", "scale"])
        w.writerows(index_rows)

    print(f"\nToplam {len(index_rows)} crop → {out}/index.csv")
    print(f"(uzak makineye scp: scp -r {out} user@remote:~/drone-tracker/data/)")


if __name__ == "__main__":
    main()
