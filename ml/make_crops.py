#!/usr/bin/env python3
"""Faz 1 uyarlama: merged_dataset (5 sinif, tam kare) -> 128^2 kesit dataseti (1 sinif).

Calisma ani (N6) ile ayni dagilim: model ROI kesiti gorur, tam kare degil.
Kural (PHASE0.md): hedef <= ~96px ise native 128 kesit; daha buyukse hedefin
~1.6x'i kadar kare kesit alip 128'e kucult (buyuk hedef downscale'i tolere eder).

Sinif eslemesi:
  drone(0), plane(1), helicopter(3) -> sinif 0 "hava-hedefi" (pozitif kesit)
  bird(2), person(4)               -> ETIKETSIZ kesit (hard negative)
  empty_* goruntuler               -> rastgele etiketsiz kesit (arka plan negatifi)

Kullanim (uzak kutuda):
  python3 make_crops.py --src ~/Auto-Yolo-Training-30.04.2026/Auto-Drone-Training/merged_dataset \
                        --dst ~/drone-tracker-ml/crops128
"""
import argparse, random, sys
from pathlib import Path
import cv2

CROP = 128
POS_CLASSES = {0, 1, 3}      # drone, plane, helicopter
NEG_CLASSES = {2, 4}         # bird, person
TRAIN_CROPS_PER_TARGET = 2   # jitter cesitliligi icin ayni hedeften 2 kesit
EMPTY_CROPS_PER_IMAGE = 2
# OLCEK AUGMENTASYONU (yalniz train): kesit kenari = hedef * U[SCALE_MIN, SCALE_MAX].
# Deploy'da arama penceresi search_scale=1.4 ile baslar, miss-expansion ile ~2.8x'e
# acilir; model HER iki rejimi de gormeli. Sabit 1.6 (eski) genis pencerede conf
# cokmesine yol acti (olculdu: 2.5x pencerede ort. conf 0.38 — egitim dagilimi disi).
SCALE_MIN, SCALE_MAX = 1.2, 2.6
# ZEMIN NEGATIFI (yalniz train): her etiketli karenin ALT yarisindan (zemin/ufuk
# clutter'i) pozitifsiz kesit. Deploy'da kutu yere suruklenince model zemini
# "drone" sanmamali — zemin hard-negative bunu ogretir.
GROUND_NEG_PER_IMAGE = 1

def yolo_to_px(line, W, H):
    p = line.split()
    cls = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:5])
    return cls, cx * W, cy * H, w * W, h * H

def crop_around(img, cx, cy, side, rng):
    """Hedef etrafinda 'side' boyutlu kare kesit; jitter ile merkezden kaydirilir.
    Kesit kare disina tasarsa icine itilir (padding yok, gercek piksel)."""
    H, W = img.shape[:2]
    side = min(side, W, H)
    # jitter: hedef kesit icinde rastgele bir yerde dursun (kenara cok yapismasin)
    jmax = side * 0.30
    x0 = cx - side / 2 + rng.uniform(-jmax, jmax)
    y0 = cy - side / 2 + rng.uniform(-jmax, jmax)
    x0 = max(0, min(W - side, x0)); y0 = max(0, min(H - side, y0))
    return int(round(x0)), int(round(y0)), int(side)

def boxes_in_crop(boxes, x0, y0, side):
    """Kesit icinde kalan pozitif kutulari kesit-normalize YOLO satirina cevir.
    Merkezi kesit icindeyse al; kutuyu kesit sinirina kirp."""
    out = []
    for (cx, cy, w, h) in boxes:
        if not (x0 <= cx < x0 + side and y0 <= cy < y0 + side):
            continue
        lx0 = max(cx - w / 2, x0); ly0 = max(cy - h / 2, y0)
        lx1 = min(cx + w / 2, x0 + side); ly1 = min(cy + h / 2, y0 + side)
        bw, bh = lx1 - lx0, ly1 - ly0
        if bw < 2 or bh < 2:
            continue  # kirpinca yok olan kutu
        bcx = (lx0 + lx1) / 2 - x0; bcy = (ly0 + ly1) / 2 - y0
        out.append(f"0 {bcx/side:.6f} {bcy/side:.6f} {bw/side:.6f} {bh/side:.6f}")
    return out

def process_split(src, dst, split, rng):
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    out_img = dst / "images" / split; out_img.mkdir(parents=True, exist_ok=True)
    out_lbl = dst / "labels" / split; out_lbl.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = n_empty = 0
    crops_per_target = TRAIN_CROPS_PER_TARGET if split == "train" else 1

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lines = lbl_path.read_text().strip().splitlines() if lbl_path.exists() else []
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        pos, neg = [], []
        for ln in lines:
            if not ln.strip():
                continue
            cls, cx, cy, w, h = yolo_to_px(ln, W, H)
            (pos if cls in POS_CLASSES else neg if cls in NEG_CLASSES else []).append((cx, cy, w, h))

        def save(x0, y0, side, label_lines, tag, idx):
            nonlocal n_pos, n_neg, n_empty
            patch = img[y0:y0 + side, x0:x0 + side]
            if side != CROP:
                patch = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)
            name = f"{img_path.stem}_{tag}{idx}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_lbl / f"{name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
            if tag == "p": n_pos += 1
            elif tag == "n": n_neg += 1
            else: n_empty += 1

        for i, (cx, cy, w, h) in enumerate(pos):
            for j in range(crops_per_target):
                # train: olcek augment (deploy pencere araligi); val: sabit 1.6
                # (deterministik metrik). Kucuk hedefte taban native 128.
                scale = rng.uniform(SCALE_MIN, SCALE_MAX) if split == "train" else 1.6
                side = max(CROP, int(max(w, h) * scale))
                x0, y0, s = crop_around(img, cx, cy, side, rng)
                lab = boxes_in_crop(pos, x0, y0, s)
                if lab:
                    save(x0, y0, s, lab, "p", i * 10 + j)

        for i, (cx, cy, w, h) in enumerate(neg):
            side = CROP if max(w, h) <= CROP * 0.75 else int(max(w, h) * 1.6)
            x0, y0, s = crop_around(img, cx, cy, side, rng)
            # icine pozitif dusmusse atla — negatif kesit SAF olmali
            if boxes_in_crop(pos, x0, y0, s):
                continue
            save(x0, y0, s, [], "n", i)

        if not pos and not neg:  # bos arka plan goruntusu
            for i in range(EMPTY_CROPS_PER_IMAGE):
                x0 = rng.randint(0, max(0, W - CROP)); y0 = rng.randint(0, max(0, H - CROP))
                save(x0, y0, min(CROP, W, H), [], "e", i)

        # zemin hard-negative: etiketli karelerin alt yarisindan pozitifsiz kesit
        # (train; bkz. GROUND_NEG_PER_IMAGE). Olcek de rastgele — deploy penceresi gibi.
        if split == "train" and pos:
            for i in range(GROUND_NEG_PER_IMAGE):
                s = min(int(CROP * rng.uniform(1.0, 2.6)), W, H)
                x0 = rng.randint(0, max(0, W - s))
                y0 = rng.randint(H // 2, max(H // 2, H - s)) if H - s >= H // 2 else max(0, H - s)
                if boxes_in_crop(pos, x0, y0, s):
                    continue  # alt yarida pozitif varsa atla — negatif SAF olmali
                save(x0, y0, s, [], "g", i)

    print(f"[{split}] pozitif={n_pos} negatif={n_neg} bos={n_empty}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    rng = random.Random(a.seed)
    for split in ("train", "val"):
        process_split(src, dst, split, rng)
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['drone']\n")
    print("data.yaml yazildi:", dst / "data.yaml")

if __name__ == "__main__":
    sys.exit(main())
