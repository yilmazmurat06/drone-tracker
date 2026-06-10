#!/usr/bin/env python3
"""Faz 2: Liftoff pseudo-label (insan-onayli clean.csv) -> 128^2 kesit dataseti.

make_crops.py'nin video+CSV varyanti. Farklar:
- Kaynak tam-kare degil VIDEO: kare seek + crop.
- TRACK-BAZLI train/val bolme: her ucusun EN YUKSEK seq_id'li track'i val'e gider.
  (Kare-bazli bolme ardisik karelerin train/val'e sizmasina yol acar = leakage.)
- DEGRADASYON AUGMENT (yalniz train, %50 olasilik): JPEG q40-70 / Gauss blur /
  parlaklik-kontrast — sahadaki (N6 kamera) bozulmayi egitimde gostermek icin.
- Negatif: ayni kareden hedef kutusundan uzak rastgele kesit (arka plan).

Kullanim (Mac'te, repo kokunde):
  python3 ml/make_liftoff_crops.py --data data --dst /tmp/liftoff128
"""
import argparse, random, sys
from collections import defaultdict
from pathlib import Path
import cv2

CROP = 128
FRAME_STRIDE = 2          # ardisik kareler ~ayni: 2'de 1 ornekle
DEGRADE_P = 0.5

def degrade(img, rng):
    """Saha bozulmasi simulasyonu: blur / jpeg / kontrast (rastgele 1-2 tanesi)."""
    ops = rng.sample(["blur", "jpeg", "contrast"], k=rng.choice([1, 2]))
    out = img
    if "blur" in ops:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if "jpeg" in ops:
        q = rng.randint(40, 70)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if "contrast" in ops:
        a = rng.uniform(0.6, 1.3); b = rng.uniform(-25, 25)
        out = cv2.convertScaleAbs(out, alpha=a, beta=b)
    return out

def crop_around(W, H, cx, cy, side, rng, jitter=0.30):
    side = min(side, W, H)
    j = side * jitter
    x0 = max(0, min(W - side, cx - side / 2 + rng.uniform(-j, j)))
    y0 = max(0, min(H - side, cy - side / 2 + rng.uniform(-j, j)))
    return int(round(x0)), int(round(y0)), int(side)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data, dst = Path(a.data), Path(a.dst).expanduser()

    # CSV'leri oku: rows[flight][seq] = [(frame_idx, x,y,w,h), ...]
    rows = defaultdict(lambda: defaultdict(list))
    for csv in sorted(data.glob("pseudo/flight_0*_clean.csv")):
        for ln in csv.read_text().splitlines()[1:]:
            p = ln.split(",")
            if len(p) < 8:
                continue
            seq, flight, fi = int(p[0]), p[1], int(p[2])
            x, y, w, h = (float(v) for v in p[3:7])
            rows[flight][seq].append((fi, x, y, w, h))

    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0, "neg": 0}
    for flight, seqs in rows.items():
        val_seq = max(seqs)   # her ucusun en yuksek seq'i val (track-bazli bolme)
        cap = cv2.VideoCapture(str(data / f"{flight}.mp4"))
        if not cap.isOpened():
            print("HATA: video yok:", flight); return 1
        # kare -> kutular (seq etiketiyle); video uzerinde TEK SIRALI gecis icin
        per_frame = defaultdict(list)
        for seq, boxes in seqs.items():
            for i, (fi, x, y, w, h) in enumerate(boxes):
                if i % FRAME_STRIDE == 0:
                    per_frame[fi].append((seq, x, y, w, h))
        for fi in sorted(per_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            H, W = img.shape[:2]
            for (seq, x, y, w, h) in per_frame[fi]:
                split = "val" if seq == val_seq else "train"
                cx, cy = x + w / 2, y + h / 2
                side = CROP if max(w, h) <= CROP * 0.75 else int(max(w, h) * 1.6)
                x0, y0, s = crop_around(W, H, cx, cy, side, rng)
                patch = img[y0:y0 + s, x0:x0 + s]
                if s != CROP:
                    patch = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)
                if split == "train" and rng.random() < DEGRADE_P:
                    patch = degrade(patch, rng)
                # kutuyu kesite normalize et (kesit disina tasani kirp)
                lx0, ly0 = max(x, x0), max(y, y0)
                lx1, ly1 = min(x + w, x0 + s), min(y + h, y0 + s)
                if lx1 - lx0 < 2 or ly1 - ly0 < 2:
                    continue
                bcx = ((lx0 + lx1) / 2 - x0) / s; bcy = ((ly0 + ly1) / 2 - y0) / s
                bw = (lx1 - lx0) / s; bh = (ly1 - ly0) / s
                name = f"{flight}_s{seq}_f{fi}"
                cv2.imwrite(str(dst / "images" / split / f"{name}.jpg"), patch,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (dst / "labels" / split / f"{name}.txt").write_text(
                    f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
                n[split] += 1
                # arka plan negatifi: hedeften uzak rastgele kesit (her 4 pozitifte 1)
                if split == "train" and n["train"] % 4 == 0:
                    for _ in range(8):  # hedefle kesismeyen bir yer ara
                        nx = rng.randint(0, W - CROP); ny = rng.randint(0, H - CROP)
                        if abs(nx + CROP/2 - cx) > CROP and abs(ny + CROP/2 - cy) > CROP:
                            npatch = img[ny:ny + CROP, nx:nx + CROP]
                            nname = f"{flight}_neg_f{fi}_{nx}"
                            cv2.imwrite(str(dst / "images/train" / f"{nname}.jpg"),
                                        npatch, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            (dst / "labels/train" / f"{nname}.txt").write_text("")
                            n["neg"] += 1
                            break
        cap.release()
        print(f"{flight}: val_seq={val_seq}")
    print(f"train={n['train']} val={n['val']} negatif={n['neg']}")
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['drone']\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
