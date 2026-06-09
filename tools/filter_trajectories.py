#!/usr/bin/env python3
"""
filter_trajectories.py — Pseudo-label trajectory'lerinden bulut-kilitlerini ele.

SORUN: NanoTrack bulutu da yüksek güvenle (0.8+) takip eder. Ham trajectory'lerin
çoğu "bulut-kilidi" = distilasyon için zehirli etiket.

ÇÖZÜM (P3 ClutterDiscriminator mantığı): gök önündeki gerçek drone KOYU + KESKİN
kenarlı + KOMPAKT; bulut-kilidi AÇIK + YUMUŞAK. İki ölçüt:
  - darkness = (ring_mean - box_mean)/255   → drone gökten koyu olduğu için pozitif
  - edge     = box içi Sobel kenar yoğunluğu → drone silüeti keskin → yüksek

Bir trajectory'nin örneklenmiş karelerinde ORTALAMA darkness ve edge eşiği geçerse
"gerçek drone" kabul edilir.

Kullanım:
    python3 tools/filter_trajectories.py data/flight_01_084727.mp4 \
        data/pseudo/flight_01_probe.csv --out data/pseudo/flight_01_clean.csv
"""
import argparse, csv, os
from collections import defaultdict

import cv2
import numpy as np


def droneness(frame_gray, box):
    """(darkness, edge) — gök önünde drone ayırt edici metrikler."""
    x, y, w, h = box
    H, W = frame_gray.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    g = frame_gray.astype(np.float32)
    patch = g[y0:y1, x0:x1]
    m = max(w, h)
    ring = g[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
    darkness = (ring.mean() - patch.mean()) / 255.0
    sob = cv2.Sobel(patch, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(patch, cv2.CV_32F, 0, 1) ** 2
    edge = float(np.sqrt(sob).mean()) / 255.0
    return darkness, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dark-thresh", type=float, default=0.02)
    ap.add_argument("--edge-thresh", type=float, default=0.12)
    ap.add_argument("--samples", type=int, default=20, help="trajectory başına örnek kare")
    args = ap.parse_args()

    path = args.video if os.path.exists(args.video) else args.video + ".mp4"

    seqs = defaultdict(list)
    header_extra = None
    with open(args.csv) as f:
        rd = csv.DictReader(f)
        for r in rd:
            seqs[int(r["seq_id"])].append(r)

    cap = cv2.VideoCapture(path)
    kept, rejected = [], []
    print(f'{"seq":>4} {"len":>4} {"dark":>7} {"edge":>6}  karar')
    for s in sorted(seqs):
        rows = seqs[s]
        idxs = np.linspace(0, len(rows) - 1, min(args.samples, len(rows))).astype(int)
        vals = []
        for i in idxs:
            r = rows[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_idx"]))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            d = droneness(g, (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])))
            if d:
                vals.append(d)
        if not vals:
            continue
        dark, edge = np.array(vals).mean(axis=0)
        ok_drone = dark >= args.dark_thresh and edge >= args.edge_thresh
        (kept if ok_drone else rejected).append(s)
        print(f'{s:>4} {len(rows):>4} {dark:>7.3f} {edge:>6.3f}  {"DRONE ✓" if ok_drone else "bulut ✗"}')

    cap.release()

    # temiz CSV yaz (sadece kabul edilen trajectory'ler)
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "flight", "frame_idx", "x", "y", "w", "h", "conf"])
        for s in kept:
            for r in seqs[s]:
                w.writerow([r["seq_id"], r["flight"], r["frame_idx"],
                            r["x"], r["y"], r["w"], r["h"], r["conf"]])
                n += 1

    print(f"\nKabul: {len(kept)} drone trajectory ({n} kare-kutu) | Red: {len(rejected)} bulut")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
