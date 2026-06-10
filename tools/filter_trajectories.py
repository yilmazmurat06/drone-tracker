#!/usr/bin/env python3
"""
filter_trajectories.py — Pseudo-label trajectory'lerinden YER ve BULUT kilitlerini ele.

ESKİ EKSEN (darkness+edge) NEDEN ÇÖPE GİTTİ: yerdeki ağaç/tribün de koyu + keskin
kenarlı → filtre onları "drone" sanıp kabul ediyordu (gözle doğrulandı: 30 kabulün
~%67'si yer kilidiydi) ve ham havuzdaki ~38 gerçek hava-hedefinin ~28'ini atıyordu.

YENİ EKSENLER (lock-integrity işinde kanıtlandı):
  1. sky_ring  — kutu ÇEVRE HALKASININ gök oranı (B>=G ∪ parlak). GEOMETRİ ayırır:
                 yer kilidi %12–50, gerçek hava-hedefi %85–99. Ana ayraç.
  2. edge      — kutu içi Sobel kenar yoğunluğu. Bulut da gök-çevrelidir ama YUMUŞAK;
                 drone silüeti keskin. Bulut reddinin tek ayracı (sky tek başına yetmez).
  3. boyut     — KARE-BAZLI: kenar>400px ya da alan>kare alanının %8'i olan kareler
                 NanoTrack patlamasıdır → o KARELER atılır (trajectory değil); kalan
                 uzunluk min-len altına düşerse trajectory atılır.

Kullanım:
    python3 tools/filter_trajectories.py data/flight_01_084727.mp4 \
        data/pseudo/flight_01_raw.csv --out data/pseudo/flight_01_clean.csv
"""
import argparse, csv, os
from collections import defaultdict

import cv2
import numpy as np


def sky_ring(frame, box, bright=180):
    """Kutu çevre halkasının gök oranı [0,1] (iç kutu hariç). C++ lock_integrity portu."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    m = max(w, h)
    ox0, oy0 = max(0, x - m), max(0, y - m)
    ox1, oy1 = min(W, x + w + m), min(H, y + h + m)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0

    def count_sky(x0, y0, x1, y1):
        reg = frame[y0:y1, x0:x1].astype(np.int32)
        if reg.size == 0:
            return 0
        B, G, R = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
        return int((((B >= G) | ((B + G + R) / 3 > bright))).sum())

    ix0, iy0, ix1, iy1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    outer_area = (ox1 - ox0) * (oy1 - oy0)
    inner_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    sky_outer = count_sky(ox0, oy0, ox1, oy1)
    sky_inner = count_sky(ix0, iy0, ix1, iy1) if inner_area > 0 else 0
    denom = outer_area - inner_area
    if denom <= 0:
        return sky_outer / outer_area if outer_area > 0 else 0.0
    return (sky_outer - sky_inner) / denom


def sky_below(frame, box, bright=180):
    """Kutunun HEMEN ALTINDAKI şeridin gök oranı. Açık gökteki drone'un altı GÖK;
    ağaç tepesinin altı gövde/zemin — halka testinin kaçırdığı tepe-kilidini yakalar."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    y0, y1 = min(H, y + h), min(H, y + 2 * h)
    x0, x1 = max(0, x), min(W, x + w)
    if y1 <= y0 or x1 <= x0:
        return 1.0  # kare altına taşıyor → ölçülemez, veto etme
    reg = frame[y0:y1, x0:x1].astype(np.int32)
    B, G, R = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
    return float((((B >= G) | ((B + G + R) / 3 > bright))).mean())


def edge_density(frame_gray, box):
    """Kutu içi Sobel kenar yoğunluğu [0,~1]. Drone keskin, bulut yumuşak."""
    H, W = frame_gray.shape
    x, y, w, h = box
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = frame_gray[y0:y1, x0:x1].astype(np.float32)
    sob = cv2.Sobel(patch, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(patch, cv2.CV_32F, 0, 1) ** 2
    return float(np.sqrt(sob).mean()) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sky-thresh", type=float, default=0.85,
                    help="trajectory ortalama gök-çevre eşiği (yer kilidi reddi)")
    ap.add_argument("--below-thresh", type=float, default=0.60,
                    help="kutu ALTI şerit gök eşiği (ağaç-tepesi kilidi reddi)")
    ap.add_argument("--edge-thresh", type=float, default=0.10,
                    help="trajectory ortalama kenar eşiği (bulut reddi)")
    ap.add_argument("--max-side", type=int, default=400,
                    help="kare-bazlı: kutu kenarı bunu aşarsa o kare atılır (patlama)")
    ap.add_argument("--max-area-frac", type=float, default=0.08,
                    help="kare-bazlı: kutu alanı kare alanının bu oranını aşarsa atılır")
    ap.add_argument("--min-len", type=int, default=25,
                    help="boyut temizliği sonrası bu kareden kısa trajectory atılır")
    ap.add_argument("--samples", type=int, default=20, help="trajectory başına örnek kare")
    args = ap.parse_args()

    path = args.video if os.path.exists(args.video) else args.video + ".mp4"

    seqs = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            seqs[int(r["seq_id"])].append(r)

    cap = cv2.VideoCapture(path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_area = args.max_area_frac * fw * fh

    kept, rejected = [], []
    kept_rows = {}
    print(f'{"seq":>4} {"len":>5} {"sky":>6} {"edge":>6}  karar')
    for s in sorted(seqs):
        rows = seqs[s]
        # 1) KARE-BAZLI boyut temizliği: patlamış kutular trajectory'den kırpılır.
        sane = [r for r in rows
                if int(r["w"]) <= args.max_side and int(r["h"]) <= args.max_side
                and int(r["w"]) * int(r["h"]) <= max_area]
        if len(sane) < args.min_len:
            rejected.append(s)
            print(f'{s:>4} {len(rows):>5} {"-":>6} {"-":>6}  kısa/patlak ✗')
            continue

        # 2) Örneklenen karelerde sky-ring + edge ölç (per-sample sakla — kuyruk kırpma için).
        idxs = np.linspace(0, len(sane) - 1, min(args.samples, len(sane))).astype(int)
        metrics = []  # (sane_index, sky, edge)
        for i in idxs:
            r = sane[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_idx"]))
            ok, fr = cap.read()
            if not ok:
                continue
            box = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            metrics.append((int(i), sky_ring(fr, box), sky_below(fr, box),
                            edge_density(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), box)))
        if not metrics:
            rejected.append(s)
            continue

        # 3) KUYRUK KIRPMA: NanoTrack trajectory sonunda buluta/yere sürüklenir (güveni
        # yüksek kalır → hasat kesmez). ÜÇ kapıyı da (halka + alt-şerit + kenar) geçen
        # SON örnekte trajectory'yi KES; sürüklenme kuyruğu eğitime girmez.
        def passes(m):
            return (m[1] >= args.sky_thresh and m[2] >= args.below_thresh
                    and m[3] >= args.edge_thresh)
        last_ok = -1
        for k, m in enumerate(metrics):
            if passes(m):
                last_ok = k
        if last_ok < 0:
            rejected.append(s)
            print(f'{s:>4} {len(sane):>5} {"-":>6} {"-":>6} {"-":>6}  hiç geçen örnek yok ✗')
            continue
        cut = len(sane) if last_ok == len(metrics) - 1 else metrics[last_ok][0] + 1
        trimmed = sane[:cut]
        used = metrics[:last_ok + 1]
        sky_m   = float(np.mean([m[1] for m in used]))
        below_m = float(np.mean([m[2] for m in used]))
        edge_m  = float(np.mean([m[3] for m in used]))

        ok_air = (sky_m >= args.sky_thresh and below_m >= args.below_thresh
                  and edge_m >= args.edge_thresh and len(trimmed) >= args.min_len)
        if ok_air:
            kept.append(s)
            kept_rows[s] = trimmed       # patlama-kırpık + kuyruk-kırpık kareler
        else:
            rejected.append(s)
        if not ok_air:
            neden = ("yer ✗" if sky_m < args.sky_thresh else
                     "tepe ✗" if below_m < args.below_thresh else "bulut/kısa ✗")
        else:
            neden = "HAVA ✓"
        kuyruk = f" (kuyruk -{len(sane)-len(trimmed)})" if ok_air and cut < len(sane) else ""
        print(f'{s:>4} {len(trimmed):>5} {sky_m:>6.2f} {below_m:>6.2f} {edge_m:>6.3f}  {neden}{kuyruk}')

    cap.release()

    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "flight", "frame_idx", "x", "y", "w", "h", "conf"])
        for s in kept:
            for r in kept_rows[s]:
                w.writerow([r["seq_id"], r["flight"], r["frame_idx"],
                            r["x"], r["y"], r["w"], r["h"], r["conf"]])
                n += 1

    print(f"\nKabul: {len(kept)} hava trajectory ({n} kare-kutu) | Red: {len(rejected)}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
