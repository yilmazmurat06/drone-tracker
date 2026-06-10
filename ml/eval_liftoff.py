#!/usr/bin/env python3
"""Egitilmis 128^2 YOLO'yu Liftoff videolarinda dene (Faz 2 oncesi domain-acigi olcumu).

Ground truth = data/pseudo/flight_0X_clean.csv (insan-onayli NanoTrack kutulari).
Her GT kutusu icin: merkezli 128^2 kesit al (buyuk hedefte 1.6x büyüt+kucult,
egitimdekiyle ayni kural) -> modele sor -> en iyi tahminin GT ile IoU'su.

Metikler: detection rate (IoU>=0.3 + conf>=0.25), ortalama IoU, ortalama conf.
Ayrica kontak-foy: ornek kesitlerde GT (yesil) vs tahmin (kirmizi) gorseli.

Kullanim: /opt/anaconda3/bin/python3 ml/eval_liftoff.py [--every 10] [--conf 0.25]
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CROP = 128

def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "ml/onnx/yolo11_drone_trained.onnx"))
    ap.add_argument("--every", type=int, default=10, help="her N. GT karesini degerlendir")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sheet", default=str(ROOT / "ml/eval_liftoff_sheet.jpg"))
    a = ap.parse_args()

    model = YOLO(a.model, task="detect")
    tiles = []          # kontak-foy icin (kesit, gt, pred) ornekleri
    stats = defaultdict(lambda: [0, 0, 0.0, 0.0])  # flight -> [n, hit, iou_sum, conf_sum]

    for csv_path in sorted((ROOT / "data/pseudo").glob("flight_0*_clean.csv")):
        rows = list(csv.DictReader(open(csv_path)))
        by_flight = defaultdict(list)
        for r in rows:
            by_flight[r["flight"]].append(r)
        for flight, frs in by_flight.items():
            cap = cv2.VideoCapture(str(ROOT / "data" / f"{flight}.mp4"))
            if not cap.isOpened():
                print("video yok:", flight); continue
            for r in frs[::a.every]:
                fi = int(r["frame_idx"])
                gx, gy, gw, gh = (int(r[k]) for k in ("x", "y", "w", "h"))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, img = cap.read()
                if not ok: continue
                H, W = img.shape[:2]
                side = CROP if max(gw, gh) <= CROP * 0.75 else int(max(gw, gh) * 1.6)
                side = min(side, W, H)
                x0 = int(np.clip(gx + gw / 2 - side / 2, 0, W - side))
                y0 = int(np.clip(gy + gh / 2 - side / 2, 0, H - side))
                patch = img[y0:y0 + side, x0:x0 + side]
                s = CROP / side
                if side != CROP:
                    patch = cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)
                gt = ((gx - x0) * s, (gy - y0) * s, gw * s, gh * s)  # kesit koordinati

                res = model.predict(patch, imgsz=CROP, conf=a.conf, verbose=False)[0]
                best, best_iou, best_conf = None, 0.0, 0.0
                for b in res.boxes:
                    x0b, y0b, x1b, y1b = b.xyxy[0].tolist()
                    pb = (x0b, y0b, x1b - x0b, y1b - y0b)
                    v = iou(gt, pb)
                    if v > best_iou or best is None:
                        best, best_iou, best_conf = pb, v, float(b.conf[0])

                st = stats[flight]
                st[0] += 1
                if best is not None and best_iou >= 0.3:
                    st[1] += 1; st[2] += best_iou; st[3] += best_conf
                if len(tiles) < 120 and st[0] % 3 == 0:
                    vis = patch.copy()
                    g = tuple(int(round(v)) for v in gt)
                    cv2.rectangle(vis, (g[0], g[1]), (g[0]+g[2], g[1]+g[3]), (0, 255, 0), 1)
                    if best is not None:
                        p = tuple(int(round(v)) for v in best)
                        cv2.rectangle(vis, (p[0], p[1]), (p[0]+p[2], p[1]+p[3]), (0, 0, 255), 1)
                        cv2.putText(vis, f"{best_conf:.2f}", (2, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    tiles.append(vis)
            cap.release()

    print(f"{'ucus':<22} {'ornek':>6} {'hit':>5} {'det%':>6} {'IoU':>6} {'conf':>6}")
    tn = th = 0
    for fl, (n, hit, isum, csum) in sorted(stats.items()):
        tn += n; th += hit
        print(f"{fl:<22} {n:>6} {hit:>5} {100*hit/max(n,1):>5.1f}% "
              f"{isum/max(hit,1):>6.3f} {csum/max(hit,1):>6.3f}")
    print(f"{'TOPLAM':<22} {tn:>6} {th:>5} {100*th/max(tn,1):>5.1f}%")

    if tiles:  # kontak-foy: 12 sutunluk mozaik
        cols = 12; rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * CROP, cols * CROP, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r*CROP:(r+1)*CROP, c*CROP:(c+1)*CROP] = t
        cv2.imwrite(a.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("kontak-foy:", a.sheet)

if __name__ == "__main__":
    main()
