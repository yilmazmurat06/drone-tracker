#!/usr/bin/env python3
"""
viz_trajectories.py — Pseudo-label trajectory'lerini videoya bas (kalite denetimi).

Her seq_id farklı renkte kutu olarak çizilir. Amaç: NanoTrack gerçekten drone mu
takip ediyor yoksa buluta/dokuya mı kilitlenmiş, GÖZLE doğrulamak.

Kullanım:
    python3 tools/viz_trajectories.py data/flight_01_084727.mp4 \
        data/pseudo/flight_01_probe.csv --out viz_probe.mp4
"""
import argparse, csv, os
from collections import defaultdict
import cv2


def color_for(seq):
    import colorsys
    h = (seq * 0.61803398875) % 1.0           # altın oran → ayrık renkler
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--out", default="viz.mp4")
    args = ap.parse_args()

    path = args.video if os.path.exists(args.video) else args.video + ".mp4"
    rows_by_frame = defaultdict(list)
    fmin, fmax = 10**9, -1
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            fi = int(r["frame_idx"])
            rows_by_frame[fi].append((int(r["seq_id"]), int(r["x"]), int(r["y"]),
                                      int(r["w"]), int(r["h"]), float(r["conf"])))
            fmin, fmax = min(fmin, fi), max(fmax, fi)

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fmin)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fi in range(fmin, fmax + 1):
        ok, frame = cap.read()
        if not ok:
            break
        for (seq, x, y, bw, bh, conf) in rows_by_frame.get(fi, []):
            col = color_for(seq)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(frame, f"#{seq} {conf:.2f}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(frame, f"frame {fi}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(frame)

    cap.release(); vw.release()
    print(f"Yazıldı: {args.out}  (kare {fmin}-{fmax})")


if __name__ == "__main__":
    main()
