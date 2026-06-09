#!/usr/bin/env python3
"""Extract sample frames from specific trajectory sequences for visual inspection."""
import argparse, csv, os
from collections import defaultdict
import cv2
import numpy as np


def droneness(frame_gray, box):
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
    ap.add_argument("--seqs", required=True, help="comma-separated seq_ids")
    ap.add_argument("--out-dir", default="data/pseudo/viz_samples")
    ap.add_argument("--n-frames", type=int, default=5, help="samples per trajectory")
    args = ap.parse_args()

    target_seqs = {int(s) for s in args.seqs.split(",")}
    os.makedirs(args.out_dir, exist_ok=True)

    path = args.video if os.path.exists(args.video) else args.video + ".mp4"
    flight = os.path.splitext(os.path.basename(path))[0]

    seq_rows = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            sid = int(r["seq_id"])
            if sid in target_seqs:
                seq_rows[sid].append(r)

    cap = cv2.VideoCapture(path)

    for sid in sorted(target_seqs):
        rows = seq_rows.get(sid)
        if not rows:
            print(f"seq {sid}: CSV'de bulunamadi")
            continue

        n = len(rows)
        idxs = np.linspace(0, n - 1, args.n_frames).astype(int)

        vals = []
        for i, idx in enumerate(idxs):
            r = rows[idx]
            fi = int(r["frame_idx"])
            x, y, w_box, h_box = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            conf = float(r["conf"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                print(f"  frame {fi}: okunamadi")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            d = droneness(gray, (x, y, w_box, h_box))

            col = (0, 255, 0)  # green
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), col, 2)
            label = f"#{sid} c={conf:.2f}"
            if d:
                label += f" D={d[0]:.3f} E={d[1]:.3f}"
                vals.append(d)
            cv2.putText(frame, label, (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            cv2.putText(frame, f"{flight} frame {fi}  sample {i+1}/{args.n_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out_path = os.path.join(args.out_dir, f"{flight}_seq{sid}_f{fi}_s{i}.jpg")
            cv2.imwrite(out_path, frame)

        if vals:
            d_avg = np.mean([v[0] for v in vals])
            e_avg = np.mean([v[1] for v in vals])
            print(f"seq {sid}: {n} kare, {len(idxs)} sample | D={d_avg:.3f} E={e_avg:.3f}")

    cap.release()
    print(f"\nDone → {args.out_dir}/")


if __name__ == "__main__":
    main()
