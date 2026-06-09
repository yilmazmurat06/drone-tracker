#!/usr/bin/env python3
"""
label_gt.py — Drone tracker için ground-truth bbox etiketleme aracı.

Kullanım:
    python3 tools/label_gt.py data/flight_01_084727.mp4 --start 3080 --end 3500 --out gt_flight01.csv

Kontroller (frame gösterilirken):
    SOL-TIKLAYIP-SÜR  : bbox çiz
    SPACE             : aynı bbox'ı bir sonraki kareye kopyala (hedef düz gidiyorsa)
    s                 : bu kareyi atla (GT yok → satır yazılmaz)
    b                 : bir önceki kareye geri dön
    q / ESC           : kaydet ve çık

Çıktı CSV formatı (header dahil):
    frame_idx,x,y,w,h
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2


# ── çizim durumu ──────────────────────────────────────────────────────────────
_drawing = False
_ix, _iy = -1, -1
_rect    = None   # (x, y, w, h) — son çizilen

def _mouse_cb(event, x, y, flags, param):
    global _drawing, _ix, _iy, _rect
    frame_disp = param["frame"]
    scale      = param["scale"]

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _ix, _iy = x, y
        _rect = None

    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        tmp = frame_disp.copy()
        cv2.rectangle(tmp, (_ix, _iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("label_gt", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        x1, y1 = min(_ix, x), min(_iy, y)
        x2, y2 = max(_ix, x), max(_iy, y)
        if x2 - x1 > 2 and y2 - y1 > 2:
            # ekran koordinatlarını → orijinal frame koordinatlarına çevir
            _rect = (int(x1 / scale), int(y1 / scale),
                     int((x2 - x1) / scale), int((y2 - y1) / scale))
        tmp = frame_disp.copy()
        cv2.rectangle(tmp, (_ix, _iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("label_gt", tmp)


def _draw_hud(frame_disp, frame_idx, rect, scale, total):
    """HUD bilgisi: kare numarası, mevcut bbox, yönergeler."""
    h_d, w_d = frame_disp.shape[:2]
    overlay = frame_disp.copy()

    # bilgi kutusu arka planı
    cv2.rectangle(overlay, (0, 0), (w_d, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_disp, 0.5, 0, frame_disp)

    box_txt = f"bbox=({rect[0]},{rect[1]},{rect[2]},{rect[3]})" if rect else "bbox=--"
    cv2.putText(frame_disp, f"frame {frame_idx}  {box_txt}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame_disp, "DRAG:çiz  SPACE:kopyala  s:atla  b:geri  q:kaydet&çık",
                (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # mevcut bbox varsa göster
    if rect:
        x, y, w, h = rect
        sx, sy = int(x * scale), int(y * scale)
        sw, sh = int(w * scale), int(h * scale)
        cv2.rectangle(frame_disp, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)


def main():
    global _rect

    ap = argparse.ArgumentParser(description="Ground-truth bbox labeling aracı")
    ap.add_argument("video",  help="MP4 dosyası")
    ap.add_argument("--start", type=int, default=0,    help="Başlangıç karesi (dahil)")
    ap.add_argument("--end",   type=int, default=-1,   help="Bitiş karesi (dahil, -1=son)")
    ap.add_argument("--out",   default="gt.csv",       help="Çıktı CSV dosyası")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="Ekran ölçeği (0=otomatik)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Video açılamadı: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame    = total_frames - 1 if args.end < 0 else min(args.end, total_frames - 1)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ekran ölçeği: 1080p ekranda 1440px genişlik → ~0.67
    if args.scale <= 0:
        scale = min(1.0, 1280 / vid_w, 720 / vid_h)
    else:
        scale = args.scale

    print(f"Video: {args.video}  ({vid_w}x{vid_h}, {total_frames} kare)")
    print(f"Aralık: {args.start}–{end_frame}  Ölçek: {scale:.2f}")
    print(f"Çıktı: {args.out}")

    # mevcut CSV varsa yükle (devam modu)
    existing = {}
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[int(row["frame_idx"])] = (
                    int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
        print(f"  → {len(existing)} mevcut etiket yüklendi (devam modu)")

    cv2.namedWindow("label_gt", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label_gt", int(vid_w * scale), int(vid_h * scale))

    labels = dict(existing)  # frame_idx → (x,y,w,h)
    idx    = args.start
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Kare {idx} okunamadı.")

    param = {"frame": None, "scale": scale}

    while True:
        frame_disp = cv2.resize(frame, (int(vid_w * scale), int(vid_h * scale)))
        # önceki etiket varsa başlangıç olarak al
        if idx in labels:
            _rect = labels[idx]
        # aynı kareden devam ediliyorsa _rect korunur

        param["frame"] = frame_disp
        cv2.setMouseCallback("label_gt", _mouse_cb, param)

        _draw_hud(frame_disp, idx, _rect, scale, end_frame - args.start + 1)
        cv2.imshow("label_gt", frame_disp)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):  # q veya ESC → kaydet çık
            if _rect:
                labels[idx] = _rect
            break

        elif key == ord(' '):  # SPACE → kopyala ve ilerle
            if _rect:
                labels[idx] = _rect
            if idx >= end_frame:
                print("Son kareye ulaşıldı.")
                break
            idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break

        elif key == ord('s'):  # atla (GT yok)
            labels.pop(idx, None)
            if idx >= end_frame:
                break
            idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            _rect = labels.get(idx)

        elif key == ord('b'):  # geri
            if _rect:
                labels[idx] = _rect
            idx = max(args.start, idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            _rect = labels.get(idx)

    cap.release()
    cv2.destroyAllWindows()

    # CSV'ye yaz (sıralı)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "x", "y", "w", "h"])
        for fi in sorted(labels):
            w.writerow([fi, *labels[fi]])

    print(f"\nKaydedildi: {out_path}  ({len(labels)} etiket)")


if __name__ == "__main__":
    main()
