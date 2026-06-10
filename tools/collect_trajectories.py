#!/usr/bin/env python3
"""
collect_trajectories.py — Distilasyon için pseudo-label üretici (Faz 3, Adım 1).

FİKİR: Çalışan NanoTrack'i "öğretmen" olarak kullan. Uçuş videolarını gez,
hareketli adaylar bul, her birine bir NanoTrack tohumla, ileri takip et,
GÜVEN yüksek kaldığı sürece kaydet. Çıkan trajectory'ler (kare → kutu) küçük
SiamFC "öğrenci" modelinin eğitim etiketleri olur.

ÖNEMLİ: Tüm koordinatlar HAM kare uzayında (stabilizasyon YOK). NanoTrack
görünüm-tabanlı çalıştığı için kamera hareketine dayanıklı; stabilizasyona
ihtiyaç duymadan ham karede takip eder → kırpma (crop) tutarlı kalır.

Tohum bulucu (seeder): basit ego-hareket telafili kare-farkı.
  - prev→cur global hareketi LK optik akış + affine ile kestir
  - prev'i cur'a warp et, farkı al, eşikle → hareketli bloblar = aday tohum

Kullanım:
    python3 tools/collect_trajectories.py data/flight_01_084727.mp4 \
        --out data/pseudo/flight_01.csv --exclude 3084:3504

    # 3 uçuşu birden:
    for f in data/flight_*.mp4; do
        python3 tools/collect_trajectories.py "$f" --out "data/pseudo/$(basename ${f%.mp4}).csv" ...
    done

Çıktı CSV:
    seq_id,flight,frame_idx,x,y,w,h,conf
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

# Kanıtlanmış geometri kapıları (lock-integrity işinden): tohum adayının çevre halkası
# ve ALTI gök olmalı — ağaç tepesi/zemin tohumunu kaynağında keser.
from filter_trajectories import sky_ring, sky_below


# ── ego-hareket telafili hareket tespiti (tohum bulucu) ─────────────────────────

def estimate_global_motion(prev_gray, cur_gray):
    """prev→cur kısmi-affine (öteleme+dönme+ölçek) matrisi. Bulamazsa None."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                  minDistance=12, blockSize=7)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
    if nxt is None:
        return None
    good_prev = pts[st.ravel() == 1]
    good_nxt  = nxt[st.ravel() == 1]
    if len(good_prev) < 12:
        return None
    M, _ = cv2.estimateAffinePartial2D(good_prev, good_nxt,
                                       method=cv2.RANSAC, ransacReprojThreshold=3)
    return M


def detect_motion_boxes(prev_gray, cur_gray, cur_bgr, seed_sky, seed_below,
                        min_area, max_area):
    """
    Ego-hareket telafili kare-farkıyla hareketli aday kutular bul.
    GÖK KAPISI (eski sabit sky_frac çizgisi YANLIŞTI: FPV'de kamera yatınca ağaç/zemin
    karenin üstüne girer → ağaç tohumlanırdı). Doğru test piksel geometrisi:
      - sky_ring(box)  ≥ seed_sky   → çevre halkası gök (zemin/ufuk tohumunu eler)
      - sky_below(box) ≥ seed_below → kutunun ALTI gök (göğe uzanan ağaç tepesini eler)
    """
    M = estimate_global_motion(prev_gray, cur_gray)
    if M is None:
        return []
    h, w = cur_gray.shape
    warped_prev = cv2.warpAffine(prev_gray, M, (w, h))

    diff = cv2.absdiff(cur_gray, warped_prev)
    # warp kenar artefaktlarını kırp (geçersiz bölge)
    diff[:4, :] = 0; diff[-4:, :] = 0; diff[:, :4] = 0; diff[:, -4:] = 0

    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        ar = bw / max(1, bh)
        if ar > 6 or ar < 1 / 6:        # aşırı uzun şeritler (bulut kenarı) → ele
            continue
        box = (x, y, bw, bh)
        if sky_ring(cur_bgr, box) < seed_sky:      # çevresi gök değil → zemin/ufuk
            continue
        if sky_below(cur_bgr, box) < seed_below:   # altı gök değil → ağaç tepesi
            continue
        boxes.append(box)

    # Dedupe (örtüşenleri birleştir) + alana göre azalan sırala: en belirgin
    # bloblar (gerçek hedef olma olasılığı yüksek) önce denensin. Top-N ile sınırla.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(iou(b, k) < 0.3 for k in kept):
            kept.append(b)
        if len(kept) >= 15:
            break
    return kept


# ── kutu örtüşmesi (aktif tracker'larla çakışmayı önle) ─────────────────────────

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── ana hasat döngüsü ───────────────────────────────────────────────────────────

def make_tracker(backbone, neckhead):
    p = cv2.TrackerNano.Params()
    p.backbone = backbone
    p.neckhead = neckhead
    return cv2.TrackerNano.create(p)


def main():
    ap = argparse.ArgumentParser(description="NanoTrack ile pseudo-label trajectory üretici")
    ap.add_argument("video", help=".mp4 dosyası (veya extension'sız prefix)")
    ap.add_argument("--out", required=True, help="Çıktı CSV")
    ap.add_argument("--backbone", default="models/nanotrack_backbone_sim.onnx")
    ap.add_argument("--neckhead", default="models/nanotrack_head_sim.onnx")
    ap.add_argument("--seed-every", type=int, default=15,
                    help="Her N karede bir yeni tohum ara")
    ap.add_argument("--max-active", type=int, default=6,
                    help="Aynı anda en fazla aktif tracker")
    ap.add_argument("--min-len", type=int, default=30,
                    help="Bu kareden kısa trajectory'leri at")
    ap.add_argument("--keep-conf", type=float, default=0.55,
                    help="Takip güveni bunun altına düşerse trajectory'yi bitir")
    ap.add_argument("--seed-sky", type=float, default=0.60,
                    help="Tohum adayı çevre-halka gök eşiği (zemin/ufuk reddi)")
    ap.add_argument("--seed-below", type=float, default=0.50,
                    help="Tohum adayı ALT şerit gök eşiği (ağaç-tepesi reddi)")
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--max-area", type=int, default=20000)
    ap.add_argument("--start", type=int, default=0, help="Bu kareden başla (seek)")
    ap.add_argument("--max-frames", type=int, default=0, help="0=tümü (start'tan itibaren)")
    ap.add_argument("--exclude", default="",
                    help="Eğitime girmemesi gereken kare aralığı 'a:b' (TEST seti). "
                         "Bu aralıkta tohum atılmaz.")
    args = ap.parse_args()

    # .mp4 yoksa ekle
    path = args.video if os.path.exists(args.video) else args.video + ".mp4"
    flight = Path(path).stem

    excl_a, excl_b = -1, -1
    if args.exclude:
        excl_a, excl_b = (int(v) for v in args.exclude.split(":"))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Video açılamadı: {path}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    active = []      # her biri: dict(tracker, seq_id, rows=[(frame_idx,x,y,w,h,conf)])
    finished = []    # tamamlanan (min-len geçen) trajectory'ler
    next_seq = 0
    prev_gray = None
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = args.start - 1
    end_frame = (args.start + args.max_frames) if args.max_frames else None

    def in_excl(fi):
        return excl_a <= fi <= excl_b

    def finalize(trk):
        if len(trk["rows"]) >= args.min_len:
            finished.append(trk)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if end_frame and frame_idx >= end_frame:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0) TEST aralığı koruması: bu karelerde NE kayıt NE tohum. Aktif olanları
        # bitir (test karelerini kaydetmesinler → veri sızıntısı önlenir).
        if in_excl(frame_idx):
            for trk in active:
                finalize(trk)
            active = []
            prev_gray = gray
            continue

        # 1) aktif tracker'ları güncelle
        still = []
        for trk in active:
            ok2, box = trk["tracker"].update(frame)
            score = float(trk["tracker"].getTrackingScore())
            x, y, bw, bh = [int(v) for v in box]
            if ok2 and score >= args.keep_conf and bw > 2 and bh > 2:
                trk["rows"].append((frame_idx, x, y, bw, bh, round(score, 4)))
                still.append(trk)
            else:
                finalize(trk)
        active = still

        # 2) periyodik yeni tohum
        if prev_gray is not None and frame_idx % args.seed_every == 0 \
                and len(active) < args.max_active and not in_excl(frame_idx):
            cands = detect_motion_boxes(prev_gray, gray, frame,
                                        args.seed_sky, args.seed_below,
                                        args.min_area, args.max_area)
            # mevcut aktiflere yakın olmayanları tohumla
            active_boxes = [t["rows"][-1][1:5] for t in active if t["rows"]]
            for box in cands:
                if len(active) >= args.max_active:
                    break
                if any(iou(box, ab) > 0.2 for ab in active_boxes):
                    continue
                trk = make_tracker(args.backbone, args.neckhead)
                trk.init(frame, box)   # NOT: init sonrası score=0; anlamlı skor
                                       # ilk update()'te gelir → rows boş başlar,
                                       # kayıt update döngüsünde yapılır. Gürültü
                                       # tohumu 1 karede ölüp min-len altı elenir.
                active.append({"tracker": trk, "seq_id": next_seq, "rows": []})
                active_boxes.append(box)
                next_seq += 1

        prev_gray = gray

    # kalan aktifleri bitir
    for trk in active:
        finalize(trk)

    cap.release()

    # CSV yaz
    n_rows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "flight", "frame_idx", "x", "y", "w", "h", "conf"])
        for trk in finished:
            for (fi, x, y, bw, bh, sc) in trk["rows"]:
                w.writerow([trk["seq_id"], flight, fi, x, y, bw, bh, sc])
                n_rows += 1

    print(f"{flight}: {len(finished)} trajectory, {n_rows} kare-kutu → {args.out}")
    if finished:
        lens = [len(t["rows"]) for t in finished]
        print(f"  uzunluk: min={min(lens)} ort={sum(lens)//len(lens)} max={max(lens)}")


if __name__ == "__main__":
    main()
