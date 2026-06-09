#!/usr/bin/env python3
"""
eval_tracker.py — Tracker dump çıktısını ground-truth ile karşılaştır.

Kullanım:
    # Önce her tracker için dump al:
    ./build/dtrack_track data/flight_01_084727.mp4 --siamese \
        --auto-lock 3080 --start 3080 --max-frames 420 --dump > dump_nano.txt
    ./build/dtrack_track data/flight_01_084727.mp4 \
        --auto-lock 3080 --start 3080 --max-frames 420 --dump > dump_ncc.txt

    # Sonra karşılaştır:
    python3 tools/eval_tracker.py gt.csv dump_nano.txt dump_ncc.txt

    # İsimle:
    python3 tools/eval_tracker.py gt.csv dump_nano.txt:NanoTrack dump_ncc.txt:NCC

Çıktı:
    - Her tracker için per-threshold success oranı (tablo)
    - Opsiyonel: matplotlib ile success plot (--plot)

Dump formatı (track.cpp --dump):
    <frame>  <STATE>  conf=<f>  aday=<n>  box=[x,y,w,h]
    Örn:  3081  TRACK   conf=0.671  aday= 0  box=[1035,586,10,10]

GT CSV formatı:
    frame_idx,x,y,w,h
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# ── yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def load_gt(path: str) -> dict:
    """GT CSV → {frame_idx: (x, y, w, h)}"""
    gt = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_idx"])
            gt[fi] = (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
    return gt


# dump satırı regex: "  3081  TRACK   conf=0.671  aday= 0  box=[1035,586,10,10]"
_DUMP_RE = re.compile(
    r"^\s*(\d+)\s+(\w+)\s+conf=([\d.]+)\s+aday=\s*\d+\s+box=\[(\d+),(\d+),(\d+),(\d+)\]"
)

def load_dump(path: str) -> dict:
    """tracker dump → {frame_idx: (x, y, w, h, state, conf)}"""
    data = {}
    with open(path) as f:
        for line in f:
            m = _DUMP_RE.match(line)
            if m:
                fi = int(m.group(1))
                state = m.group(2)
                conf  = float(m.group(3))
                x, y, w, h = int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
                data[fi] = (x, y, w, h, state, conf)
    return data


def iou(a, b) -> float:
    """(x,y,w,h) çiftleri için IoU."""
    ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
    bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b) -> float:
    """İki (x,y,w,h) kutusunun merkez mesafesi (piksel)."""
    cx_a = a[0] + a[2] / 2
    cy_a = a[1] + a[3] / 2
    cx_b = b[0] + b[2] / 2
    cy_b = b[1] + b[3] / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def evaluate(gt: dict, dump: dict):
    """
    GT ile eşleşen TRACK/SUSPECT kareler üzerinde IoU + merkez uzaklığı hesapla.
    Sadece gt AND dump'ta var olan ve dump'ta w,h>0 olan kareler değerlendirilir.
    Döndürür: (ious, dists, n_skip)  — list[float], list[float], int
    """
    common = sorted(set(gt.keys()) & set(dump.keys()))
    ious  = []
    dists = []
    n_skip = 0
    for fi in common:
        pred = dump[fi]
        if pred[2] <= 0 or pred[3] <= 0:   # w veya h sıfır → tracker yok
            n_skip += 1
            continue
        ious.append(iou(gt[fi], pred[:4]))
        dists.append(center_dist(gt[fi], pred[:4]))
    return ious, dists, n_skip


def success_rate(ious, threshold) -> float:
    if not ious:
        return 0.0
    return sum(v >= threshold for v in ious) / len(ious)


def precision_rate(dists, threshold_px) -> float:
    if not dists:
        return 0.0
    return sum(v <= threshold_px for v in dists) / len(dists)


# ── tablo yazdırma ─────────────────────────────────────────────────────────────

def print_table(results: list, gt_count: int):
    """
    results: [(name, ious, dists, n_skip), ...]
    """
    iou_thresholds  = [0.10, 0.25, 0.50, 0.75]
    dist_thresholds = [5, 10, 20, 40]  # piksel

    print(f"\nGT kareler: {gt_count}")
    print()

    # ── Success @ IoU ──
    header = f"{'Tracker':<18}" + "".join(f"  IoU≥{t:.2f}" for t in iou_thresholds)
    print(header)
    print("-" * len(header))
    for name, ious, dists, n_skip in results:
        evaluated = len(ious)
        row = f"{name:<18}"
        for t in iou_thresholds:
            row += f"  {success_rate(ious, t)*100:6.1f}%"
        row += f"   [{evaluated} değerlendirilen, {n_skip} atlanan]"
        print(row)

    print()

    # ── Precision @ dist ──
    header2 = f"{'Tracker':<18}" + "".join(f"  d≤{t:2d}px" for t in dist_thresholds)
    print(header2)
    print("-" * len(header2))
    for name, ious, dists, n_skip in results:
        row = f"{name:<18}"
        for t in dist_thresholds:
            row += f"  {precision_rate(dists, t)*100:6.1f}%"
        print(row)

    print()

    # ── Özet istatistik ──
    print(f"{'Tracker':<18}  mean_IoU  median_IoU  mean_dist  median_dist  mean_conf")
    print("-" * 75)
    for name, ious, dists, n_skip in results:
        if not ious:
            print(f"{name:<18}  -- veri yok --")
            continue
        mean_iou    = sum(ious) / len(ious)
        sorted_ious = sorted(ious)
        med_iou     = sorted_ious[len(sorted_ious) // 2]
        mean_dist   = sum(dists) / len(dists)
        sorted_d    = sorted(dists)
        med_dist    = sorted_d[len(sorted_d) // 2]
        print(f"{name:<18}  {mean_iou:.3f}     {med_iou:.3f}       "
              f"{mean_dist:7.1f}px  {med_dist:7.1f}px")


# ── opsiyonel matplotlib grafiği ───────────────────────────────────────────────

def plot_success(results):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib kurulu değil, grafik atlanıyor.")
        return

    thresholds = [t / 100 for t in range(0, 101)]
    plt.figure(figsize=(8, 5))
    for name, ious, dists, _ in results:
        rates = [success_rate(ious, t) for t in thresholds]
        auc = sum(rates) / len(rates)
        plt.plot(thresholds, rates, label=f"{name} (AUC={auc:.3f})")

    plt.xlabel("IoU eşiği")
    plt.ylabel("Başarı oranı")
    plt.title("Success Plot — NCC vs NanoTrack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("success_plot.png", dpi=150)
    print("Grafik kaydedildi: success_plot.png")
    plt.show()


def plot_precision(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    thresholds = list(range(0, 101))
    plt.figure(figsize=(8, 5))
    for name, ious, dists, _ in results:
        rates = [precision_rate(dists, t) for t in thresholds]
        auc = sum(rates) / len(rates)
        plt.plot(thresholds, rates, label=f"{name} (AUC={auc:.3f})")

    plt.xlabel("Merkez uzaklığı eşiği (px)")
    plt.ylabel("Hassasiyet oranı")
    plt.title("Precision Plot — NCC vs NanoTrack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("precision_plot.png", dpi=150)
    print("Grafik kaydedildi: precision_plot.png")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tracker değerlendirme aracı")
    ap.add_argument("gt",      help="GT CSV dosyası (label_gt.py çıktısı)")
    ap.add_argument("dumps",   nargs="+",
                    help="Tracker dump dosyaları. İsim eklemek için 'dosya.txt:İsim' formatı")
    ap.add_argument("--plot",  action="store_true", help="matplotlib ile grafik çiz")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    print(f"GT yüklendi: {args.gt}  ({len(gt)} etiketli kare)")

    results = []
    for spec in args.dumps:
        if ":" in spec:
            path, name = spec.rsplit(":", 1)
        else:
            path = spec
            name = Path(spec).stem

        dump = load_dump(path)
        print(f"Dump yüklendi: {path} → '{name}'  ({len(dump)} kare)")
        ious, dists, n_skip = evaluate(gt, dump)
        results.append((name, ious, dists, n_skip))

    print_table(results, len(gt))

    if args.plot:
        plot_success(results)
        plot_precision(results)


if __name__ == "__main__":
    main()
