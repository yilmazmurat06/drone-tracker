#!/usr/bin/env python3
"""
shrink_sweep.py — Faz 0a devamı: stok nano bütçeyi aştı → NE KADAR küçültmeli?

nc=1 (tek sınıf) + GENİŞLİK ölçeği taraması. Her varyantın parametre sayısı ve
projekte int8 ağırlık MB'si. Hedef: ağırlık ≲ 2.0 MB (kalan ~0.5 MB aktivasyona).

ultralytics scale = [depth, width, max_channels]; nano 'n' = [0.50, 0.25, 1024].
Genişliği düşürmek param'ı ~kuadratik azaltır (conv ağırlığı ~ Cin·Cout).
"""
import copy
import os

import yaml
import ultralytics
from ultralytics.nn.tasks import DetectionModel

_CFG = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "models")
_PATHS = {"yolo11.yaml": f"{_CFG}/11/yolo11.yaml",
          "yolov8.yaml": f"{_CFG}/v8/yolov8.yaml"}


def build(cfg_name, nc, width, depth=0.50, max_ch=1024):
    """yolo11/yolov8 yaml'ını verilen scale ile nc sınıfında kur, param say."""
    with open(_PATHS[cfg_name]) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["nc"] = nc
    # scale'i sabitle (tek varyant kur)
    cfg["scales"] = {"x": [depth, width, max_ch]}
    cfg["scale"] = "x"
    m = DetectionModel(cfg, nc=nc, verbose=False)
    n = sum(p.numel() for p in m.parameters())
    return n


def report(base, label):
    print(f"\n{label}  (nc=1, depth=0.50)")
    print(f"  {'genişlik':>8} {'param':>12} {'int8 MB':>9}  bütçe")
    for w in (0.25, 0.20, 0.1875, 0.15, 0.125):
        try:
            n = build(base, 1, w)
        except Exception as e:
            print(f"  {w:>8} HATA {type(e).__name__}: {e}")
            continue
        mb = n * 1.05 / 1e6  # int8 + ~%5 meta
        ok = "✅" if mb <= 2.0 else ("⚠ <2.5" if mb <= 2.5 else "❌")
        print(f"  {w:>8.4f} {n:>12,} {mb:>9.2f}  {ok}")


def main():
    print("Hedef: ağırlık ≲ 2.0 MB (kalan ~0.5 MB aktivasyon). Stok nano width=0.25.")
    report("yolo11.yaml", "YOLO11")
    report("yolov8.yaml", "YOLOv8")
    print("\nNOT: nc=1 head'i stok 80-sınıfa göre küçültür. Kesin RAM + op-mapping ST "
          "Edge AI Cloud'da; bu tarama hangi GENİŞLİĞİN sığdığını seçmek için.")


if __name__ == "__main__":
    main()
