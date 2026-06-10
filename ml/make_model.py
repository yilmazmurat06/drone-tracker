#!/usr/bin/env python3
"""
make_model.py — Faz 0a kilidi: seçilen config'i (YOLO11, nc=1, genişlik 0.20) gerçek bir
ultralytics yaml'ına dök + 128x128 int8 ONNX'e çıkar (ST Edge AI Cloud derlenirlik kontrolü).

Çıktı:
  ml/yolo11_drone.yaml         — eğitim+export için model config (genişlik 0.20, nc=1)
  ml/onnx/yolo11_drone.onnx    — 128x128 fp32 ONNX (ST Cloud'a yüklenecek)
  ml/onnx/yolo11_drone_int8.onnx — int8 (ağırlık boyutu doğrulaması)

Bu ONNX'ler RASTGELE ağırlıklı (henüz eğitilmedi) — ST derlenirlik + RAM/op-mapping kontrolü
ağırlık değerinden bağımsızdır; gating soru "bu MİMARİ N6'ya derleniyor + sığıyor mu?".
"""
import copy
import os
from pathlib import Path

import yaml
import ultralytics
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

WIDTH, DEPTH, MAXCH, NC, IMGSZ = 0.20, 0.50, 1024, 1, 128
SRC = os.path.join(os.path.dirname(ultralytics.__file__),
                   "cfg", "models", "11", "yolo11.yaml")
OUT_YAML = Path("ml/yolo11_drone.yaml")
OUT = Path("ml/onnx"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    with open(SRC) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["nc"] = NC
    cfg["scales"] = {"d": [DEPTH, WIDTH, MAXCH]}  # tek scale 'd' (drone)
    cfg.pop("scale", None)
    OUT_YAML.write_text(
        "# YOLO11 drone tracker — Faz 0a kilidi: genişlik 0.20, nc=1 (tek sınıf 'drone').\n"
        "# STM32N6 bütçesi: ~1.83M param → int8 ~1.92MB ağırlık (2.5MB pay, ~0.5MB aktivasyona).\n"
        + yaml.safe_dump(cfg, sort_keys=False))
    print(f"yaml → {OUT_YAML}")

    model = YOLO(str(OUT_YAML))
    n = sum(p.numel() for p in model.model.parameters())
    print(f"parametre: {n:,}  → int8 ~{n*1.05/1e6:.2f} MB")

    p = model.export(format="onnx", imgsz=IMGSZ, opset=13,
                     dynamic=False, simplify=True, nms=False)
    dst = OUT / "yolo11_drone.onnx"
    os.replace(p, dst)
    print(f"ONNX fp32 → {dst}  ({dst.stat().st_size/1e6:.2f} MB)")

    q = OUT / "yolo11_drone_int8.onnx"
    quantize_dynamic(str(dst), str(q), weight_type=QuantType.QInt8)
    print(f"ONNX int8 → {q}  ({q.stat().st_size/1e6:.2f} MB)")
    print("\nSıradaki: bu ONNX'i ST Edge AI Developer Cloud'a yükle → derlenirlik + RAM/Flash "
          "+ hangi op CPU'ya düşüyor. (Pilot çalıştırma değil, mimari doğrulama.)")


if __name__ == "__main__":
    main()
