#!/usr/bin/env python3
"""
budget_probe.py — Faz 0a: YOLO model BELLEK BÜTÇESİ doğrulaması (STM32N6).

NEDEN İLK İŞ: yolo11n/v8n stok ~2.6/3.2M param → int8 ~2.6/3.2 MB. Bizim payımız
2.5 MB (aktivasyon HARİÇ!). "Sığmıyor" cevabı tüm model seçimini değiştirir → veri
toplamadan ÖNCE öğrenilmeli. Bu script ilk-mertebe cevabı YERELDE verir; kesin RAM/Flash
ve op-mapping için çıktı ONNX'leri ST Edge AI Developer Cloud'a yüklenir.

Ölçülenler (her model için, hem stok 80-sınıf hem bizim nc=1 varyantı):
  - parametre sayısı (≈ int8 ağırlık MB)
  - 128x128 ONNX export + dosya boyutu
  - int8 statik quantize (onnxruntime) → GERÇEK ağırlık boyutu
  - ONNX op envanteri (Neural-ART'a map olmayabilecekleri işaretle)

Kullanım: /opt/anaconda3/bin/python3 ml/budget_probe.py
"""
import collections
import os
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

OUT = Path("ml/onnx")
OUT.mkdir(parents=True, exist_ok=True)
IMGSZ = 128
BUDGET_MB = 2.5

# Neural-ART (ST) int8'de tipik SORUNLU/CPU'ya düşen op'lar — kabaca işaretleme.
SUSPECT_OPS = {"Softmax", "Split", "Resize", "ScatterND", "TopK", "NonMaxSuppression",
               "ReduceMax", "Exp", "Sigmoid", "Mul", "Concat", "Transpose"}


def param_count(model):
    return sum(p.numel() for p in model.model.parameters())


def op_histogram(onnx_path):
    m = onnx.load(str(onnx_path))
    hist = collections.Counter(n.op_type for n in m.graph.node)
    return hist


def probe(tag, weights):
    print(f"\n{'='*64}\n{tag}\n{'='*64}")
    model = YOLO(weights)
    n = param_count(model)
    print(f"parametre: {n:,}  → int8 ağırlık kabaca ~{n/1e6:.2f} MB")

    # 128x128 ONNX export (opset 13, sabit boyut — NPU)
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, opset=13,
                             dynamic=False, simplify=True, nms=False)
    onnx_path = Path(onnx_path)
    dst = OUT / f"{tag}.onnx"
    os.replace(onnx_path, dst)
    fp32_mb = dst.stat().st_size / 1e6
    print(f"ONNX fp32 ({IMGSZ}x{IMGSZ}): {fp32_mb:.2f} MB  → {dst}")

    # int8 dinamik quantize (gerçek ağırlık boyutu için kaba alt-sınır)
    q = OUT / f"{tag}_int8.onnx"
    try:
        quantize_dynamic(str(dst), str(q), weight_type=QuantType.QInt8)
        int8_mb = q.stat().st_size / 1e6
        verdict = "✅ SIĞAR" if int8_mb <= BUDGET_MB else "❌ BÜTÇE AŞIMI"
        print(f"ONNX int8 (dinamik): {int8_mb:.2f} MB  [{verdict} / {BUDGET_MB} MB]")
    except Exception as e:
        print(f"int8 quantize başarısız: {type(e).__name__}: {e}")

    # op envanteri
    hist = op_histogram(dst)
    print(f"op çeşidi: {len(hist)}, toplam düğüm: {sum(hist.values())}")
    suspects = {o: c for o, c in hist.items() if o in SUSPECT_OPS}
    if suspects:
        print(f"  ⚠ NPU'da dikkat (CPU'ya düşebilir): {dict(suspects)}")
    return n


def main():
    print(f"Hedef bütçe: {BUDGET_MB} MB (int8 ağırlık; aktivasyon HARİÇ)")
    # Stok 80-sınıf = ÜST SINIR (nc=1 head'i biraz küçültür → stok sığarsa nc=1 de sığar).
    probe("yolo11n_coco", "yolo11n.pt")
    probe("yolov8n_coco", "yolov8n.pt")
    print("\nNOT: dinamik int8 ≈ ağırlık alt-sınırı. Aktivasyon RAM'i + statik int8 için "
          "bu ONNX'leri ST Edge AI Developer Cloud'a yükle → kesin Flash/RAM + op-mapping.")


if __name__ == "__main__":
    main()
