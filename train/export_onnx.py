#!/usr/bin/env python3
"""
export_onnx.py — Eğitilmiş DroneSiam'ı STM32Cube.AI için ONNX'e çıkar.

STM32N6 BÖLÜŞÜMÜ: xcorr dinamik ağırlık (template runtime'da hesaplanır) gerektirir
→ NPU'ya gitmez, M55 CPU'da yapılır. Bu yüzden SADECE BACKBONE'u export ederiz.
Backbone tam-konvolüsyonel ama STM32Cube.AI SABİT giriş boyutu ister → iki ayrı
ONNX çıkarırız:
  - dronesiam_backbone_z.onnx : giriş [1,3,127,127] → [1,64,6,6]   (template kolu)
  - dronesiam_backbone_x.onnx : giriş [1,3,255,255] → [1,64,22,22] (search kolu)
Aynı ağırlıklar; sadece sabit giriş boyutu farklı. xcorr (C/M55) bu iki çıktıyı eşler.

opset 11, dynamic_axes YOK (NPU sabit boyut ister).

Kullanım:
    python3 train/export_onnx.py --ckpt train/runs/exp1/best.pth --out models/
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import DroneSiam


class BackboneOnly(nn.Module):
    """Sadece backbone'u dışa veren sarmalayıcı (export için)."""
    def __init__(self, full):
        super().__init__()
        self.backbone = full.backbone

    def forward(self, x):
        return self.backbone(x)


def export_one(net, size, path, opset=11):
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        net, dummy, str(path),
        input_names=["input"], output_names=["feat"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,           # SABİT boyut (NPU)
    )
    print(f"  → {path}  (giriş 1x3x{size}x{size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    full = DroneSiam()
    full.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    full.eval()
    net = BackboneOnly(full).eval()

    print("Backbone ONNX export (sabit boyut, opset", args.opset, "):")
    export_one(net, 127, out / "dronesiam_backbone_z.onnx", args.opset)
    export_one(net, 255, out / "dronesiam_backbone_x.onnx", args.opset)

    # xcorr response ölçek/bias'ı (C tarafı için) + meta
    meta = {
        "exemplar": 127, "instance": 255, "stride": 8, "response": 17, "embed": 64,
        "resp_scale": float(full.resp_scale.detach()),
        "resp_bias": float(full.resp_bias.detach()),
        "note": "backbone NPU(int8), xcorr M55 CPU. response = xcorr(x,z)*scale+bias",
    }
    (out / "dronesiam_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  → {out}/dronesiam_meta.json")
    print("\nSonraki: STM32Cube.AI ile bu ONNX'leri kalibrasyon dataset'iyle int8'e çevir.")


if __name__ == "__main__":
    main()
