#!/usr/bin/env python3
"""Faz 2 etiket rafinasyonu: NanoTrack pseudo-kutulari GEVSEK/KAYIK (tracker cikti
kalitesi); box-regression'i bozuyor (fine-tune v1: mAP50=0.105). Cozum: public
modelimiz (mAP 0.953) OGRETMEN olur — kesitte pseudo-kutuya yakin tespit varsa
etiket ogretmenin SIKI kutusuyla degistirilir.

- conf >= 0.30 VE merkez-uzaklik < 0.35 kesit → etiketi degistir (refined)
- yoksa → kesiti AYIR (drop): "hava araci var" bilgisi dogru olsa da kutusu
  guvenilmez; gevsek kutuyla egitmek zarar (v1 kaniti). Recall kaybi kabul.
- Negatif (bos etiketli) kesitlere dokunulmaz.

NOT (durustluk): val da ayni ogretmenle rafine ediliyor → val metrigi "ogretmen
kutusuna gore" olcer, mutlak degil. Asil dogrulama uctan-uca track testi (C++).

Kullanim: python3 ml/refine_labels.py --dset /tmp/liftoff128 --model models/yolo11_drone.onnx
"""
import argparse, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--maxdist", type=float, default=0.35)
    a = ap.parse_args()
    dset = Path(a.dset)
    sess = ort.InferenceSession(a.model)

    for split in ("train", "val"):
        img_dir = dset / "images" / split
        lbl_dir = dset / "labels" / split
        drop_dir = dset / "dropped" / split
        drop_dir.mkdir(parents=True, exist_ok=True)
        n_ref = n_drop = n_neg = 0
        for lbl in sorted(lbl_dir.glob("*.txt")):
            txt = lbl.read_text().strip()
            if not txt:
                n_neg += 1
                continue                       # negatif kesit: dokunma
            p = txt.split()
            pcx, pcy = float(p[1]), float(p[2])   # pseudo merkez (normalize)
            img_p = img_dir / (lbl.stem + ".jpg")
            img = cv2.imread(str(img_p))
            blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out = sess.run(None, {"images": blob.transpose(2, 0, 1)[None]})[0][0]  # (5,N)
            best, best_s = -1, -1.0
            for i in range(out.shape[1]):
                conf = out[4, i]
                if conf < a.conf:
                    continue
                d = np.hypot(out[0, i] / 128 - pcx, out[1, i] / 128 - pcy)
                if d > a.maxdist:
                    continue
                if conf > best_s:
                    best_s, best = conf, i
            if best < 0:
                # ogretmen onaylamadi → kesiti egitimden cikar (gozle denetlenebilir)
                shutil.move(str(img_p), drop_dir / img_p.name)
                lbl.unlink()
                n_drop += 1
                continue
            cx, cy, w, h = (out[j, best] / 128 for j in range(4))
            lbl.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n_ref += 1
        print(f"[{split}] rafine={n_ref} atilan={n_drop} negatif={n_neg}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
