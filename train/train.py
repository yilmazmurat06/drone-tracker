#!/usr/bin/env python3
"""
train.py — DroneSiam (tiny SiamFC) distilasyon eğitimi. Device-agnostic (CUDA/MPS/CPU).

Kullanım (uzak sunucuda):
    python3 train/train.py --data data/dataset --epochs 30 --batch 64 \
        --out train/runs/exp1

Loss: ağırlıklı BCE (pozitif/negatif dengesi dataset'ten gelen weight ile).
Metrik: response tepe lokalizasyon hatası (argmax vs label merkezi, cell cinsinden).
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DroneSiam
from dataset import SiamPairs, RESP, RESP_CTR


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_error(pred, label):
    """Tahmin tepe noktası ile gerçek tepe arasındaki ortalama L2 (cell)."""
    B = pred.shape[0]
    pf = pred.view(B, -1).argmax(1)
    lf = label.view(B, -1).argmax(1)
    py, px = pf // RESP, pf % RESP
    ly, lx = lf // RESP, lf % RESP
    return torch.sqrt(((py - ly) ** 2 + (px - lx) ** 2).float()).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset", help="index.csv'nin olduğu klasör")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=20000)
    ap.add_argument("--out", default="train/runs/exp1")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out}")

    ds = SiamPairs(f"{args.data}/index.csv", samples_per_epoch=args.samples_per_epoch)
    print(f"trajectory sayısı: {len(ds.trajs)}  | epoch başına örnek: {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=(device == "cuda"),
                    drop_last=True)

    model = DroneSiam().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_err = 1e9
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss, run_err, nb = 0.0, 0.0, 0
        for template, search, label, weight in dl:
            template = template.to(device, non_blocking=True)
            search   = search.to(device, non_blocking=True)
            label    = label.to(device, non_blocking=True)
            weight   = weight.to(device, non_blocking=True)

            pred = model(template, search).squeeze(1)         # [B,17,17]
            loss = F.binary_cross_entropy_with_logits(pred, label, weight=weight,
                                                      reduction="sum") / pred.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += loss.item()
            run_err  += peak_error(pred.detach(), label)
            nb += 1
        sched.step()

        avg_loss = run_loss / nb
        avg_err  = run_err / nb
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"peak_err={avg_err:.2f}cell  lr={sched.get_last_lr()[0]:.1e}  "
              f"{time.time()-t0:.0f}s")

        torch.save(model.state_dict(), out / "last.pth")
        if avg_err < best_err:
            best_err = avg_err
            torch.save(model.state_dict(), out / "best.pth")
            print(f"   ↑ yeni en iyi (peak_err={best_err:.2f}) → best.pth")

    print(f"\nBitti. En iyi peak_err={best_err:.2f} cell. Ağırlıklar: {out}/best.pth")


if __name__ == "__main__":
    main()
