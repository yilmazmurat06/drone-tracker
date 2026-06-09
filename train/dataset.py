"""
dataset.py — SiamFC eğitim çiftleri (template, search, label) üreten PyTorch Dataset.

build_dataset.py'nin ürettiği 255×255 instance crop'larını (hedef ORTALI) kullanır:
  - template (127×127) = crop_a'nın MERKEZ 127'si  (referans gömme z)
  - search   (255×255) = crop_b, RASTGELE KAYDIRILMIŞ (augment)  (arama bölgesi x)
  - label    (17×17)   = kaydırma offset'ine yerleştirilmiş Gaussian tepe

NEDEN KAYDIRMA? Her iki crop'ta hedef ortalı olduğu için, augment olmadan label
hep merkezde olur → model "hep merkez" demeyi öğrenir (işe yaramaz). Search'ü
rastgele (±max_shift px) kaydırınca hedef merkezden kayar, model OFF-CENTER
hedefi bulmayı öğrenir. SiamFC'nin standart augment'i.

Response geometrisi: search 255, exemplar 127, toplam stride 8 → response 17×17.
  response merkezi = indeks 8. Hedef piksel offset'i (tx,ty) → response offset (tx/8, ty/8).
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXEMPLAR = 127
INSTANCE = 255
STRIDE   = 8
RESP     = 17           # (255-127)/8 + 1
RESP_CTR = RESP // 2    # = 8


def _make_label(offset_x, offset_y, sigma=1.5, r_pos=2.0):
    """17×17 Gaussian label + denge ağırlığı.
    offset = hedefin response-merkezinden kayması (cell). Gaussian tepe TAM
    (merkez+offset)'te → argmax = merkez (metrik doğru çalışır)."""
    ys, xs = np.mgrid[0:RESP, 0:RESP]
    cx = RESP_CTR + offset_x
    cy = RESP_CTR + offset_y
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    label = np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)   # tepe=1, merkezde
    # pozitif/negatif dengesi: dist<=r_pos pozitif bölge sayılır
    pos = dist2 <= r_pos ** 2
    npos = max(1, int(pos.sum()))
    nneg = max(1, int((~pos).sum()))
    weight = np.where(pos, 0.5 / npos, 0.5 / nneg).astype(np.float32)
    return label, weight


class SiamPairs(Dataset):
    def __init__(self, index_csv, max_gap=30, max_shift=48, scale_jitter=0.08,
                 samples_per_epoch=20000):
        root = Path(index_csv).parent
        self.root = root
        traj = defaultdict(list)
        with open(index_csv) as f:
            for r in csv.DictReader(f):
                traj[(r["flight"], int(r["seq"]))].append((int(r["frame"]), r["path"]))
        # en az 2 kareli trajectory'leri tut, kareye göre sırala
        self.trajs = []
        for key, rows in traj.items():
            if len(rows) >= 2:
                rows.sort()
                self.trajs.append([p for _, p in rows])
        self.max_gap = max_gap
        self.max_shift = max_shift
        self.scale_jitter = scale_jitter
        self.n = samples_per_epoch
        if not self.trajs:
            raise RuntimeError(f"{index_csv}: kullanılabilir trajectory yok")

    def __len__(self):
        return self.n

    def _load(self, rel_path):
        img = cv2.imread(str(self.root / rel_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _center_crop(self, img, size):
        h, w = img.shape[:2]
        x0 = (w - size) // 2
        y0 = (h - size) // 2
        return img[y0:y0 + size, x0:x0 + size]

    def __getitem__(self, _):
        traj = random.choice(self.trajs)
        i = random.randrange(len(traj))
        j = min(len(traj) - 1, max(0, i + random.randint(-self.max_gap, self.max_gap)))
        crop_z = self._load(traj[i])      # 255, hedef ortalı
        crop_x = self._load(traj[j])      # 255, hedef ortalı

        # template = z'nin merkez 127'si
        template = self._center_crop(crop_z, EXEMPLAR)

        # search = x'i rastgele kaydır + hafif ölçek jitter
        tx = random.randint(-self.max_shift, self.max_shift)
        ty = random.randint(-self.max_shift, self.max_shift)
        s  = 1.0 + random.uniform(-self.scale_jitter, self.scale_jitter)
        avg = crop_x.mean(axis=(0, 1))
        M = np.array([[s, 0, tx + (1 - s) * INSTANCE / 2],
                      [0, s, ty + (1 - s) * INSTANCE / 2]], dtype=np.float32)
        search = cv2.warpAffine(crop_x, M, (INSTANCE, INSTANCE),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=avg.tolist())

        # label: hedef merkezi (127.5,127.5)+shift → response cell offset
        off_x = tx / STRIDE
        off_y = ty / STRIDE
        label, weight = _make_label(off_x, off_y)

        def to_t(im):
            return torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)

        return (to_t(template), to_t(search),
                torch.from_numpy(label), torch.from_numpy(weight))
