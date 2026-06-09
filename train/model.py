"""
model.py — DroneSiam: STM32N6 (Neural-ART, int8) hedefli tiny SiamFC tracker.

NEDEN SiamFC (NanoTrack'in RPN kafası yerine)?
  - En BASİT Siamese: tek cross-correlation → tek response haritası. RPN/anchor yok.
  - STM32Cube.AI export'u temiz: standart conv/dwconv/pool/relu op'ları.
  - Öğretici: "template ile search'ü eşleştir, tepe = hedef" mantığı çıplak görünür.

MİMARİ (toplam stride = 8, VALID conv = no padding):
  Backbone (paylaşılan, depthwise-separable):
    conv1   3→32   11x11 s2  +BN+ReLU + maxpool 3x3 s2
    ds2    32→64   dw5 s1 + pw1  +BN+ReLU + maxpool 3x3 s2
    ds3    64→64   dw3 s1 + pw1  +BN+ReLU
    ds4    64→64   dw3 s1 + pw1  +BN+ReLU
    ds5    64→64   dw3 s1 + pw1            (embedding, BN'siz/ReLU'suz)

  Boyut akışı:
    template 127 → 6x6x64   (z, referans gömme)
    search   255 → 22x22x64 (x, arama bölgesi gömme)
    xcorr(z, x) → 17x17 response  (22-6+1 = 17)

  ÖNEMLİ SiamFC özelliği: VALID conv + stride'ın (255-127)=128'i tam bölmesi →
  template'in search içinde 16 hücrelik kayması response'ta 17x17 ızgaraya düşer.
  Bu "translation equivariance" SiamFC'nin çalışma sebebidir; padding bunu bozar.

STM32N6 BÖLÜŞÜMÜ:
  - Backbone (forward_backbone) → Neural-ART NPU'da int8. Statik ağırlıklar.
  - xcorr → M55 CPU'da. Çünkü template "kernel"i çalışma anında hesaplanır
    (dinamik ağırlık); NPU statik ağırlık ister. xcorr ~670K MAC, M55'te ucuz.

Parametre sayısı ~32K → int8 ~32KB ağırlık. 2.5MB bütçeye rahat sığar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSepConv(nn.Module):
    """Depthwise-separable conv (VALID/no-pad): dw(kxk) + pw(1x1). BN+ReLU opsiyonel."""
    def __init__(self, in_ch, out_ch, k, stride=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=0,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x):
        x = self.pw(self.dw(x))
        x = self.bn(x)
        if self.act:
            x = F.relu(x, inplace=True)
        return x


class Backbone(nn.Module):
    """Paylaşılan gömme ağı. template ve search aynı ağırlıklardan geçer."""
    def __init__(self, embed=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 11, stride=2, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.ds2   = DWSepConv(32, 64, 5, stride=1, act=True)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.ds3   = DWSepConv(64, 64, 3, stride=1, act=True)
        self.ds4   = DWSepConv(64, 64, 3, stride=1, act=True)
        self.ds5   = DWSepConv(64, embed, 3, stride=1, act=False)  # son gömme

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)
        x = self.ds2(x)
        x = self.pool2(x)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.ds5(x)
        return x


def xcorr_depthwise(search_feat, templ_feat):
    """
    SiamFC cross-correlation: template'i kernel olarak search üzerinde kaydır.
    search_feat: [B, C, Hs, Ws]   templ_feat: [B, C, Ht, Wt]
    dönüş:       [B, 1, Hs-Ht+1, Ws-Wt+1]  (tek kanallı response)

    Batch içinde her örneğin kendi template'i kernel olduğundan grouped conv
    hilesi kullanırız (B'yi kanal grubuna katlarız).
    """
    B, C, Hs, Ws = search_feat.shape
    Ht, Wt = templ_feat.shape[2], templ_feat.shape[3]
    x = search_feat.reshape(1, B * C, Hs, Ws)
    k = templ_feat.reshape(B * C, 1, Ht, Wt)
    out = F.conv2d(x, k, groups=B * C)          # [1, B*C, Ho, Wo]
    out = out.reshape(B, C, out.shape[2], out.shape[3]).sum(dim=1, keepdim=True)
    return out                                   # [B, 1, Ho, Wo]


class DroneSiam(nn.Module):
    """Tam tracker: iki kol aynı backbone + xcorr → response map."""
    def __init__(self, embed=64):
        super().__init__()
        self.backbone = Backbone(embed)
        # Response'u eğitilebilir ölçek/bias ile normalize et (SiamFC "adjust" katmanı)
        self.resp_scale = nn.Parameter(torch.tensor(1e-3))
        self.resp_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, template, search):
        z = self.backbone(template)   # [B,C,6,6]
        x = self.backbone(search)     # [B,C,22,22]
        resp = xcorr_depthwise(x, z)  # [B,1,17,17]
        return resp * self.resp_scale + self.resp_bias

    # STM32 export için: backbone'u tek başına çağırabilmek (NPU parçası)
    def forward_backbone(self, img):
        return self.backbone(img)


if __name__ == "__main__":
    # Boyut doğrulaması (template 127, search 255 → response 17x17)
    m = DroneSiam()
    z = torch.randn(2, 3, 127, 127)
    x = torch.randn(2, 3, 255, 255)
    print("template feat:", m.backbone(z).shape)   # [2,64,6,6]
    print("search   feat:", m.backbone(x).shape)   # [2,64,22,22]
    print("response:", m(z, x).shape)              # [2,1,17,17]
    n = sum(p.numel() for p in m.parameters())
    print(f"parametre: {n:,}  (~int8 {n/1024:.0f} KB)")
