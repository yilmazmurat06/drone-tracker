# Faz 0 — YOLO-in-ROI: kilitlenen kararlar + bütçe doğrulaması

Tracking-by-detection planının (DroneSiam yerine) temel parametreleri. Sonradan değişirse
tüm dataset yeniden üretilir → burada SABİTLENDİ.

## Bütçe doğrulaması (ml/budget_probe.py, ml/shrink_sweep.py)

Hedef donanım **STM32N6**: Neural-ART NPU int8, payımız **2.5 MB ağırlık** (aktivasyon hariç).

| Model | param | int8 ağırlık | bütçe |
|-------|-------|-------------|-------|
| yolo11n stok (nc=80) | 2.62M | 2.89 MB | ❌ |
| yolov8n stok (nc=80) | 3.16M | 3.34 MB | ❌ |
| **YOLO11 w0.20 nc=1** | **1.81M** | **2.04 MB** | ✅ ~0.46 MB pay |
| YOLOv8 w0.20 nc=1 | 2.76M | 2.90 MB | ❌ |

**Bulgular:** (1) stok nano modeller olduğu gibi SIĞMIYOR; (2) nc=1 tek başına yetmez
(head küçülmesi marjinal) — genişlik indirimi şart; (3) YOLO11 ≫ YOLOv8 param verimliliğinde.

## Kilitlenen kararlar

- **Model:** YOLO11, genişlik 0.20, derinlik 0.50, **nc=1** ("drone"). Config: `ml/yolo11_drone.yaml`.
  1.81M param, 4.8 GFLOPs@640 → ~0.19 GFLOP@128² (latency rahat; kesin ölçüm Faz 6 N6 kartı).
- **Girdi/crop:** **128×128**, full-res kesit. Kural: ROI ≤ 128 → native pikseller (5px hedef
  5px kalır); ROI > 128 → downscale (büyük hedef zaten kolay). lock-integrity `dynamic_roi`
  (roi_margin=2×kutu) ile uyumlu; README cue-margin (200px) bu kuralın downscale tarafına düşer.
- **Sınıf:** tek sınıf "drone"; kuş/bulut/paralaks/zeplin-değil-hedef = background (negatif).
- **Export:** head decode'suz (`nms=False`, çıktı (1,5,336)=4box+1cls). Decode + NMS CPU'da
  (M55, ucuz). DFL Softmax + neck Resize NPU'da izlenecek op'lar.
- **Latency hedefi:** TRACK modunda kare başı Detection(CPU) + YOLO-in-ROI(NPU) < 50 ms.

## Sıradaki: ST derlenirlik kontrolü (manuel, kullanıcı ST hesabı)

`ml/onnx/yolo11_drone.onnx` (rastgele ağırlık — derlenirlik ağırlıktan bağımsız) →
**ST Edge AI Developer Cloud**'a yükle. Beklenen rapor: Flash (ağırlık), RAM (aktivasyon),
hangi op NPU'da / hangisi M55'e düşüyor. Sığmazsa fallback: genişlik 0.1875/0.15 ya da girdi 96².
