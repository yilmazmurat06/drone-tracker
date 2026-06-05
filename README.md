# drone-tracker

CPU-only, onboard (gömülü) hava-hava küçük İHA tespit & takip sistemi.
GPU yok; klasik CV + el yazımı Kalman/IMM. Hedef: foton → direksiyon komutu < 50 ms.

> Bu, sınıflandırma değil **takip** problemi: sistem zaten hedefe yönlendirilmiş
> (cued) gelir; iş, hedefin karede *nerede* olduğunu yeterince hızlı bulmak.

## Klasör yapısı

Her modül kendi `include/dtrack/<modül>/` (arayüzler) + `src/` (implementasyon)
ikilisine sahip bağımsız bir kütüphanedir. Pipeline somut sınıfları değil
**soyut arayüzleri** (interface) tanır → modüller takılıp çıkarılabilir.

```
common/         Paylaşılan tipler (Frame, Detection, Track...) + lock-free ring buffer
pipeline/       Stage thread iskeleti + Pipeline orkestratör (çekirdek)
io/             Kamera & IMU kaynak arayüzleri
stabilization/  Ego-motion telafisi (gyro + optical flow → homography warp) [Problem 1]
detection/      Arka plan çıkarma (MOG2) + blob + drone skorlama       [Problem 2,3]
tracking/       Kalman/IMM takip + M-of-N yaşam döngüsü                 [Problem 4]
fusion/         Görünür + termal track-seviyesi (geç) füzyon            [Problem 5]
app/            Uygulama girişi (pipeline'ı kurar)                      [Problem 6]
tests/          Birim testleri
```

## Veri akışı

```
[capture] → [stabilize] → [detect+score] → [track] → [fuse] → [output]
          ↑ lock-free SPSC ring buffer her ok'ta ↑
[IMU] ────┘ (stabilize'a)
```

Her ok, tek-üretici-tek-tüketici (SPSC) lock-free ring buffer. Stage'ler örtüşür
(tracker N'de iken detector N+1'de) → throughput frame hızına yetişir.

## Derleme

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

OpenCV **kurulu değilse** yalnızca çekirdek (common + pipeline) ve testler
derlenir — ring buffer'ı hemen doğrulayabilirsin. Tüm modüller için:

```bash
brew install opencv          # macOS
sudo apt install libopencv-dev   # Ubuntu
```

## Durum

- [x] Klasör yapısı + modüler CMake
- [x] Lock-free SPSC ring buffer (test edildi: 1M eleman, 2 thread)
- [x] Stage thread iskeleti + Pipeline orkestratör (drop-oldest politikası)
- [x] Tüm modül arayüzleri (interface)
- [x] io: kamera/IMU somut kaynakları
  - [x] `SyntheticCameraSource` (kayan arka plan + alt-piksel hedef + ego titreşim)
  - [x] `SyntheticImuSource` (gyro = sahne hızı + bias drift + gürültü)
  - [x] `VideoCameraSource` (simülatör RTSP/UDP / .mp4 / webcam / V4L2)
  - [x] `CameraStage` (kaynağı pipeline'a sarar) + uçtan uca demo (`drone_tracker`)
- [x] stabilization: gyro+flow füzyon, ego-motion warp
  - [x] `KltGyroStabilizer` — KLT optical flow + RANSAC similarity, gyro predict,
        online gyro-bias kalibrasyonu (OF çıpalar), gökyüzünde gyro-fallback
  - [x] `StabilizeStage` (IMU'yu drain eder) + demo `camera→stabilize→sink`
  - [x] Test: artık arka plan hareketi 2.5px→0.1px; enjekte bias 0.030→0.031 yakınsadı
- [x] detection: top-hat + MOG2 küçük hedef tespiti
  - [x] `MogDetector` — çoklu ölçek top-hat (3×3 + 5×5) + DoG ⊕ MOG2 + kümülatif referans
        kaydı (referans sıfırlamalı), alt-piksel ağırlıklı centroid, geometrik eleme
  - [x] `DetectStage` + demo `camera→stabilize→detect→sink`
  - [x] Test: recall=1.00, konum hatası 0.11px, yanlış pozitif 0.27/kare
  - [x] Benchmark: 15 senaryo, 14/15 recall≥0.79, 14/15 continuity≥0.67 (bkz. bench_pipeline)
- [x] detection: drone skorlama (IDiscriminator, Problem 3)
  - [x] `StabilityDiscriminator` — uzamsal yakınlık + yaş ağırlıklı geçmiş eşleme
  - [x] Geometrik skor (alan, en/boy, parlaklık) + zamansal varlık + özellik kararlılığı
  - [x] Test: hedef skoru FP'den %35 yüksek, zamansal kararlılıkla artar
- [x] tracking: α-β çoklu-hedef takip + M-of-N yaşam döngüsü
  - [x] `AlphaBetaTracker` — α-β filtresi (fixed-gain, Kalman'a göre manevrada daha sağlam)
  - [x] İki-nokta hız başlatma (hız akıl kontrollü) + Öklid kapısı (adaptive: coast ile genişler)
  - [x] Onaylı track öncelikli aç gözlü atama (coasting track "çalmasın" diye)
  - [x] `TrackStage` + `TrackVizSink` (kutu + iz + video) + demo `camera→stab→detect→track→viz`
  - [x] Test: continuity=0.98 (kesintisiz kilit), konum hatası=1.70px, tek ID (kimlik atlaması yok),
        boşluk köprüleme (5 kare tespitsiz hayatta kalır + iyileşir)
- [x] fusion: track-seviyesi (geç) füzyon
  - [x] `SimpleTrackFusion` — uzamsal kapı + aç gözlü atama + güven ağırlıklı birleştirme
  - [x] Hemfikirlik bonusu (C_fused = C_v + C_t - C_v*C_t), tek modalite cezası (×0.7)
  - [x] Test: 78/80 kilitli, hemfikirlik güveni 1.0, termal kesintide görünür devam
- [x] app: çift kamera pipeline'ı + füzyon + viz demo
  - [x] Görünür + termal paralel pipeline → füzyon → görselleştirme
  - [x] Demo: 542/551 kilitli (%98.4), mor=hemfikir, yeşil=görünür, sarı=coast
```
