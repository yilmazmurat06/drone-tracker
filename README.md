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
          ↑ lock-free SPSC ring buffer her ok'ta ↑       │
[IMU] ────┘ (stabilize'a)                                 │ cue (track→detect)
          └───────────── kapalı-döngü geri besleme ◄──────┘
```

Her ileri ok, tek-üretici-tek-tüketici (SPSC) lock-free ring buffer. Stage'ler örtüşür
(tracker N'de iken detector N+1'de) → throughput frame hızına yetişir.

**Kapalı-döngü (cued tracking):** Hedefe KİLİTLENDİKTEN sonra tracker, hedefin bir
sonraki kare için tahminini (Kalman ileri tahmini) küçük bir **CueBoard** yan kanalıyla
(seqlock, lock-free) detektöre geri verir. Global tespit o karede hedefi kaçırırsa
(düşük SNR / hover), detektör tahmin etrafındaki ROI'de düşük eşikle (track-before-detect)
hedefi **kurtarır** → kilit kesilmez. Bu IRST (Infrared Search & Track) literatüründeki
"Kalman-tahminli ROI + kayıp telafisi" yaklaşımıdır. Sentetik worst-case'de recall
**0.84 → 1.00**, coast oranı **%10 → %0** (bkz. `bench_pipeline` A/B).

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
  - [x] Test: recall=1.00, konum hatası 0.18px (gerçek değer)
  - [x] **Kapalı-döngü cued ROI kurtarma** (track-before-detect-lite): tracker tahmini
        ROI'sinde düşük eşik + MOG2-AND'siz tek tepe → düşük-SNR/hover'da kilit korunur
  - [x] Benchmark: 17 senaryo, hız-fix+cued sonrası **17/17 recall=1.00 & continuity=1.00**
        (worst 0.84→1.00, coast %10→%0; bkz. bench_pipeline A/B)
- [x] detection: drone skorlama (IDiscriminator, Problem 3)
  - [x] `StabilityDiscriminator` — uzamsal yakınlık + yaş ağırlıklı geçmiş eşleme
  - [x] Geometrik skor (alan, en/boy, parlaklık) + zamansal varlık + özellik kararlılığı
  - [x] Test: hedef skoru FP'den %35 yüksek, zamansal kararlılıkla artar
- [x] tracking: Kalman/IMM çoklu-hedef takip + M-of-N yaşam döngüsü
  - [x] `KalmanTracker` (VARSAYILAN) — gerçek 4-durumlu CV Kalman (iz başına iki `Kf2`):
        Joseph-form kovaryans (sayısal kararlı) + σ_a/σ_r ile ayarlı kazanç (≈α-β) + NIS kapısı
  - [x] `AlphaBetaTracker` (YEDEK) — α-β sabit-kazanç filtresi (kararlı-durumda Kalman'a denk)
  - [x] `ImmTracker` — iki Kalman modelli (yumuşak+çevik) IMM, S-tabanlı mod olabilirliği
  - [x] İki-nokta hız başlatma (hız akıl kontrollü) + NIS/Mahalanobis kapısı (coast'ta doğal genişler)
  - [x] **PDAF** (Probabilistic Data Association, VARSAYILAN): yerleşik izler için kapı içi
        tüm adayları olabilirliğe göre ağırlıklandıran Bayesçi yumuşak ilişkilendirme +
        spread-of-innovations kovaryans. Clutter'da ID kararlılığı (test: NN 5 ID → PDAF 1 ID)
  - [x] Per-tespit ölçüm gürültüsü R ipucu (kapalı-döngü kurtarma ölçümü az güvenle işlenir)
  - [x] Onaylı track öncelikli aç gözlü atama (NN yedek yol; coasting track "çalmasın" diye)
  - [x] `TrackStage` + `TrackVizSink` (kutu + iz + video) + demo `camera→stab→detect→track→viz`
  - [x] Test: continuity=0.98 (kesintisiz kilit), konum hatası≈1.7px, tek ID, boşluk köprüleme.
        Çekirdek `Kf2` OpenCV'siz `test_kalman_math` ile doğrulanır (P-SPD kararlı, kazanç≈α-β)
- [x] fusion: track-seviyesi (geç) füzyon
  - [x] `SimpleTrackFusion` — uzamsal kapı + aç gözlü atama + güven ağırlıklı birleştirme
  - [x] Hemfikirlik bonusu (C_fused = C_v + C_t - C_v*C_t), tek modalite cezası (×0.7)
  - [x] Test: 78/80 kilitli, hemfikirlik güveni 1.0, termal kesintide görünür devam
- [x] app: çift kamera pipeline'ı + füzyon + viz demo
  - [x] Görünür + termal paralel pipeline → füzyon → görselleştirme
  - [x] Demo: 542/551 kilitli (%98.4), mor=hemfikir, yeşil=görünür, sarı=coast
```
