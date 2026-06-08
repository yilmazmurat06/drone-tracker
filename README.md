# drone-tracker

**CPU-only, hava-hava (air-to-air) drone tespit & takip sistemi.**
Bir interceptor drone üzerinde, GPU olmadan, gerçek-zamanlı (<50 ms) çalışacak şekilde tasarlandı.
Küçük (2–6 px) düşman İHA'larını klasik CV + Kalman filtresiyle bulup takip eder.
**C++17 + OpenCV**, hedef platform: gömülü ARM (Cortex-A78).

> **Durum: Çalışan prototip.** P1–P4 tamamlandı. Tam zincir:
> stabilize → tespit → P3 ayraç → Kalman takip → kapalı-döngü cue.
> 30/30 birim test geçiyor.

## Veri akışı

```
mp4 + csv
    │
    ▼
GyroFlowStabilizer (P1)   ← KLT optik akış + gyro füzyonu → homography warp
    │
    ▼
MovingTargetDetector (P2) ← kare-farkı ∪ MOG2 → sky-gate → blob'lar
    │
    ▼
ClutterDiscriminator (P3) ← compactness · aspect-ratio · alan skoru [0,1]
    │
    ▼
MultiTargetTracker (P4)   ← Kalman(x,y,vx,vy) · M-of-N · min_travel · vel-consistency
    │
    ▼
Kapalı-döngü cue          ← confirmed izlerden ROI → bir sonraki kareye geri beslenir
```

## Klasörler

| Klasör | İçerik |
|--------|--------|
| `core/` | Tipler (`Frame`, `Telemetry`, `Detection`, `Track`) + `SpscRingBuffer` |
| `io/` | `RecordedFrameSource` (.mp4), `TelemetryLog` (.csv, ~100 Hz interpolasyon) |
| `stabilization/` | P1: `GyroFlowStabilizer` — KLT+RANSAC → homoğrafi warp; gyro yedek |
| `detection/` | P2: `MovingTargetDetector` + P3: `ClutterDiscriminator` |
| `tracking/` | P4: `MultiTargetTracker` — Kalman + hız tutarlılık filtresi |
| `pipeline/` | P6: aşama soyutlaması (çok-thread runner ertelendi) |
| `app/` | CLI araçları (`dtrack_track`, `dtrack_detect`, `dtrack_stabilize`, `dtrack_replay`, `dtrack_calibrate`) |
| `tests/` | 30 birim testi (GoogleTest) |
| `data/` | 3 Liftoff uçuşu — `flight_XX.mp4` + `flight_XX.telemetry.csv` (mp4 git'e girmez) |

## Derleme

```bash
# Önkoşul: OpenCV, CMake ≥ 3.16, C++17
# Ubuntu:  sudo apt install libopencv-dev cmake
# macOS:   brew install opencv cmake

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # 30/30 test
```

## Kullanım

### Tam zincir — tespit + takip

```bash
# Canlı pencere (q = çık)
build/app/dtrack_track

# Başka bir uçuş
build/app/dtrack_track data/flight_02_084915

# Video kaydet
build/app/dtrack_track --save cikti.mp4

# Sayısal çıktı (300 kare)
build/app/dtrack_track --dump 300

# Kapalı-döngü cue'yu kapat (karşılaştırma için)
build/app/dtrack_track --no-cue

# Tüm parametreler
build/app/dtrack_track --help  # henüz yok; aşağıya bak
```

**CLI parametreleri:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--score-thresh F` | 0.70 | P3 eşiği: altı → tracker'a gitmez |
| `--gate-dist F` | 25 | Kalman ilişkilendirme kapısı (px) |
| `--confirm-hits N` | 5 | Tentative→Confirmed için gereken isabet |
| `--min-travel F` | 25 | Onay için asgari yol (px) |
| `--cue-margin N` | 200 | Kapalı-döngü cue padding (px) |
| `--no-cue` | — | Cue kapalı, tam kare arama |
| `--roi x,y,w,h` | — | Manuel sabit ROI |
| `--show-tentative` | — | Sarı: henüz onaylanmamış izler |
| `--speed F` | 1.0 | Oynatma hızı çarpanı |
| `--save dosya.mp4` | — | Çıktıyı video olarak kaydet |
| `--dump N` | — | GUI yok; N kare sayısal çıktı |
| `--max-frames N` | — | Durma noktası |

### Diğer araçlar

```bash
# Sadece tespit (P1+P2)
build/app/dtrack_detect [--mask] [--roi x,y,w,h]

# Stabilizasyon karşılaştırması (sol=ham, sağ=stab)
build/app/dtrack_stabilize [--dump 2000]

# Gyro eksen/işaret kalibrasyonu
build/app/dtrack_calibrate data/flight_01_084727

# Telemetri + video overlay
build/app/dtrack_replay [--speed 0.5] [--save out.mp4]
```

## Ölçülen performans (flight_01_084727, 300 kare)

| Yapılandırma | Ort. confirmed iz/kare | Boş kare |
|---|---|---|
| Filtresiz baseline | ~50 | %10 |
| P3 + tracker tuning | ~4.1 | %17 |
| + Kapalı-döngü cue | ~1.1 | %36 |
| Son 100 kare (cue oturmuş) | ~1.2 | %20 |

> Kalan ~1 iz büyük ölçüde gerçek hedef. %20 boş kare = soğuk başlangıç + hedef geçici kayıp.

## Önemli bulgular

- **Gyro birimi DERECE/s** (rad/s değil) — quaternion çapraz-doğrulamasıyla keşfedildi.
- **Eksen eşlemesi:** `cam_x = −pitch`, `cam_y = +yaw`, `cam_z = −roll`, `FOV ≈ 128°`
- **3B paralaks tavan:** Klasik yöntemlerle ~4 yanlış-iz/kare sınırına ulaşıldı. Tam-kare arama paralaksla swamp olur; kapalı-döngü cue bunu ~1'e indiriyor.
- **P3 şekil skoru** (compactness · AR · alan): warp artefaktlarını ve büyük zemin bloblarını eler.

## Açık sorunlar (GitHub Issues)

| # | Konu |
|---|------|
| [#2](https://github.com/yilmazmurat06/drone-tracker/issues/2) | Kapalı-döngü cue ✓ (bu sürümde) |
| [#3](https://github.com/yilmazmurat06/drone-tracker/issues/3) | GrabCut silüet iyileştirmesi |
| [#4](https://github.com/yilmazmurat06/drone-tracker/issues/4) | 2–6 px nokta hedef ayrımı |
| [#5](https://github.com/yilmazmurat06/drone-tracker/issues/5) | Çoklu-cue desteği |
| [#6](https://github.com/yilmazmurat06/drone-tracker/issues/6) | Patch-level NN ayraç |
| [#7](https://github.com/yilmazmurat06/drone-tracker/issues/7) | Çok-thread'li lock-free pipeline |
| [#8](https://github.com/yilmazmurat06/drone-tracker/issues/8) | Termal+görünür füzyon (ertelendi) |

## Veri formatı

`data/flight_XX.telemetry.csv` — ~100 Hz, 28 sütun.
Kritik sütunlar: `t_rel` (video-hizalı saniye), `gyro_pitch/roll/yaw` (**DERECE/s**),
`att_x/y/z/w` (attitude quaternion), `pos_*` (m), `vel_*` (m/s).
Senkron: kare `i` → `t_rel = i / fps`.

> Detaylı problem tanımı: [`initial.txt`](initial.txt).
> P5 (termal füzyon) ertelendi: Liftoff termal sensör vermiyor.
