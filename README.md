# drone-tracker

**CPU-only, hava-hava (air-to-air) drone tespit & takip sistemi.**
Bir interceptor drone üzerinde, GPU olmadan, gerçek-zamanlı (<50 ms) çalışacak;
küçük (2–6 piksel) düşman İHA'larını klasik CV + Kalman/IMM ile bulup takip eder.
Geliştirme C++17 + OpenCV ile yapılır. Detaylı problem tanımı: [`initial.txt`](initial.txt).

> **Durum: Adım 1 — proje iskeleti.** Klasör yapısı, CMake build, modül arayüzleri
> (header), lock-free ring buffer + testleri. Algoritma implementasyonları sonraki
> adımlarda (aşağıdaki yol haritası).

## Mimari (hedeflenen veri akışı)

```
FrameSource ─┐
             ├─> Stabilize (P1) ─> Detect (P2) ─> Discriminate (P3) ─> Track (P4) ─> çıktı
TelemSource ─┘   gyro+flow→warp    MOG2→blob       şekil+hareket        Kalman/IMM
```

Modüller arayüzlerle (`I...`) konuşur; somut sınıflar enjekte edilir. Pipeline önce
**tek-thread'li/deterministik** koşar (debug kolay); çok-thread'li lock-free sürüm
Adım 7'de eklenir (ring buffer çekirdeği şimdiden hazır ve test edilmiş).

| Klasör | İçerik |
|---|---|
| `core/` | Paylaşılan tipler (`Frame`, `Telemetry`, `Detection`, `Track`) + lock-free `SpscRingBuffer` |
| `io/` | Veri kaynakları: `RecordedFrameSource` (.mp4), `RecordedTelemetrySource` (.csv) |
| `stabilization/` | P1: ego-motion stabilizasyon (gyro+flow → homography warp) |
| `detection/` | P2 tespit (MOG2→blob) + P3 ayraç (şekil/hareket) |
| `tracking/` | P4: Kalman/IMM takip |
| `pipeline/` | P6: aşama soyutlaması + runner |
| `app/` | Çalıştırılabilir giriş noktası |
| `tests/` | Birim testleri (GoogleTest) |
| `data/` | Test verisi: 3 uçuş `flight_XX.mp4` + `flight_XX.telemetry.csv` (mp4 git'e girmez) |

## Veri

`data/` altında 3 kayıtlı Liftoff uçuşu (1920×1080 @ **60 fps** mp4 + **~100 Hz**,
28 sütunlu telemetri csv). Senkron: csv'deki `t_rel=0 ≈ video başı`; kare `i` için
`t = i/60`, telemetri o `t_rel`'e interpolasyonla eşlenir. Önemli sütunlar:
`gyro_pitch/roll/yaw` (rad/s), `att_x/y/z/w` (attitude quaternion), `pos_*`, `vel_*`.

## Derleme & test

**Önkoşul:** OpenCV (Ubuntu: `sudo apt install libopencv-dev`; macOS: `brew install opencv`),
CMake ≥ 3.16, C++17 derleyici. GoogleTest CMake tarafından otomatik indirilir (internet gerekir).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # birim testleri
./build/app/dtrack_app                        # iskelet doğrulama

# Replay oynatıcı (Adım 2):
./build/app/dtrack_replay                      # canlı: data/flight_01 + overlay (q=çık, boşluk=duraklat)
./build/app/dtrack_replay data/flight_02_084915 --speed 0.5
./build/app/dtrack_replay --dump 10            # GUI yok: kare→telemetri eşlemesi
./build/app/dtrack_replay --save out.mp4       # ekransız: annotasyonlu mp4 yaz

# Stabilizasyon demo (Adım 3a): sol=ham fark, sağ=stabilize fark
./build/app/dtrack_stabilize                   # canlı yan-yana kare-farkı
./build/app/dtrack_stabilize --dump 2000       # GUI yok: ham vs stab sayısal (iyileşme oranı)
```

## Yol haritası

1. **(✓) İskelet** — yapı + CMake + arayüzler + ring buffer.
2. **(✓) io/ replay** — `.mp4`/`.csv` okuma + `TelemetryLog` interpolasyon + `dtrack_replay` overlay oynatıcı (t_rel↔kare senkronu).
3. **P1 stabilizasyon** — **3a (✓)** `OpticalFlowStabilizer` (KLT + RANSAC homografi + warp; gerçek veride kare-arası farkta ~1.7–2x azalma). **3b (sırada)** gyro/attitude füzyonu (özelliksiz gökyüzü/parallax için).
4. **P2 tespit** — MOG2 background subtraction → blob → centroid.
5. **P3 ayraç** — şekil kararlılığı + hareket tutarlılığı skoru.
6. **P4 takip** — Kalman → IMM, M-of-N başlatma.
7. **P6 thread'leme** — aşamaları ring buffer'larla ayrı thread'lere taşı.

> P5 (termal + görünür füzyon) ertelendi: Liftoff termal sensör vermiyor.
