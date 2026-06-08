---
name: project-drone-tracker-decisions
description: "drone-tracker mimari kararları (host, veri kaynağı, kapsam, threading, dil, test, yol haritası)"
metadata: 
  node_type: memory
  type: project
  originSessionId: cf2c3044-9c99-49c9-9ffc-fea607ff4d12
---

`drone-tracker`: CPU-only, hava-hava drone tespit & takip (C++17/OpenCV, <50ms).
Önce çalışan ürün, sonra optimizasyon.

**Temel kararlar:**
1. Host = Linux. Kod taşınabilir. Mac = editör.
2. Veri = `data/` altındaki 3 kayıtlı Liftoff uçuşu. Gyro DERECE/s (rad/s değil — doğrulandı).
3. P5 termal ertelendi. P3 şekil+hareket aktif.
4. C++17, GoogleTest. Pipeline = tek-thread şimdi, çok-thread Adım 7.
5. **Operating mode = CUED** (tam-kare paralaks swamp'u aşılamaz — teyit edildi).
6. **Önce çalışan ürün, sonra optimizasyon** (kullanıcı tercihi).

**Tamamlanan adımlar:**
- Adım 1–3b: iskelet, io/, P1 stabilizasyon (OF+gyro füzyon), kalibrasyon. cam_x=−pitch, cam_y=+yaw, cam_z=−roll, FOV≈128°
- Adım 4 (P2): `MovingTargetDetector` — kare-farkı ∪ MOG2, sky-gate, merge_gap, tight_box.
- Adım 5 (P4): `MultiTargetTracker` — Kalman(x,y,vx,vy), gate+greedy, M-of-N, min_travel, coasting, vel consistency filtresi.
- Adım 6 (P3): `ClutterDiscriminator` — shape skoru (compactness + aspect-ratio log-Gauss + alan bandı) [0,1].
- **Kapalı-döngü cue (Issue #2 ✓):** confirmed izlerin bir-kare-sonraki tahmini → ROI → detektöre geri besleme. `--no-cue` ile kapatılabilir. Cue ROI görselleştirmede soluk mavi kutu olarak çiziliyor.
- **README güncellendi** (Issue #10 ✓): mevcut durum, tüm CLI args, performans tablosu.

**Mevcut parametre defaults (track.cpp + MultiTargetTracker::Params):**
- `score_thresh = 0.70`, `gate_dist = 25` px, `confirm_hits = 5`, `min_travel = 25` px
- `max_accel_sigma = 4.0`, `max_angle_sigma = 0.8` rad, `vel_history_n = 8`
- `cue_margin = 200` px, `closed_loop = true`

**Ölçülen performans (flight_01_084727, 300 kare):**
- Filtresiz baseline: ~50 confirmed/kare
- P3 + tracker tuning: ~4.1 confirmed/kare
- + Kapalı-döngü cue: ~1.1 confirmed/kare (%36 boş)
- Son 100 kare (cue oturmuş): ~1.2 confirmed / %20 boş ← mevcut en iyi

**Klasik tavan:** ~1-2 yanlış-pozitif. Kalan sorun: 3B paralaks tutarlı hareket eder, klasik yöntemle ayırt edilemiyor. Patch NN (Issue #6) ile aşılabilir ama şimdilik ertelendi.

**30/30 test geçiyor.**

**GitHub issues:** #1(P3)✓ #2(kapalı-döngü cue)✓ #3(GrabCut) #4(küçük nokta) #5(çoklu-cue) #6(NN patch) #7(pipeline) #8(termal) #9(paralaks sınır) #10(README)✓

**Sıradaki seçenekler:** #3 GrabCut silüet, #4 küçük nokta, #5 çoklu-cue, #7 pipeline.
