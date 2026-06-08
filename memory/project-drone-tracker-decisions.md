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

**Tamamlanan adımlar:**
- Adım 1–3b: iskelet, io/, P1 stabilizasyon (OF+gyro füzyon), kalibrasyon. **cam_x=−pitch, cam_y=+yaw, cam_z=−roll, FOV≈128°**
- Adım 4 (P2): `MovingTargetDetector` — kare-farkı ∪ MOG2, sky-gate, merge_gap, tight_box.
- Adım 5 (P4): `MultiTargetTracker` — Kalman(x,y,vx,vy), gate+greedy, M-of-N, min_travel, coasting.
- **Adım 6 (P3): `ClutterDiscriminator`** — shape skoru (compactness + aspect-ratio log-Gauss + alan bandı) [0,1]. track.cpp'de detect→score→filter zinciri. 6 birim test.

**Mevcut parametre defaults (track.cpp + MultiTargetTracker::Params):**
- `score_thresh = 0.70` (P3 filtresi)
- `gate_dist = 25` px
- `confirm_hits = 5`
- `min_travel = 25` px
- `max_accel_sigma = 4.0`, `max_angle_sigma = 0.8` rad (vel consistency filtresi)
- `vel_history_n = 8` kare

**Ölçülen performans (flight_01_084727, 300 kare):**
- Filtresiz baseline: ~50 confirmed/kare
- Mevcut (tüm filtreler): ~4.1 confirmed/kare (%85 karede ≥1 confirmed)
- **Klasik tavan:** ~4 — kalan yanlış-pozitifler tutarlı paralaks (drone gibi hareket eder)
- Çözüm sınırı: patch-level görünüm bilgisi gerekiyor (Issue #6 NN classifier)

**30/30 test geçiyor.** CLI args: `--score-thresh`, `--gate-dist`, `--confirm-hits`, `--min-travel`

**GitHub issues:** #1(P3)✓ #2(kapalı-döngü cue) #3(GrabCut) #4(küçük nokta) #5(çoklu-cue) #6(NN patch) #7(pipeline) #8(termal) #9(paralaks sınır) #10(README stale)

**Sıradaki:** Patch-level NN discriminator (Issue #6) VEYA kapalı-döngü cue (Issue #2). Kullanıcı "önce çalışan ürün" önceliği koydu — en az kodla en görünür kazancı getiren adım seçilmeli.
