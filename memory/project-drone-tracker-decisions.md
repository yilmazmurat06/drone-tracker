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

---

## Güdüm katmanı + NPU yol haritası (Haziran 2026)

**Kısıtlar:** NPU, 4MB RAM (bize maks 2.5MB), int8 (~0.6 TOPS).  
**Hibrit:** klasik P2/P3 detector CPU; tek-hedef Siamese tracker NPU.

### Tamamlanan: Faz 1 Spike
- `ISingleTargetTracker` arayüzü + `NccTemplateTracker` (CPU stub)
- `NanoSiameseTracker` (cv::TrackerNano, iki ONNX dosyası)
- `GuidanceController` durum makinesi: SEARCH → pilot seç → TRACK → SUSPECT → SEARCH
- `--siamese`, `--auto-lock N`, `--dump` CLI flags
- **A/B sonuç:** NanoTrack conf 0.60-0.73 (anlamlı, pürüzsüz bbox); NCC conf ~0.98 (saturated, sıçrayan bbox)
- → Generic Siamese domain-specific eğitim olmadan çalışıyor; kendi modelini eğitmek değer

### Tamamlanan: Faz 2 Değerlendirme araçları
- `tools/label_gt.py` — Manuel GT bbox etiketleme (drag çiz, SPACE kopyala, s atla, b geri)
- `tools/eval_tracker.py` — GT vs dump karşılaştırma; IoU@threshold success tablosu + precision@dist tablosu + opsiyonel matplotlib grafiği

**Faz 2 iş akışı:**
```bash
# 1. GT etiketle (frame 3080-3500, FPV drone görünür olduğu segment)
python3 tools/label_gt.py data/flight_01_084727.mp4 --start 3080 --end 3500 --out gt_flight01.csv

# 2. Her tracker için dump al
./build/dtrack_track data/flight_01_084727.mp4 --siamese --auto-lock 3080 \
    --start 3075 --max-frames 430 --dump > dump_nano.txt
./build/dtrack_track data/flight_01_084727.mp4 --auto-lock 3080 \
    --start 3075 --max-frames 430 --dump > dump_ncc.txt

# 3. Karşılaştır
python3 tools/eval_tracker.py gt_flight01.csv dump_nano.txt:NanoTrack dump_ncc.txt:NCC --plot
```

### Sıradaki: Faz 3
- Kendi DroneSiam P5 Siamese modelini eğit (Faz 2 GT + ek veri)
- Static int8 quantization (calibration dataset, NOT quantize_dynamic)
- DW-xcorr ONNX export sorunu çöz
- NPU vendor belirleme (RKNN/Hailo/other) → export stratejisi

### Sıradaki: Faz 4
- P3 NNDiscriminator (Siamese backbone paylaşan küçük sınıflandırma kafası)
- re-acquire ClutterDiscriminator → NNDiscriminator geçişi

---

## Faz 3 ilerleme (donanım netleşti: STM32N6 + Neural-ART, int8)

**Donanım:** STM32N6, Neural-ART NPU (~0.6 TOPS int8), 4MB SRAM (2.5MB pay).
Toolchain: **STM32Cube.AI** (ONNX → C kodu, statik int8 + kalibrasyon). NOT onnxruntime.
**Eğitim:** uzak masaüstü, NVIDIA RTX 6000 Ada. Transfer: SSH/scp. Eğitim kodu device-agnostic.

**Veri stratejisi: DİSTİLASYON** (kullanıcı kararı) — NanoTrack öğretmen → pseudo-label → tiny SiamFC öğrenci.
**TEST seti:** flight_01 kare 3084–3504 (425 insan-GT). Bu aralıkta ASLA eğitim yok.

### Model: DroneSiam (train/model.py) — DOĞRULANDI
- tiny SiamFC: depthwise-separable backbone (paylaşılan) + xcorr head
- conv1(3→32,11,s2)+pool → ds2(64,5)+pool → ds3,ds4,ds5(64,3) hepsi VALID (no-pad)
- toplam stride 8: template 127→6×6, search 255→22×22, xcorr→17×17 response
- **29,058 param (~28KB int8)** → 2.5MB bütçeye rahat sığar
- STM32 bölüşümü: backbone→NPU (statik int8), xcorr→M55 CPU (dinamik template kernel)

### Veri üretim pipeline'ı (yerel) — DOĞRULANDI
1. `tools/collect_trajectories.py` — NanoTrack öğretmen ile ham karelerde trajectory hasadı.
   Ego-hareket telafili (LK+affine) kare-farkı tohum bulucu; sky-gate; çoklu aktif tracker;
   keep-conf filtresi. ÇIKTI: seq_id,flight,frame_idx,x,y,w,h,conf (HAM koordinat).
   BUG düzeltildi: NanoTrack init sonrası getTrackingScore()=0 → tohum boş başlar, ilk update'ten kaydet.
2. `tools/filter_trajectories.py` — **KRİTİK**: NanoTrack bulutu da 0.8 güvenle takip eder.
   Droneness filtresi (P3 mantığı): darkness=(ring-box)/255 >0 VE edge(Sobel)>0.12 → gerçek drone.
   Probe (flight_01, 3505-4004): 13 trajectory → 3 drone tutuldu, 10 bulut elendi. Görsel doğrulandı.
3. `tools/viz_trajectories.py` — trajectory'leri videoya bas (kalite denetimi, renk=seq).

### Sıradaki Faz 3 adımları
- Veri üretimini 3 uçuşun tamamına ölçekle (collect→filter→viz) → tam eğitim seti
- `train/dataset.py` — trajectory'lerden (template, search, Gaussian-label) çiftleri
- `train/train.py` — device-agnostic eğitim (logistic/BCE loss, MPS/CUDA)
- `train/export_onnx.py` — sabit boyut, opset 11
- C++ `DroneSiamTracker : ISingleTargetTracker` (cv::dnn backbone + manuel xcorr)
- Faz 2 araçlarıyla 425 GT'de değerlendir → NanoTrack/NCC ile karşılaştır
