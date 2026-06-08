---
name: project-drone-tracker-decisions
description: "drone-tracker mimari kararları (host, veri kaynağı, kapsam, threading, dil, test, yol haritası)"
metadata: 
  node_type: memory
  type: project
  originSessionId: cf2c3044-9c99-49c9-9ffc-fea607ff4d12
---

`drone-tracker`: CPU-only, hava-hava drone tespit & takip (C++17/OpenCV, <50ms).
2026-06-08'de grill-me röportajıyla netleşen kararlar:

1. **Tertemiz başlangıç** — eski koda bakılmaz (bkz. [[feedback-clean-slate-teach]]).
2. **Host = Linux** (Ubuntu sunucu / Windows+WSL). Kod taşınabilir, platforma özel API yok. Mac = editör.
3. **Veri = `data/` altındaki 3 kayıtlı uçuş, "kaydet-sonra-oynat".** Canlı yakalama (ekran+UDP)
   ve recorder aracı **gerçek Liftoff aşamasına ertelendi**. (CSV şeması: [[reference-telemetry-csv]].)
4. **P5 (termal+görünür füzyon) ve P3'ün termal ayracı ertelendi** — Liftoff termal vermiyor, kuş yok.
   Aktif kapsam: P1 stabilizasyon, P2 tespit, P3 (şekil+hareket), P4 Kalman/IMM, P6 pipeline.
5. **Pipeline = hibrit:** ortak `IStage` arayüzü; önce tek-thread'li/deterministik runner;
   lock-free SPSC ring buffer şimdi yazıldı+test edildi; çok-thread'li runner Adım 7'ye bırakıldı.
6. **C++17.**  7. **GoogleTest** (CMake FetchContent).
8. **Yol haritası (veri akışı sırası):** İskelet → io/ replay → P1 → P2 → P3 → P4 → P6.

**Yapı:** `dtrack` namespace; modüller header-only INTERFACE lib (`core/io/stabilization/
detection/tracking/pipeline`), `app/` executable, `tests/`. Tipler: `Frame`, `Telemetry`,
`Detection`, `Track`, `Cue`. Ring buffer = `core/include/dtrack/core/ring_buffer.hpp`.

**Durum (2026-06-08):** Adım 1 (iskelet) + Adım 2 (io/ replay) tamam. io/ artık STATIC
lib; `TelemetryLog` (csv→interpolasyon, `interpolate_telemetry` lineer+nlerp), `RecordedFrameSource`
(VideoCapture), `dtrack_replay` overlay oynatıcı (canlı/--save/--dump). 11 test geçiyor.
Adım 3a (P1 optik akış stabilizasyon) tamam: `OpticalFlowStabilizer` (KLT + RANSAC
findHomography + warpPerspective, cur→prev). Gerçek veride kare-arası fark ~1.74x
(yüksek-hareket ~1.95x) azaldı; artık fark = parallax (3B sahne, ileri uçuş) → bu
3b gyro füzyonunu motive ediyor. Adım 3b (gyro füzyonu) tamam: `GyroFlowStabilizer`
(IStabilizer). Saf dönme homografisi `H=K·R·K⁻¹`; gyro ω trapez integralle θ
(prev→cur), Rodrigues→R; cur→prev için −θ. Füzyon = "seç-ya da-düş": OF güvenilirse
Flow, çökerse Gyro (Mode enum). `AxisMap` (gyro→kamera işaret/eksen eşlemesi, şu an
kimlik) + `fov_h_deg`(120)→K. `app/dtrack_stabilize` artık telemetri yükleyip
[önceki,geçerli] kare penceresi besliyor, `yol` (FLOW/GYRO) gösteriyor. **18 test geçiyor.**
İki tuzak çözüldü: (1) Rodrigues çıkışı giriş derinliğini alır → rvec CV_64F kurulmalı;
(2) OF çökünce `out=in` ile aliasing → warpPerspective yerinde no-op olur, ayrı dst'e yaz.
Adım 3b-ii (kalibrasyon) tamam: `dtrack_calibrate` aracı (OF homografisini "doğru cevap"
alır: `t=Rodrigues(K⁻¹·H_of·K)`, 48 işaretli-permütasyon × FOV ızgarası tarar; att_*
quaternion açısı = bağımsız çapraz-doğrulama). **BÜYÜK BULGU: gyro DERECE/s'ymiş** (rad/s
değil — quaternion oranı tam π/180 çıktı). Düzeltince oran 1.000, RMS 11.8°→0.36°/kare.
3 uçuşta da TUTARLI eşleme: **cam_x=−pitch, cam_y=+yaw, cam_z=−roll, FOV≈128°** — bunlar
`GyroFlowStabilizer::Params` varsayılanına gömüldü. **18 test geçiyor.** Dürüst sonuç:
saf gyro 0.08x→0.75x (kalibrasyon felaketi düzeltti ama hâlâ <1x); çünkü bu veride
kare-arası dönme küçük (~0.12°/kare), hareket çoğunlukla ÖTELEME/paralaks → saf dönme
gideremez. Gyro'nun değeri OF çökünce YEDEK olması; füzyon (auto) 1.74x.

**VERİ GERÇEĞİ (teyit edildi):** `data/` Liftoff uçuşlarında gerçek hedefler VAR — haritanın
AI hava trafiği: **zeplin** (büyük) + **küçük uçaklar/noktalar** (2–6px) + planörler, çoğunlukla
gökyüzü önünde. Ground-truth kutu yok (telemetri sadece bizim drone) → doğrulama: gözle (overlay)
+ sentetik enjeksiyonlu birim test.

Adım 4 (P2 tespit) tamam: `MovingTargetDetector` (IDetector) — **HİBRİT**: kare-farkı
(küçük/hızlı) ∪ MOG2 (büyük/yavaş zeplin) → morfoloji → connectedComponents → `Detection`.
`border_margin` warp kenarını yok sayar; **sky-gate** (HSV: parlak ∧ (soluk ∨ mavi)) zemin
paralaksını eler; **cue ROI** (`set_roi`) aramayı daraltır. detection/ artık STATIC lib.
`app/dtrack_detect` (stabilize+detect+kutu; --save/--dump/--mask/--roi). **21 test geçiyor.**
DÜRÜST SONUÇ: tam-karede zemin/ağaç paralaksı çok yanlış-pozitif üretir (~260/kare, sky-gate
sonrası) — P2 **recall-odaklı aday üreticisi**; precision aşağı akıştan gelir. Cue ROI ile
(zeplin etrafında) ~13/kare, temiz izole → "cued" mimarisi doğrulandı. Gerçek hedefler (zeplin,
uçaklar) yakalanıyor.

Adım 5 (P4 takip) tamam: `MultiTargetTracker` (ITracker) — track başına sabit-hız
`cv::KalmanFilter` (x,y,vx,vy) + gate'li aç-gözlü ilişkilendirme + M-of-N onay + `min_travel`
(doğuştan net yol) kapısı + coasting. tracking/ artık STATIC lib. `app/dtrack_track` (tam zincir
stabilize→detect→track; yalnız Confirmed çizer; --roi/--show-tentative/--save/--dump). **24 test geçiyor.**
**ÇOK ÖNEMLİ DÜRÜST BULGU:** takip TEK BAŞINA tam-kare gürültüyü çözMEZ — paralaks gürültüsü
o kadar yoğun ki gate hep dolu → her şey onaylanıp coasting ile birikiyor (~570 confirmed/kare).
`min_travel` bile yetmedi çünkü ötelenen kamerada zemin paralaksı TUTARLI AKAR (gerçek hareket
gibi net yol biriktirir). Sonuç: **3B sahnede tam-kare hareket-tespiti paralaksla swamp olur;
takip bunu kurtaramaz.** Çözüm mimaride: **sistem CUED.** Cue ROI ile (zeplin) ~16 confirmed/kare,
kare temiz, hedefe KİLİTLENİR (Kalman hız okları). Yani işletim modu tam-kare DEĞİL, cued.
Büyük-hedef birleştirme eklendi: `MovingTargetDetector::Params.merge_gap` (union-find ile
yakın blobları tek `Detection`'a topla) → zeplin 8 parça → ~1-2; cued onaylı iz 16→2.3/kare.
SİLÜET KUTUSU eklendi (`tight_box`): hareket maskesi cismin yalnız kenarını görür; gök önünde
TAM silüet = ¬sky_raw'ın hedef merkezini içeren bağlı bileşeninin bbox'ı (hareketsiz iç gövde
dahil). `Track`'e `cv::Rect bbox` eklendi (tespit→izleme taşınır, coasting'de pred'e ortalanır);
`dtrack_track` sabit 28px yerine bunu çizer → kutu zeplinin TAMAMINI sarıyor. **24 test geçiyor.**
NOT: cue tek bir MANUEL ROI (zeplinin üstüne biz koyduk) → diğer uçaklar kutu dışında kaldığı
için görünmez; gerçek sistemde cue dışarıdan/her hedef için ayrı gelir (çoklu-cue ileride).
Onay gecikmesi = M-of-N(4) + min_travel(25px) (cued temizken düşürülebilir).
Sıradaki seçenekler: **Adım 6 (P3 ayraç)** (şekil/hareket skoru), çoklu-cue, ya da **P6 pipeline**.
