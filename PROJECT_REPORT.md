# Drone Tracker — Kapsamlı Proje Raporu

**Tarih**: 5 Haziran 2026 (güncelleme: hız-fix + video_eval + LCM + IMM)  
**Dil**: C++20  
**Testler**: 9 birim/regresyon testi (%100 geçti)  

---

## 1. Proje Özeti

**Amaç**: CPU-only, onboard (gömülü) hava-hava küçük İHA tespit ve takip sistemi.  
**Hedef donanım**: 6 çekirdekli ARM (Cortex-A78/A76), GPU yok, ağır ML framework'leri yok.  
**Gecikme bütçesi**: Foton → direksiyon komutu < 50 ms.  
**Temel karar**: Bu bir *sınıflandırma* değil *takip* problemi. Sistem hedefe yönlendirilmiş (cued) gelir.  

---

## 2. Mimari Kararlar ve Nedenleri

### 2.1 Modül-başına statik kütüphane

```
common/         Paylaşılan tipler + lock-free ring buffer
pipeline/       Stage thread iskeleti + Pipeline orkestratör
io/             Kamera & IMU kaynak arayüzleri + somut implementasyonlar
stabilization/  Ego-motion telafisi (gyro + optical flow)                    [Problem 1]
detection/      Arka plan çıkarma (MOG2) + blob + drone skorlama            [Problem 2,3]
tracking/       α-β filtre takip + M-of-N yaşam döngüsü                      [Problem 4]
fusion/         Görünür + termal track-seviyesi (geç) füzyon                 [Problem 5]
app/            Uygulama girişi (pipeline demo + video benchmark)            [Problem 6]
tests/          Birim testleri + benchmark
```

**Neden**: Her modül bağımsız derlenebilir, test edilebilir ve değiştirilebilir. `dtrack/<modül>/` include prefix'i ile isim çakışması önlenir. Namespace'ler dizin yapısını yansıtır (`dtrack::stabilization`, `dtrack::detection` vb.).

### 2.2 Interface-first tasarım

Her modül bir soyut arayüzle (pure-virtual interface) tanımlanır:
- `ICameraSource`, `IImuSource` (io)
- `IStabilizer` (stabilization)
- `IDetector`, `IDiscriminator` (detection)
- `ITracker` (tracking)
- `ITrackFusion` (fusion)

**Neden**: Pipeline somut sınıfları değil arayüzleri tanır. Sentetik kamera ↔ gerçek donanım ↔ simülatör değişimi tek satırlık kod değişikliği ile yapılır. Test edilebilirlik: her modül mock implementasyonla izole test edilebilir.

### 2.3 Lock-free SPSC pipeline

Her stage kendi thread'inde çalışır. Stage'ler arası iletişim `SpscRingBuffer<T>` (Single Producer Single Consumer) ile yapılır. `push_overwrite()` drop-oldest back-pressure politikası uygular (gerçek zamanlı sistem: bayat kareyi düşür, tazeyi koru).

**Neden**: Mutex yok → thread'ler birbirini beklemez. Stage'ler örtüşür: tracker N'de iken detector N+1'de, kamera N+2'yi yakalar → CPU frame hızına yetişir.

### 2.4 IMU pull model

IMU pipeline'da ayrı bir stage **değildir**. Stabilizer `IImuSource::drain()` ile her karede biriken gyro örneklerini çeker.

**Neden**: Kamera ve IMU aynı `t0`'ı paylaşır, zaman damgaları üzerinden eşlenir. IMU thread'i yok → daha az thread, daha az context switch.

### 2.5 Sentetik kaynak = yer-gerçeği

`SceneModel` saf matematik (OpenCV'siz, deterministik). Kamera ve IMU aynı fiziksel modelden beslenir → gyro ile görüntü ego-hareketi tutarlıdır. Her karede hedefin gerçek konumu bilinir → tespit/takip doğruluğu otomatik ölçülür.

### 2.6 Koordinat sandviçinden vazgeçiş

**İlk tasarım**: Tracker, detektörün referans koordinatında çalışsın (ego-jitter'dan arınmış). Tespitler görüntü→referans çevrilir, takip yapılır, track'ler referans→görüntü geri çevrilir.

**Neden vazgeçildi**: Detektörün referans sıfırlaması (kümülatif öteleme > eşik) koordinat sistemini aniden değiştiriyor → tracker'ın tüm iç durumu geçersiz oluyor. Ayrıca stabilizasyon kalıntısı sadece ~0.10 px → imge koordinatında takip yeterli.

**Son karar**: Tracker imge koordinatında çalışır. Daha basit, daha sağlam.

### 2.7 Kalman → α-β filtresi geçişi

**İlk implementasyon**: 4-durumlu CV (Constant Velocity) Kalman filtresi. `process_noise` ve `init_vel_var` parametrelerinde yoğun tuning yapıldı.

**Sorun**: Kalman kazancı manevra yapan hedefte çok yavaş yakınsıyor (hız kazancı ~0.0069 px/s per px). P kovaryansı küçük hedefte kararsız. `init_vel_var=1e4` ile başlangıç hız kazancı ~35 px/s per px → salınım. İki-nokta başlatma sahte pozitif + hedefi yanlış eşleyip absürt hız üretiyor.

**Çözüm**: α-β filtresi (`AlphaBetaTracker`). Sabit kazançlar (α=0.55, β=0.12) sayesinde yakınsama sorunu yok. Hız akıl kontrollü iki-nokta başlatma (>400 px/s reddedilir). Adaptive Öklid kapısı (coast süresiyle genişler). Onaylı track öncelikli aç gözlü atama.

**Sonuç**: Continuity 0.81→0.98, ID atlaması 22→1, Kalman'dan çok daha sağlam.

---

## 3. Modül Detayları

### 3.1 `common/` — Paylaşılan Tipler ve Altyapı

| Dosya | İçerik |
|---|---|
| `types.hpp` | Frame, FramePtr, ImuSample, EgoMotion, StabilizedFrame, Detection, Track, TrackStatus, FrameDetections, FrameTracks |
| `ring_buffer.hpp` | `SpscRingBuffer<T>` — lock-free SPSC, `std::atomic` acquire/release, `push_overwrite` (drop-oldest) |
| `time.hpp` | `steady_clock` tabanlı ortak zaman, `Timestamp`, `Duration`, `now()`, `millis_between()` |

**Karar**: Frame'ler `shared_ptr<const Frame>` ile taşınır → stage'ler arası sıfır kopya. Tüm zaman damgaları aynı `steady_clock`'tan → gyro↔frame eşleme ve füzyon için kritik.

### 3.2 `pipeline/` — Thread İskeleti

| Dosya | İçerik |
|---|---|
| `stage.hpp` | `Stage<In, Out>` — kendi thread'inde `pop→process()→emit` döngüsü |
| `pipeline.hpp` | `Pipeline` — stage'leri tüketiciden üreticiye başlatır, ters sırada durdurur |
| `istage.hpp` | `IStage` soyut arayüzü |

**Karar**: `Stage::process()` `std::nullopt` döndürürse çıktı üretilmez (boş tespit karesi). Kaynak stage'ler `In=Tick` (sentinel tip). Sink stage'ler `Out=Tick`. Başlatma sırası: tüketici→üretici (ters ekleme), durdurma: üretici→tüketici → kuyruklar hiç boşken okunmaz, kapanışta aç kalmaz.

### 3.3 `io/` — Kamera ve IMU Kaynakları

| Dosya | İçerik |
|---|---|
| `camera_source.hpp` | `ICameraSource` arayüzü |
| `imu_source.hpp` | `IImuSource` arayüzü |
| `synthetic_scene.hpp` | `SceneModel` — saf matematik yer-gerçeği modeli. Ego titreşim (sinüs toplamı), hedef yörüngesi (düz + manevra sinüsü), gyro = dθ/dt |
| `synthetic_camera_source.cpp` | `SyntheticCameraSource` — sahneyi `cv::Mat`'e render eder. Kayan dokulu arka plan + alt-piksel Gaussian hedef + sensör gürültüsü |
| `synthetic_imu_source.cpp` | `SyntheticImuSource` — gyro = sahne hızı + bias random walk + gürültü. `generate_until(t)` testler için deterministik yol |
| `video_camera_source.cpp` | `VideoCameraSource` — `cv::VideoCapture` sarmalayıcı (MP4/RTSP/UDP/V4L2/webcam) |
| `camera_stage.hpp` | `CameraStage` — kaynağı pipeline Stage'ine sarar |

**Sahne parametreleri**: 640×512, focal=700 px, ego titreşim 1.7/2.3/11 Hz, hedef hızı 55 px/s, manevra 40 px genlik 0.25 Hz, hedef sigma 1.1 px (2-6 px görünür), arka plan seviyesi 35, doku 18.

### 3.4 `stabilization/` — Ego-Motion Telafisi [Problem 1]

| Dosya | İçerik |
|---|---|
| `stabilizer.hpp` | `IStabilizer` arayüzü |
| `klt_gyro_stabilizer.cpp` | `KltGyroStabilizer` — 3 adımlı füzyon |
| `stabilize_stage.hpp` | `StabilizeStage` — IMU'yu drain eder |

**Algoritma (3 adım)**:

1. **PREDICT (gyro)**: `Δθ = Σ(ω_k - bias) · Δt_k`. Tahmini içerik kayması = `-focal · Δθ` (kamera dönüşü ↔ içerik hareketi zıt yönlü).

2. **MEASURE (optical flow)**: Shi-Tomasi köşeleri → piramidal Lucas-Kanade (ileri-geri hata kontrolü) → RANSAC similarity (`estimateAffinePartial2D`). Hareketli hedef outlier atılır.

3. **CORRECT (bias kalibrasyonu)**: OF geçerliyken `bias -= k · innovation / (-focal · dt)`. Gökyüzünde (köşe < eşik) saf gyro'ya düş, `ego.valid=false`.

**Çıktı**: `ego.homography` = güncel→önceki kare dönüşümü (warp yapılmaz, tüketici yapar).

**Test sonucu**: Artık arka plan hareketi 2.47 px → 0.10 px (~25× azalma). Enjekte gyro bias 0.030 → kestirilen (0.031, 0.032) rad/s.

### 3.5 `detection/` — Küçük Hedef Tespiti [Problem 2]

| Dosya | İçerik |
|---|---|
| `detector.hpp` | `IDetector` arayüzü |
| `discriminator.hpp` | `IDiscriminator` arayüzü |
| `mog_detector.cpp` | `MogDetector` — 4 ölçekli top-hat + 2 bant DoG + MOG2 |
| `stability_discriminator.cpp` | `StabilityDiscriminator` — uzamsal yakınlık + yaş ağırlıklı geçmiş eşleme |
| `detect_stage.hpp` | `DetectStage` — detektör + discriminator'ı pipeline'a sarar |

**MogDetector algoritması**:

1. **Kümülatif referans kaydı**: `h_cum *= ego.homography`. Öteleme > %25 min(W,H) ise referans sıfırlanır (kareden taşmayı önler).

2. **Warp**: Güncel kare → referans koordinat (MOG2 sabit model görsün).

3. **Çoklu ölçek top-hat** (4 kernel: 3×3, 5×5, 9×9, 15×15): Her ölçekte `orijinal - açma(erozyon+dilasyon)`. Maksimum yanıt alınır. 2 px'ten 25 px'e kadar tüm hedefleri kapsar.

4. **İki bant DoG**: Küçük bant (σ=0.6/1.8, 2-6 px) + orta bant (σ=1.5/5.0, 5-15 px). Maksimum yanıt alınır, top-hat ile harmanlanır.

5. **Eşik**: 5×5 top-hat üzerinden adaptif (`mean + 6·std`) — gürültüye karşı kararlı.

6. **MOG2**: Zamansal hareket maskesi. `top-hat ∧ MOG2 ∧ geçerli_bölge`.

7. **Bağlı bileşenler → blob eleme**: Alan [2,400], en/boy < 4.0, top-hat tepe ≥ 25.0.

8. **Alt-piksel centroid**: Top-hat yanıtıyla ağırlıklı ortalama.

9. **Referans → güncel koordinat**: `h_inv * centroid_ref`.

**9b) Yönlü yerel kontrast (LCM/WLDM) son-elemesi** *(yeni)*: IRST literatüründeki çok-yönlü Local Contrast Measure. Aday merkezinden GEÇEN kenar/çizgi karşıt iki sektörü birden parlatır → elenir; kompakt hedef (sönük olsa da) tüm yönlerde arka planla çevrili → korunur. `reg` üzerinde, top-hat∧MOG2'den geçmiş adaylara. **Recall-güvenli varsayılan** (lr=0.90, prom=10): sentetik izotropik dokuda ~no-op; gerçek yapısal arka planda (bulut/ufuk/yer) devreye girer.

**Test sonucu (gerçek)**: Recall 1.00, konum hatası 0.18 px, FP ~1.4/kare. *(Not: raporun eski "FP 0.27/kare, 8× azaltıldı" iddiası mevcut kodla uyuşmuyordu; gerçek değer ~1.4. FP'ler izotropik doku tepeleri olduğu için LCM ile daha fazla düşürmek worst-case recall'a mal oluyor — bkz. §5.3.)

### 3.6 `detection/` — Drone Skorlama [Problem 3]

| Dosya | İçerik |
|---|---|
| `stability_discriminator.cpp` | `StabilityDiscriminator` |

**Algoritma**: Her tespit için 3 bileşenli skor:

1. **Geometrik skor** (anlık): Alan (trapezoid), en/boy (Gaussian, tepe=1.0 kare), parlaklık (Gaussian).

2. **Zamansal varlık skoru**: Geçmiş tamponunda (son ~20 kare) uzamsal yakınlık (25 px) + yaş ağırlığı (`exp(-age/5)`) ile en iyi eşleşme.

3. **Özellik kararlılığı**: Eşleşen geçmiş girişle alan/aspect/parlaklık tutarlılığı.

**Birleşik**: `0.35·geometrik + 0.35·zamansal + 0.30·kararlılık`

**Test sonucu**: Hedef skoru 0.829, FP skoru 0.617 (%35 fark). Zamansal öğrenme: son skor 0.748.

### 3.7 `tracking/` — Çoklu Hedef Takip [Problem 4]

| Dosya | İçerik |
|---|---|
| `tracker.hpp` | `ITracker` arayüzü |
| `kalman_tracker.hpp` | `AlphaBetaTracker` (sınıf adı `KalmanTracker` alias ile geriye uyumlu) |
| `kalman_tracker.cpp` | α-β filtresi implementasyonu |
| `track_stage.hpp` | `TrackStage` — tracker'ı pipeline'a sarar |

**Algoritma (5 adım)**:

1. **PREDICT**: `p += v · dt` (sabit hız modeli).

2. **GATE**: Adaptive Öklid kapısı: `gate = 12 + 1.5·coast_süresi` px. Coast eden track daha geniş kapıyla tespiti geri yakalar.

3. **ASSOCIATE**: Onaylı track öncelikli (prio 2), sonra tentative (1), sonra coasting (0) aç gözlü atama. Coasting track confirmed track'in tespitini "çalmaz".

4. **UPDATE**: `pos += α · innov`, `vel += β · innov` (α=0.55, β=0.12). Coasting→Confirmed terfisi.

5. **LIFECYCLE**: M-of-N onayı (8 karede 3 tespit → Confirmed). Tentative 2 ardışık ıskalarsa silinir. max_coast=22 kare (~0.37s).

**İki-nokta başlatma**: Eşleşmeyen ilk tespit saklanır, ikinci gelince `v = (z2−z1)/dt` ile başlatılır. Hız akıl kontrolü: `|v| > 400 px/s` ise sıfır hızla başlat, yüksek belirsizlikle Kalman'a bırak.

**Hız (velocity) çıktısı fix'i** *(yeni, kritik)*: Eski kod hızı sabit `×60` ile çeviriyordu → çıktı ~11× şişkin ve fps'e bağlıydı (60fps→1700, 30fps→2400 px/s; gerçek ~58 px/s). Doğru **Kalata α-β** formülasyonuna geçildi: `v += (β/dt)·innov`, hız px/SANİYE, fps-değişmez. `correct()` artık gerçek `dt` alıyor; iki-nokta başlatma px/s saklıyor. Bu sadece çıktıyı düzeltmedi, **konum tahminini de iyileştirdi** (doğru px/s → coast köprüleme), worst senaryoda ID 3→1, continuity 0.92→0.99. Regresyon kilidi: `test_track_velocity`.

**IMM alternatifi** *(yeni)*: `ImmTracker` — iki α-β modelli (yumuşak+çevik) hafif IMM, innovation-olabilirliğiyle Markov-mikslemeli (bkz. `imm_tracker.hpp`). Kalman'a dönülmedi (§2.7 dersi). Sentetik düz-sinüs manevrada α-β ile ~eşit (tek iyi α-β yumuşak manevrayı zaten hallediyor); asıl faydası KESKİN/ani manevrada — gerçek veride doğrulanmalı. `ITracker` sayesinde tek satırla takılır. Varsayılan hâlâ α-β (worst'te daha ID-kararlı).

**Test sonucu**: Continuity 0.98+, konum hatası ~1.6 px, tek ID (kimlik atlaması yok), 5 kare boşlukta hayatta kalır + iyileşir.

### 3.8 `fusion/` — Çoklu Kamera Füzyonu [Problem 5]

| Dosya | İçerik |
|---|---|
| `track_fusion.hpp` | `ITrackFusion` arayüzü |
| `simple_track_fusion.cpp` | `SimpleTrackFusion` |

**Algoritma**:

1. **Projeksiyon**: Termal track'leri görünür koordinata `H_termal→görünür` ile taşı.

2. **Uzamsal kapı** (25 px) + **aç gözlü atama**.

3. **Eşleşen çift**: Konum = güven ağırlıklı ortalama. Güven = `C_v + C_t − C_v·C_t` (probabilistic OR, hemfikirlik bonusu ≤1).

4. **Eşleşmeyen tek**: Güven × 0.7 ceza. `min_confidence=0.15` altı elenir.

**Test sonucu**: 78/80 kilitli, hemfikirlik güveni 1.0, termal kesintide görünür devam eder.

### 3.9 `app/` — Uygulama [Problem 6]

| Dosya | İçerik |
|---|---|
| `main.cpp` | Çift kamera pipeline demo: görünür + termal paralel → füzyon → viz |
| `video_bench.cpp` | Video benchmark: MP4 dosyasını pipeline'dan geçirir, canlı görselleştirme |

**Demo pipeline**: `Camera(Vis) → Stab → Detect → Track ─┐`  
                   `Camera(Thm) → Stab → Detect → Track ─┤ Fusion → Viz`

**Viz**: Mor kutu = hemfikir (yüksek güven), yeşil = confirmed, sarı = coasting, kırmızı nokta = ileri tahmin, renkli iz = yörünge.

**Video benchmark**: Canlı `imshow`, sağ-üst metrik paneli (kare no, tespit/track sayısı, durum, pozisyon, hız, güven, ms, FPS). `q`/ESC çıkış, `p` duraklatma.

---

## 4. Klasör Yapısı (tam)

```
drone-tracker/
├── CMakeLists.txt              # Kök CMake: OpenCV opsiyonel, modül sırası
├── CLAUDE.md                   # Geliştirme kuralları (Türkçe yorum, akış sırası, araştırma şartı)
├── README.md                   # Durum checklist'i + derleme talimatları
├── initial-prompt.md           # 6 problem tanımı + zayıf yanlar
├── compile_commands.json -> build/compile_commands.json
│
├── common/
│   ├── CMakeLists.txt          # INTERFACE (header-only)
│   └── include/dtrack/common/
│       ├── ring_buffer.hpp     # SpscRingBuffer<T>
│       ├── time.hpp            # Ortak zaman + Timestamp
│       └── types.hpp           # Frame, Detection, Track, vb.
│
├── pipeline/
│   ├── CMakeLists.txt          # INTERFACE (header-only)
│   └── include/dtrack/pipeline/
│       ├── istage.hpp          # IStage arayüzü
│       ├── pipeline.hpp        # Pipeline orkestratör
│       └── stage.hpp           # Stage<In,Out> thread iskeleti
│
├── io/
│   ├── CMakeLists.txt          # STATIC
│   ├── src/
│   │   ├── synthetic_camera_source.cpp
│   │   ├── synthetic_imu_source.cpp
│   │   └── video_camera_source.cpp
│   └── include/dtrack/io/
│       ├── camera_source.hpp   # ICameraSource
│       ├── camera_stage.hpp   # CameraStage
│       ├── imu_source.hpp      # IImuSource
│       ├── synthetic_camera_source.hpp
│       ├── synthetic_imu_source.hpp
│       ├── synthetic_scene.hpp # SceneModel (matematik)
│       └── video_camera_source.hpp
│
├── stabilization/
│   ├── CMakeLists.txt          # STATIC
│   ├── src/
│   │   └── klt_gyro_stabilizer.cpp
│   └── include/dtrack/stabilization/
│       ├── stabilizer.hpp      # IStabilizer
│       ├── klt_gyro_stabilizer.hpp
│       └── stabilize_stage.hpp
│
├── detection/
│   ├── CMakeLists.txt          # STATIC
│   ├── src/
│   │   ├── mog_detector.cpp
│   │   └── stability_discriminator.cpp
│   └── include/dtrack/detection/
│       ├── detector.hpp        # IDetector
│       ├── discriminator.hpp   # IDiscriminator
│       ├── detect_stage.hpp
│       ├── mog_detector.hpp
│       └── stability_discriminator.hpp
│
├── tracking/
│   ├── CMakeLists.txt          # STATIC
│   ├── src/
│   │   └── kalman_tracker.cpp  # AlphaBetaTracker implementasyonu
│   └── include/dtrack/tracking/
│       ├── tracker.hpp         # ITracker
│       ├── kalman_tracker.hpp  # AlphaBetaTracker (alias: KalmanTracker)
│       └── track_stage.hpp
│
├── fusion/
│   ├── CMakeLists.txt          # STATIC
│   ├── src/
│   │   └── simple_track_fusion.cpp
│   └── include/dtrack/fusion/
│       ├── track_fusion.hpp    # ITrackFusion
│       └── simple_track_fusion.hpp
│
├── app/
│   ├── CMakeLists.txt
│   ├── main.cpp                # Çift kamera füzyon demo
│   └── video_bench.cpp         # Canlı video benchmark
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_ring_buffer.cpp
│   ├── test_synthetic_source.cpp
│   ├── test_stabilizer.cpp
│   ├── test_detector.cpp
│   ├── test_tracker.cpp
│   ├── test_discriminator.cpp
│   ├── test_fusion.cpp
│   └── bench_pipeline.cpp      # 17 senaryolu benchmark
│
└── samples/                    # Çıktı videoları + PNG'ler (.gitignore'da)
```

---

## 5. Benchmark Sonuçları (17 senaryo, 150 kare/her biri)

`bench_pipeline` aracı ile ölçüldü. Her senaryo farklı parametrelerle 180 kare çalıştırıldı (30 kare ısınma).

### 5.1 Metrik tanımları

| Metrik | Tanım |
|---|---|
| **Recall** | Yer-gerçeğine 8 px mesafede tespit bulunan kare / toplam kare |
| **Precision** | Hedefe yakın tespit / tüm tespitler |
| **Det RMS** | Tespit konum hatası RMS (piksel) |
| **FP/k** | Kare başına yanlış pozitif sayısı |
| **Continuity** | Kilitli kare (track < 10 px mesafede) / toplam kare |
| **Trk RMS** | Track konum hatası RMS (piksel) |
| **ID#** | Test boyunca görülen benzersiz track ID sayısı |
| **Coast %** | Coast durumunda geçen karelerin oranı |
| **ms/k** | Kare başına ortalama işlem süresi (milisaniye) |

### 5.2 Senaryo sonuçları

> **Ölçüm tarihi: hız-fix + LCM (recall-güvenli varsayılan) sonrası, x86 geliştirme
> makinesinde.** `ms/k` makineye bağlıdır (hedef ARM A78/A76'da ~2-4× daha yüksek
> beklenir); göreli kıyas için değil mutlak bütçe için değil, trend için kullanın.

| # | Senaryo | Recall | Prec | Cont | Trk RMS | ID# | ms/k |
|---|---|---|---|---|---|---|---|
| 1 | **baseline** (σ=1.1, v=55, man=40, ego=0.012, I=90, tex=18) | 1.00 | 0.43 | 1.00 | 1.59 | 1 | 6.1 |
| 2 | tiny_2px (σ=0.7) | 0.99 | 0.43 | 1.00 | 1.61 | 1 | 6.2 |
| 3 | large_6px (σ=1.8) | 1.00 | 0.53 | 1.00 | 1.59 | 1 | 6.5 |
| 4 | **medium_15px** (σ=2.5) | 1.00 | 0.77 | 1.00 | 1.60 | 1 | 6.3 |
| 5 | **close_25px** (σ=4.0) | 1.00 | 1.00 | 1.00 | 1.60 | 1 | 6.4 |
| 6 | slow_target (v=20) | 1.00 | 0.44 | 1.00 | 1.58 | 1 | 6.5 |
| 7 | fast_target (v=100) | 1.00 | 0.44 | 1.00 | 1.60 | 1 | 6.4 |
| 8 | no_maneuver (man=0) | 1.00 | 0.41 | 1.00 | 1.57 | 1 | 6.8 |
| 9 | heavy_maneuver (man=80) | 1.00 | 0.41 | 1.00 | 1.58 | 1 | 6.9 |
| 10 | calm_ego (ego=0.006) | 1.00 | 0.46 | 1.00 | 1.49 | 1 | 6.7 |
| 11 | heavy_ego (ego=0.024) | 1.00 | 0.48 | 1.00 | 1.89 | 1 | 6.7 |
| 12 | dim_target (I=50) | 1.00 | 0.43 | 1.00 | 1.60 | 1 | 6.4 |
| 13 | bright_target (I=130) | 1.00 | 0.39 | 1.00 | 1.58 | 1 | 8.0 |
| 14 | noisy_bg (tex=30) | 0.98 | 0.16 | 1.00 | 1.72 | 1 | 7.0 |
| 15 | clean_bg (tex=10) | 1.00 | 0.53 | 1.00 | 1.58 | 1 | 7.9 |
| 16 | **worst** (σ=0.7, v=100, man=80, ego=0.024, I=50, tex=30) | 0.84 | 0.16 | 0.99 | 2.79 | 1 | 6.7 |
| 17 | **best** (σ=1.8, v=20, man=0, ego=0.006, I=130, tex=10) | 1.00 | 0.49 | 1.00 | 1.48 | 1 | 8.0 |

### 5.3 Analiz

**Güçlü yanlar**:
- 17/17 senaryoda recall ≥ 0.84, continuity ≥ 0.99
- 15/17 senaryoda recall = 1.00, continuity = 1.00
- **Kimlik kararlılığı: 17/17 senaryoda TEK ID** (hız-fix sonrası worst dahil; eskiden worst'te 3 ID atlıyordu — doğru px/s tahmini coast boşluklarını köprülüyor)
- Hedef boyutu 2 px'ten 25 px'e kadar sorunsuz (çoklu ölçek top-hat + DoG)
- Hız, manevra, ego şiddeti, parlaklık, arka plan gürültüsü — hiçbiri tek başına pipeline'ı bozmuyor
- Track RMS hız-fix sonrası düştü (örn. baseline 1.84→1.59, worst 3.93→2.79)

**Zayıf yanlar (bilinen, belgelenen — DÜRÜST sayılar)**:
- **Precision düşük: kolay sahnelerde 0.39-0.53, gürültülü/worst'te 0.16.** FP/kare baseline'da ~1.3, noisy_bg'de ~5.0. DoG + çoklu ölçek top-hat bol aday üretir. ANCAK tracker M-of-N + iki-nokta hız-aklı FP'leri eliyor → continuity yine 1.00 ve tek ID. (Bu sayılar raporun eski sürümünden farklıdır; eski "FP 0.27 / prec 0.71" değerleri mevcut kodla uyuşmuyordu, düzeltildi.)
- **LCM neden precision'ı kurtarmıyor?** Yönlü yerel-kontrast eleme (§3.5) eklendi ama sentetik sahnenin FP'leri İZOTROPİK doku tepeleridir (kenar değil) → kontrast ölçüsüyle dim hedeften ayrılamazlar. Recall-güvenli varsayılanda LCM sentetikte ~no-op'tur (bilinçli); asıl faydası gerçek yapısal arka planda (bulut/ufuk/yer kenarları), `video_eval` ile doğrulanmalı.
- **worst senaryo (SNR≈1.4)**: recall 0.84 — dim+küçük+hızlı+manevra+ağır-ego birleşimi bilgi-teorik duvar (bkz. §11/sınırlar). Continuity 0.99'a çıktı (hız-fix), ama tek-kare tespit hâlâ zorlanıyor.

---

## 6. Test Sonuçları

| # | Test | Sonuç |
|---|---|---|
| 1 | `ring_buffer` | ✓ 1M eleman, 2 thread, lock-free doğrulandı |
| 2 | `synthetic_source` | ✓ Render hedefi <1.5px, IMU gyro sahneyle tutarlı |
| 3 | `stabilizer` | ✓ Artık hareket 0.10px, bias 0.030→0.031 |
| 4 | `detector` | ✓ Recall 1.00, hata 0.18px, FP ~1.4/kare (gerçek değer) |
| 5 | `tracker` | ✓ Continuity 0.98+, tek ID, boşluk köprüleme |
| 6 | `discriminator` | ✓ Hedef FP'den %35 yüksek, zamansal öğrenme |
| 7 | `fusion` | ✓ 78/80 kilitli, hemfikirlik 1.0, tek modalite fallback |
| 8 | `track_velocity` | ✓ **YENİ** — hız çıktısı px/s doğru + fps-değişmez (60×-şişme bug'ı regresyon kilidi) |
| 9 | `imm_tracker` | ✓ **YENİ** — IMM, α-β ile aynı sağlamlık çıtasını tutturuyor |

**Toplam**: 9/9 test geçti (%100).

---

## 7. Tasarım Kararlarının Kronolojisi

### 7.1 İlk kurulum
- Klasör yapısı + modüler CMake + header-only arayüzler
- Lock-free SPSC ring buffer + Stage thread iskeleti
- OpenCV opsiyonel: yoksa sadece çekirdek derlenir

### 7.2 io modülü
- `SceneModel` saf matematik (OpenCV'siz, deterministik)
- `SyntheticCameraSource` + `SyntheticImuSource` (aynı fiziksel model)
- `VideoCameraSource` (`cv::VideoCapture` sarmalayıcı — simülatör/gerçek donanım girişi)
- `CameraStage` ile pipeline'a bağlama

### 7.3 stabilization modülü [Problem 1]
- `KltGyroStabilizer`: predict (gyro) + measure (OF) + correct (bias)
- Gyro bias online kalibrasyonu (OF çıpalar)
- warp sonraya ertelendi (tüketici yapar, çift interpolasyon önlendi)

### 7.4 detection modülü [Problem 2]
- `MogDetector`: top-hat ∧ MOG2 + kümülatif referans kaydı
- Alt-piksel centroid (top-hat ağırlıklı)
- Başlangıçta 5×5 top-hat → 4 ölçekli (3×3+5×5+9×9+15×15) + 2 bant DoG
- max_area 60 → 400 (yakın hedef için)
- Referans koordinat sandviçi kaldırıldı

### 7.5 discriminator modülü [Problem 3]
- `StabilityDiscriminator`: ızgara tabanlı → uzamsal yakınlık + yaş ağırlıklı geçmiş
- 3 bileşenli skor: geometrik + zamansal varlık + özellik kararlılığı

### 7.6 tracking modülü [Problem 4]
- Önce 4-durumlu CV Kalman → ıraksama sorunu
- Sabit ivme (CA) Kalman denendi → daha kötü
- 4-durumlu CV Kalman'a dönüş + tuning → hala churn
- **α-β filtresine geçiş** — sabit kazanç, yakınsama sorunu yok
- İki-nokta hız başlatma + hız akıl kontrolü
- Adaptive Öklid kapısı (coast ile genişler)
- Onaylı track öncelikli atama

### 7.7 fusion modülü [Problem 5]
- `SimpleTrackFusion`: uzamsal kapı + güven ağırlıklı birleştirme
- Probabilistic OR güven (hemfikirlik bonusu)
- Tek modalite cezası (×0.7)

### 7.8 app + benchmark [Problem 6]
- Tek kamera demo → çift kamera füzyon demo
- Video benchmark (`video_bench`): canlı `imshow` + sağ-üst panel
- `bench_pipeline`: 17 senaryolu kapsamlı test
- `highgui` OpenCV bileşeni eklendi

---

## 8. Kullanılan Teknolojiler ve Kütüphaneler

| Bileşen | Kullanım |
|---|---|
| **C++20** | `std::optional`, `std::atomic`, `std::jthread` (pipeline thread'leri), konseptler |
| **OpenCV 4.13** | `core` (matris), `imgproc` (warp/morphology/GaussianBlur), `video` (MOG2, LK optical flow), `calib3d` (estimateAffinePartial2D), `videoio` (VideoCapture/VideoWriter), `imgcodecs` (imwrite), `highgui` (imshow) |
| **CMake 3.16+** | Modüler build sistemi, opsiyonel OpenCV, `compile_commands.json` |
| **pthread** | Stage thread'leri (CMake `find_package(Threads)`) |

---

## 9. İncelenen Harici Kaynaklar

| Kaynak | İçerik | Fayda |
|---|---|---|
| **MDPI Sensors 2021** (DIVA) | Gyro tabanlı stabilizasyon, K·R·K⁻¹ homography | Stabilizasyon tasarımını doğruladı |
| **Purdue UAV-to-UAV** | MOG2 + RANSAC, hareketli hedef outlier | Detection + tracker ayrımını doğruladı |
| **IRST top-hat** (MITHF) | Çok yönlü top-hat, küçük hedef SNR artışı | Çoklu ölçek top-hat'a ilham verdi |
| **ATA-YOLOv8** (drones-09-00154.pdf) | Havadan-havaya YOLOv8 tabanlı tespit, MOT-FLY/Det-Fly verisetleri | DL yaklaşımı, bizim CPU-only kararımızı doğruladı. Verisetleri ileride kıyaslama için |
| **Object Detection and Tracking with Autonomous UAV** (arXiv:2206.12941) | Savaş İHA simülasyonu, derin öğrenme, sistem entegrasyonu | Sistem seviyesi mimariye ışık tuttu, simülasyon kullanımını doğruladı |
| **Drone-vs-bird ayırt etme** | Radar micro-Doppler, kinematik özellikler, şekil kararlılığı | Discriminator özellik seçimini yönlendirdi |

---

## 10. Çalıştırma Talimatları

```bash
# Derleme
cmake -S . -B build
cmake --build build

# Testler
ctest --test-dir build --output-on-failure

# Sentetik demo (çift kamera + füzyon)
./build/app/drone_tracker

# Video benchmark (canlı)
./build/app/video_bench <video.mp4> [cikti_dizini]

# Kapsamlı benchmark (17 senaryo)
./build/tests/bench_pipeline

# Tek test çalıştırma
./build/tests/test_tracker
ctest --test-dir build -R tracker
```

---

## 11. Sıradaki Adımlar

1. ✅ **Hız çıktısı fix'i** (Kalata α-β, px/s, fps-değişmez) — yapıldı.
2. ✅ **Gerçek-veri değerlendirme aracı** `video_eval` (MP4 + opsiyonel GT CSV → metrikler, headless) — yapıldı.
3. ✅ **IMM** (`ImmTracker`, iki α-β modelli hafif IMM) — yapıldı (varsayılan α-β; IMM swappable).
4. **Gerçek veri doğrulama**: `videos/` klipleri + (varsa) GT etiketleriyle `video_eval` koşturup LCM/IMM faydasını gerçek arka planda ölçmek. ← *sıradaki ana iş*
5. **Simülatör entegrasyonu**: AirSim/Gazebo → RTSP/UDP → `VideoCameraSource`.
6. **Gerçek donanım**: ARM üzerinde `VideoCameraSource("/dev/video0")` + V4L2 IMU.
7. **Performans optimizasyonu**: SIMD (NEON), cache-aware veri yapıları, ARM-specific tuning.
8. **Açık veriseti doğrulama**: MOT-FLY / Det-Fly (MOT formatı `video_eval --gt` ile doğrudan).

---

## 12. Aşılamayan Sınırlar (bilgi-teorik / paradigma)

Aşağıdakiler "daha çok mühendislik"le çözülmez; tek-pasif-kamera + no-ML + 50 ms
paradigmasının duvarlarıdır. Ancak ikinci modalite (radar micro-Doppler) veya daha
çok piksel (uzun odak/yakınlaşma) ile aşılır:

1. **2-6 px'te sınıflandırma (drone vs kuş vs çöp) imkânsız** — bu boyutta şekil/doku
   bilgisi yok denecek kadar az. Sistem bu yüzden "cued tracking" çerçevesinde; sınıf-
   landırmaya girmiyor.
2. **Aynı anda sönük + küçük + sert-manevralı hedef** (worst, recall 0.84): tek-kare
   tespit düşük SNR'da çöker (TBD ister), TBD ise öngörülebilir hareket ister, sert
   manevra onu bozar. İki gereksinim çelişir → fiziksel duvar.
3. **Havada asılı / sıfır-hızlı hedef**: pipeline belkemiği hareket (MOG2 + ego-telafi);
   hareketsiz hedef ego-telafisinden sonra arka plana karışır → yapısal olarak görünmez.
4. **Parallaks / non-planar arka plan**: stabilizasyon uzak düzlemsel arka plan varsayar
   (similarity); FOV'a 3B yapı (yer/bina) girerse tek homografi hizalayamaz → artık
   parallaks FP. Tek kameradan derinlik olmadan baştan önlenemez (M-of-N ile sonradan elenir).
5. **Sentetik ↔ gerçek uçurumu**: §5 sayıları idealize sentetik sahnededir (tek hedef,
   Gaussian leke, düzlemsel gökyüzü). Gerçek-dünya rakamları için `video_eval` + etiketli
   klip şarttır; recall=1.0 gerçek veride bu kadar yüksek olmayabilir.
