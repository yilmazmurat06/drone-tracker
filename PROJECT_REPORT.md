# Drone Tracker — Kapsamlı Proje Raporu

**Tarih**: 7 Haziran 2026 (güncelleme: **kapalı-döngü cued tracking + PDAF**)  
**Dil**: C++20  
**Testler**: 13 birim/regresyon testi (%100 geçti)  

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

### 2.8 α-β → gerçek Kalman (geri dönüş; §2.7 kök sebepleri giderildi)

**Karar**: Varsayılan tracker yeniden GERÇEK 4-durumlu CV Kalman'a alındı (`KalmanTracker`).
§2.7'de Kalman'ı bozan sorunlar artık tek tek adreslendi — yani §2.7 bir "Kalman kötü"
kanıtı değil, "naif Kalman + kötü tuning kötü" kanıtıydı:

| §2.7 sorunu | Bu implementasyondaki çözüm |
|---|---|
| "P kovaryansı küçük hedefte kararsız" | **Joseph-form** kovaryans güncellemesi → P daima simetrik+pozitif (naif `(I−KH)P` float'ta negatife kayar; Joseph kaymaz). `test_kalman_math` her adımda SPD'yi doğrular: 0 ihlal. |
| "`init_vel_var=1e4` → salınım" | **Ölçülü P0**: başlangıç hız std=60 px/s (1e4 değil). İlk karelerde aşırı kazanç yok. |
| "kazanç manevrada yavaş yakınsar" | Süreç gürültüsü σ_a (manevra ivmesi) ile **fiziksel** ayar. Varsayılanlar (σ_r=1.5px, σ_a=1500px/s²) kararlı-durum kazancını α-β ile EŞLER: `test_kalman_math` k0=0.523 (α-β'de α=0.55) ölçtü. Yani en kötü ihtimalle α-β kadar iyi. |
| "iki-nokta başlatma absürt hız" | Hız akıl kontrolü korundu (>400 px/s reddedilir). |

**α-β'ye göre net kazanım**: kapı (gating) artık **NIS/Mahalanobis** tabanlı (S = HPHᵀ+R).
Coast'ta P büyür → S büyür → kapı DOĞAL genişler; §2.7-α-β'deki elle `gate_expand_per_frame`
katsayısı gereksiz. Değişken dt (kare atlama/jitter) ilkesel olarak ele alınır.

**Mimari**: Çekirdek matematik `kalman_core.hpp`'de `Kf2` olarak — köşegen R ile 4-durumlu CV
filtresi iki bağımsız 2-durumlu eksene ayrışır (her iz iki Kf2). Saf C++ (OpenCV'siz) olduğu
için `tests/test_kalman_math.cpp` ile OpenCV kurmadan koşturulabilir/doğrulanabilir. α-β
(`AlphaBetaTracker`) YEDEK olarak korundu (`alpha_beta_tracker.hpp`; kararlı-durumda Kalman'a
denk, karşılaştırma temeli). IMM artık iki GERÇEK Kalman modeli kullanıyor (§3.7).

**Doğrulama**: `test_kalman_math` (saf çekirdek) + `kalman_selftest.py` (Python eşi, birebir
aynı sayılar: k0=0.523, RMS≈0.97px, KalmanTracker kilit sürekliliği 1.00 boşluk+clutter ile) +
tüm C++ dosyaları gerçek OpenCV başlıklarına karşı derlenir. Tam pipeline testleri (`test_tracker`
vb.) OpenCV gerektirir → `ctest` ile koşturulur.

### 2.9 Kapalı-döngü "cued tracking" — track → detect geri beslemesi (ANA İYİLEŞTİRME)

**Sorun**: Pipeline açık-döngüydü (detect → track). Hedefe *kilitlendikten sonra* tracker
hedefin nerede olacağını (Kalman tahmini) biliyordu ama bu bilgi tespite GERİ DÖNMÜYORDU.
Sonuç: global tespit (top-hat ∧ MOG2) bir karede hedefi kaçırırsa (düşük SNR, hover, ego
artığı), iz yalnızca körlemesine coast ediyordu → worst-case recall 0.84, coast %10.

**Çözüm — kapalı-döngü ROI kurtarma (IRST literatürü)**:
1. **CueBoard** (`common/cue.hpp`): tek-yazar/çok-okur lock-free anlık görüntü (seqlock).
   Tracker her karede en güvenli Confirmed/Coasting izin **ileri tahminini** (`predicted`)
   ve coast'la büyüyen kapı yarıçapını yazar; detektör okur. IMU gibi cue de pipeline
   kuyruğuna girmez — küçük bir YAN KANALdır (bir kare bayatlık zararsız: cue zaten bir
   tahmindir). Threaded app'te `TrackStage` yazar, `DetectStage` okur; bench/video_eval'de
   senkron `det.set_cue(make_cue(tracks))`.
2. **MogDetector kurtarma geçişi** (`mog_detector.cpp` adım 7): global pass tahmin kapısı
   içinde aday bulamazsa, tahmin etrafındaki ROI'de **DÜŞÜK eşik** (k=3, global k=6) +
   **MOG2-AND şartı OLMADAN** (track-before-detect-lite) tek en iyi top-hat tepesini
   alt-piksel centroid'le kurtarır. MOG2-AND'siz olması kritik: hover/düşük-SNR hedef
   zamansal hareket maskesini tetiklemese de yakalanır. Lokal SNR tabanı (mean+3σ, floor 12)
   saf gürültünün kurtarma üretmesini engeller (test ile doğrulandı: boş cue → 0 sahte).
3. **Az-güvenli ölçüm**: kurtarma tespiti `meas_std` ipucuyla (σ_r=3px) işaretlenir; tracker
   bunu daha büyük R ile (Kalman + IMM) işler → daha az ağırlıklı, daha geniş etkin kapı.

**Sonuç (bench A/B, cue KAPALI→AÇIK)**: worst-case recall **0.84 → 1.00**, coast **%10 → %0**,
worst trk_rms 2.30 → 2.02. **17/17 senaryoda recall=1.00 & continuity=1.00.** Latency maliyeti
yok (kurtarma yalnız kaçırılan karede, küçük ROI'de çalışır; ms/kare ~6 değişmedi). Kurtarma
gerçek görüntü kanıtına dayanır (det_rms 0.18px korundu → tahmin yankısı değil).

### 2.10 PDAF — clutter-dayanıklı veri ilişkilendirme (varsayılan)

**Sorun**: "En yakını seç" (nearest-neighbor) ilişkilendirme, hedefe yakın clutter'da
(kapı içinde birden çok aday) yanlış adaya kilitlenir; ayrıca her clutter kümesinden
tentative iz açılıp **kimlik parçalanır** (sahte rakip izler).

**Çözüm — PDAF** (`Probabilistic Data Association Filter`, Bar-Shalom): yerleşik
(Confirmed/Coasting) izler için kapı içindeki TÜM adaylar olabilirliğe göre
ağırlıklandırılır:
- Olabilirlik `L_i = exp(−NIS_i/2)`; clutter terimi `b = λ·2π·√|S|·(1−P_D·P_G)/P_D`.
- Ağırlıklar `β_i = L_i/(b+ΣL)`, "ölçüm yok" `β_0 = b/(b+ΣL)`.
- Birleşik innovation `ỹ = Σ β_i y_i`; durum `x += K·ỹ`; kovaryans moment-eşleme:
  `P = β_0·P_pred + (1−β_0)·P^c + K·(Σβ_i y_i² − ỹ²)·Kᵀ` (innovation yayılımı).
- Köşegen R + ayrışık eksen sayesinde GÜNCELLEME eksen-bazında `Kf2::correct_pda` ile yapılır.
- Kapı içi adaylar yerleşik ize "ait" sayılır → **clutter sahte iz DOĞURAMAZ**. Tentative
  izler hâlâ NN (başlatma sade kalsın). `use_pdaf=false` → eski saf-NN yolu (bayt-aynı).

**Sonuç (`test_pdaf_clutter`, hedefe yakın clutter + %20 kaçırma)**: kimlik kararlılığı
**NN 5 ID → PDAF 1 ID**, continuity 1.00; RMS NN ile kıyaslanabilir (1.62 vs 1.73 — NN'in
düşük RMS'i "kare başına en yakın izi seç" metriğinin çoklu-iz parçalanmasını ödüllendiren
yanılsamasıdır). Sentetik bench'te (FP'ler hedeften uzak) PDAF≈NN → regresyon yok.

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

**10) Kapalı-döngü cued ROI kurtarma** *(YENİ — §2.9)*: `set_cue()` ile tracker'ın bir
sonraki kare tahmini alınır. Global pass tahmin kapısı içinde aday bulamazsa, ROI'de düşük
eşik + MOG2-AND'siz tek en iyi top-hat tepesi alt-piksel kurtarılır (`meas_std` ile az-güven
işaretli). Hover/düşük-SNR'da kilidi korur; lokal SNR tabanı sahte üretmeyi engeller.

**Test sonucu (gerçek)**: Recall 1.00, konum hatası 0.18 px, FP ~1.4/kare. Kapalı-döngü ile
worst-case sistem-recall 0.84→1.00 (FP'ler iz-seviyesinde önemsiz; bkz. §5.3).

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
| `kalman_core.hpp` | `Kf2` — 1B CV Kalman çekirdeği (Joseph-form), OpenCV'siz, birim-test edilebilir |
| `kalman_tracker.hpp/.cpp` | `KalmanTracker` (VARSAYILAN) — gerçek 4-durumlu CV Kalman (iz başına iki Kf2) |
| `alpha_beta_tracker.hpp/.cpp` | `AlphaBetaTracker` (YEDEK) — α-β sabit-kazanç filtresi |
| `imm_tracker.hpp/.cpp` | `ImmTracker` — iki Kalman modelli (yumuşak+çevik) IMM |
| `track_stage.hpp` | `TrackStage` — tracker'ı pipeline'a sarar |

**Algoritma (5 adım, `KalmanTracker`)**:

1. **PREDICT**: her eksen `x = F x`, `P = F P Fᵀ + Q` (σ_a² beyaz-gürültü ivme; Joseph-form correct).

2. **GATE**: **NIS/Mahalanobis** kapısı: `NIS = yᵀS⁻¹y ≤ 13.8` (2-dof χ², %99.9). Coast'ta P (dolayısıyla S) büyür → kapı DOĞAL genişler (elle katsayı yok).

3. **ASSOCIATE**: İki yol. **PDAF (varsayılan, §2.10)**: yerleşik (Confirmed/Coasting) izler
   kapı içi TÜM adayları olabilirliğe göre ağırlıklandırır (clutter sahte iz doğuramaz, ID
   kararlı); tentative izler NN. **NN yedek** (`use_pdaf=false`): onaylı öncelikli aç gözlü atama
   (prio 2/1/0, eşitlikte küçük NIS önce; coasting track confirmed'ın tespitini "çalmaz").

4. **UPDATE**: Kalman düzeltmesi (Joseph-form, her eksen; PDAF'te `correct_pda` birleşik
   innovation + spread-of-innovations). Kararlı-durum kazancı σ_a/σ_r ile ayarlanır
   (varsayılan ≈ α-β α=0.55). Per-tespit R ipucu (`meas_std`) varsa o kullanılır
   (kapalı-döngü kurtarma ölçümü daha büyük R ile). Coasting→Confirmed terfisi.

5. **LIFECYCLE**: M-of-N onayı (8 karede 3 tespit → Confirmed). Tentative 2 ardışık ıskalarsa silinir. max_coast=22 kare (~0.37s).

**İki-nokta başlatma**: Eşleşmeyen ilk tespit saklanır, ikinci gelince `v = (z2−z1)/dt` ile başlatılır. Hız akıl kontrolü: `|v| > 400 px/s` ise sıfır hızla başlat, yüksek belirsizlikle Kalman'a bırak.

**Hız (velocity) çıktısı fix'i** *(yeni, kritik)*: Eski kod hızı sabit `×60` ile çeviriyordu → çıktı ~11× şişkin ve fps'e bağlıydı (60fps→1700, 30fps→2400 px/s; gerçek ~58 px/s). Doğru **Kalata α-β** formülasyonuna geçildi: `v += (β/dt)·innov`, hız px/SANİYE, fps-değişmez. `correct()` artık gerçek `dt` alıyor; iki-nokta başlatma px/s saklıyor. Bu sadece çıktıyı düzeltmedi, **konum tahminini de iyileştirdi** (doğru px/s → coast köprüleme), worst senaryoda ID 3→1, continuity 0.92→0.99. Regresyon kilidi: `test_track_velocity`.

**IMM alternatifi**: `ImmTracker` — iki GERÇEK Kalman modelli (yumuşak σ_a=400 + çevik σ_a=4000) IMM. Mod olabilirliği artık Kalman'ın innovation kovaryansı S'ten gelir (Λ_j = N(y;0,S_j)), sabit-sigma varsayımı yok; mod olasılıkları Markov geçiş matrisiyle karışır, durum+kovaryans tam IMM-karışımıyla harmanlanır (bkz. `imm_tracker.hpp`). §2.7 endişesi (P kararsızlığı) Joseph-form + ölçülü P0 ile giderildiği için artık Kalman-tabanlı IMM güvenli. Asıl faydası KESKİN/ani manevrada — gerçek veride doğrulanmalı. `ITracker` sayesinde tek satırla takılır. Varsayılan tek-modelli `KalmanTracker` (worst'te daha ID-kararlı).

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
│   ├── CMakeLists.txt              # STATIC
│   ├── src/
│   │   ├── kalman_tracker.cpp      # KalmanTracker (gerçek CV Kalman, varsayılan)
│   │   ├── alpha_beta_tracker.cpp  # AlphaBetaTracker (yedek)
│   │   └── imm_tracker.cpp         # ImmTracker (iki Kalman modelli IMM)
│   └── include/dtrack/tracking/
│       ├── tracker.hpp             # ITracker
│       ├── kalman_core.hpp         # Kf2 (1B CV Kalman çekirdek, OpenCV'siz)
│       ├── kalman_tracker.hpp      # KalmanTracker (varsayılan)
│       ├── alpha_beta_tracker.hpp  # AlphaBetaTracker (yedek)
│       ├── imm_tracker.hpp         # ImmTracker
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

> **Ölçüm: kapalı-döngü cued tracking (§2.9) + PDAF (§2.10) sonrası, x86 geliştirme
> makinesinde.** `ms/k` makineye bağlıdır (hedef ARM A78/A76'da ~2-4× daha yüksek
> beklenir); mutlak bütçe için değil trend için kullanın. Cued recovery'nin etkisini
> görmek için `bench_pipeline` çıktısının "A/B" bölümüne bakın (cue KAPALI→AÇIK).

| # | Senaryo | Recall | Prec | Cont | Trk RMS | ID# | ms/k |
|---|---|---|---|---|---|---|---|
| 1 | **baseline** (σ=1.1, v=55, man=40, ego=0.012, I=90, tex=18) | 1.00 | 0.43 | 1.00 | 1.64 | 1 | 5.9 |
| 2 | tiny_2px (σ=0.7) | 1.00 | 0.43 | 1.00 | 1.63 | 1 | 5.9 |
| 3 | large_6px (σ=1.8) | 1.00 | 0.53 | 1.00 | 1.63 | 1 | 6.0 |
| 4 | **medium_15px** (σ=2.5) | 1.00 | 0.77 | 1.00 | 1.64 | 1 | 6.0 |
| 5 | **close_25px** (σ=4.0) | 1.00 | 1.00 | 1.00 | 1.65 | 1 | 6.0 |
| 6 | slow_target (v=20) | 1.00 | 0.44 | 1.00 | 1.63 | 1 | 5.8 |
| 7 | fast_target (v=100) | 1.00 | 0.44 | 1.00 | 1.64 | 1 | 5.8 |
| 8 | no_maneuver (man=0) | 1.00 | 0.41 | 1.00 | 1.62 | 1 | 5.9 |
| 9 | heavy_maneuver (man=80) | 1.00 | 0.41 | 1.00 | 1.62 | 1 | 5.8 |
| 10 | calm_ego (ego=0.006) | 1.00 | 0.46 | 1.00 | 1.58 | 1 | 5.9 |
| 11 | heavy_ego (ego=0.024) | 1.00 | 0.48 | 1.00 | 1.80 | 1 | 5.9 |
| 12 | dim_target (I=50) | 1.00 | 0.43 | 1.00 | 1.64 | 1 | 5.9 |
| 13 | bright_target (I=130) | 1.00 | 0.39 | 1.00 | 1.63 | 1 | 7.4 |
| 14 | noisy_bg (tex=30) | 1.00 | 0.17 | 1.00 | 1.68 | 1 | 5.9 |
| 15 | clean_bg (tex=10) | 1.00 | 0.53 | 1.00 | 1.62 | 1 | 7.5 |
| 16 | **worst** (σ=0.7, v=100, man=80, ego=0.024, I=50, tex=30) | 1.00 | 0.18 | 1.00 | 2.02 | 1 | 5.9 |
| 17 | **best** (σ=1.8, v=20, man=0, ego=0.006, I=130, tex=10) | 1.00 | 0.49 | 1.00 | 1.57 | 1 | 7.6 |

**A/B (cued recovery KAPALI → AÇIK), zor senaryolar:**

| Senaryo | Recall | Coast % | Trk RMS |
|---|---|---|---|
| worst_dim_tiny_fast | 0.84 → **1.00** | 9.3 → **0.0** | 2.31 → **2.02** |
| noisy_bg | 0.98 → **1.00** | 0.7 → **0.0** | 1.70 → 1.68 |
| tiny_2px | 0.99 → **1.00** | 0.0 → 0.0 | 1.63 → 1.63 |

### 5.3 Analiz

**Güçlü yanlar**:
- **17/17 senaryoda recall = 1.00 VE continuity = 1.00** (kapalı-döngü cued recovery sonrası;
  worst-case recall 0.84→1.00, coast %10→%0 — eskiden tek-kare tespit düşük SNR'da çökerken
  şimdi tahmin ROI'sinden kurtarılıyor)
- **Kimlik kararlılığı: 17/17 senaryoda TEK ID** (sentetik; clutter testinde PDAF NN'in
  5-ID parçalanmasını 1-ID'ye indiriyor — bkz. §2.10)
- Hedef boyutu 2 px'ten 25 px'e kadar sorunsuz (çoklu ölçek top-hat + DoG)
- Hız, manevra, ego şiddeti, parlaklık, arka plan gürültüsü — hiçbiri tek başına pipeline'ı bozmuyor
- Latency artmadı: cued recovery yalnız kaçırılan karede küçük ROI'de çalışır (ms/kare ~6)

**Zayıf yanlar (bilinen, belgelenen — DÜRÜST sayılar)**:
- **Detektör-seviyesi precision düşük: kolay sahnelerde 0.39-0.53, gürültülü/worst'te 0.16-0.18.**
  FP/kare baseline'da ~1.3, noisy_bg'de ~5.0. DoG + çoklu ölçek top-hat bol aday üretir.
  ANCAK iz-seviyesinde bu önemsiz: tracker M-of-N + NIS kapısı + (yeni) **PDAF** FP'leri
  eliyor → continuity 1.00 ve tek ID. Kapalı-döngü cue de kilitliyken ilgiyi tahmin ROI'sine
  topladığı için FP'ler izi etkilemiyor. (Yani "precision" sayısı yanıltıcı bir ham-detektör
  metriği; sistem-seviyesi performans tam.)
- **LCM neden ham precision'ı kurtarmıyor?** Sentetik sahnenin FP'leri İZOTROPİK doku
  tepeleridir (kenar değil) → kontrast ölçüsüyle dim hedeften ayrılamazlar. Asıl faydası
  gerçek yapısal arka planda (bulut/ufuk/yer kenarları), `video_eval` ile doğrulanmalı.
- **worst senaryo, tek-kare global tespit hâlâ SNR≈1.4'te zorlanır** (ham recall ~0.84).
  Bu bilgi-teorik bir duvar (§12). ANCAK kapalı-döngü cued recovery, kilitliyken bu
  kaçırmaları tahmin ROI'sinden kurtardığı için **sistem-seviyesi recall 1.00'a, coast %0'a**
  çıkar. Yani duvar, *kilitten önce* (acquisition) hâlâ geçerli; *kilitten sonra* (asıl
  hedef: kesintisiz takip) pratikte aşıldı.

---

## 6. Test Sonuçları

| # | Test | Sonuç |
|---|---|---|
| 1 | `ring_buffer` | ✓ 1M eleman, 2 thread, lock-free doğrulandı |
| 2 | `synthetic_source` | ✓ Render hedefi <1.5px, IMU gyro sahneyle tutarlı |
| 3 | `stabilizer` | ✓ Artık hareket 0.10px, bias 0.030→0.031 |
| 4 | `detector` | ✓ Recall 1.00, hata 0.18px, FP ~1.4/kare (gerçek değer) |
| 5 | `tracker` | ✓ Continuity 0.98+, tek ID, boşluk köprüleme (artık `KalmanTracker`) |
| 6 | `discriminator` | ✓ Hedef FP'den %35 yüksek, zamansal öğrenme |
| 7 | `fusion` | ✓ 78/80 kilitli, hemfikirlik 1.0, tek modalite fallback |
| 8 | `track_velocity` | ✓ hız çıktısı px/s doğru + fps-değişmez (60×-şişme bug'ı regresyon kilidi) |
| 9 | `imm_tracker` | ✓ IMM (iki Kalman modelli), tek Kalman ile aynı sağlamlık çıtası |
| 10 | `alpha_beta_tracker` | ✓ yedek α-β aynı çıta (Kalman karşılaştırma temeli) |
| 11 | `kalman_math` | ✓ Kf2 çekirdek (OpenCV'siz): P-SPD kararlı, kazanç≈α-β, fps-değişmez |
| 12 | `cue_recovery` | ✓ **YENİ** — kapalı-döngü ROI kurtarma: düşük-SNR'da kurtarır (hata 0.18px), boş cue'da sahte üretmez, entegrasyon recall 0.84→1.00 |
| 13 | `pdaf_clutter` | ✓ **YENİ** — PDAF hedefe yakın clutter'da kimlik kararlılığı (NN 5 ID → PDAF 1 ID), continuity 1.00 |

**Toplam**: 13/13 test geçti (%100). (`kalman_math` OpenCV olmadan da koşar.)

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
- α-β filtresine geçiş — sabit kazanç, yakınsama sorunu yok
- İki-nokta hız başlatma + hız akıl kontrolü
- Adaptive Öklid kapısı (coast ile genişler)
- Onaylı track öncelikli atama
- **GERÇEK Kalman'a dönüş (§2.8)** — Joseph-form kovaryans (P kararsızlığı çözüldü) + ölçülü P0 (salınım yok) + σ_a/σ_r ile kararlı-durum kazancı α-β'ye eşitlendi; NIS kapısı coast'ta doğal genişler. α-β yedek olarak korundu; IMM iki Kalman modeline geçti. Çekirdek `Kf2` OpenCV'siz birim-test edilir.

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
| **IRST kapalı-döngü ROI** (Sage J. Aerospace 2019; ScienceDirect grid-density+KCF 2022; IRSDT 2023) | "Kalman ile ROI tahmini + kayıp telafisi", track-before-detect vs detect-before-track | **Kapalı-döngü cued recovery (§2.9)** tasarımını doğruladı |
| **PDAF** (Bar-Shalom & Tse; "The Probabilistic Data Association Filter" IEEE CSM 2009) | Tek hedef + clutter'da Bayesçi yumuşak ilişkilendirme; NN'e üstünlük (2 sahte ölçümde NN ~1/3 iz kaybı), veri kaybında daha iyi | **PDAF (§2.10)** formülasyonu + clutter dayanıklılığı |

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
3. ✅ **IMM** (`ImmTracker`, iki Kalman modelli) — yapıldı (swappable).
4. ✅ **Kapalı-döngü cued tracking (§2.9)** — track→detect geri besleme + ROI kurtarma; worst recall 0.84→1.00, coast %10→%0.
5. ✅ **PDAF (§2.10)** — clutter-dayanıklı yumuşak ilişkilendirme (varsayılan); ID kararlılığı NN 5→1.
6. **Gerçek veri doğrulama**: `videos/` klipleri + (varsa) GT etiketleriyle `video_eval` koşturup cued recovery/PDAF/LCM faydasını gerçek arka planda ölçmek. ← *sıradaki ana iş*
7. **Simülatör entegrasyonu**: AirSim/Gazebo → RTSP/UDP → `VideoCameraSource`.
8. **Gerçek donanım**: ARM üzerinde `VideoCameraSource("/dev/video0")` + V4L2 IMU.
9. **Performans optimizasyonu**: SIMD (NEON), cache-aware veri yapıları; pipeline yoklama
   (200µs sleep-poll) yerine bloklamalı bekleme ile stage-handoff latency'sini kısmak.
10. **Açık veriseti doğrulama**: MOT-FLY / Det-Fly (MOT formatı `video_eval --gt` ile doğrudan).

---

## 12. Aşılamayan Sınırlar (bilgi-teorik / paradigma)

Aşağıdakiler "daha çok mühendislik"le çözülmez; tek-pasif-kamera + no-ML + 50 ms
paradigmasının duvarlarıdır. Ancak ikinci modalite (radar micro-Doppler) veya daha
çok piksel (uzun odak/yakınlaşma) ile aşılır:

1. **2-6 px'te sınıflandırma (drone vs kuş vs çöp) imkânsız** — bu boyutta şekil/doku
   bilgisi yok denecek kadar az. Sistem bu yüzden "cued tracking" çerçevesinde; sınıf-
   landırmaya girmiyor.
2. **Aynı anda sönük + küçük + sert-manevralı hedef** (worst, *ham* tek-kare recall ~0.84):
   tek-kare global tespit düşük SNR'da çöker. Bu duvar *acquisition'da* (kilitten önce) hâlâ
   geçerli. ANCAK **kilitten sonra** kapalı-döngü cued recovery (§2.9) bu kaçırmaları tahmin
   ROI'sinden kurtardığı için sistem-seviyesi recall 1.00, coast %0 → "kesintisiz takip"
   hedefi pratikte sağlandı. (Asıl iş zaten *cued tracking*, sınırsız acquisition değil.)
3. **Havada asılı / sıfır-hızlı hedef**: global belkemiği hareket (MOG2 + ego-telafi);
   hareketsiz hedef ego-telafisinden sonra arka plana karışır. Kapalı-döngü ROI kurtarma
   MOG2-AND şartını ROI'de düşürdüğü için (track-before-detect) *kilitliyken* hover'ı
   uzamsal saliency'den (top-hat) yakalayabilir; ama *kilitten önce* hover hâlâ görünmez.
4. **Parallaks / non-planar arka plan**: stabilizasyon uzak düzlemsel arka plan varsayar
   (similarity); FOV'a 3B yapı (yer/bina) girerse tek homografi hizalayamaz → artık
   parallaks FP. Tek kameradan derinlik olmadan baştan önlenemez (M-of-N ile sonradan elenir).
5. **Sentetik ↔ gerçek uçurumu**: §5 sayıları idealize sentetik sahnededir (tek hedef,
   Gaussian leke, düzlemsel gökyüzü). Gerçek-dünya rakamları için `video_eval` + etiketli
   klip şarttır; recall=1.0 gerçek veride bu kadar yüksek olmayabilir.
   **Somut ölçüm (7 Haz 2026, `video_eval videos/flight_02_clip...mp4`, GT'siz)**: kapalı-döngü
   gerçek videoda çöksüz koşar (~52 FPS, x86), ANCAK **~560 tespit/kare, ~82 iz/kare** üretir.
   Sebep: gerçek klipte IMU yok → stabilizasyon saf-OF → titreşimde artık hareket → MOG2 her
   yerde tetikler (clutter seli). Cued recovery + PDAF *birincil* izi korur ama clutter selini
   *bastırmaz*. Bu bir DETEKTÖR/EŞİK ayarı problemidir ve GT olmadan sorumluca çözülemez
   ("tahmin-ayarı" recall'ı bozma riski taşır). Doğru sıralı çözüm: etiketli klip → `video_eval`
   ile LCM eşiği / CFAR-tipi adaptif eşik / max-iz sınırı tuning (bkz. §11.6). Mimari (kapalı-
   döngü + PDAF) bu işe hazır; eksik olan veridir, kod değil.

   **Seyrek-GT ölçümü (7 Haz 2026, flight_02_clip, 14 kare elle işaretli, hedef=zeplin/balon):**
   Klip aslında *yavaş büyük bir zeplin* içeriyor (2-6px dron değil) — `SkyRegionDetector`
   tam bu rejim için. Gerçek sayılar (match=80px):
   - `mog` detektör: arazide ~560 tespit/kare sel → kullanılamaz (yanlış araç).
   - `sky` detektör (baseline): zeplin recall **0.21**, continuity **0.14**, trk RMS 70px →
     zeplin yavaş olduğu için hareket-tabanlı (frame-diff) tespit onu çoğunlukla kaçırıyor.
   - **Denenen ve GERİ ALINAN iyileştirmeler** (GT olmadan "kazanım" sanılırdı):
     (a) gökyüzü-kapılı dark-saliency → marjinal + latency 2× + bulut-kenarı FP;
     (b) sky cue-recovery → proxy %40 ama görselde sahte-iz uzatması, doğrulanamadı;
     (c) gökyüzü-"delik" tespiti → zeplin recall **0.21→0.79** AMA FP **18/kare**, **35 iz/kare**
     (kapalı bulut boşlukları da "delik") → görselde kullanılamaz. GT + görsel olmadan (c)
     "harika kazanım" sanılırdı; **GT bunun aldatıcı olduğunu kanıtladı** → şart.
   - **Dürüst sonuç**: zeplini bulut-boşluğu clutter'ından temiz ayırmak, çok-kareli hareket
     tutarlılığı (zeplin bulut sürüklenmesinden BAĞIMSIZ hareket eder) veya daha zengin
     görünüm/şekil modeli ister; her ikisi de YOĞUN GT ile geliştirilip doğrulanmalıdır.
     Sentetik-doğrulanmış kazanımlar (kapalı-döngü cued recovery + PDAF) bağımsızca geçerli;
     bu gerçek-klip ayrımı ayrı bir araştırma kalemidir (§11.6).
