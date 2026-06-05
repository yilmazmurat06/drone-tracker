# Proje: CPU-Only Onboard Air-to-Air Drone Detection & Tracking

## Bağlam ve Amaç

Bir interceptor (önleyici) drone üzerinde çalışacak, küçük düşman İHA'larını
tespit edip takip eden bir gömülü görüntü işleme sistemi geliştiriyorum.
Geometri "hava-hava" (air-to-air): hem benim platformum hem de hedef uçuyor.

**Kesin kısıtlar:**
- Dil: **C++** (modern C++17/20).
- Donanım: **GPU YOK.** Sadece CPU. Hedef platform: 6 çekirdekli ARM
  (Cortex-A78 / A76 sınıfı, 1.5–2.0 GHz) veya benzeri gömülü işlemci.
- SWaP (Size, Weight, Power): tüm hesaplama + görüntüleme yükü ~100–200 gram,
  birkaç watt. Aktif soğutma yok.
- Gerçek zamanlı: foton → direksiyon komutu gecikmesi **< 50 ms** olmalı.
- Kütüphane: OpenCV serbest. Ağır ML framework'leri (PyTorch/TF) YOK.
  Gerekirse hafif klasik CV + el yazımı Kalman/IMM.

**Temel tasarım kararı:** Bu bir *sınıflandırma* değil *takip* problemi.
Sistem zaten hedefe yönlendirilmiş (cued) durumda geliyor; bizim işimiz
hedefin karede *nerede* olduğunu yeterince hızlı bulup direksiyon komutu
üretmek. Bu yüzden ağır object detector'lar (YOLO vb.) kullanmıyoruz —
hedef zaten 2–6 piksel, öğrenilecek feature yok. Klasik sinyal/istatistik
tabanlı tespit + güçlü tracker kullanıyoruz. CPU-only bu yüzden uygulanabilir.

---

## Çözmem Gereken 6 Ana Problem

Aşağıdaki 6 problem, çözüm yaklaşımları ve bilinen zayıf yanlarıyla birlikte
listelenmiştir. Kodu bu mimariye göre kurmak istiyorum.

### Problem 1 — Ego-Motion (Kendi Hareketimden Kaynaklanan Bozulma)
Kamera titreşen, dönen, ivmelenen bir platformda. Arka plandaki hareket iki
kaynaktan geliyor: kendi platformum + hedef. Ayıramazsam, kendi hareketimden
kayan her piksel sahte hedef gibi görünür ve "hareket eden pikseli bul"
yaklaşımı çöker.

**Çözüm:** Gyroscope (IMU içinde) + sparse optical flow füzyonu.
- Gyro: açısal hızı hızlı/temiz verir ama zamanla **drift** eder.
- Sparse optical flow: birkaç köşe noktası takibi, drift yok ama gürültülü.
- Kalman filtresi ile birleştir: gyro → predict adımı, optical flow → correct.
- Sonuçtaki kamera hareketini **homography (3×3)** olarak ifade et, yeni
  frame'i warp ederek arka planı sabitle. Geriye sadece hedefin gerçek
  hareketi kalır. Warp < 1 ms.

**Zayıf yanlar:** Homography düzlemsel/uzak sahne varsayar → karmaşık 3B
arazide parallax yüzünden bozulur. Özelliksiz gökyüzünde optical flow takip
edecek nokta bulamaz → saf gyro'ya düşer, drift döner. IMU–kamera zaman
senkronu ms seviyesinde olmalı. Stabilizasyon her zaman yaklaşık, artık
hareket kalır.

### Problem 2 — Sub-Pixel Tespit (Hedef 2–6 Piksel)
Birkaç yüz metrede küçük quadcopter sadece 2–6 piksel. Bu, YOLO gibi derin
öğrenme detektörlerinin çalıştığı sınırın altında — tanınacak doku/şekil yok.
Bu bir sinyal-gürültü problemi: arka plandan sıyrılan soluk piksel kümesini
bulmak.

**"Arka plan modeli" ne demek:** Sahnenin hiçbir şey uçmazkenki normal halini
sisteme öğret (her pikselin normal değeri: soğuk gökyüzü, güneşli taraf,
geçen bulutlar). Drone girince sıcak motorları birkaç pikseli normalden
saptırır; bu sapma tespittir.

**Çözüm:** Gaussian Mixture Model (GMM) — her piksel için tek değer değil,
birkaç Gaussian dağılımı tut (çünkü "normal" birden çok durumdur). Standart
implementasyon: OpenCV'deki **MOG2** (Mixture of Gaussians v2). Birkaç frame
ile arka planı öğret, sonra her frame için foreground/background maskesi al.
Foreground pikselleri **connected-component labeling** ile blob'lara grupla.
Her blob'un centroid / alan / aspect ratio'su ile ilk eleme: çok büyük = kuş/
bulut, çok küçük = gürültü, aralıkta = aday. Birkaç ms'de çalışır.

**Zayıf yanlar:** Arka planın ön plandan daha kararlı olduğunu varsayar →
hareketli platformda artık hareket sahte blob üretir. Sabit duran (hover)
hedef arka plana karışıp kaybolur. Learning rate kritik: hızlı → gerçek
hedefi yutar, yavaş → buluta adapte olmaz. Drone'un ne olduğunu bilmez →
ardından kinematik filtre şart.

### Problem 3 — Drone'u Diğer Nesnelerden Ayırmak
Sadece hıza göre filtrelersem, drone hız aralığındaki bir kuş ne olacak?
Hız tek başına yetersiz; birkaç zayıf ayırt ediciyi birleştir.

**Çözüm (çoklu özellik):**
- Termal imza: drone motoru 50–80°C, kuş gövdesi ~40–42°C — yakın ama
  sistematik fark.
- Blob şekli ve kararlılığı: drone sabit geometrik şekil; kuş kanat çırpar,
  şekli değişir → aspect ratio'nun zamanla değişkenliği ayırt edici.
- Hareket tutarlılığı: motor tahrikli hareket düzgün; kuşun çırpışı periyodik
  hız dalgalanması yapar.
- İkinci kameradan teyit (görünür kamerada pervane imzası).

Hiçbiri tek başına kesinlik vermez → sistem **olasılık skoru** üretir
("%78 drone"). Eşiği menzile göre değiştir: uzakta düşük (kaybetme), yakında
yüksek (teyit). Bu kinematik kapı tracker'dan ayrı değil — Kalman'ın kendi
**innovation gate**'i, fiziksel olarak imkânsız hareketi reddeder.

**Zayıf yanlar:** Her özellik tek başına zayıf. Termal fark 2 piksellik
menzilde soğuk gökyüzüyle ortalanınca azalır. Şekil ayrımı yeterli piksel
ister. Hareket analizi birkaç frame ister → ayrım tam da track başlangıcında
en zayıf. Düşman, biyolojik imzayı taklit eden bir drone tasarlayıp mantığı
bozabilir.

### Problem 4 — Kalman Filtresi ile Takip
Tespit "hedef var mı?", takip "nerede ve sonraki an nerede olacak?" sorusunu
cevaplar. Bu ileri tahmin, interceptor'ın hedefin bayat konumunu kovalamak
yerine bir kesişim noktasına yönelmesini sağlar.

**Çözüm:** Kalman state'i en az 2B görüntü konumu + hız tutar. Hava-havada
derinlik önemli ama tek kamerayla ancak kaba tahmin edilir (büyüyen blob =
yaklaşıyor). Sabit hız modeli manevra yapan drone için yetersiz; en güçlüsü
**IMM (Interacting Multiple Model)** — birkaç hareket modelini (sabit hız,
sabit ivme, koordineli dönüş) paralel çalıştırıp uyuma göre harmanlar,
CPU'da uygun. Track başlatma için **M-of-N** kuralı (örn. 5 frame'de 3 tespit)
gürültüyü bastırır; tahmin, kısa boşluklarda track'i taşır.

**Zayıf yanlar:** Kalman yalnızca lineer + Gaussian varsayımda optimal; sert
manevra ihlal eder; IMM yalnızca içine konmuş manevra sınıflarını kapsar.
Tek kameradan boyut-bazlı derinlik kaba. M-of-N kaçınılmaz başlatma gecikmesi
ekler, o sürede hızlı hedef çok yol alır. En kötüsü: insan pilot, tahmin
modelini bilerek sömüren manevralar yapabilir (adversarial) — model-tabanlı
hiçbir tracker bunu tam çözmez.

### Problem 5 — Termal + Görünür Kamera Füzyonu
Görünür kamera gecede/karmaşık arka planda başarısız; termal orada çalışır
ama düşük çözünürlük + motorların ısınmasını ister. İki kamera birbirinin
açığını kapatır ama ikinci kamera ağırlık/güç yer; füzyon ucuz olmalı.

**Çözüm:** **Geç (track-seviyesi) füzyon**, erken füzyon değil. Her kamera
kendi pipeline'ında bağımsız blob üretir; yalnızca track seviyesinde birleştir.
İki kamera aynı bölgede hemfikirse güven artar; tek kamera = temkinli. Basit
ağırlıklı oylama yeter. Zaman senkronu şart — ideali donanım sync, yoksa
timestamp eşleştirme.

**Zayıf yanlar:** Geç füzyon hassasiyet kaybeder — iki kamerada da tek başına
zayıf olan hedef hiç tespit edilmez (erken füzyon kurtarabilirdi). Farklı FOV/
optik merkez → geometrik kalibrasyon gerekir, ısı/titreşimle kayar. Donanım
sync yoksa timestamp eşleştirmesi küçük zaman ofseti bırakır, hızlı hedefte
önemli. İkinci kamera yine de ağırlık/güç ister.

### Problem 6 — Pipeline Mimarisi ve Gerçek Zamanlı Entegrasyon
Foton → direksiyon komutu yolu < ~50 ms bitmeli (yüksek kapanış hızında
kontrol için). Pozlama, stabilizasyon, background subtraction, blob detection,
kinematik filtre, Kalman update, çıktı + bellek kopyaları + thread koordinasyonu
hepsi bu bütçeyi paylaşır.

**Çözüm:** Çok thread'li, **lock-free** pipeline. Ayrı thread'ler: kamera
capture, IMU, preprocessing/stabilizasyon, detection, tracking, output.
Aralarında lock-free kuyruklar (C++'ta `std::atomic` + ring buffer) → hiçbir
stage mutex'te beklemez. Stage'leri örtüştür: tracker frame N'de çalışırken
detector N+1'de, kamera N+2'yi yakalar → CPU frame hızına yetişir.

**Zayıf yanlar:** Lock-free kod doğru yazmak zor; ince atomic-ordering hataları
nadiren tekrarlanabilen race'ler üretir. Örtüşme throughput'u artırır ama tek
frame latency'sini değil — sonuç her zaman birkaç stage eski. Küçük CPU'da
çok thread bellek bandwidth'i/cache için yarışır. Frame hızına sıkı bağ →
bir stage taşarsa frame düşer, terminal manevrada düşen frame track kaybı.

---

## Genel Mimari (Hedeflenen Veri Akışı)

```
[Camera capture thread] ─┐
                         ├─> [Preprocess/Stabilize thread] (gyro+flow füzyon, warp)
[IMU thread] ────────────┘            │
                                      v
                          [Detection thread] (MOG2 → blob → centroid)
                                      │
                                      v
                          [Tracking thread] (kinematik filtre + Kalman/IMM)
                                      │
                                      v
                          [Output thread] (track state → flight controller)
```
Thread'ler arası: lock-free ring buffer (`std::atomic`).

---

## Senden İstediğim İlk Adım

Önce **proje iskeletini** kuralım. Şunları istiyorum:
1. Mantıklı bir klasör yapısı öner (modüller: `stabilization/`, `detection/`,
   `tracking/`, `fusion/`, `pipeline/`, `io/`, `tests/`).
2. CMake tabanlı build kurulumu (OpenCV bağımlılığı ile).
3. Her modül için header arayüzlerini (interface) tanımla — implementasyona
   geçmeden önce temiz API'ler görmek istiyorum.
4. Lock-free ring buffer ve thread iskeletini içeren `pipeline/` çekirdeğini
   en başta kur.

Önce klasör yapısı + CMake + boş header arayüzleri ile başla. Kod yazmadan
önce mimari kararları kısaca açıkla, sonra dosyaları oluştur. Tek seferde her
şeyi implemente etme — adım adım, her modülü ayrı ele alacağız.

**Not:** Ben bu alanda yeniyim, bu yüzden attığın her adımda kısa ama net
açıklama yap ve kullandığın kısaltmaların açılımını ver.