#pragma once
//
// Pipeline boyunca akan ortak veri tipleri. Tüm modüller bu tipleri konuşur;
// böylece modüller birbirinin iç detayını değil yalnızca bu sözleşmeyi bilir.

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/time.hpp"

namespace dtrack::common {

// Kaynak (source) stage'lerde "giriş tipi yok" anlamına gelen sentinel.
// Stage<Tick, FramePtr>: "girdi almadan kare üret" demektir.
struct Tick {};

// Hangi kameradan geldiği. Geç (track-seviyesi) füzyonda kaynak takibi için.
enum class Modality : std::uint8_t {
    Visible = 0,  // görünür ışık kamerası
    Thermal = 1,  // termal (LWIR) kamera
};

// IMU'dan tek bir örnek. Stabilizasyonun "predict" adımını gyro besler.
struct ImuSample {
    Timestamp stamp{};
    cv::Vec3f angular_velocity{0, 0, 0};  // rad/s  (gyroscope)
    cv::Vec3f acceleration{0, 0, 0};      // m/s^2  (accelerometer)
};

// Tek bir kamera karesi. shared_ptr ile taşınır -> stage'ler arası kopya yok.
struct Frame {
    std::uint64_t index{0};   // monoton kare numarası (drop tespiti için)
    Timestamp stamp{};        // poz alındığı an
    Modality modality{Modality::Visible};
    cv::Mat image;            // ham veya stabilize görüntü (referans-sayımlı)
};
using FramePtr = std::shared_ptr<const Frame>;

// Kamera hareketini ifade eden 3x3 homography. Yeni kareyi warp edip arka planı
// sabitlemek için (bkz. Problem 1). Birim matris = hareket yok.
struct EgoMotion {
    Timestamp stamp{};
    cv::Matx33f homography = cv::Matx33f::eye();
    bool valid{false};  // false ise (örn. özelliksiz gökyüzü) saf gyro'ya düşülmüş
};

// Stabilizasyon çıktısı: HAM (orijinal) kare + kestirilen kare-arası ego-motion.
// Stabilizer warp YAPMAZ; sadece ego.homography'yi (güncel->önceki) kestirir.
// Gerçek warp'ı tüketici (detection) kendi kalıcı referansına kümülatif yapar.
struct StabilizedFrame {
    FramePtr frame;   // orijinal kare (warp edilmemiş)
    EgoMotion ego;    // güncel -> önceki kare dönüşümü + valid bayrağı
};

// Detektörün ürettiği ham aday (henüz "drone mu?" kararı yok).
struct Detection {
    Timestamp stamp{};
    Modality modality{Modality::Visible};
    cv::Point2f centroid{0, 0};  // alt-piksel konum (görüntü koordinatı)
    float area_px{0};            // blob alanı (piksel)
    float aspect_ratio{1.0f};    // şekil ayrımı için (kuş kanat çırpar -> değişir)
    float intensity{0};          // ortalama/tepe parlaklık (termalde ~sıcaklık vekili)
    // [0,1] "drone olma" olasılık skoru. Tek özellik kesinlik vermez; çoklu
    // zayıf ayırt edici birleştirilir (bkz. Problem 3). 0 = henüz skorlanmadı.
    float drone_score{0};
};

// Takip durumu. Kalman/IMM state'inin dışarıya açılan özeti.
enum class TrackStatus : std::uint8_t {
    Tentative = 0,  // M-of-N doğrulaması bekliyor (henüz onaylanmadı)
    Confirmed = 1,  // onaylı track
    Coasting = 2,   // tespit yok, tahminle taşınıyor (kısa boşluk)
    Lost = 3,       // kaybedildi, silinecek
};

struct Track {
    std::uint32_t id{0};
    Timestamp stamp{};
    TrackStatus status{TrackStatus::Tentative};

    cv::Point2f position{0, 0};   // tahmini görüntü konumu (alt-piksel)
    cv::Point2f velocity{0, 0};   // piksel/saniye (image-plane)
    cv::Point2f predicted{0, 0};  // bir sonraki an için ileri tahmin (kesişim için)

    float confidence{0};          // birleşik güven [0,1]
    float scale{1.0f};            // blob ölçeği -> kaba menzil vekili (büyüyen=yaklaşıyor)
    int hits{0};                  // toplam eşleşen tespit sayısı
    int time_since_update{0};     // tespitsiz geçen kare (coasting süresi)
};

// Demet (bundle) tipleri: kareyi aşağı akışa (tracking, görselleştirme, çıktı)
// taşımak için. Tespit/track listesinin yanında orijinal görüntü de gerekir
// (örn. track kutusunu çizmek ya da uçuş kontrolcüsüne kare-ilişkili çıktı vermek).
struct FrameDetections {
    FramePtr frame;
    std::vector<Detection> detections;
    // Güncel görüntü -> kararlı referans koordinat dönüşümü. Tracking, ego-jitter'dan
    // arınmış pürüzsüz referans koordinatında takip etmek için kullanır.
    cv::Matx33f image_to_ref = cv::Matx33f::eye();
};

struct FrameTracks {
    FramePtr frame;
    std::vector<Track> tracks;
};

}  // namespace dtrack::common
