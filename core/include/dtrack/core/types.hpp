#pragma once
// ============================================================================
//  Çekirdek veri tipleri — tüm modüller bunları paylaşır.
//  Bu tipler modüller arası "sözleşme"dir: io bir Frame üretir, detection
//  Detection üretir, tracking Track üretir. Arayüzler bunlarla konuşur.
// ============================================================================
#include <cstdint>
#include <opencv2/core.hpp>

namespace dtrack {

// Zaman damgası: nanosaniye (steady_clock tabanlı, bkz. time.hpp).
// İşaretli 64-bit → ~292 yıl aralık, taşma derdi yok.
using Timestamp = std::int64_t;

// ---------------------------------------------------------------------------
// Frame: tek bir kamera karesi.
// ---------------------------------------------------------------------------
struct Frame {
    cv::Mat      image;     // görüntü (BGR veya gri). Sahibi bu struct.
    std::int64_t id = -1;   // kare sırası: 0, 1, 2, ...
    Timestamp    t  = 0;    // yakalama/oynatma zamanı (ns)
};

// ---------------------------------------------------------------------------
// Telemetry: bir telemetri örneği (Liftoff .csv satırından).
//   gyro_*  → açısal hız (DERECE/s — rad/s DEĞİL; quaternion çapraz-doğrulaması ile
//             saptandı). GERÇEK IMU karşılığı; P1 stabilizasyonu bunu kullanır (rad'a çevirir).
//   att_*   → attitude quaternion (mutlak yönelim, driftsiz). Sim'e özel referans/doğrulama.
//   pos_*/vel_* → dünya koordinatında konum/hız (kendi drone'umuz).
// ---------------------------------------------------------------------------
struct Telemetry {
    double t_rel = 0.0;                              // video-hizalı saniye (t_rel=0 ≈ video başı)
    double gyro_pitch = 0.0, gyro_roll = 0.0, gyro_yaw = 0.0;  // rad/s
    double att_x = 0.0, att_y = 0.0, att_z = 0.0, att_w = 1.0; // quaternion (w=1 → birim/dönüşsüz)
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;    // metre (dünya çerçevesi)
    double vel_x = 0.0, vel_y = 0.0, vel_z = 0.0;    // m/s
};

// ---------------------------------------------------------------------------
// Detection: ham bir aday blob. Henüz "drone" demiyoruz — sadece "ilginç leke".
// ---------------------------------------------------------------------------
struct Detection {
    cv::Point2f centroid;          // piksel konumu (alt-piksel olabilir)
    cv::Rect    bbox;              // sınırlayıcı kutu
    float       area = 0.f;        // piksel alanı
    float       aspect_ratio = 1.f;// en/boy oranı (şekil ipucu)
    float       score = 0.f;       // discriminator "drone olma" skoru [0,1]
    Timestamp   t = 0;
};

// ---------------------------------------------------------------------------
// Track: zaman içinde takip edilen bir hedef (Kalman/IMM state'inin dışa açık özeti).
// ---------------------------------------------------------------------------
struct Track {
    enum class Status { Tentative, Confirmed, Lost };  // M-of-N başlatma durumları

    int         id = -1;
    cv::Point2f pos;                    // tahmini konum (piksel)
    cv::Point2f vel;                    // tahmini hız (piksel/s)
    cv::Rect    bbox;                   // ölçülen hedef kutusu (silüet) — çizim/boyut
    float       score = 0.f;
    int         age = 0;                // kaç kare yaşadı
    int         hits = 0;               // kaç kez bir tespitle eşleşti
    int         misses = 0;             // ardışık kaç kez eşleşmedi
    Status      status = Status::Tentative;
    Timestamp   t = 0;
};

// ---------------------------------------------------------------------------
// Cue: sistem hedefe "yönlendirilmiş" (cued) gelir — beklenen başlangıç bölgesi.
// Tespiti/aramayı bu bölgeye odaklarız (initial.txt: "zaten cued durumda").
// ---------------------------------------------------------------------------
struct Cue {
    cv::Point2f expected_pos;   // beklenen piksel konumu
    float       radius = 0.f;   // arama yarıçapı (piksel)
};

} // namespace dtrack
