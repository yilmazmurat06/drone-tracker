#pragma once
//
// AlphaBetaTracker: çoklu-hedef takip (ITracker, Problem 4).
//
// α-β filtresi (fixed-gain tracker). Kalman'a göre avantajı: sabit kazançlar
// sayesinde yakınsama sorunu yoktur; manevra hedefine anında tepki verir.
// Küçük (2-6 px) hedef takibinde Kalman'dan daha sağlam çalışır çünkü:
//   - Kalman'in P kovaryansı küçük hedefte yavaş yakınsar
//   - α-β doğrudan α ile konumu, β ile hızı düzeltir
//   - Basit Öklid kapısı (Mahalanobis yerine) daha öngörülebilir
//
// State: [px, vx, py, vy] (4B, CV modeliyle aynı).
//
// Her update():
//   1) PREDICT  : tüm track'leri dt kadar ilerlet (x += v*dt).
//   2) GATE     : her track çevresinde Öklid kapısı; fiziksel imkânsızı reddet.
//   3) ASSOCIATE: kapı içinde onaylı track'lere öncelikli aç gözlü ata.
//   4) UPDATE   : eşleşen track'i α-β düzelt; eşleşmeyen tespit -> yeni track.
//   5) LIFECYCLE: M-of-N onayı; uzun coast -> sil.
//
// Çıktı: aktif track'lerin anlık görüntüsü (konum + hız + ileri tahmin).

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

struct TrackerConfig {
    // α-β kazançları. α = konum düzeltme (0..1), β = hız düzeltme (0..1).
    // Büyük α -> hızlı konum yakınsama ama gürültülü.
    // Büyük β -> hızlı hız adaptasyonu ama salınımlı.
    double alpha = 0.55;         // konum kazancı
    double beta  = 0.12;         // hız kazancı

    // Öklid kapısı (piksel). Tespitin track tahmininden bu kadar uzaksa eşleşmez.
    // Temel yarıçap; coast süresiyle genişler (bkz. gate_expand_per_frame).
    double gate_px = 12.0;
    // Coast eden track'in kapısı kare-başına bu kadar genişler.
    double gate_expand_per_frame = 1.5;

    // M-of-N track yaşam döngüsü.
    int confirm_hits = 3;        // M: onay için gereken tespit sayısı
    int confirm_window = 8;      // N: ilk bu kadar karede M tutmalı
    int max_coast = 22;          // Confirmed track tespitsiz bu kadar kare yaşar (~0.37s@60fps)
    int tentative_max_miss = 2;  // Tentative ardışık bu kadar ıskalarsa silinir

    // İki-nokta hız başlatma: eşleşmeyen ilk tespiti sakla, ikinciden hız hesapla.
    // Hız akıl kontrolü: |v| > max_init_speed ise sıfır hızla başlat.
    double max_init_speed = 400.0;  // px/s
    int pending_max_age = 10;       // bekletme karesi
};

class AlphaBetaTracker : public ITracker {
public:
    explicit AlphaBetaTracker(TrackerConfig cfg = {}) : cfg_(cfg) {}

    std::vector<common::Track> update(
        const std::vector<common::Detection>& detections,
        common::Timestamp stamp) override;

    void reset() override;

private:
    struct TrackImpl {
        std::uint32_t id{0};
        cv::Vec4f x{0, 0, 0, 0};  // [px, vx, py, vy]
        common::TrackStatus status{common::TrackStatus::Tentative};
        int hits{1};
        int age{0};
        int consecutive_miss{0};
        int time_since_update{0};
        float scale{1.0f};
        float intensity{0};
    };

    void predict(TrackImpl& t, double dt) const;
    void correct(TrackImpl& t, const common::Detection& z) const;
    common::Track to_public(const TrackImpl& t, common::Timestamp stamp, double dt_pred) const;

    TrackerConfig cfg_;
    std::vector<TrackImpl> tracks_;
    std::uint32_t next_id_{1};
    common::Timestamp last_stamp_{};
    bool has_last_{false};

    // İki-nokta başlatma.
    struct PendingInit {
        cv::Point2f position;
        common::Timestamp stamp;
    };
    std::optional<PendingInit> pending_init_;
    int pending_age_{0};
};

// Geriye dönük uyumluluk için takma ad.
using KalmanTracker = AlphaBetaTracker;

}  // namespace dtrack::tracking
