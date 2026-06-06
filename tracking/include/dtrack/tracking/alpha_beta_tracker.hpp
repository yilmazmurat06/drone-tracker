#pragma once
//
// AlphaBetaTracker: çoklu-hedef takip (ITracker, Problem 4) — YEDEK/karşılaştırma.
//
// NOT: Varsayılan tracker artık gerçek CV Kalman'dır (bkz. kalman_tracker.hpp).
// Bu α-β filtresi sabit-kazanç bir alternatif olarak korunur (kararlı-durumda
// Kalman'a denktir; tuning gerektirmez, karşılaştırma temeli olarak kullanışlıdır).
//
// α-β filtresi (fixed-gain tracker): sabit kazançlar sayesinde yakınsama sorunu
// yoktur; manevra hedefine anında tepki verir. Matematiksel olarak, CV modeli için
// kararlı-durum Kalman kazancı tam olarak α-β kazancına eşittir — Kalman bunu
// dt'ye göre uyarlar, α-β sabit tutar.
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
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

struct TrackerConfig {
    // α-β kazançları (Kalata formülasyonu, dt-değişmez).
    //   konum: x += α·innovation         (α boyutsuz, 0..1)
    //   hız  : v += (β/dt)·innovation     (β boyutsuz; gerçek hız kazancı β/dt)
    // Hız state'i px/SANİYE birimindedir (fps'ten bağımsız). Büyük α -> hızlı
    // konum yakınsama ama gürültülü; büyük β -> hızlı hız adaptasyonu ama salınımlı.
    double alpha = 0.55;         // konum kazancı
    double beta  = 0.12;         // hız kazancı (boyutsuz; uygulanan = β/dt)

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
    void correct(TrackImpl& t, const common::Detection& z, double dt) const;
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

}  // namespace dtrack::tracking
