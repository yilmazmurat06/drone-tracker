#pragma once
//
// KalmanTracker: çoklu-hedef takip — GERÇEK 4-durumlu CV Kalman (ITracker, Problem 4).
//
// State: [px, vx, py, vy], hız px/SANİYE. Köşegen ölçüm gürültüsüyle 4-durumlu CV
// filtresi iki bağımsız 2-durumlu eksene ayrışır; her iz iki `Kf2` tutar (bkz.
// kalman_core.hpp). Bu sayede matematik kapalı-form, hızlı (gerçek-zaman) ve
// OpenCV'siz birim testi yapılabilir (tests/test_kalman_math.cpp).
//
// §2.7 (PROJECT_REPORT) DERSİ ELE ALINDI: İlk Kalman küçük (2-6 px) hedefte terk
// edilmişti. Sebepler ve bu implementasyondaki çözümleri:
//   1) "P kovaryansı kararsız"        -> Joseph-form güncelleme (P daima SPD).
//   2) "init_vel_var=1e4 -> salınım"  -> ÖLÇÜLÜ P0 (init_vel_std=60 px/s).
//   3) "kazanç yavaş yakınsar"        -> σ_a (manevra ivmesi) ile ayarlanan süreç
//        gürültüsü; varsayılanlar kararlı-durum kazancını α-β (α≈0.52) ile eşler
//        (test_kalman_math [B] bunu doğrular) -> en kötü ihtimalle α-β kadar iyi.
//   4) "iki-nokta başlatma absürt hız"-> hız akıl kontrolü (>max_init_speed reddedilir).
//
// α-β'ye GÖRE KAZANIM:
//   - Kapı (gating) NIS/Mahalanobis tabanlı: coast'ta P büyür -> kapı DOĞAL
//     genişler (α-β'deki elle "gate_expand_per_frame" katsayısı gerekmez).
//   - Değişken dt'yi (kare atlama / jitter) ilkesel olarak handle eder.
//
// Her update(): 1) PREDICT (her iz, her eksen)  2) GATE (NIS) + öncelikli aç gözlü
// atama  3) UPDATE (Joseph)  4) eşleşmeyen -> iki-nokta başlatma  5) M-of-N yaşam döngüsü.

#include <cstdint>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/tracking/kalman_core.hpp"
#include "dtrack/tracking/tracker.hpp"
// Geriye dönük uyumluluk: bu başlığı dahil eden eski kod AlphaBetaTracker/TrackerConfig'i
// (yedek tracker) de görsün diye burada ekleniyor.
#include "dtrack/tracking/alpha_beta_tracker.hpp"

namespace dtrack::tracking {

struct KalmanConfig {
    // Fiziksel gürültü parametreleri.
    double meas_noise_std    = 1.5;     // σ_r: ölçüm (centroid) gürültüsü (px)
    double process_accel_std = 1500.0;  // σ_a: süreç (manevra) ivme gürültüsü (px/s²)
    // Başlangıç kovaryansı P0 (ÖLÇÜLÜ tutulur, bkz. §2.7).
    double init_pos_std = 2.0;          // konum std (px); ~σ_r
    double init_vel_std = 60.0;         // hız std (px/s); büyük değer -> ilk karelerde salınım

    // NIS (Normalized Innovation Squared = Mahalanobis²) kapısı. 2 serbestlik
    // dereceli χ²: 9.21 (%99), 13.8 (%99.9). Coast'ta S=HPHᵀ+R büyüdüğü için
    // aynı NIS daha geniş piksel yarıçapına karşılık gelir (kapı doğal genişler).
    double gate_nis = 13.8;

    // M-of-N track yaşam döngüsü (α-β ile aynı varsayılanlar).
    int confirm_hits = 3;        // M: onay için gereken tespit
    int confirm_window = 8;      // N: ilk bu kadar karede M tutmalı
    int max_coast = 22;          // Confirmed track tespitsiz bu kadar kare yaşar (~0.37s@60fps)
    int tentative_max_miss = 2;  // Tentative ardışık bu kadar ıskalarsa silinir

    // İki-nokta hız başlatma + akıl kontrolü.
    double max_init_speed = 400.0;  // px/s; üstündeki hızlar sıfırlanır
    int pending_max_age = 10;       // bekletme karesi
};

class KalmanTracker : public ITracker {
public:
    explicit KalmanTracker(KalmanConfig cfg = {}) : cfg_(cfg) {}

    std::vector<common::Track> update(
        const std::vector<common::Detection>& detections,
        common::Timestamp stamp) override;

    void reset() override;

private:
    struct TrackImpl {
        std::uint32_t id{0};
        Kf2 kx, ky;  // bağımsız x ve y CV Kalman filtreleri (durum [p,v] her biri)
        common::TrackStatus status{common::TrackStatus::Tentative};
        int hits{1};
        int age{0};
        int consecutive_miss{0};
        int time_since_update{0};
        float scale{1.0f};
        float intensity{0};
    };

    common::Track to_public(const TrackImpl& t, common::Timestamp stamp, double dt_pred) const;

    KalmanConfig cfg_;
    std::vector<TrackImpl> tracks_;
    std::uint32_t next_id_{1};
    common::Timestamp last_stamp_{};
    bool has_last_{false};

    struct PendingInit {
        cv::Point2f position;
        common::Timestamp stamp;
    };
    std::optional<PendingInit> pending_init_;
    int pending_age_{0};
};

}  // namespace dtrack::tracking
