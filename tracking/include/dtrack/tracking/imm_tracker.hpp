#pragma once
//
// ImmTracker: Interacting Multiple Model (IMM) çoklu-hedef takip (ITracker, Problem 4).
//
// NEDEN IMM: tek Kalman tek bir hareket modeli (sabit hız) varsayar. Düz uçuşta
// düşük süreç gürültüsü pürüzsüzdür ama manevrada geride kalır; yüksek süreç
// gürültüsü manevrayı yakalar ama düz uçuşta gürültülüdür. IMM bu ikilemi iki
// modeli PARALEL koşturup olabilirliklerine göre HARMANLAYARAK çözer: hedef düz
// uçarken "yumuşak" model, manevra yaparken "çevik" model baskın olur.
//
// TASARIM: Klasik IMM her modeli Kalman ile koşturur. Bu projede ilk denemede
// Kalman küçük (2-6 px) hedefte P kovaryans kararsızlığı (§2.7) yüzünden terk
// edilip α-β'ye geçilmişti. O sorun GİDERİLDİĞİ için (Joseph-form kovaryans +
// ölçülü P0; bkz. kalman_core.hpp / kalman_tracker.hpp), IMM artık iki GERÇEK
// Kalman CV modeli üzerine kurulur:
//   - YUMUŞAK model: düşük σ_a  -> düz uçuş, az gürültü
//   - ÇEVİK  model: yüksek σ_a  -> manevra, hızlı tepki
// Model olabilirliği, Kalman'ın innovation kovaryansı S'ten Gaussian olarak
// hesaplanır (Λ_j = N(y; 0, S_j)) — sabit innovation-std varsayımına gerek yok.
// Mod olasılıkları Markov geçiş matrisiyle karışır (mod kalıcılığı).
//
// Köşegen R ile her model iki bağımsız 2-durumlu eksene (x, y) ayrışır; her
// model her eksende bir `Kf2` tutar. Mod olasılığı (μ) eksenler arası PAYLAŞILIR
// (manevra modu hedefin özelliğidir, eksenin değil).
//
// State (her model, her eksen): [konum, hız], hız px/SANİYE.
// Çıktı: mod-olasılığı ağırlıklı KOMBİNE state.

#include <cstdint>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/tracking/kalman_core.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

struct ImmConfig {
    // Ölçüm gürültüsü σ_r (px).
    double meas_noise_std = 1.5;
    // İki model: süreç (manevra) ivme gürültüsü σ_a (px/s²).
    // Yumuşak düşük (pürüzsüz düz uçuş), çevik yüksek (keskin manevra).
    double process_accel_std_gentle = 400.0;
    double process_accel_std_agile  = 4000.0;
    // Başlangıç kovaryansı P0 (ÖLÇÜLÜ; §2.7 salınımını önler).
    double init_pos_std = 2.0;     // px
    double init_vel_std = 60.0;    // px/s

    // Markov geçiş matrisi P[i][j]=i->j (mod kalıcılığı). Yumuşak çok yapışkan ->
    // çevik moda yalnız ısrarlı büyük innovation'da geçilir.
    double p_stay_gentle = 0.97;   // gentle->gentle (kalan 0.03 -> agile)
    double p_stay_agile  = 0.80;   // agile->agile  (kalan 0.20 -> gentle, çabuk geri döner)
    double mu_floor = 0.02;        // mod olasılığı tabanı (tek moda kilitlenmeyi önler)

    // NIS (Mahalanobis²) kapısı, 2-dof χ²: 9.21(%99) / 13.8(%99.9). Kombine S üzerinden.
    double gate_nis = 13.8;

    // M-of-N yaşam döngüsü (Kalman tracker ile aynı varsayılanlar).
    int confirm_hits = 3;
    int confirm_window = 8;
    int max_coast = 22;
    int tentative_max_miss = 2;
    double max_init_speed = 400.0;  // px/s
    int pending_max_age = 10;
};

class ImmTracker : public ITracker {
public:
    explicit ImmTracker(ImmConfig cfg = {}) : cfg_(cfg) {}

    std::vector<common::Track> update(
        const std::vector<common::Detection>& detections,
        common::Timestamp stamp) override;

    void reset() override;

private:
    struct TrackImpl {
        std::uint32_t id{0};
        Kf2 gx, gy;   // YUMUŞAK model: x ve y CV Kalman
        Kf2 ax, ay;   // ÇEVİK   model: x ve y CV Kalman
        double mu_g{0.5}, mu_a{0.5};  // mod olasılıkları (eksenler arası paylaşılır)
        // Kombine (çıktı + gating) önbelleği — mix_predict/correct sonrası güncellenir.
        double cpx{0}, cvx{0}, cpy{0}, cvy{0};  // kombine konum/hız (x,y)
        double cPx00{1}, cPy00{1};              // kombine konum varyansı (gating S için)
        common::TrackStatus status{common::TrackStatus::Tentative};
        int hits{1};
        int age{0};
        int consecutive_miss{0};
        int time_since_update{0};
        float scale{1.0f};
        float intensity{0};
    };

    void mix_predict(TrackImpl& t, double dt) const;  // IMM karışım + her model predict + kombine
    // her model düzelt + mod güncelle. r: bu tespitin ölçüm varyansı (meas_std ipucu
    // varsa per-tespit, yoksa varsayılan σ_r²) -> kapalı-döngü kurtarma ölçümü az güvenle işlenir.
    void correct(TrackImpl& t, const common::Detection& z, double r) const;
    void recombine(TrackImpl& t) const;               // kombine state + kovaryans önbelleğini güncelle
    common::Track to_public(const TrackImpl& t, common::Timestamp stamp, double dt_pred) const;

    ImmConfig cfg_;
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
