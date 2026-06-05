#pragma once
//
// ImmTracker: Interacting Multiple Model (IMM) çoklu-hedef takip (ITracker, Problem 4).
//
// NEDEN IMM: tek α-β filtresi tek bir hareket modeli (sabit hız) varsayar. Düz
// uçuşta düşük kazanç pürüzsüzdür ama manevrada geride kalır; yüksek kazanç
// manevrayı yakalar ama düz uçuşta gürültülüdür. IMM bu ikilemi iki modeli
// PARALEL koşturup olabilirliklerine göre HARMANLAYARAK çözer: hedef düz uçarken
// "yumuşak" model, manevra yaparken "çevik" model baskın olur.
//
// TASARIM KARARI: Klasik IMM her modeli Kalman ile koşturur. Ancak bu projede
// Kalman küçük (2-6 px) hedefte P kovaryans kararsızlığı yüzünden terk edilip
// α-β'ye geçilmişti (bkz. PROJECT_REPORT §2.7). O dersi korumak için IMM'i iki
// α-β CV filtresi üzerine kurarız (kovaryans yok -> kararsızlık yok):
//   - YUMUŞAK model: düşük α,β  -> düz uçuş, az gürültü
//   - ÇEVİK  model: yüksek α,β  -> manevra, hızlı tepki
// Model olabilirliği, her modelin innovation'ından sabit bir innovation std ile
// Gaussian olarak hesaplanır (Kalman S yerine). Mod olasılıkları Markov geçiş
// matrisiyle karışır (mod kalıcılığı). Bu "hafif IMM" yaklaşımı SAFE-IMM gibi
// gerçek-zaman odaklı çalışmalarla aynı ruhtadır.
//
// State (her model): [px, vx, py, vy], hız px/SANİYE (Kalata α-β, dt-değişmez).
// Çıktı: mod-olasılığı ağırlıklı KOMBİNE state.

#include <cstdint>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

struct ImmConfig {
    // İki model: yumuşak (düz uçuş) + çevik (manevra). α=konum, β=hız kazancı.
    // Yumuşak model, kanıtlanmış tek-filtre α-β değerleriyle (0.55/0.12) eşittir;
    // böylece düz/yumuşak uçuşta IMM ≈ tek α-β (regresyon yok). Çevik model yalnız
    // büyük innovation'da (keskin manevra / coast sonrası yakalama) baskın olur.
    double alpha_gentle = 0.55, beta_gentle = 0.12;
    double alpha_agile = 0.85, beta_agile = 0.40;
    // Model olabilirlik (likelihood) innovation std (px): küçük σ -> küçük
    // innovation'da yüksek olabilirlik (yumuşak düz uçuşta kazanır); büyük σ ->
    // büyük innovation'a toleranslı (çevik manevrada kazanır).
    double sigma_gentle = 3.0;
    double sigma_agile = 12.0;
    // Markov geçiş matrisi P[i][j]=i->j (mod kalıcılığı). Yumuşak çok yapışkan ->
    // çevik moda yalnız ısrarlı büyük innovation'da geçilir.
    double p_stay_gentle = 0.97;  // gentle->gentle (kalan 0.03 -> agile)
    double p_stay_agile = 0.80;   // agile->agile  (kalan 0.20 -> gentle, çabuk geri döner)
    double mu_floor = 0.02;       // mod olasılığı tabanı (tek moda kilitlenmeyi önler)

    // Gating + yaşam döngüsü (α-β ile aynı varsayılanlar).
    double gate_px = 12.0;
    double gate_expand_per_frame = 1.5;
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
        cv::Vec4f xg{0, 0, 0, 0};   // yumuşak model state [px,vx,py,vy]
        cv::Vec4f xa{0, 0, 0, 0};   // çevik model state
        float mu_g{0.5f};           // yumuşak mod olasılığı
        float mu_a{0.5f};           // çevik mod olasılığı
        cv::Vec4f xc{0, 0, 0, 0};   // kombine (çıktı) state; predict sonrası gating'de kullanılır
        common::TrackStatus status{common::TrackStatus::Tentative};
        int hits{1};
        int age{0};
        int consecutive_miss{0};
        int time_since_update{0};
        float scale{1.0f};
        float intensity{0};
    };

    void mix_predict(TrackImpl& t, double dt) const;  // IMM karışım + her model predict + kombine
    void correct(TrackImpl& t, const common::Detection& z, double dt) const;  // her model düzelt + mod güncelle
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
