#pragma once
// ============================================================================
//  MultiTargetTracker — P4, Adım 5: çok-hedefli takip (sabit-hız Kalman +
//  gate'li veri ilişkilendirme + M-of-N başlatma).
//
//  NEDEN?  P2 tespit YÜKSEK RECALL ama gürültülü (paralaks pırıltısı yüzlerce
//  yanlış-pozitif). Gerçek hedef kareler boyunca DÜZGÜN, tutarlı bir İZ çizer;
//  gürültü ise titreşir, yer değiştirir, kalıcı iz oluşturamaz. Takip bunu
//  temporal tutarlılıkla ayırır: yalnız M-of-N onayını geçen izler "Confirmed".
//
//  HER TRACK BİR KALMAN FİLTRESİ:
//    durum  x = [px, py, vx, vy]   (konum + hız, piksel & piksel/kare)
//    geçiş  sabit hız: p' = p + v·dt   (dt = 1 kare)
//    ölçüm  z = [px, py]            (tespit centroid'i)
//  Döngü: predict (ileri tahmin) → associate (gate'li en yakın) → correct.
//
//  YAŞAM DÖNGÜSÜ:
//    yeni tespit → Tentative. hits ≥ confirm_hits → Confirmed (çizilir).
//    eşleşmeyen track → misses++ (Kalman tahminiyle "coasting"). misses >
//    max_misses → Lost → silinir.
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>   // cv::KalmanFilter

#include "dtrack/core/types.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack {

class MultiTargetTracker : public ITracker {
public:
    struct Params {
        double gate_dist    = 25.0;  // ilişkilendirme kapısı (piksel) — bundan uzak eşleşmez
        int    confirm_hits = 5;     // Tentative→Confirmed için gereken asgari isabet
        double min_travel   = 25.0;  // onay için doğuştan asgari NET yer değiştirme (px).
                                     // KRİTİK: yerinde titreşen paralaks kümeleri elenir;
                                     // yalnız gerçekten yol alan hedefler onaylanır.
        int    max_misses   = 5;     // ardışık kaçış > bu → Lost (coasting hayaletini kısalt)
        float  process_noise = 1.0f; // Q: model belirsizliği (büyük = daha çevik)
        float  measure_noise = 4.0f; // R: ölçüm gürültüsü (büyük = tespite az güven)
    };

    MultiTargetTracker();
    explicit MultiTargetTracker(Params params);

    // ITracker: bu karenin tespitleriyle track'leri günceller, aktifleri döndürür.
    void update(const std::vector<Detection>& detections,
                std::vector<Track>& out_tracks) override;

    void reset();

private:
    // Bir track'in iç durumu: açık Track özeti + kendi Kalman filtresi.
    struct Internal {
        Track          t;
        cv::KalmanFilter kf;
        cv::Point2f      birth;   // doğum konumu → net yer değiştirme ölçümü
    };

    void init_kf(Internal& in, const cv::Point2f& p) const;

    Params               p_;
    std::vector<Internal> tracks_;
    int                  next_id_ = 0;
};

} // namespace dtrack
