#pragma once
//
// SkyRegionDetector: ARAZİ-ÜSTÜ uçuşta GÖKYÜZÜ önündeki belirgin hava hedefini
// (zeplin / büyük dron / balon) tespit eder (IDetector, "option (a)" rejimi).
//
// NEDEN AYRI DETEKTÖR: MogDetector 2-6 px PARLAK NOKTA + temiz gökyüzü içindir.
// Gerçek arazi görüntüsünde (ağaç/bina/yol) top-hat her yere yanar (kare başına
// ~500 FP) ve ~100 px zeplini göremez (alan sınırı + top-hat büyük nesneyi bastırır).
//
// STRATEJİ (gökyüzü-kapılı hareket + zamansal tutarlılık):
//   1) GÖKYÜZÜ MASKESİ: parlak + düşük-doku + üst-kenara bağlı bölge. Arazi (alt,
//      dokulu) tamamen dışlanır -> clutter FP patlaması biter.
//   2) HAREKET: gökyüzü ÖZELLİKLERİNDEN (sonsuzdaki arka plan -> rotasyon homografi
//      birebir hizalar) kare-arası dönüşüm kestirilir; önceki kare hizalanıp fark
//      alınır. Hedef (gökyüzüne göre hareket eden katı blob) parlar; bulut sürüklenmesi
//      yavaş/dağınıktır. (Arazi global homografisini KULLANMAYIZ; parallaks bozar.)
//   3) Bağlı bileşen -> geniş alan aralığı (büyük zeplin de geçer) -> centroid.
// Kalan bulut-artığı FP'lerini downstream tracker'ın TRAJEKTORİ TUTARLILIĞI eler
// (zeplin pürüzsüz iz çizer; bulut-artıkları tutarlı iz çizmez).
//
// Çıktı Detection'lar GÜNCEL kare koordinatındadır.

#include <opencv2/core.hpp>

#include <vector>

#include "dtrack/common/types.hpp"
#include "dtrack/detection/detector.hpp"

namespace dtrack::detection {

struct SkyDetectorConfig {
    // --- Gökyüzü segmentasyonu ---
    float sky_texture_std = 12.0f;  // yerel std bu altındaysa "düzgün" (gökyüzü adayı)
    int sky_texture_win = 15;       // yerel std penceresi (tek)
    int sky_top_rows = 6;           // bileşen ilk bu satırlara değiyorsa üst-kenara bağlı
    int sky_min_area_cols_mult = 8; // gökyüzü bileşeni min alan = width * bu
    int sky_morph = 15;             // gökyüzü maskesi kapama/açma çekirdeği
    int sky_erode = 13;             // horizon kenarını dışlamak için maske erozyonu

    // --- Hareket tespiti (gökyüzü-kapılı) ---
    int max_features = 400;         // gökyüzünde takip noktası
    double feature_quality = 0.01;
    double min_feature_distance = 8.0;
    int lk_window = 21;
    int lk_pyramid = 3;
    float fb_error_px = 1.5f;       // ileri-geri tutarlılık
    int min_inliers = 12;
    double ransac_thresh_px = 2.5;
    double motion_thresh = 12.0;    // hizalanmış fark eşiği (gri seviye)
    int motion_open = 3;            // tek-piksel gürültü açma
    int motion_close = 9;           // hedef parçalarını birleştir

    // --- Blob eleme (geniş: küçük dron -> büyük zeplin) ---
    double min_area = 12.0;
    double max_area = 60000.0;      // ~245 px çapa kadar (büyük zeplin)
    double max_aspect = 6.0;
};

class SkyRegionDetector : public IDetector {
public:
    explicit SkyRegionDetector(SkyDetectorConfig cfg = {}) : cfg_(cfg) {}

    std::vector<common::Detection> detect(const common::StabilizedFrame& sf) override;
    void reset() override;

    cv::Matx33f image_to_reference() const override { return cv::Matx33f::eye(); }

    // Teşhis/görselleştirme: son karenin gökyüzü maskesi.
    const cv::Mat& last_sky_mask() const { return sky_mask_; }

private:
    cv::Mat compute_sky_mask(const cv::Mat& gray) const;

    SkyDetectorConfig cfg_;
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;
    cv::Mat sky_mask_;   // son kare (teşhis)
    bool has_prev_{false};
};

}  // namespace dtrack::detection
