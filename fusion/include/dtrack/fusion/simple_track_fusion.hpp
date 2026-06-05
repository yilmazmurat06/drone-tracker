#pragma once
//
// SimpleTrackFusion: iki kameranın (görünür + termal) track listelerini
// geç (track-seviyesi) füzyonla birleştirir (ITrackFusion, Problem 5).
//
// Algoritma:
//   1) Termal track'leri görünür koordinata projekte et (H_termal_gorunur).
//   2) Her görünür-termal çifti için uzamsal mesafe < kapı -> eşleme adayı.
//   3) Aç gözlü atama (en yakın önce).
//   4) Eşleşen çiftler: konum/hız/güven birleştir.
//   5) Eşleşmeyen tekler: düşük güvenle tek başına geçir.
//
// Hemfikirlik (iki kamera da görüyor) -> yüksek güven.
// Tek kamera -> düşük güven (0.7 çarpanı).
// Hiçbiri -> track yok.
//
// Geometrik hizalama: H termal piksel -> görünür piksel. Sentetik sahnede
// iki kamera aynı optik merkez/FOV -> H = I (kimlik). Gerçek donanımda
// stereo kalibrasyonla ölçülür.

#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/fusion/track_fusion.hpp"

namespace dtrack::fusion {

struct FusionConfig {
    // Termal -> görünür piksel eşleme homografisi (3x3).
    cv::Matx33f thermal_to_visible = cv::Matx33f::eye();

    // İki track'in eşleşmesi için maksimum uzamsal mesafe (görünür piksel).
    double gate_px = 25.0;

    // Tek kamerada görünen track'in güven çarpanı (ceza).
    float single_modality_factor = 0.7f;

    // Minimum güven: bunun altındaki track'ler çıktıya eklenmez.
    float min_confidence = 0.15f;

    // Zaman penceresi: track'lerin stamp'leri arasındaki maksimum fark (saniye).
    double time_window_s = 0.05;
};

class SimpleTrackFusion : public ITrackFusion {
public:
    explicit SimpleTrackFusion(FusionConfig cfg = {});

    std::vector<common::Track> fuse(
        const std::vector<common::Track>& visible_tracks,
        const std::vector<common::Track>& thermal_tracks) override;

    void reset() override;

    // Son füzyon istatistikleri (debug).
    int last_matched() const { return last_matched_; }
    int last_visible_only() const { return last_visible_only_; }
    int last_thermal_only() const { return last_thermal_only_; }

private:
    cv::Point2f project(const cv::Matx33f& H, const cv::Point2f& p) const;

    FusionConfig cfg_;
    int last_matched_{0};
    int last_visible_only_{0};
    int last_thermal_only_{0};
};

}  // namespace dtrack::fusion
