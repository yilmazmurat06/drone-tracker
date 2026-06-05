#include "dtrack/detection/stability_discriminator.hpp"

#include <algorithm>
#include <cmath>

namespace dtrack::detection {

StabilityDiscriminator::StabilityDiscriminator(DiscriminatorConfig cfg) : cfg_(cfg) {}

void StabilityDiscriminator::reset() {
    history_.clear();
    frame_count_ = 0;
}

float StabilityDiscriminator::geom_score(const common::Detection& d) const {
    float s_area = 0.0f;
    const float a = d.area_px;
    if (a >= cfg_.area_opt_min && a <= cfg_.area_opt_max) {
        s_area = 1.0f;
    } else if (a >= cfg_.area_min && a < cfg_.area_opt_min) {
        s_area = (a - cfg_.area_min) / (cfg_.area_opt_min - cfg_.area_min);
    } else if (a > cfg_.area_opt_max && a <= cfg_.area_max) {
        s_area = 1.0f - (a - cfg_.area_opt_max) / (cfg_.area_max - cfg_.area_opt_max);
    }

    const float s_aspect = std::exp(-(d.aspect_ratio - 1.0f) * (d.aspect_ratio - 1.0f) /
                                    (2.0f * cfg_.aspect_sigma * cfg_.aspect_sigma));

    const float s_intensity = std::exp(-(d.intensity - cfg_.intensity_center) *
                                        (d.intensity - cfg_.intensity_center) /
                                        (2.0f * cfg_.intensity_sigma * cfg_.intensity_sigma));

    return (s_area + s_aspect + s_intensity) / 3.0f;
}

void StabilityDiscriminator::score(std::vector<common::Detection>& detections) {
    const float r2 = cfg_.match_radius * cfg_.match_radius;

    // Mevcut geçmiş girişlerini yaşlandır.
    for (auto& h : history_) ++h.age;

    for (auto& d : detections) {
        const float gs = geom_score(d);

        // Geçmişte en yakın tespiti bul (yaşla ağırlıklandırılmış).
        float best_score = -1.0f;
        const HistoryEntry* best_h = nullptr;
        for (const auto& h : history_) {
            const float dx = d.centroid.x - h.pos.x;
            const float dy = d.centroid.y - h.pos.y;
            const float d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            // Zamansal yakınlık: yeni tespitler daha değerli.
            const float age_w = std::exp(-static_cast<float>(h.age) / 5.0f);
            const float prox_w = 1.0f - std::sqrt(d2) / cfg_.match_radius;
            const float w = age_w * prox_w;
            if (w > best_score) {
                best_score = w;
                best_h = &h;
            }
        }

        float ts = 0.1f;  // geçmiş yok = düşük
        float ss = 0.5f;

        if (best_h && best_score > 0.0f) {
            ts = std::clamp(best_score, 0.0f, 1.0f);

            const float area_rel =
                best_h->area > 1.0f ? (d.area_px - best_h->area) / best_h->area : 0.0f;
            const float sa = std::exp(-area_rel * area_rel / 0.5f);
            const float sa2 = std::exp(-(d.aspect_ratio - best_h->aspect) *
                                        (d.aspect_ratio - best_h->aspect) / 0.18f);
            const float si = std::exp(-(d.intensity - best_h->intensity) *
                                       (d.intensity - best_h->intensity) / 800.0f);
            ss = (sa + sa2 + si) / 3.0f;
        }

        float score = 0.35f * gs + 0.35f * ts + 0.30f * ss;
        score = std::clamp(score, 0.0f, 1.0f);
        if (score < cfg_.min_score) score = 0.0f;
        d.drone_score = score;
    }

    // Bu karenin tespitlerini geçmişe ekle (yaş=0).
    for (const auto& d : detections) {
        history_.push_back({d.centroid, d.area_px, d.aspect_ratio, d.intensity, 0});
    }

    // Yaşlı girişleri temizle.
    while (!history_.empty() && history_.front().age > cfg_.history_frames) {
        history_.pop_front();
    }

    // Güvenlik sınırı: tampon çok büyürse buda.
    const size_t max_entries = static_cast<size_t>(cfg_.history_frames * 10);
    while (history_.size() > max_entries) {
        history_.pop_front();
    }

    ++frame_count_;
}

}  // namespace dtrack::detection
