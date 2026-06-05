#include "dtrack/fusion/simple_track_fusion.hpp"

#include <algorithm>
#include <cmath>

namespace dtrack::fusion {

SimpleTrackFusion::SimpleTrackFusion(FusionConfig cfg) : cfg_(cfg) {}

void SimpleTrackFusion::reset() {
    last_matched_ = 0;
    last_visible_only_ = 0;
    last_thermal_only_ = 0;
}

cv::Point2f SimpleTrackFusion::project(const cv::Matx33f& H, const cv::Point2f& p) const {
    const cv::Vec3f q = H * cv::Vec3f(p.x, p.y, 1.0f);
    const float w = (std::abs(q[2]) > 1e-6f) ? 1.0f / q[2] : 1.0f;
    return {q[0] * w, q[1] * w};
}

std::vector<common::Track> SimpleTrackFusion::fuse(
    const std::vector<common::Track>& visible_tracks,
    const std::vector<common::Track>& thermal_tracks) {
    const size_t nv = visible_tracks.size(), nt = thermal_tracks.size();
    const double g2 = cfg_.gate_px * cfg_.gate_px;

    // --- 1) Termal track'leri görünür koordinata projekte et. ---
    struct ProjTherm {
        cv::Point2f pos;
        size_t idx;
        float conf;
    };
    std::vector<ProjTherm> proj;
    proj.reserve(nt);
    for (size_t i = 0; i < nt; ++i) {
        const cv::Point2f p = project(cfg_.thermal_to_visible, thermal_tracks[i].position);
        proj.push_back({p, i, thermal_tracks[i].confidence});
    }

    // --- 2) Uzamsal kapı + eşleme adayları. ---
    struct Pair { double dist2; size_t vi; size_t ti; };
    std::vector<Pair> pairs;
    for (size_t vi = 0; vi < nv; ++vi) {
        const auto& vt = visible_tracks[vi];
        for (size_t ti = 0; ti < nt; ++ti) {
            const auto& pt = proj[ti];
            const double dx = vt.position.x - pt.pos.x;
            const double dy = vt.position.y - pt.pos.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 <= g2) {
                pairs.push_back({d2, vi, pt.idx});
            }
        }
    }

    // --- 3) Aç gözlü atama (en yakın önce). ---
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.dist2 < b.dist2; });
    std::vector<bool> vis_used(nv, false), thm_used(nt, false);
    std::vector<size_t> vis_match(nv, SIZE_MAX);  // visible -> thermal indeks

    for (const auto& p : pairs) {
        if (vis_used[p.vi] || thm_used[p.ti]) continue;
        vis_used[p.vi] = true;
        thm_used[p.ti] = true;
        vis_match[p.vi] = p.ti;
    }

    // --- 4) Birleşik track listesini üret. ---
    std::vector<common::Track> out;
    out.reserve(nv + nt);

    // Eşleşen çiftler.
    for (size_t vi = 0; vi < nv; ++vi) {
        if (vis_match[vi] == SIZE_MAX) continue;  // eşleşmedi -> sonra işle

        const auto& v = visible_tracks[vi];
        const auto& t = thermal_tracks[vis_match[vi]];
        const float cv = std::clamp(v.confidence, 0.0f, 1.0f);
        const float ct = std::clamp(t.confidence, 0.0f, 1.0f);

        common::Track m;
        m.id = v.id;                      // görünür ID'sini taşı (baskın modalite)
        m.stamp = v.stamp;
        m.status = v.status;              // görünür track durumunu yansıt
        // Konum: güven ağırlıklı ortalama.
        const float wsum = cv + ct;
        if (wsum > 0) {
            m.position = {(cv * v.position.x + ct * proj[vis_match[vi]].pos.x) / wsum,
                          (cv * v.position.y + ct * proj[vis_match[vi]].pos.y) / wsum};
            m.velocity = {(cv * v.velocity.x + ct * t.velocity.x) / wsum,
                          (cv * v.velocity.y + ct * t.velocity.y) / wsum};
            m.predicted = {(cv * v.predicted.x + ct * proj[vis_match[vi]].pos.x) / wsum,
                           (cv * v.predicted.y + ct * proj[vis_match[vi]].pos.y) / wsum};
        } else {
            m.position = v.position;
            m.velocity = v.velocity;
            m.predicted = v.predicted;
        }
        // Güven: olasılıksal VEYA (hemfikirlik bonusu).
        m.confidence = std::min(1.0f, cv + ct - cv * ct);
        m.scale = (v.scale + t.scale) * 0.5f;
        m.hits = v.hits + t.hits;
        m.time_since_update = std::min(v.time_since_update, t.time_since_update);
        if (m.confidence >= cfg_.min_confidence) out.push_back(m);
    }

    // Eşleşmeyen görünür track'ler (tek modalite).
    for (size_t vi = 0; vi < nv; ++vi) {
        if (vis_match[vi] != SIZE_MAX) continue;
        auto t = visible_tracks[vi];
        t.confidence *= cfg_.single_modality_factor;
        if (t.confidence >= cfg_.min_confidence) out.push_back(t);
    }

    // Eşleşmeyen termal track'ler (tek modalite, görünür koordinata projekte).
    for (size_t ti = 0; ti < nt; ++ti) {
        if (thm_used[ti]) continue;
        auto t = thermal_tracks[ti];
        t.position = proj[ti].pos;
        t.predicted = project(cfg_.thermal_to_visible, t.predicted);
        t.confidence *= cfg_.single_modality_factor;
        if (t.confidence >= cfg_.min_confidence) out.push_back(t);
    }

    // İstatistik.
    last_matched_ = 0;
    for (size_t vi = 0; vi < nv; ++vi)
        if (vis_match[vi] != SIZE_MAX) ++last_matched_;
    last_visible_only_ = static_cast<int>(nv) - last_matched_;
    last_thermal_only_ = static_cast<int>(nt) - last_matched_;

    return out;
}

}  // namespace dtrack::fusion
