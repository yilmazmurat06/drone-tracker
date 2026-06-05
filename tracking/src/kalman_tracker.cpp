#include "dtrack/tracking/kalman_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "dtrack/common/time.hpp"

namespace dtrack::tracking {

void AlphaBetaTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    has_last_ = false;
    pending_init_.reset();
    pending_age_ = 0;
}

void AlphaBetaTracker::predict(TrackImpl& t, double dt) const {
    // Sabit hız tahmini: p += v * dt.
    const float d = static_cast<float>(dt);
    t.x[0] += t.x[1] * d;
    t.x[2] += t.x[3] * d;
}

void AlphaBetaTracker::correct(TrackImpl& t, const common::Detection& z) const {
    // α-β düzeltmesi: konum = tahmin + α*(ölçüm-tahmin),
    //                 hız   = tahmin + (β/dt)*(ölçüm-tahmin).
    const float innov_x = z.centroid.x - t.x[0];
    const float innov_y = z.centroid.y - t.x[2];
    const float a = static_cast<float>(cfg_.alpha);
    const float b = static_cast<float>(cfg_.beta);
    // dt = 1/fps ~ 0.0167. kare-başı dt kullanıyoruz; kazançları buna göre
    // ayarladık. actual dt çağrı yerinde gelir, burada gerekli değil çünkü
    // β zaten "ölçüm başına hız düzeltme birimi" olarak tanımlandı.
    t.x[0] += a * innov_x;
    t.x[1] += b * innov_x;   // hız birimi: px/kare (1/60 s). Dışarı çevirirken *fps.
    t.x[2] += a * innov_y;
    t.x[3] += b * innov_y;
    t.scale = z.area_px > 0 ? std::sqrt(z.area_px) : t.scale;
    t.intensity = z.intensity;
}

common::Track AlphaBetaTracker::to_public(const TrackImpl& t, common::Timestamp stamp,
                                          double dt_pred) const {
    common::Track o;
    o.id = t.id;
    o.stamp = stamp;
    o.status = t.status;
    o.position = {t.x[0], t.x[2]};
    // Hız birimi px/kare -> px/s çevir (60 fps varsayımı).
    const float fps = 60.0f;
    o.velocity = {t.x[1] * fps, t.x[3] * fps};
    // Bir kare ileri tahmin (kesişim için).
    const float d = static_cast<float>(dt_pred);
    o.predicted = {t.x[0] + t.x[1] * d, t.x[2] + t.x[3] * d};
    o.scale = t.scale;
    o.hits = t.hits;
    o.time_since_update = t.time_since_update;
    float conf = std::min(1.0f, static_cast<float>(t.hits) / cfg_.confirm_hits);
    conf *= std::max(0.0f, 1.0f - static_cast<float>(t.time_since_update) / cfg_.max_coast);
    o.confidence = conf;
    return o;
}

std::vector<common::Track> AlphaBetaTracker::update(
    const std::vector<common::Detection>& detections, common::Timestamp stamp) {
    double dt = 1.0 / 60.0;
    if (has_last_) {
        dt = common::millis_between(last_stamp_, stamp) / 1000.0;
        if (dt < 1e-3) dt = 1e-3;
        if (dt > 0.5) dt = 0.5;
    }
    last_stamp_ = stamp;
    has_last_ = true;

    // --- 1) PREDICT. ---
    for (auto& t : tracks_) {
        predict(t, dt);
        ++t.age;
        ++t.time_since_update;
        ++t.consecutive_miss;
    }

    // --- 2+3) GATE + greedy association (onaylıya öncelik, adaptive kapı). ---
    const size_t nt = tracks_.size(), nd = detections.size();
    std::vector<int> det_to_track(nd, -1);
    std::vector<bool> track_taken(nt, false);

    struct Pair { double dist; int ti; int di; int prio; };
    std::vector<Pair> pairs;

    for (size_t ti = 0; ti < nt; ++ti) {
        const auto& trk = tracks_[ti];
        int prio = 0;
        if (trk.status == common::TrackStatus::Confirmed) prio = 2;
        else if (trk.status == common::TrackStatus::Tentative) prio = 1;
        // Adaptive kapı: coast süresiyle genişler.
        const double gate = cfg_.gate_px + cfg_.gate_expand_per_frame * trk.time_since_update;
        const double g2 = gate * gate;
        for (size_t di = 0; di < nd; ++di) {
            const double dx = trk.x[0] - detections[di].centroid.x;
            const double dy = trk.x[2] - detections[di].centroid.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 <= g2)
                pairs.push_back({d2, (int)ti, (int)di, prio});
        }
    }
    // Önce önceliğe göre, sonra mesafeye göre sırala (küçük = iyi).
    // prio büyük = öncelikli, dist küçük = yakın. Skor = dist - prio*katsayı.
    // Ama daha basit: önce prio'ya göre sırala, eşitse dist.
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) {
                  if (a.prio != b.prio) return a.prio > b.prio;  // yüksek öncelik önce
                  return a.dist < b.dist;                         // yakın olan önce
              });

    for (const auto& p : pairs) {
        if (track_taken[p.ti] || det_to_track[p.di] != -1) continue;
        track_taken[p.ti] = true;
        det_to_track[p.di] = p.ti;
    }

    // --- 4) UPDATE eşleşenler; eşleşmeyen -> iki-nokta başlatma. ---
    const common::Detection* unmatched_z = nullptr;
    for (size_t di = 0; di < nd; ++di) {
        const int ti = det_to_track[di];
        if (ti >= 0) {
            auto& tr = tracks_[ti];
            correct(tr, detections[di]);
            ++tr.hits;
            tr.time_since_update = 0;
            tr.consecutive_miss = 0;
            if (tr.status == common::TrackStatus::Coasting)
                tr.status = common::TrackStatus::Confirmed;
        } else {
            unmatched_z = &detections[di];
        }
    }

    if (unmatched_z) {
        bool created = false;
        if (pending_init_) {
            const double dt_pend = common::millis_between(pending_init_->stamp, stamp) / 1000.0;
            if (dt_pend > 1e-3 && dt_pend < 0.5) {
                const float fps = static_cast<float>(1.0 / dt_pend);
                const float vx = (unmatched_z->centroid.x - pending_init_->position.x) * fps;
                const float vy = (unmatched_z->centroid.y - pending_init_->position.y) * fps;
                const float speed = std::hypot(vx, vy);
                TrackImpl t;
                t.id = next_id_++;
                t.x = {unmatched_z->centroid.x,
                       speed <= cfg_.max_init_speed ? vx / 60.0f : 0.0f,
                       unmatched_z->centroid.y,
                       speed <= cfg_.max_init_speed ? vy / 60.0f : 0.0f};
                t.status = common::TrackStatus::Tentative;
                t.hits = 1;
                t.age = 1;
                t.scale = unmatched_z->area_px > 0 ? std::sqrt(unmatched_z->area_px) : 1.0f;
                t.intensity = unmatched_z->intensity;
                tracks_.push_back(t);
                created = true;
            }
            pending_init_.reset();
            pending_age_ = 0;
        }
        if (!created) {
            pending_init_ = PendingInit{unmatched_z->centroid, stamp};
            pending_age_ = 0;
        }
    }

    // --- 5) LIFECYCLE. ---
    if (pending_init_) {
        ++pending_age_;
        if (pending_age_ > cfg_.pending_max_age) {
            pending_init_.reset();
            pending_age_ = 0;
        }
    } else {
        pending_age_ = 0;
    }

    for (auto& t : tracks_) {
        if (t.status == common::TrackStatus::Tentative) {
            if (t.hits >= cfg_.confirm_hits && t.age <= cfg_.confirm_window) {
                t.status = common::TrackStatus::Confirmed;
            }
        }
        if (t.status == common::TrackStatus::Confirmed && t.time_since_update > 0) {
            t.status = common::TrackStatus::Coasting;
        }
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const TrackImpl& t) {
                           if (t.status == common::TrackStatus::Tentative &&
                               (t.consecutive_miss > cfg_.tentative_max_miss ||
                                t.age > cfg_.confirm_window))
                               return true;
                           if (t.time_since_update > cfg_.max_coast) return true;
                           return false;
                       }),
        tracks_.end());

    // --- Çıktı: Confirmed + Coasting track'ler. ---
    std::vector<common::Track> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.status == common::TrackStatus::Confirmed ||
            t.status == common::TrackStatus::Coasting) {
            out.push_back(to_public(t, stamp, dt));
        }
    }
    return out;
}

}  // namespace dtrack::tracking
