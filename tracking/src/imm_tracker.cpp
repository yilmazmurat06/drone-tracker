#include "dtrack/tracking/imm_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "dtrack/common/time.hpp"

namespace dtrack::tracking {

namespace {
constexpr double kTwoPi = 6.283185307179586;
// 2B izotropik Gaussian olabilirlik: innovation karesi r2, std sigma.
inline double gauss_like(double r2, double sigma) {
    const double s2 = sigma * sigma;
    return std::exp(-0.5 * r2 / s2) / (kTwoPi * s2);
}
}  // namespace

void ImmTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    has_last_ = false;
    pending_init_.reset();
    pending_age_ = 0;
}

// IMM adım 1-2-3: mod karışımı (mixing) + her modeli dt kadar predict + kombine.
void ImmTracker::mix_predict(TrackImpl& t, double dt) const {
    const float d = static_cast<float>(dt);
    // Markov geçiş matrisi P[i][j] = i->j.
    const double Pgg = cfg_.p_stay_gentle, Pga = 1.0 - cfg_.p_stay_gentle;
    const double Paa = cfg_.p_stay_agile, Pag = 1.0 - cfg_.p_stay_agile;
    const double mg = t.mu_g, ma = t.mu_a;

    // Tahmini mod olasılıkları: cbar[j] = Σ_i P[i][j]·μ[i].
    double cbar_g = Pgg * mg + Pag * ma;
    double cbar_a = Pga * mg + Paa * ma;
    const double eps = 1e-9;
    cbar_g = std::max(cbar_g, eps);
    cbar_a = std::max(cbar_a, eps);

    // Karışım olasılıkları μ_{i|j} = P[i][j]·μ[i] / cbar[j].
    const float w_g_in_g = static_cast<float>(Pgg * mg / cbar_g);
    const float w_a_in_g = static_cast<float>(Pag * ma / cbar_g);
    const float w_g_in_a = static_cast<float>(Pga * mg / cbar_a);
    const float w_a_in_a = static_cast<float>(Paa * ma / cbar_a);

    // Karışmış başlangıç state'leri x0[j] = Σ_i μ_{i|j}·x[i].
    const cv::Vec4f x0g = w_g_in_g * t.xg + w_a_in_g * t.xa;
    const cv::Vec4f x0a = w_g_in_a * t.xg + w_a_in_a * t.xa;

    // Her model CV predict: p += v·dt.
    t.xg = {x0g[0] + x0g[1] * d, x0g[1], x0g[2] + x0g[3] * d, x0g[3]};
    t.xa = {x0a[0] + x0a[1] * d, x0a[1], x0a[2] + x0a[3] * d, x0a[3]};

    // Tahmini mod olasılıkları (ölçüm gelmezse bunlar kalır).
    t.mu_g = static_cast<float>(cbar_g);
    t.mu_a = static_cast<float>(cbar_a);

    // Kombine (gating + coast çıktısı için).
    t.xc = t.mu_g * t.xg + t.mu_a * t.xa;
}

// IMM adım 4-5: her modeli α-β düzelt, olabilirlikten mod olasılığı güncelle, kombine et.
void ImmTracker::correct(TrackImpl& t, const common::Detection& z, double dt) const {
    const float zx = z.centroid.x, zy = z.centroid.y;
    const float inv_dt = static_cast<float>(1.0 / std::max(dt, 1e-3));

    // --- Yumuşak model ---
    const float rgx = zx - t.xg[0], rgy = zy - t.xg[2];
    const double rg2 = static_cast<double>(rgx) * rgx + static_cast<double>(rgy) * rgy;
    {
        const float a = static_cast<float>(cfg_.alpha_gentle);
        const float b = static_cast<float>(cfg_.beta_gentle) * inv_dt;
        t.xg[0] += a * rgx; t.xg[1] += b * rgx;
        t.xg[2] += a * rgy; t.xg[3] += b * rgy;
    }
    // --- Çevik model ---
    const float rax = zx - t.xa[0], ray = zy - t.xa[2];
    const double ra2 = static_cast<double>(rax) * rax + static_cast<double>(ray) * ray;
    {
        const float a = static_cast<float>(cfg_.alpha_agile);
        const float b = static_cast<float>(cfg_.beta_agile) * inv_dt;
        t.xa[0] += a * rax; t.xa[1] += b * rax;
        t.xa[2] += a * ray; t.xa[3] += b * ray;
    }

    // Mod olasılığı güncelle: μ[j] ∝ cbar[j]·Λ[j] (cbar predict'te μ'ye yazılmıştı).
    const double Lg = gauss_like(rg2, cfg_.sigma_gentle);
    const double La = gauss_like(ra2, cfg_.sigma_agile);
    double ug = t.mu_g * Lg;
    double ua = t.mu_a * La;
    const double sum = ug + ua;
    if (sum > 1e-12) { ug /= sum; ua /= sum; } else { ug = ua = 0.5; }
    // Taban (tek moda kilitlenmeyi önle).
    ug = std::clamp(ug, cfg_.mu_floor, 1.0 - cfg_.mu_floor);
    ua = 1.0 - ug;
    t.mu_g = static_cast<float>(ug);
    t.mu_a = static_cast<float>(ua);

    // Kombine state.
    t.xc = t.mu_g * t.xg + t.mu_a * t.xa;
    t.scale = z.area_px > 0 ? std::sqrt(z.area_px) : t.scale;
    t.intensity = z.intensity;
}

common::Track ImmTracker::to_public(const TrackImpl& t, common::Timestamp stamp,
                                    double dt_pred) const {
    common::Track o;
    o.id = t.id;
    o.stamp = stamp;
    o.status = t.status;
    o.position = {t.xc[0], t.xc[2]};
    o.velocity = {t.xc[1], t.xc[3]};  // px/s
    const float d = static_cast<float>(dt_pred);
    o.predicted = {t.xc[0] + t.xc[1] * d, t.xc[2] + t.xc[3] * d};
    o.scale = t.scale;
    o.hits = t.hits;
    o.time_since_update = t.time_since_update;
    float conf = std::min(1.0f, static_cast<float>(t.hits) / cfg_.confirm_hits);
    conf *= std::max(0.0f, 1.0f - static_cast<float>(t.time_since_update) / cfg_.max_coast);
    o.confidence = conf;
    return o;
}

std::vector<common::Track> ImmTracker::update(
    const std::vector<common::Detection>& detections, common::Timestamp stamp) {
    double dt = 1.0 / 60.0;
    if (has_last_) {
        dt = common::millis_between(last_stamp_, stamp) / 1000.0;
        if (dt < 1e-3) dt = 1e-3;
        if (dt > 0.5) dt = 0.5;
    }
    last_stamp_ = stamp;
    has_last_ = true;

    // --- 1) MIX + PREDICT (her track, her model). ---
    for (auto& t : tracks_) {
        mix_predict(t, dt);
        ++t.age;
        ++t.time_since_update;
        ++t.consecutive_miss;
    }

    // --- 2+3) GATE + greedy association (kombine tahmin konumu üzerinden). ---
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
        const double gate = cfg_.gate_px + cfg_.gate_expand_per_frame * trk.time_since_update;
        const double g2 = gate * gate;
        for (size_t di = 0; di < nd; ++di) {
            const double dx = trk.xc[0] - detections[di].centroid.x;
            const double dy = trk.xc[2] - detections[di].centroid.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 <= g2) pairs.push_back({d2, (int)ti, (int)di, prio});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        if (a.prio != b.prio) return a.prio > b.prio;
        return a.dist < b.dist;
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
            correct(tr, detections[di], dt);
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
                const float speed = std::hypot(vx, vy);  // px/s
                const bool sane = speed <= cfg_.max_init_speed;
                TrackImpl t;
                t.id = next_id_++;
                const cv::Vec4f x0{unmatched_z->centroid.x, sane ? vx : 0.0f,
                                   unmatched_z->centroid.y, sane ? vy : 0.0f};
                t.xg = x0;
                t.xa = x0;
                t.xc = x0;
                t.mu_g = 0.5f;
                t.mu_a = 0.5f;
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
            if (t.hits >= cfg_.confirm_hits && t.age <= cfg_.confirm_window)
                t.status = common::TrackStatus::Confirmed;
        }
        if (t.status == common::TrackStatus::Confirmed && t.time_since_update > 0)
            t.status = common::TrackStatus::Coasting;
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

    std::vector<common::Track> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.status == common::TrackStatus::Confirmed ||
            t.status == common::TrackStatus::Coasting)
            out.push_back(to_public(t, stamp, dt));
    }
    return out;
}

}  // namespace dtrack::tracking
