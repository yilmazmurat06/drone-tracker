#include "dtrack/tracking/imm_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "dtrack/common/time.hpp"

namespace dtrack::tracking {

namespace {
constexpr double kTwoPi = 6.283185307179586;

// İki Kf2 modelini IMM karışımıyla birleştir (tek eksen):
//   x0 = wA·xA + wB·xB
//   P0 = Σ w·(P + (x-x0)(x-x0)ᵀ)   <- yayılım terimi (spread of means) dahil.
Kf2 imm_mix(const Kf2& A, const Kf2& B, double wA, double wB) {
    Kf2 r;
    r.p = wA * A.p + wB * B.p;
    r.v = wA * A.v + wB * B.v;
    const double dpA = A.p - r.p, dvA = A.v - r.v;
    const double dpB = B.p - r.p, dvB = B.v - r.v;
    r.P00 = wA * (A.P00 + dpA * dpA) + wB * (B.P00 + dpB * dpB);
    r.P01 = wA * (A.P01 + dpA * dvA) + wB * (B.P01 + dpB * dvB);
    r.P10 = wA * (A.P10 + dvA * dpA) + wB * (B.P10 + dvB * dpB);
    r.P11 = wA * (A.P11 + dvA * dvA) + wB * (B.P11 + dvB * dvB);
    return r;
}
}  // namespace

void ImmTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    has_last_ = false;
    pending_init_.reset();
    pending_age_ = 0;
}

// Kombine (çıktı + gating) state ve konum varyansı önbelleğini güncelle.
//   xc = Σ μ_j x_j ;  Pc00 = Σ μ_j (P_j,00 + (p_j - pc)²)
void ImmTracker::recombine(TrackImpl& t) const {
    t.cpx = t.mu_g * t.gx.p + t.mu_a * t.ax.p;
    t.cvx = t.mu_g * t.gx.v + t.mu_a * t.ax.v;
    t.cpy = t.mu_g * t.gy.p + t.mu_a * t.ay.p;
    t.cvy = t.mu_g * t.gy.v + t.mu_a * t.ay.v;
    const double dgx = t.gx.p - t.cpx, dax = t.ax.p - t.cpx;
    const double dgy = t.gy.p - t.cpy, day = t.ay.p - t.cpy;
    t.cPx00 = t.mu_g * (t.gx.P00 + dgx * dgx) + t.mu_a * (t.ax.P00 + dax * dax);
    t.cPy00 = t.mu_g * (t.gy.P00 + dgy * dgy) + t.mu_a * (t.ay.P00 + day * day);
}

// IMM adım 1-2-3: mod karışımı (mixing) + her modeli dt kadar predict + kombine.
void ImmTracker::mix_predict(TrackImpl& t, double dt) const {
    // Markov geçiş matrisi P[i][j] = i->j.
    const double Pgg = cfg_.p_stay_gentle, Pga = 1.0 - cfg_.p_stay_gentle;
    const double Paa = cfg_.p_stay_agile, Pag = 1.0 - cfg_.p_stay_agile;
    const double mg = t.mu_g, ma = t.mu_a;

    // Tahmini mod olasılıkları: cbar[j] = Σ_i P[i][j]·μ[i].
    const double eps = 1e-9;
    double cbar_g = std::max(Pgg * mg + Pag * ma, eps);
    double cbar_a = std::max(Pga * mg + Paa * ma, eps);

    // Karışım olasılıkları μ_{i|j} = P[i][j]·μ[i] / cbar[j].
    const double wgg = Pgg * mg / cbar_g, wag = Pag * ma / cbar_g;  // -> yumuşak prior
    const double wga = Pga * mg / cbar_a, waa = Paa * ma / cbar_a;  // -> çevik prior

    // Karışmış başlangıç state'leri (her iki priors'u ESKİ değerlerden hesapla, sonra yaz).
    const Kf2 ng_x = imm_mix(t.gx, t.ax, wgg, wag);
    const Kf2 na_x = imm_mix(t.gx, t.ax, wga, waa);
    t.gx = ng_x; t.ax = na_x;
    const Kf2 ng_y = imm_mix(t.gy, t.ay, wgg, wag);
    const Kf2 na_y = imm_mix(t.gy, t.ay, wga, waa);
    t.gy = ng_y; t.ay = na_y;

    // Her model CV predict (kendi süreç gürültüsü σ_a² ile).
    const double qg = cfg_.process_accel_std_gentle * cfg_.process_accel_std_gentle;
    const double qa = cfg_.process_accel_std_agile * cfg_.process_accel_std_agile;
    t.gx.predict(dt, qg); t.gy.predict(dt, qg);
    t.ax.predict(dt, qa); t.ay.predict(dt, qa);

    // Tahmini mod olasılıkları (ölçüm gelmezse bunlar kalır).
    t.mu_g = cbar_g;
    t.mu_a = cbar_a;

    recombine(t);
}

// IMM adım 4-5: her modeli Kalman düzelt, S-olabilirliğinden mod olasılığı güncelle.
void ImmTracker::correct(TrackImpl& t, const common::Detection& z) const {
    const double r = cfg_.meas_noise_std * cfg_.meas_noise_std;
    const double zx = z.centroid.x, zy = z.centroid.y;

    // Olabilirlik Λ_j = N(y; 0, S_j) — DÜZELTMEDEN ÖNCE (predicted S ile).
    auto like = [&](const Kf2& kx, const Kf2& ky) -> double {
        const double Sx = kx.P00 + r, Sy = ky.P00 + r;
        const double yx = zx - kx.p, yy = zy - ky.p;
        const double nis = yx * yx / Sx + yy * yy / Sy;  // 2-dof
        const double detS = Sx * Sy;
        return std::exp(-0.5 * nis) / (kTwoPi * std::sqrt(detS));
    };
    const double Lg = like(t.gx, t.gy);
    const double La = like(t.ax, t.ay);

    // Her modeli Kalman düzelt (Joseph form, her eksen).
    t.gx.correct(zx, r); t.gy.correct(zy, r);
    t.ax.correct(zx, r); t.ay.correct(zy, r);

    // Mod olasılığı güncelle: μ[j] ∝ cbar[j]·Λ[j] (cbar predict'te μ'ye yazılmıştı).
    double ug = t.mu_g * Lg;
    double ua = t.mu_a * La;
    const double sum = ug + ua;
    if (sum > 1e-300) { ug /= sum; ua /= sum; } else { ug = ua = 0.5; }
    // Taban (tek moda kilitlenmeyi önle).
    ug = std::clamp(ug, cfg_.mu_floor, 1.0 - cfg_.mu_floor);
    ua = 1.0 - ug;
    t.mu_g = ug;
    t.mu_a = ua;

    t.scale = z.area_px > 0 ? std::sqrt(z.area_px) : t.scale;
    t.intensity = z.intensity;
    recombine(t);
}

common::Track ImmTracker::to_public(const TrackImpl& t, common::Timestamp stamp,
                                    double dt_pred) const {
    common::Track o;
    o.id = t.id;
    o.stamp = stamp;
    o.status = t.status;
    o.position = {static_cast<float>(t.cpx), static_cast<float>(t.cpy)};
    o.velocity = {static_cast<float>(t.cvx), static_cast<float>(t.cvy)};  // px/s
    o.predicted = {static_cast<float>(t.cpx + t.cvx * dt_pred),
                   static_cast<float>(t.cpy + t.cvy * dt_pred)};
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

    const double r = cfg_.meas_noise_std * cfg_.meas_noise_std;

    // --- 1) MIX + PREDICT (her track, her model). ---
    for (auto& t : tracks_) {
        mix_predict(t, dt);
        ++t.age;
        ++t.time_since_update;
        ++t.consecutive_miss;
    }

    // --- 2+3) GATE (NIS, kombine S üzerinden) + öncelikli aç gözlü atama. ---
    const size_t nt = tracks_.size(), nd = detections.size();
    std::vector<int> det_to_track(nd, -1);
    std::vector<bool> track_taken(nt, false);

    struct Pair { double nis; int ti; int di; int prio; };
    std::vector<Pair> pairs;
    for (size_t ti = 0; ti < nt; ++ti) {
        const auto& trk = tracks_[ti];
        int prio = 0;
        if (trk.status == common::TrackStatus::Confirmed) prio = 2;
        else if (trk.status == common::TrackStatus::Tentative) prio = 1;
        const double Sx = trk.cPx00 + r, Sy = trk.cPy00 + r;
        for (size_t di = 0; di < nd; ++di) {
            const double yx = trk.cpx - detections[di].centroid.x;
            const double yy = trk.cpy - detections[di].centroid.y;
            const double nis = yx * yx / Sx + yy * yy / Sy;
            if (nis <= cfg_.gate_nis) pairs.push_back({nis, (int)ti, (int)di, prio});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        if (a.prio != b.prio) return a.prio > b.prio;
        return a.nis < b.nis;
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
                const double inv = 1.0 / dt_pend;
                const double vx = (unmatched_z->centroid.x - pending_init_->position.x) * inv;
                const double vy = (unmatched_z->centroid.y - pending_init_->position.y) * inv;
                const double speed = std::hypot(vx, vy);  // px/s
                const bool sane = speed <= cfg_.max_init_speed;
                const double pos_var = cfg_.init_pos_std * cfg_.init_pos_std;
                const double vel_var = cfg_.init_vel_std * cfg_.init_vel_std;
                TrackImpl t;
                t.id = next_id_++;
                // Her iki model AYNI ölçülü P0 ile başlar (§2.7 salınımını önler).
                t.gx.init(unmatched_z->centroid.x, sane ? vx : 0.0, pos_var, vel_var);
                t.gy.init(unmatched_z->centroid.y, sane ? vy : 0.0, pos_var, vel_var);
                t.ax = t.gx;
                t.ay = t.gy;
                t.mu_g = 0.5;
                t.mu_a = 0.5;
                t.status = common::TrackStatus::Tentative;
                t.hits = 1;
                t.age = 1;
                t.scale = unmatched_z->area_px > 0 ? std::sqrt(unmatched_z->area_px) : 1.0f;
                t.intensity = unmatched_z->intensity;
                recombine(t);
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
