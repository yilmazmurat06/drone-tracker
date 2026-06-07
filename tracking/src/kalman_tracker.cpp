#include "dtrack/tracking/kalman_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "dtrack/common/time.hpp"

namespace dtrack::tracking {

namespace {
constexpr double kTwoPi = 6.283185307179586;
}  // namespace

void KalmanTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    has_last_ = false;
    pending_init_.reset();
    pending_age_ = 0;
}

common::Track KalmanTracker::to_public(const TrackImpl& t, common::Timestamp stamp,
                                       double dt_pred) const {
    common::Track o;
    o.id = t.id;
    o.stamp = stamp;
    o.status = t.status;
    o.position = {static_cast<float>(t.kx.p), static_cast<float>(t.ky.p)};
    // Hız state'i zaten px/s -> doğrudan dışarı ver.
    o.velocity = {static_cast<float>(t.kx.v), static_cast<float>(t.ky.v)};
    // Bir kare ileri tahmin (kesişim için): p + v·dt.
    o.predicted = {static_cast<float>(t.kx.p + t.kx.v * dt_pred),
                   static_cast<float>(t.ky.p + t.ky.v * dt_pred)};
    o.scale = t.scale;
    o.hits = t.hits;
    o.time_since_update = t.time_since_update;
    float conf = std::min(1.0f, static_cast<float>(t.hits) / cfg_.confirm_hits);
    conf *= std::max(0.0f, 1.0f - static_cast<float>(t.time_since_update) / cfg_.max_coast);
    o.confidence = conf;
    return o;
}

std::vector<common::Track> KalmanTracker::update(
    const std::vector<common::Detection>& detections, common::Timestamp stamp) {
    double dt = 1.0 / 60.0;
    if (has_last_) {
        dt = common::millis_between(last_stamp_, stamp) / 1000.0;
        if (dt < 1e-3) dt = 1e-3;
        if (dt > 0.5) dt = 0.5;
    }
    last_stamp_ = stamp;
    has_last_ = true;

    const double q = cfg_.process_accel_std * cfg_.process_accel_std;  // σ_a²
    const double r = cfg_.meas_noise_std * cfg_.meas_noise_std;        // σ_r²

    // Per-tespit ölçüm gürültüsü R: Detection.meas_std ipucu varsa onu kullan
    // (kapalı-döngü ROI kurtarma tespitleri daha gürültülü -> daha büyük R -> daha
    // az ağırlıklı güncelleme, daha geniş etkin kapı). Yoksa varsayılan σ_r².
    std::vector<double> rd(detections.size(), r);
    for (size_t di = 0; di < detections.size(); ++di) {
        if (detections[di].meas_std > 0.0f)
            rd[di] = static_cast<double>(detections[di].meas_std) * detections[di].meas_std;
    }

    // --- 1) PREDICT (her iz, her eksen). ---
    for (auto& t : tracks_) {
        t.kx.predict(dt, q);
        t.ky.predict(dt, q);
        ++t.age;
        ++t.time_since_update;
        ++t.consecutive_miss;
    }

    // --- 2+3+4) İLİŞKİLENDİRME + GÜNCELLEME. ---
    // İki yol:
    //   use_pdaf=true  -> yerleşik (Confirmed/Coasting) izler PDAF (clutter-dayanıklı
    //                     yumuşak ilişkilendirme); tentative izler NN (başlatma sade).
    //   use_pdaf=false -> hepsi öncelikli aç gözlü NN (eski, doğrulanmış yol).
    // GATE: NIS = (yx²/Sx)+(yy²/Sy), 2-dof χ². Coast'ta P->S büyür -> kapı doğal genişler.
    const size_t nt = tracks_.size(), nd = detections.size();
    const common::Detection* unmatched_z = nullptr;
    std::vector<bool> det_claimed(nd, false);  // yerleşik ize ait tespit -> yeni iz başlatmaz

    if (cfg_.use_pdaf) {
        // === FAZ 1: Yerleşik izler -> PDAF (Bayesçi yumuşak ilişkilendirme). ===
        std::vector<int> idx;
        std::vector<double> Lw, yx, yy;
        for (auto& trk : tracks_) {
            if (trk.status != common::TrackStatus::Confirmed &&
                trk.status != common::TrackStatus::Coasting)
                continue;
            idx.clear(); Lw.clear(); yx.clear(); yy.clear();
            const double Sx0 = trk.kx.P00 + r, Sy0 = trk.ky.P00 + r;  // b için temsili |S|
            for (size_t di = 0; di < nd; ++di) {
                const double Sxi = trk.kx.P00 + rd[di], Syi = trk.ky.P00 + rd[di];
                const double iyx = detections[di].centroid.x - trk.kx.p;
                const double iyy = detections[di].centroid.y - trk.ky.p;
                const double nis = iyx * iyx / Sxi + iyy * iyy / Syi;
                if (nis > cfg_.gate_nis) continue;
                idx.push_back((int)di);
                Lw.push_back(std::exp(-0.5 * nis));
                yx.push_back(iyx);
                yy.push_back(iyy);
            }
            if (idx.empty()) continue;  // kapıda aday yok -> coast (predict ilerletti)
            // Parametrik PDAF: b = λ·2π·√|S|·(1−P_D·P_G)/P_D ; β_i = L_i/(b+ΣL) ; β0 = b/(b+ΣL).
            const double b = cfg_.pdaf_clutter_density * kTwoPi * std::sqrt(Sx0 * Sy0) *
                             (1.0 - cfg_.pdaf_pd * cfg_.pdaf_pg) / std::max(1e-6, cfg_.pdaf_pd);
            double sumL = 0.0;
            for (double L : Lw) sumL += L;
            const double denom = b + sumL;
            const double beta0 = b / denom;
            std::vector<double> beta(idx.size());
            double r_eff_num = 0.0, w_sum = 0.0;
            int best_i = 0;
            double best_b = -1.0;
            for (size_t i = 0; i < idx.size(); ++i) {
                beta[i] = Lw[i] / denom;
                r_eff_num += beta[i] * rd[idx[i]];
                w_sum += beta[i];
                det_claimed[idx[i]] = true;
                if (beta[i] > best_b) { best_b = beta[i]; best_i = (int)i; }
            }
            const double r_eff = w_sum > 1e-9 ? r_eff_num / w_sum : r;
            trk.kx.correct_pda(yx.data(), beta.data(), (int)idx.size(), beta0, r_eff);
            trk.ky.correct_pda(yy.data(), beta.data(), (int)idx.size(), beta0, r_eff);
            // Yaşam döngüsü: birleşik tespit olasılığı (1−β0) yeterliyse "tespit edildi".
            if ((1.0 - beta0) >= 0.5) {
                const auto& bd = detections[idx[best_i]];
                trk.scale = bd.area_px > 0 ? std::sqrt(bd.area_px) : trk.scale;
                trk.intensity = bd.intensity;
                ++trk.hits;
                trk.time_since_update = 0;
                trk.consecutive_miss = 0;
                if (trk.status == common::TrackStatus::Coasting)
                    trk.status = common::TrackStatus::Confirmed;
            }
        }

        // === FAZ 2: Tentative izler -> sahiplenilmemiş tespitlerle aç gözlü NN. ===
        std::vector<int> det_to_track(nd, -1);
        std::vector<bool> track_taken(nt, false);
        struct Pair { double nis; int ti; int di; };
        std::vector<Pair> pairs;
        for (size_t ti = 0; ti < nt; ++ti) {
            if (tracks_[ti].status != common::TrackStatus::Tentative) continue;
            const auto& trk = tracks_[ti];
            for (size_t di = 0; di < nd; ++di) {
                if (det_claimed[di]) continue;
                const double nis = trk.kx.nis(detections[di].centroid.x, rd[di]) +
                                   trk.ky.nis(detections[di].centroid.y, rd[di]);
                if (nis <= cfg_.gate_nis) pairs.push_back({nis, (int)ti, (int)di});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) { return a.nis < b.nis; });
        for (const auto& p : pairs) {
            if (track_taken[p.ti] || det_to_track[p.di] != -1) continue;
            track_taken[p.ti] = true;
            det_to_track[p.di] = p.ti;
        }
        for (size_t di = 0; di < nd; ++di) {
            const int ti = det_to_track[di];
            if (ti >= 0) {
                auto& tr = tracks_[ti];
                tr.kx.correct(detections[di].centroid.x, rd[di]);
                tr.ky.correct(detections[di].centroid.y, rd[di]);
                tr.scale = detections[di].area_px > 0 ? std::sqrt(detections[di].area_px) : tr.scale;
                tr.intensity = detections[di].intensity;
                ++tr.hits;
                tr.time_since_update = 0;
                tr.consecutive_miss = 0;
            } else if (!det_claimed[di]) {
                unmatched_z = &detections[di];
            }
        }
    } else {
        // === Saf NN (eski, doğrulanmış yol). ===
        std::vector<int> det_to_track(nd, -1);
        std::vector<bool> track_taken(nt, false);
        struct Pair { double nis; int ti; int di; int prio; };
        std::vector<Pair> pairs;
        for (size_t ti = 0; ti < nt; ++ti) {
            const auto& trk = tracks_[ti];
            int prio = 0;
            if (trk.status == common::TrackStatus::Confirmed) prio = 2;
            else if (trk.status == common::TrackStatus::Tentative) prio = 1;
            for (size_t di = 0; di < nd; ++di) {
                const double nis = trk.kx.nis(detections[di].centroid.x, rd[di]) +
                                   trk.ky.nis(detections[di].centroid.y, rd[di]);
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
        for (size_t di = 0; di < nd; ++di) {
            const int ti = det_to_track[di];
            if (ti >= 0) {
                auto& tr = tracks_[ti];
                tr.kx.correct(detections[di].centroid.x, rd[di]);
                tr.ky.correct(detections[di].centroid.y, rd[di]);
                tr.scale = detections[di].area_px > 0 ? std::sqrt(detections[di].area_px) : tr.scale;
                tr.intensity = detections[di].intensity;
                ++tr.hits;
                tr.time_since_update = 0;
                tr.consecutive_miss = 0;
                if (tr.status == common::TrackStatus::Coasting)
                    tr.status = common::TrackStatus::Confirmed;
            } else {
                unmatched_z = &detections[di];
            }
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
                // ÖLÇÜLÜ P0 (§2.7 salınımını önler): konum ~σ_r, hız ~init_vel_std.
                t.kx.init(unmatched_z->centroid.x, sane ? vx : 0.0, pos_var, vel_var);
                t.ky.init(unmatched_z->centroid.y, sane ? vy : 0.0, pos_var, vel_var);
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
