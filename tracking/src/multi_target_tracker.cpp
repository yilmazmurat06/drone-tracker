// ============================================================================
//  MultiTargetTracker implementasyonu.
// ============================================================================
#include "dtrack/tracking/multi_target_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace dtrack {

MultiTargetTracker::MultiTargetTracker() : MultiTargetTracker(Params{}) {}
MultiTargetTracker::MultiTargetTracker(Params params) : p_(params) {}

void MultiTargetTracker::reset() {
    tracks_.clear();
    next_id_ = 0;
}

// 4 durum (x,y,vx,vy), 2 ölçüm (x,y), sabit-hız modeli.
void MultiTargetTracker::init_kf(Internal& in, const cv::Point2f& pos) const {
    cv::KalmanFilter& kf = in.kf;
    kf.init(4, 2, 0, CV_32F);

    // Geçiş A: sabit hız, dt=1 → p' = p + v
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    // Ölçüm H: yalnız konumu gözlemleriz
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(p_.process_noise));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(p_.measure_noise));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(50));  // başlangıç belirsizliği

    kf.statePost = (cv::Mat_<float>(4, 1) << pos.x, pos.y, 0, 0);
}

void MultiTargetTracker::update(const std::vector<Detection>& dets,
                                std::vector<Track>& out_tracks) {
    out_tracks.clear();

    // --- 1) Tahmin: her track'i bir kare ileri taşı.
    std::vector<cv::Point2f> pred(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) {
        const cv::Mat s = tracks_[i].kf.predict();
        pred[i] = {s.at<float>(0), s.at<float>(1)};
        tracks_[i].t.age++;
    }

    // --- 2) İlişkilendirme: gate içindeki tüm (track,tespit) çiftlerini mesafeye
    //        göre sırala, aç-gözlü ata (her ikisi de boşsa eşle). Basit ve etkili.
    struct Pair { float d; int ti; int di; };
    std::vector<Pair> pairs;
    for (size_t ti = 0; ti < tracks_.size(); ++ti)
        for (size_t di = 0; di < dets.size(); ++di) {
            const float d = static_cast<float>(cv::norm(pred[ti] - dets[di].centroid));
            if (d <= p_.gate_dist) pairs.push_back({d, (int)ti, (int)di});
        }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.d < b.d; });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> det_used(dets.size(), false);
    for (const Pair& pr : pairs) {
        if (track_used[pr.ti] || det_used[pr.di]) continue;
        track_used[pr.ti] = det_used[pr.di] = true;

        Internal& in = tracks_[pr.ti];
        const Detection& d = dets[pr.di];
        // --- 3a) Eşleşen track → Kalman düzeltme.
        cv::Mat z = (cv::Mat_<float>(2, 1) << d.centroid.x, d.centroid.y);
        const cv::Mat s = in.kf.correct(z);
        in.t.pos = {s.at<float>(0), s.at<float>(1)};
        in.t.vel = {s.at<float>(2), s.at<float>(3)};
        in.t.bbox = d.bbox;                    // ölçülen silüet kutusu
        in.t.hits++;
        in.t.misses = 0;
        in.t.score = std::max(in.t.score, d.score);
        in.t.t = d.t;
        // Onay: yeterli isabet VE doğuştan yeterli NET yol. İkinci koşul, yerinde
        // titreşen paralaks kümelerini (yol almayanları) eler.
        const double travel = cv::norm(in.t.pos - in.birth);
        if (in.t.status == Track::Status::Tentative &&
            in.t.hits >= p_.confirm_hits && travel >= p_.min_travel)
            in.t.status = Track::Status::Confirmed;
    }

    // --- 3b) Eşleşmeyen track'ler → coasting (tahminle yaşa), misses++.
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (track_used[ti]) continue;
        Internal& in = tracks_[ti];
        // Boyutu koru, kutuyu tahmini konuma ortala (coasting'de hedefle birlikte gider).
        if (in.t.bbox.width > 0)
            in.t.bbox = cv::Rect(cvRound(pred[ti].x) - in.t.bbox.width / 2,
                                 cvRound(pred[ti].y) - in.t.bbox.height / 2,
                                 in.t.bbox.width, in.t.bbox.height);
        in.t.pos = pred[ti];
        in.t.misses++;
    }

    // --- 3c) Eşleşmeyen tespitler → yeni Tentative track.
    for (size_t di = 0; di < dets.size(); ++di) {
        if (det_used[di]) continue;
        Internal in;
        init_kf(in, dets[di].centroid);
        in.birth = dets[di].centroid;
        in.t.id = next_id_++;
        in.t.pos = dets[di].centroid;
        in.t.vel = {0, 0};
        in.t.bbox = dets[di].bbox;
        in.t.score = dets[di].score;
        in.t.age = 1;
        in.t.hits = 1;
        in.t.misses = 0;
        in.t.status = Track::Status::Tentative;
        in.t.t = dets[di].t;
        tracks_.push_back(std::move(in));
    }

    // --- 4) Yaşam yönetimi: çok kaçıran track'i sil.
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const Internal& in) { return in.t.misses > p_.max_misses; }),
        tracks_.end());

    // --- 5) Aktif track'leri döndür (çağıran Confirmed'leri filtreleyebilir).
    out_tracks.reserve(tracks_.size());
    for (const Internal& in : tracks_) out_tracks.push_back(in.t);
}

} // namespace dtrack
