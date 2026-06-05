#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "dtrack/common/time.hpp"

namespace dtrack::stabilization {

namespace {
// Kamera dönüşü ile görüntü içeriği hareketi ZIT yöndedir: kamera saga dönerse
// içerik sola kayar. Sentetik render de bu konvansiyonu izler (içerik -focal*aci
// kadar kayar). predicted_content_motion = kGyroSign * focal * dtheta.
constexpr float kGyroSign = -1.0f;

float median(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}
}  // namespace

void KltGyroStabilizer::reset() {
    prev_gray_.release();
    prev_pts_.clear();
    has_prev_ = false;
    bias_ = {0.0f, 0.0f};
}

cv::Vec2f KltGyroStabilizer::integrate_gyro(const std::vector<common::ImuSample>& imu,
                                            common::Timestamp frame_stamp,
                                            double& dt_out) const {
    // [prev_stamp, frame_stamp] aralığında (omega - bias) * dt entegrali -> dtheta.
    // angular_velocity[0] -> yatay (yaw), [1] -> dikey (pitch) eksen kayması.
    cv::Vec2f dtheta{0.0f, 0.0f};
    double total_dt = 0.0;
    common::Timestamp prev_t = prev_stamp_;
    for (const auto& s : imu) {
        if (s.stamp <= prev_stamp_ || s.stamp > frame_stamp) continue;
        const double dt = common::millis_between(prev_t, s.stamp) / 1000.0;
        if (dt <= 0.0) continue;
        dtheta[0] += static_cast<float>((s.angular_velocity[0] - bias_[0]) * dt);
        dtheta[1] += static_cast<float>((s.angular_velocity[1] - bias_[1]) * dt);
        total_dt += dt;
        prev_t = s.stamp;
    }
    dt_out = total_dt;
    // Tahmini kare-arası içerik kayması (piksel), arka plan prev->curr.
    return {kGyroSign * cfg_.focal_px * dtheta[0], kGyroSign * cfg_.focal_px * dtheta[1]};
}

common::StabilizedFrame KltGyroStabilizer::stabilize(
    const common::FramePtr& frame, const std::vector<common::ImuSample>& imu_samples) {
    common::StabilizedFrame out;
    out.frame = frame;
    out.ego.stamp = frame->stamp;
    out.ego.homography = cv::Matx33f::eye();
    out.ego.valid = false;

    // Tek kanal gri varsayıyoruz (io kaynakları öyle üretiyor).
    const cv::Mat& gray = frame->image;

    // --- İlk kare: referans kur, kimlik dönüşüm döndür. ---
    if (!has_prev_ || prev_gray_.empty()) {
        gray.copyTo(prev_gray_);
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.max_features,
                                cfg_.feature_quality, cfg_.min_feature_distance);
        prev_stamp_ = frame->stamp;
        has_prev_ = true;
        // Stabilize kare = kopya (henüz hizalanacak önceki yok).
        out.frame = frame;
        return out;
    }

    // --- 1) PREDICT: gyro entegrali. ---
    double dt = 0.0;
    const cv::Vec2f predicted = integrate_gyro(imu_samples, frame->stamp, dt);
    if (dt <= 0.0) dt = common::millis_between(prev_stamp_, frame->stamp) / 1000.0;

    // --- 2) MEASURE: LK optical flow + ileri-geri kontrol + RANSAC. ---
    cv::Matx23f M_curr_to_prev(1, 0, 0, 0, 1, 0);  // güncel -> önceki koord.
    bool of_valid = false;
    cv::Vec2f measured{0.0f, 0.0f};

    if (prev_pts_.size() >= static_cast<size_t>(cfg_.min_inliers)) {
        std::vector<cv::Point2f> cur_pts, back_pts;
        std::vector<uchar> st1, st2;
        std::vector<float> e1, e2;
        const cv::Size win(cfg_.lk_window, cfg_.lk_window);

        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts_, cur_pts, st1, e1, win,
                                 cfg_.lk_pyramid);
        // İleri-geri: curr->prev tekrar takip et, başlangıca dönmeyenleri ele.
        cv::calcOpticalFlowPyrLK(gray, prev_gray_, cur_pts, back_pts, st2, e2, win,
                                 cfg_.lk_pyramid);

        std::vector<cv::Point2f> good_prev, good_cur;
        good_prev.reserve(prev_pts_.size());
        good_cur.reserve(prev_pts_.size());
        for (size_t i = 0; i < prev_pts_.size(); ++i) {
            if (!st1[i] || !st2[i]) continue;
            const float fb = static_cast<float>(cv::norm(prev_pts_[i] - back_pts[i]));
            if (fb > cfg_.fb_error_px) continue;
            good_prev.push_back(prev_pts_[i]);
            good_cur.push_back(cur_pts[i]);
        }

        if (good_cur.size() >= static_cast<size_t>(cfg_.min_inliers)) {
            std::vector<uchar> inliers;
            // Güncel -> önceki eşleyen robust similarity (rotasyon+ölçek+öteleme).
            cv::Mat A = cv::estimateAffinePartial2D(good_cur, good_prev, inliers,
                                                    cv::RANSAC, cfg_.ransac_thresh_px);
            int n_in = inliers.empty() ? 0 : cv::countNonZero(inliers);
            if (!A.empty() && n_in >= cfg_.min_inliers) {
                M_curr_to_prev = cv::Matx23f(A.ptr<double>(0)[0], A.ptr<double>(0)[1],
                                             A.ptr<double>(0)[2], A.ptr<double>(1)[0],
                                             A.ptr<double>(1)[1], A.ptr<double>(1)[2]);
                // Ölçülen arka plan kayması (prev->curr), inlier'lar üzerinden medyan.
                std::vector<float> dx, dy;
                dx.reserve(n_in);
                dy.reserve(n_in);
                for (size_t i = 0; i < inliers.size(); ++i) {
                    if (!inliers[i]) continue;
                    dx.push_back(good_cur[i].x - good_prev[i].x);
                    dy.push_back(good_cur[i].y - good_prev[i].y);
                }
                measured = {median(dx), median(dy)};
                of_valid = true;
                // Sonraki kare için takip noktalarını güncelle (inlier güncel konumlar).
                std::vector<cv::Point2f> kept;
                kept.reserve(n_in);
                for (size_t i = 0; i < inliers.size(); ++i)
                    if (inliers[i]) kept.push_back(good_cur[i]);
                prev_pts_ = std::move(kept);
            }
        }
    }

    // --- 3) CORRECT: OF geçerliyse gyro bias'ını çıpala (online kalibrasyon). ---
    if (of_valid) {
        // innovation = ölçülen - tahmin. predicted = +focal*bias*dt terimi içerir;
        // d(predicted)/d(bias) = kGyroSign*focal*(-dt). bias'ı innovation'ı
        // kapatacak yönde güncelle.
        const cv::Vec2f innov = measured - predicted;
        if (dt > 1e-6) {
            bias_[0] -= cfg_.k_bias * innov[0] / (kGyroSign * cfg_.focal_px * static_cast<float>(dt));
            bias_[1] -= cfg_.k_bias * innov[1] / (kGyroSign * cfg_.focal_px * static_cast<float>(dt));
        }
    } else {
        // OF yok (gökyüzü/az köşe): bias-düzeltilmiş gyro ile öteleme-only undo.
        M_curr_to_prev = cv::Matx23f(1, 0, -predicted[0], 0, 1, -predicted[1]);
    }
    out.ego.valid = of_valid;

    // --- 4) ÇIKTI: ham kareyi + kestirilen kare-arası ego-motion'ı geçir. ---
    // NOT: Burada WARP YAPMIYORUZ. Stabilizer yalnızca ego-motion'ı KESTİRİR;
    // gerçek warp'ı tüketici (detection) kendi referansına kümülatif olarak yapar.
    // Böylece çift interpolasyon (kalite kaybı) ve gereksiz CPU yükü olmaz.
    out.frame = frame;

    // EgoMotion 3x3 (güncel -> önceki).
    out.ego.homography = cv::Matx33f(M_curr_to_prev(0, 0), M_curr_to_prev(0, 1),
                                     M_curr_to_prev(0, 2), M_curr_to_prev(1, 0),
                                     M_curr_to_prev(1, 1), M_curr_to_prev(1, 2), 0, 0, 1);

    // --- Bakım: köşe sayısı düşükse yeniden tespit et. ---
    gray.copyTo(prev_gray_);
    prev_stamp_ = frame->stamp;
    if (prev_pts_.size() < static_cast<size_t>(cfg_.redetect_below)) {
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.max_features,
                                cfg_.feature_quality, cfg_.min_feature_distance);
    }

    return out;
}

}  // namespace dtrack::stabilization
