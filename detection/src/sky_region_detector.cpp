#include "dtrack/detection/sky_region_detector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace dtrack::detection {

void SkyRegionDetector::reset() {
    prev_gray_.release();
    prev_pts_.clear();
    sky_mask_.release();
    has_prev_ = false;
}

// Gökyüzü maskesi: parlak (Otsu) ∧ düşük-doku (yerel std) -> üst-kenara bağlı büyük
// bileşen. Arazi (alt, dokulu, koyu) elenir.
cv::Mat SkyRegionDetector::compute_sky_mask(const cv::Mat& gray) const {
    // Yerel std (doku ölçüsü): sqrt(E[I^2] - E[I]^2).
    cv::Mat f;
    gray.convertTo(f, CV_32F);
    const cv::Size win(cfg_.sky_texture_win, cfg_.sky_texture_win);
    cv::Mat m, m2;
    cv::boxFilter(f, m, CV_32F, win);
    cv::boxFilter(f.mul(f), m2, CV_32F, win);
    cv::Mat var = m2 - m.mul(m), sd;
    cv::sqrt(cv::max(var, 0.0f), sd);

    cv::Mat bright;
    cv::threshold(gray, bright, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat smooth = sd < cfg_.sky_texture_std;
    smooth.convertTo(smooth, CV_8U, 255);

    cv::Mat sky;
    cv::bitwise_and(bright, smooth, sky);
    const cv::Mat kc = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                 {cfg_.sky_morph, cfg_.sky_morph});
    cv::morphologyEx(sky, sky, cv::MORPH_CLOSE, kc);
    cv::morphologyEx(sky, sky, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    // Üst-kenara değen ve yeterince büyük bileşenleri tut (gerçek gökyüzü).
    cv::Mat lab, stats, cent;
    const int n = cv::connectedComponentsWithStats(sky, lab, stats, cent, 8);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8U);
    const int min_area = gray.cols * cfg_.sky_min_area_cols_mult;
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_TOP) <= cfg_.sky_top_rows &&
            stats.at<int>(i, cv::CC_STAT_AREA) > min_area) {
            out.setTo(255, lab == i);
        }
    }
    return out;
}

std::vector<common::Detection> SkyRegionDetector::detect(const common::StabilizedFrame& sf) {
    std::vector<common::Detection> out;
    if (!sf.frame || sf.frame->image.empty()) return out;
    const cv::Mat& gray = sf.frame->image;

    sky_mask_ = compute_sky_mask(gray);

    // İlk kare: referansı kur, tespit yok.
    if (!has_prev_ || prev_gray_.empty()) {
        gray.copyTo(prev_gray_);
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.max_features,
                                cfg_.feature_quality, cfg_.min_feature_distance, sky_mask_);
        has_prev_ = true;
        return out;
    }

    // --- Gökyüzü özelliklerinden kare-arası dönüşüm (önceki -> güncel) ---
    cv::Matx23f M_prev_to_cur(1, 0, 0, 0, 1, 0);
    bool aligned = false;
    if (prev_pts_.size() >= static_cast<size_t>(cfg_.min_inliers)) {
        std::vector<cv::Point2f> cur_pts, back_pts;
        std::vector<uchar> st1, st2;
        std::vector<float> e1, e2;
        const cv::Size w(cfg_.lk_window, cfg_.lk_window);
        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts_, cur_pts, st1, e1, w, cfg_.lk_pyramid);
        cv::calcOpticalFlowPyrLK(gray, prev_gray_, cur_pts, back_pts, st2, e2, w, cfg_.lk_pyramid);
        std::vector<cv::Point2f> gp, gc;
        for (size_t i = 0; i < prev_pts_.size(); ++i) {
            if (!st1[i] || !st2[i]) continue;
            if (cv::norm(prev_pts_[i] - back_pts[i]) > cfg_.fb_error_px) continue;
            gp.push_back(prev_pts_[i]);
            gc.push_back(cur_pts[i]);
        }
        if (gp.size() >= static_cast<size_t>(cfg_.min_inliers)) {
            std::vector<uchar> inl;
            cv::Mat A = cv::estimateAffinePartial2D(gp, gc, inl, cv::RANSAC, cfg_.ransac_thresh_px);
            if (!A.empty() && cv::countNonZero(inl) >= cfg_.min_inliers) {
                M_prev_to_cur = cv::Matx23f(A.ptr<double>(0)[0], A.ptr<double>(0)[1],
                                            A.ptr<double>(0)[2], A.ptr<double>(1)[0],
                                            A.ptr<double>(1)[1], A.ptr<double>(1)[2]);
                aligned = true;
            }
        }
    }

    // Hizalama başarısızsa (gökyüzü köşesi az / RANSAC çöktü): ham fark her şeyi
    // hareketli gösterir -> FP flood. Bu kareyi atla, sadece durumu güncelle.
    if (!aligned) {
        gray.copyTo(prev_gray_);
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.max_features,
                                cfg_.feature_quality, cfg_.min_feature_distance, sky_mask_);
        return out;
    }

    // --- Önceki kareyi güncele hizala, gökyüzü içinde fark al ---
    cv::Mat prev_aligned, diff;
    cv::warpAffine(prev_gray_, prev_aligned, M_prev_to_cur, gray.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::absdiff(gray, prev_aligned, diff);

    // Horizon kenarını dışlamak için maskeyi erit (warp/kenar artığı oraya düşer).
    cv::Mat sky_er;
    cv::erode(sky_mask_, sky_er,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, {cfg_.sky_erode, cfg_.sky_erode}));
    cv::Mat motion;
    cv::bitwise_and(diff, diff, motion, sky_er);

    cv::Mat mask;
    cv::threshold(motion, mask, cfg_.motion_thresh, 255, cv::THRESH_BINARY);
    if (cfg_.motion_open > 0)
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   {cfg_.motion_open, cfg_.motion_open}));
    if (cfg_.motion_close > 0)
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   {cfg_.motion_close, cfg_.motion_close}));

    // --- Bağlı bileşen -> blob eleme -> centroid ---
    cv::Mat lab, stats, cent;
    const int n = cv::connectedComponentsWithStats(mask, lab, stats, cent, 8);
    for (int i = 1; i < n; ++i) {
        const double area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < cfg_.min_area || area > cfg_.max_area) continue;
        const int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const float aspect = static_cast<float>(std::max(bw, bh)) /
                             std::max(1, std::min(bw, bh));
        if (aspect > cfg_.max_aspect) continue;

        common::Detection d;
        d.stamp = sf.frame->stamp;
        d.modality = sf.frame->modality;
        d.centroid = {static_cast<float>(cent.at<double>(i, 0)),
                      static_cast<float>(cent.at<double>(i, 1))};
        d.area_px = static_cast<float>(area);
        d.aspect_ratio = aspect;
        // Parlaklık: blob bölgesinin ham görüntü ortalaması (kaba).
        const int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int by = stats.at<int>(i, cv::CC_STAT_TOP);
        cv::Rect roi(bx, by, bw, bh);
        roi &= cv::Rect(0, 0, gray.cols, gray.rows);
        d.intensity = roi.area() > 0 ? static_cast<float>(cv::mean(gray(roi))[0]) : 0.0f;
        d.drone_score = 0.0f;
        out.push_back(d);
    }

    // Sonraki kare için: güncel kareyi sakla + gökyüzü özelliklerini yenile.
    gray.copyTo(prev_gray_);
    cv::goodFeaturesToTrack(prev_gray_, prev_pts_, cfg_.max_features,
                            cfg_.feature_quality, cfg_.min_feature_distance, sky_mask_);
    return out;
}

}  // namespace dtrack::detection
