// ============================================================================
//  OpticalFlowStabilizer implementasyonu.
// ============================================================================
#include "dtrack/stabilization/optical_flow_stabilizer.hpp"

#include <vector>

#include <opencv2/calib3d.hpp>   // findHomography, RANSAC
#include <opencv2/imgproc.hpp>   // cvtColor, goodFeaturesToTrack, warpPerspective
#include <opencv2/video.hpp>     // calcOpticalFlowPyrLK

namespace dtrack {

namespace {
cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}
}  // namespace

OpticalFlowStabilizer::OpticalFlowStabilizer() : OpticalFlowStabilizer(Params{}) {}
OpticalFlowStabilizer::OpticalFlowStabilizer(Params params) : p_(params) {}

void OpticalFlowStabilizer::reset() {
    has_prev_ = false;
    prev_gray_.release();
}

bool OpticalFlowStabilizer::stabilize(const Frame& in,
                                      const std::vector<Telemetry>& /*telemetry*/,
                                      Frame& out,
                                      cv::Matx33f& homography) {
    homography = cv::Matx33f::eye();
    last_tracked_ = 0;
    last_inliers_ = 0;

    cv::Mat gray = to_gray(in.image);

    // İlk kare: hizalanacak referans yok.
    if (!has_prev_ || prev_gray_.empty()) {
        out = in;
        gray.copyTo(prev_gray_);
        has_prev_ = true;
        return false;
    }

    bool ok = false;
    cv::Mat best_h;  // CV_64F, cur→prev

    // 1) Önceki karede köşeler.
    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(prev_gray_, prev_pts, p_.max_corners,
                            p_.quality_level, p_.min_distance);

    if (prev_pts.size() >= static_cast<size_t>(p_.min_inliers)) {
        // 2) KLT ile bu kareye takip.
        std::vector<cv::Point2f> cur_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts, cur_pts, status, err,
                                 cv::Size(p_.lk_window, p_.lk_window), 3);

        std::vector<cv::Point2f> p0, p1;
        p0.reserve(prev_pts.size());
        p1.reserve(prev_pts.size());
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                p0.push_back(prev_pts[i]);  // önceki konum
                p1.push_back(cur_pts[i]);   // geçerli konum
            }
        }
        last_tracked_ = static_cast<int>(p0.size());

        // 3) cur(p1) → prev(p0) homografisi, RANSAC.
        if (p0.size() >= static_cast<size_t>(p_.min_inliers)) {
            std::vector<uchar> inliers;
            cv::Mat h = cv::findHomography(p1, p0, cv::RANSAC, p_.ransac_reproj, inliers);
            if (!h.empty()) {
                last_inliers_ = cv::countNonZero(inliers);
                if (last_inliers_ >= p_.min_inliers) {
                    best_h = h;
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            homography(r, c) = static_cast<float>(h.at<double>(r, c));
                    ok = true;
                }
            }
        }
    }

    // 4) Warp veya güvenli geri dönüş.
    if (ok) {
        cv::warpPerspective(in.image, out.image, best_h, in.image.size(),
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        out.id = in.id;
        out.t = in.t;
    } else {
        out = in;  // ileride (3b) gyro devralır
    }

    gray.copyTo(prev_gray_);
    return ok;
}

} // namespace dtrack
