// ============================================================================
//  LockIntegrity implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/lock_integrity.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace dtrack {
namespace lock_integrity {

namespace {
// Bir dikdörtgendeki "gök" piksel sayısı: B>=G (renk varsa) ∪ parlaklık>bright.
long count_sky(const cv::Mat& frame, const cv::Rect& r, float bright) {
    if (r.area() <= 0) return 0;
    const cv::Mat sub = frame(r);
    cv::Mat gray, sky_mask, bright_mask, mask;
    if (sub.channels() == 3) {
        std::vector<cv::Mat> ch;
        cv::split(sub, ch);             // ch[0]=B, ch[1]=G, ch[2]=R
        sky_mask = ch[0] >= ch[1];      // mavi >= yeşil → gök (çimen/ağaç yeşil-baskın)
        cv::cvtColor(sub, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = sub;                     // gri görüntü: renk yok
        sky_mask = cv::Mat::zeros(sub.size(), CV_8U);
    }
    bright_mask = gray > bright;        // parlak (bulut/açık gök)
    cv::bitwise_or(sky_mask, bright_mask, mask);
    return cv::countNonZero(mask);
}
}  // namespace

float sky_ring(const cv::Mat& frame, const cv::Rect& box, int ring_px, float bright_thresh) {
    if (frame.empty() || box.width <= 0 || box.height <= 0) return 0.f;
    const int m = ring_px > 0 ? ring_px : std::max(box.width, box.height);
    const cv::Rect full(0, 0, frame.cols, frame.rows);
    const cv::Rect outer = cv::Rect(box.x - m, box.y - m,
                                    box.width + 2 * m, box.height + 2 * m) & full;
    const cv::Rect inner = box & full;
    if (outer.area() <= 0) return 0.f;

    const long sky_outer = count_sky(frame, outer, bright_thresh);
    const long sky_inner = count_sky(frame, inner, bright_thresh);
    const long denom = static_cast<long>(outer.area()) - static_cast<long>(inner.area());
    if (denom <= 0)  // halka kalmadıysa (iç ≈ dış) → dış oranına düş
        return outer.area() > 0 ? static_cast<float>(sky_outer) / outer.area() : 0.f;
    return static_cast<float>(sky_outer - sky_inner) / static_cast<float>(denom);
}

bool size_sane(const cv::Rect& box, const cv::Size& frame_size,
               const cv::Rect& lock_box0, float max_area_frac, float max_growth) {
    const double frame_area = static_cast<double>(frame_size.width) * frame_size.height;
    const double area = static_cast<double>(box.width) * box.height;
    if (frame_area > 0 && area > max_area_frac * frame_area) return false;  // tüm kareyi kaplıyor
    double area0 = static_cast<double>(lock_box0.width) * lock_box0.height;
    // MİNİK kutu tabanı: 5×4 gibi benek kilidinde oran aşırı hassas olur (tracker'ın
    // ~10×10 minimumu bile "5× büyüme" sayılır); uzak gerçek hedefin doğal kutu
    // titreşimi de yanlış alarm verir. Referansı en az 20×20 (400 px²) kabul et.
    if (area0 > 0) area0 = std::max(area0, 400.0);
    if (area0 > 0 && area > max_growth * area0) return false;               // kilitten beri patladı
    return true;
}

float edge_density(const cv::Mat& frame, const cv::Rect& box) {
    const cv::Rect r = box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (r.area() <= 0) return 0.f;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame(r), gray, cv::COLOR_BGR2GRAY);
    else                       gray = frame(r);
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0);
    cv::Sobel(gray, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, mag);
    return static_cast<float>(cv::mean(mag)[0]) / 255.f;
}

bool motion_sane(const cv::Rect& box, const cv::Rect& prev_box, float max_jump_px) {
    if (prev_box.area() <= 0) return true;  // referans yok (ilk kare)
    const cv::Point2f c0(prev_box.x + prev_box.width * 0.5f, prev_box.y + prev_box.height * 0.5f);
    const cv::Point2f c1(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    return cv::norm(c1 - c0) <= max_jump_px;
}

}  // namespace lock_integrity
}  // namespace dtrack
