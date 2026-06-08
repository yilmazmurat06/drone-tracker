// ============================================================================
//  MovingTargetDetector birim testi.
//
//  STRATEJİ: Gerçek footage'da hedefin kesin piksel kutusu yok (telemetri sadece
//  bizim drone). Bu yüzden BİLİNEN konum/hızda küçük bir hedef ENJEKTE edip
//  (ground-truth biz koyduk) detektörün onu yakaladığını ölçüyoruz. Statik
//  sahnede de yanlış-pozitif üretmediğini doğruluyoruz.
// ============================================================================
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/detection/moving_target_detector.hpp"

using namespace dtrack;

namespace {

// Statik gökyüzü + birkaç sabit bulut (düşük frekanslı, hareketsiz arka plan).
cv::Mat make_sky(int w = 480, int h = 360) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(205, 200, 190));
    cv::circle(img, {120, 80}, 40, cv::Scalar(235, 235, 235), cv::FILLED);
    cv::circle(img, {320, 120}, 55, cv::Scalar(230, 232, 235), cv::FILLED);
    cv::GaussianBlur(img, img, {31, 31}, 0);
    return img;
}

Frame as_frame(const cv::Mat& img, int64_t id) {
    Frame f; f.image = img; f.id = id; f.t = id; return f;
}

}  // namespace

// Bilinen hızda hareket eden küçük koyu hedef → yakalanmalı, konumu yakın olmalı.
TEST(Detector, DetectsMovingTarget) {
    MovingTargetDetector det;

    const int y = 180, r = 5, step = 10, x0 = 60;
    std::vector<Detection> out;
    bool ran = false;
    int last_x = 0;

    // warmup(8) + birkaç kare daha çalıştır.
    for (int k = 0; k <= 13; ++k) {
        cv::Mat fr = make_sky();
        last_x = x0 + step * k;
        cv::circle(fr, {last_x, y}, r, cv::Scalar(40, 40, 40), cv::FILLED);  // koyu hedef
        ran = det.detect(as_frame(fr, k), out);
    }

    EXPECT_TRUE(ran);                       // warmup geçti → çalıştı
    ASSERT_FALSE(out.empty());              // en az bir aday

    // Adaylardan biri güncel hedef konumuna yakın olmalı.
    bool near = false;
    for (const auto& d : out) {
        const double dist = std::hypot(d.centroid.x - last_x, d.centroid.y - y);
        if (dist < 14.0) near = true;
    }
    EXPECT_TRUE(near) << "hedefe yakin tespit yok (x=" << last_x << ")";
}

// Tamamen statik sahne (hedef yok) → warmup sonrası yanlış-pozitif olmamalı.
TEST(Detector, NoFalsePositiveOnStatic) {
    MovingTargetDetector det;
    const cv::Mat sky = make_sky();
    std::vector<Detection> out;

    for (int k = 0; k <= 13; ++k)
        det.detect(as_frame(sky, k), out);

    EXPECT_TRUE(out.empty()) << "statik sahnede " << out.size() << " yanlis tespit";
}

// ROI dışındaki hareket yok sayılmalı (cue-odaklı arama).
TEST(Detector, RespectsRoi) {
    MovingTargetDetector det;
    // Arama bölgesini sol-üste sınırla; hedef sağ-altta hareket etsin.
    det.set_roi(cv::Rect(30, 30, 120, 120));

    std::vector<Detection> out;
    for (int k = 0; k <= 13; ++k) {
        cv::Mat fr = make_sky();
        cv::circle(fr, {300 + 8 * k, 250}, 5, cv::Scalar(40, 40, 40), cv::FILLED);
        det.detect(as_frame(fr, k), out);
    }
    EXPECT_TRUE(out.empty()) << "ROI disindaki hareket tespit edildi";
}
