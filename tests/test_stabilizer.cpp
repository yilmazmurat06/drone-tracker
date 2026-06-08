// ============================================================================
//  OpticalFlowStabilizer birim testi.
//  Dokulu bir görüntü üretip BİLİNEN bir dönüşümle ikinci kareyi oluşturuyoruz;
//  stabilizer'ın bu dönüşümü geri kazanıp kareleri hizaladığını ölçüyoruz.
// ============================================================================
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/stabilization/optical_flow_stabilizer.hpp"

using namespace dtrack;

namespace {

// goodFeaturesToTrack'in köşe bulabilmesi için bol dokulu sahte görüntü.
cv::Mat make_textured(int w = 480, int h = 360) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(60, 60, 60));
    cv::RNG rng(12345);  // sabit tohum → deterministik
    for (int i = 0; i < 120; ++i) {
        cv::Point p(rng.uniform(0, w), rng.uniform(0, h));
        cv::Scalar c(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        if (i % 2)
            cv::circle(img, p, rng.uniform(4, 16), c, cv::FILLED);
        else
            cv::rectangle(img, cv::Rect(p.x, p.y, rng.uniform(6, 24), rng.uniform(6, 24)), c, cv::FILLED);
    }
    return img;
}

double mean_abs_diff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat ga, gb;
    cv::cvtColor(a, ga, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b, gb, cv::COLOR_BGR2GRAY);
    // kenar etkilerini dışla
    cv::Rect roi(a.cols / 6, a.rows / 6, a.cols * 2 / 3, a.rows * 2 / 3);
    cv::Mat d;
    cv::absdiff(ga(roi), gb(roi), d);
    return cv::mean(d)[0];
}

Frame as_frame(const cv::Mat& img, int64_t id) {
    Frame f; f.image = img; f.id = id; f.t = id; return f;
}

}  // namespace

// Bilinen ufak dönme+öteleme → stabilizer hizalamayı geri kazanmalı.
TEST(Stabilizer, RecoversKnownMotion) {
    cv::Mat frame1 = make_textured();

    // Bilinen dönüşüm: merkez etrafında 3° dönme + (10, 6) öteleme.
    const cv::Point2f center(frame1.cols / 2.f, frame1.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(center, 3.0, 1.0);
    M.at<double>(0, 2) += 10.0;
    M.at<double>(1, 2) += 6.0;
    cv::Mat frame2;
    cv::warpAffine(frame1, frame2, M, frame1.size());

    OpticalFlowStabilizer stab;
    Frame out1, out2;
    cv::Matx33f H1, H2;

    EXPECT_FALSE(stab.stabilize(as_frame(frame1, 0), {}, out1, H1));  // ilk kare
    const bool ok = stab.stabilize(as_frame(frame2, 1), {}, out2, H2);
    EXPECT_TRUE(ok);
    EXPECT_GE(stab.last_inliers(), 15);

    // Hizalama öncesi vs sonrası fark.
    const double before = mean_abs_diff(frame2, frame1);        // ham (kaymış)
    const double after = mean_abs_diff(out2.image, frame1);     // stabilize edilmiş
    EXPECT_LT(after, before * 0.5);  // belirgin iyileşme
    EXPECT_LT(after, 18.0);          // mutlak olarak da küçük kalmalı
}

// Düz (özelliksiz) görüntü → köşe yok → güvenli geri dönüş (false, kimlik).
TEST(Stabilizer, FeaturelessFallsBack) {
    cv::Mat flat(360, 480, CV_8UC3, cv::Scalar(120, 120, 120));
    OpticalFlowStabilizer stab;
    Frame o1, o2; cv::Matx33f H1, H2;
    stab.stabilize(as_frame(flat, 0), {}, o1, H1);
    const bool ok = stab.stabilize(as_frame(flat, 1), {}, o2, H2);
    EXPECT_FALSE(ok);                 // güvenilir homografi yok
    EXPECT_EQ(H2, cv::Matx33f::eye()); // kimlik dönüşümü
}
