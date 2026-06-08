// ============================================================================
//  GyroFlowStabilizer birim testi.
//
//  STRATEJİ: Optik akışın ÇÖKTÜĞÜ durumu (özelliksiz gökyüzü) taklit etmek için
//  OF'un güven eşiğini (min_inliers) ulaşılamaz yapıyoruz → stabilizer her zaman
//  GYRO yoluna düşer. Sonra dokulu bir kareyi BİLİNEN bir kamera dönmesiyle warp
//  edip, aynı dönmeyi üreten bir gyro penceresi besliyoruz; stabilizer'ın bu
//  dönmeyi geri kazanıp kareleri hizaladığını ölçüyoruz.
// ============================================================================
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"

using namespace dtrack;

namespace {

cv::Mat make_textured(int w = 480, int h = 360) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(60, 60, 60));
    cv::RNG rng(12345);
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
    cv::Rect roi(a.cols / 6, a.rows / 6, a.cols * 2 / 3, a.rows * 2 / 3);
    cv::Mat d;
    cv::absdiff(ga(roi), gb(roi), d);
    return cv::mean(d)[0];
}

Frame as_frame(const cv::Mat& img, int64_t id) {
    Frame f; f.image = img; f.id = id; f.t = id; return f;
}

// Sabit ω'lı, [t0, t0+dt] aralığını kapsayan 2 örnekli gyro penceresi.
std::vector<Telemetry> gyro_window(double dt, double g_pitch, double g_roll, double g_yaw) {
    Telemetry a, b;
    a.t_rel = 0.0;  b.t_rel = dt;
    a.gyro_pitch = b.gyro_pitch = g_pitch;
    a.gyro_roll  = b.gyro_roll  = g_roll;
    a.gyro_yaw   = b.gyro_yaw   = g_yaw;
    return {a, b};
}

}  // namespace

// --- Saf matematik: sıfır dönme → kimlik homografi -------------------------
TEST(GyroStabilizer, ZeroRotationIsIdentity) {
    GyroFlowStabilizer stab;
    const cv::Matx33f K = stab.intrinsics({480, 360});
    const cv::Matx33f H = GyroFlowStabilizer::rotation_homography(K, {0, 0, 0});
    const cv::Matx33f I = cv::Matx33f::eye();
    // K float ve asal nokta büyük (≈180) → K·K⁻¹ artığı ~1e-5 mertebesinde.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(H(r, c), I(r, c), 1e-3);
}

// --- AxisMap işaret/eksen eşlemesini uygular -------------------------------
TEST(GyroStabilizer, AxisMapMapsAndSigns) {
    GyroFlowStabilizer::Params p;
    p.axes.src[0]  = 2;  p.axes.sign[0] = -1.f;  // cam_x <- (-yaw)
    p.axes.src[1]  = 0;  p.axes.sign[1] = +1.f;  // cam_y <- (+pitch)
    p.axes.src[2]  = 1;  p.axes.sign[2] = +1.f;  // cam_z <- (+roll)
    GyroFlowStabilizer stab(p);

    // dt=1s, ω=(pitch=0.1, roll=0.2, yaw=0.3) deg/s → θ_cam = (-yaw, pitch, roll),
    // integrate_rotation deg→rad çevirir (gyro DERECE/s).
    const float d2r = static_cast<float>(CV_PI) / 180.f;
    const auto w = gyro_window(1.0, 0.1, 0.2, 0.3);
    const cv::Vec3f theta = stab.integrate_rotation(w);
    EXPECT_NEAR(theta[0], -0.3f * d2r, 1e-6);
    EXPECT_NEAR(theta[1], 0.1f * d2r, 1e-6);
    EXPECT_NEAR(theta[2], 0.2f * d2r, 1e-6);
}

// --- Gyro yolu BİLİNEN kamera dönmesini geri kazanır ------------------------
TEST(GyroStabilizer, GyroRecoversKnownRotation) {
    cv::Mat frame1 = make_textured();

    GyroFlowStabilizer::Params p;
    p.of.min_inliers = 1000000;  // OF'u kasıtlı güvenilmez yap → daima gyro yolu
    p.axes = GyroFlowStabilizer::AxisMap{{0, 1, 2}, {1.f, 1.f, 1.f}};  // bu test için kimlik
    GyroFlowStabilizer stab(p);

    const cv::Matx33f K = stab.intrinsics(frame1.size());

    // Bilinen prev→cur kamera dönmesi (R_{cur<-prev} ekseni). Küçük açılar.
    const cv::Vec3f theta_true(0.012f, 0.009f, 0.030f);  // ~0.7°, 0.5°, 1.7°
    const cv::Matx33f H_fwd = GyroFlowStabilizer::rotation_homography(K, theta_true);

    cv::Mat frame2;
    cv::warpPerspective(frame1, frame2, cv::Mat(H_fwd), frame1.size());

    // Aynı θ_true'yu üreten gyro penceresi (kimlik AxisMap). DİKKAT: gyro DERECE/s,
    // integrate deg→rad çevirir → ω'yu rad2deg ile büyüt ki sonuç θ_true olsun.
    const double rad2deg = 180.0 / CV_PI;
    const double dt = 0.02;  // ~iki kare arası (50 Hz örnek)
    const auto w = gyro_window(dt,
                               theta_true[0] / dt * rad2deg,   // gyro_pitch → cam_x
                               theta_true[1] / dt * rad2deg,   // gyro_roll  → cam_y
                               theta_true[2] / dt * rad2deg);  // gyro_yaw   → cam_z

    Frame out1, out2;
    cv::Matx33f H1, H2;

    // 1. kare: önceki yok → false, Mode::None.
    EXPECT_FALSE(stab.stabilize(as_frame(frame1, 0), w, out1, H1));
    EXPECT_EQ(stab.last_mode(), GyroFlowStabilizer::Mode::None);

    // 2. kare: OF güvenilmez → gyro devralır.
    const bool ok = stab.stabilize(as_frame(frame2, 1), w, out2, H2);
    EXPECT_TRUE(ok);
    EXPECT_EQ(stab.last_mode(), GyroFlowStabilizer::Mode::Gyro);

    const double before = mean_abs_diff(frame2, frame1);     // hizalanmamış
    const double after  = mean_abs_diff(out2.image, frame1); // gyro ile hizalı
    EXPECT_LT(after, before * 0.5);  // belirgin iyileşme
    EXPECT_LT(after, 18.0);          // mutlak olarak da küçük
}

// --- Köşe bolsa optik akış yolu seçilir ------------------------------------
TEST(GyroStabilizer, UsesFlowWhenFeaturesPresent) {
    cv::Mat frame1 = make_textured();
    const cv::Point2f center(frame1.cols / 2.f, frame1.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(center, 3.0, 1.0);
    M.at<double>(0, 2) += 10.0;
    M.at<double>(1, 2) += 6.0;
    cv::Mat frame2;
    cv::warpAffine(frame1, frame2, M, frame1.size());

    GyroFlowStabilizer stab;  // varsayılan: OF güvenilir
    const auto w = gyro_window(0.02, 0.0, 0.0, 0.0);
    Frame out1, out2; cv::Matx33f H1, H2;

    stab.stabilize(as_frame(frame1, 0), w, out1, H1);
    const bool ok = stab.stabilize(as_frame(frame2, 1), w, out2, H2);
    EXPECT_TRUE(ok);
    EXPECT_EQ(stab.last_mode(), GyroFlowStabilizer::Mode::Flow);
}

// --- Özelliksiz VE telemetri yok → kimlik (Mode::None, false) ---------------
TEST(GyroStabilizer, NoFeaturesNoTelemetryFallsBack) {
    cv::Mat flat(360, 480, CV_8UC3, cv::Scalar(120, 120, 120));
    GyroFlowStabilizer stab;
    Frame o1, o2; cv::Matx33f H1, H2;
    stab.stabilize(as_frame(flat, 0), {}, o1, H1);          // 1. kare
    const bool ok = stab.stabilize(as_frame(flat, 1), {}, o2, H2);  // boş pencere
    EXPECT_FALSE(ok);
    EXPECT_EQ(stab.last_mode(), GyroFlowStabilizer::Mode::None);
    EXPECT_EQ(H2, cv::Matx33f::eye());
}
