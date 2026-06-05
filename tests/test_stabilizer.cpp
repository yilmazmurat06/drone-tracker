// Stabilizer doğrulama testleri (OpenCV gerekir).
//
// A) Stabilizasyon doğruluğu: kestirilen kare-arası dönüşümle hizalandıktan SONRA
//    artık arka plan hareketi, ham (hizalanmamış) harekete göre çok küçük olmalı.
//    Bu ölçüm konvansiyon-bağımsızdır: doğrudan "dönüşüm arka planı önceki kareye
//    oturtuyor mu?" sorusunu sınar.
// B) Gyro bias kalibrasyonu: gyro'ya BİLİNEN sabit bir bias enjekte et; optical
//    flow gerçek hareketi ölçtüğü için stabilizer'ın online bias kestirimi bu
//    değere yakınsamalı (füzyon döngüsünün ve işaretin doğruluğu).

#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"

using namespace dtrack;

static int g_failures = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// İki gri kare arası medyan arka plan kayması (px). Köşe bulup LK ile takip eder.
static double median_flow(const cv::Mat& a, const cv::Mat& b) {
    std::vector<cv::Point2f> pa;
    cv::goodFeaturesToTrack(a, pa, 200, 0.01, 8.0);
    if (pa.size() < 20) return -1.0;
    std::vector<cv::Point2f> pb;
    std::vector<uchar> st;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(a, b, pa, pb, st, err, cv::Size(21, 21), 3);
    std::vector<float> mags;
    for (size_t i = 0; i < pa.size(); ++i) {
        if (!st[i]) continue;
        mags.push_back(static_cast<float>(cv::norm(pb[i] - pa[i])));
    }
    if (mags.size() < 20) return -1.0;
    std::nth_element(mags.begin(), mags.begin() + mags.size() / 2, mags.end());
    return mags[mags.size() / 2];
}

static void test_stabilization_reduces_motion() {
    std::printf("test_stabilization_reduces_motion\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0, false);
    io::SyntheticImuSource imu(cfg, 1000.0, t0);
    cam.open();
    imu.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    stabilization::KltGyroStabilizer stab(scfg);

    cv::Mat prev_raw;
    double sum_raw = 0, sum_res = 0;
    int n = 0;
    for (int i = 0; i < 40; ++i) {
        auto f = cam.next_frame();
        CHECK(f.has_value());
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        auto out = stab.stabilize(*f, imu_batch);

        const cv::Mat& curr_raw = (*f)->image;
        if (!prev_raw.empty() && i >= 5) {  // ısınma sonrası
            // Stabilizer artık warp yapmıyor -> ego.homography ile kendimiz hizalayıp
            // artık hareketi ölçeriz (güncel -> önceki koordinat).
            cv::Matx23f M(out.ego.homography(0, 0), out.ego.homography(0, 1),
                          out.ego.homography(0, 2), out.ego.homography(1, 0),
                          out.ego.homography(1, 1), out.ego.homography(1, 2));
            cv::Mat aligned;
            cv::warpAffine(curr_raw, aligned, M, curr_raw.size(), cv::INTER_LINEAR,
                           cv::BORDER_REPLICATE);
            const double raw = median_flow(prev_raw, curr_raw);
            const double res = median_flow(prev_raw, aligned);
            if (raw > 0 && res >= 0) {
                sum_raw += raw;
                sum_res += res;
                ++n;
            }
        }
        prev_raw = curr_raw.clone();
    }
    CHECK(n > 20);
    const double avg_raw = sum_raw / n;
    const double avg_res = sum_res / n;
    std::printf("   ham hareket ort=%.2f px, stabilize sonrasi artik=%.2f px\n",
                avg_raw, avg_res);
    // Ham hareket anlamlı olmalı, artık çok küçük olmalı.
    CHECK(avg_raw > 1.0);
    CHECK(avg_res < 0.5);
    CHECK(avg_res < avg_raw * 0.3);
}

static void test_gyro_bias_converges() {
    std::printf("test_gyro_bias_converges\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0, false);
    io::SyntheticImuSource imu(cfg, 1000.0, t0);
    cam.open();
    imu.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    scfg.k_bias = 0.15f;  // test için daha hızlı yakınsama
    stabilization::KltGyroStabilizer stab(scfg);

    const float kInjected = 0.03f;  // rad/s, her iki eksene eklenen sabit bias
    for (int i = 0; i < 120; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        for (auto& s : imu_batch) {
            s.angular_velocity[0] += kInjected;
            s.angular_velocity[1] += kInjected;
        }
        stab.stabilize(*f, imu_batch);
    }

    const cv::Vec2f b = stab.estimated_bias();
    std::printf("   enjekte=%.3f  kestirilen bias=(%.3f, %.3f) rad/s\n", kInjected,
                b[0], b[1]);
    // OF gerçek hareketi gördüğü için bias, enjekte edilen ofsete yakınsamalı.
    CHECK(std::abs(b[0] - kInjected) < 0.01f);
    CHECK(std::abs(b[1] - kInjected) < 0.01f);
}

int main() {
    test_stabilization_reduces_motion();
    test_gyro_bias_converges();
    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
