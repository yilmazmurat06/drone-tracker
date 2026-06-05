// Sentetik kaynak doğrulama testleri (OpenCV gerekir).
//
// 1) Kamera: render edilen karedeki en parlak leke, sahnenin GÖZLEMLENEN hedef
//    yer-gerçeğine alt-piksel yakınlıkta mı? (Render doğru hedefi koyuyor mu?)
// 2) IMU: üretilen gyro, sahnenin gerçek açısal hızına (bias=0 başlangıçta)
//    gürültü toleransı içinde mi? Zaman damgaları monoton ve doğru sayıda mı?

#include <cmath>
#include <cstdio>

#include <opencv2/core.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"

using namespace dtrack;

static int g_failures = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

static void test_camera_target_position() {
    std::printf("test_camera_target_position\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    // realtime=false -> sanal saat, deterministik (t = index/fps).
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0,
                                  /*realtime=*/false);
    cam.open();

    for (int i = 0; i < 30; ++i) {
        auto f = cam.next_frame();
        CHECK(f.has_value());
        if (!f) continue;

        // En parlak piksel.
        double maxv = 0;
        cv::Point loc;
        cv::minMaxLoc((*f)->image, nullptr, &maxv, nullptr, &loc);

        // Sahnenin söylediği gözlemlenen hedef konumu (yer-gerçeği).
        const io::Vec2 gt = cam.last_target_observed_px();

        const double err =
            std::hypot(loc.x - gt.x, loc.y - gt.y);
        // En parlak piksel, alt-piksel merkeze 1.5 px'ten yakın olmalı.
        CHECK(err < 1.5);
        // Hedef arka plandan belirgin ayrılmalı.
        CHECK(maxv > 100.0);
    }
}

static void test_imu_matches_scene() {
    std::printf("test_imu_matches_scene\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    io::SyntheticImuSource imu(cfg, /*imu_rate_hz=*/1000.0, t0);
    imu.open();
    io::SceneModel scene(cfg);

    // 0.1 sn = ~100 örnek üret (deterministik).
    auto samples = imu.generate_until(0.1);
    CHECK(samples.size() >= 95 && samples.size() <= 105);

    // İlk örneklerde bias ~0 -> gyro, gerçek açısal hıza yakın olmalı (gürültü hariç).
    int checked = 0;
    common::Timestamp prev{};
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        if (i > 0) CHECK(s.stamp > prev);  // monoton zaman
        prev = s.stamp;

        if (i < 5) {
            const double t = i / 1000.0;
            const io::Vec2 truth = scene.ego_rate(t);
            // Gürültü std ~0.004, bias küçük; 0.05 rad/s tolerans bol.
            CHECK(std::abs(s.angular_velocity[0] - truth.x) < 0.05);
            CHECK(std::abs(s.angular_velocity[1] - truth.y) < 0.05);
            ++checked;
        }
    }
    CHECK(checked == 5);
}

int main() {
    test_camera_target_position();
    test_imu_matches_scene();
    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
