// Track HIZ çıktısı doğrulama testi (OpenCV gerekir).
//
// Konum takibinin (position) doğru olması yetmez: track.velocity ve track.predicted
// alanları aşağı akışa (füzyon, uçuş kontrolcüsü, ileri-tahminli kapı) gider ve
// FİZİKSEL BİRİMDE (piksel/saniye) doğru olmalıdır.
//
// Bu test iki şeyi sınar:
//   A) Birim doğruluğu: raporlanan |velocity|, hedefin yer-gerçeği konumundan
//      türetilen gerçek hız ile aynı mertebede mi? (kaba bir [0.5, 2.0] kat bandı)
//   B) FPS değişmezliği: aynı sahne 30/60/120 fps'te koşturulduğunda raporlanan
//      hız ~aynı çıkmalı (fiziksel hız fps'ten bağımsızdır). Eğer kod hızı sabit
//      bir fps katsayısıyla ölçeklerse bu test bunu yakalar.
//
// NOT: Tracker imge koordinatında takip ettiği için referans hız, GÖZLEMLENEN
// (ego dahil) hedef konumunun sonlu-farkıdır.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"

using namespace dtrack;

static int g_failures = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

struct VelResult { double reported; double truth; int n; };

// Belirli bir fps'te sahneyi koşturup (raporlanan_hiz, gercek_hiz) ortalamasını döndürür.
static VelResult run_fps(double fps) {
    io::SceneConfig cfg;  // target_vel=(55,-18) + manevra -> ~58-170 px/s gözlemlenen
    const auto t0 = common::now();
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, fps, t0, false);
    io::SyntheticImuSource imu(cfg, 1000.0, t0);
    cam.open();
    imu.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    stabilization::KltGyroStabilizer stab(scfg);
    detection::MogDetector det;
    tracking::KalmanTracker trk;

    double sum_rep = 0, sum_truth = 0;
    int n = 0;
    io::Vec2 last_gt{};
    common::Timestamp last_stamp{};
    bool have_prev = false;

    const int kFrames = 220;
    const int kWarmup = 120;  // tespit + onay + hız kestirimi otursun
    for (int i = 0; i < kFrames; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto im = imu.generate_until((i + 1) / fps);
        auto sf = stab.stabilize(*f, im);
        auto dets = det.detect(sf);
        auto tracks = trk.update(dets, (*f)->stamp);

        const io::Vec2 gt = cam.last_target_observed_px();
        if (i >= kWarmup && have_prev && !tracks.empty()) {
            const double dt = common::millis_between(last_stamp, (*f)->stamp) / 1000.0;
            if (dt > 1e-4) {
                const double truth_spd =
                    std::hypot(gt.x - last_gt.x, gt.y - last_gt.y) / dt;
                const auto& t = tracks.front();  // tek hedefli sahne
                const double rep_spd = std::hypot(t.velocity.x, t.velocity.y);
                sum_rep += rep_spd;
                sum_truth += truth_spd;
                ++n;
            }
        }
        last_gt = gt;
        last_stamp = (*f)->stamp;
        have_prev = true;
    }
    return {n ? sum_rep / n : 0.0, n ? sum_truth / n : 0.0, n};
}

int main() {
    std::printf("test_track_velocity\n");

    VelResult r60 = run_fps(60.0);
    VelResult r30 = run_fps(30.0);
    VelResult r120 = run_fps(120.0);

    std::printf("   fps=60   raporlanan=%.1f px/s  gercek(GT)=%.1f px/s  (n=%d)\n",
                r60.reported, r60.truth, r60.n);
    std::printf("   fps=30   raporlanan=%.1f px/s  gercek(GT)=%.1f px/s  (n=%d)\n",
                r30.reported, r30.truth, r30.n);
    std::printf("   fps=120  raporlanan=%.1f px/s  gercek(GT)=%.1f px/s  (n=%d)\n",
                r120.reported, r120.truth, r120.n);

    CHECK(r60.n > 30 && r30.n > 30 && r120.n > 30);

    // A) Birim doğruluğu: raporlanan hız, gerçek hızla aynı mertebede.
    const double ratio60 = r60.truth > 1.0 ? r60.reported / r60.truth : 0.0;
    std::printf("   60fps oran (raporlanan/gercek) = %.2f  (ideal ~1.0)\n", ratio60);
    CHECK(ratio60 > 0.5 && ratio60 < 2.0);

    // B) FPS değişmezliği: 30/60/120'de raporlanan hız ~aynı olmalı.
    const double mx = std::max({r60.reported, r30.reported, r120.reported});
    const double mn = std::min({r60.reported, r30.reported, r120.reported});
    const double spread = mn > 1.0 ? mx / mn : 1e9;
    std::printf("   fps yayilimi (max/min raporlanan) = %.2f  (ideal ~1.0)\n", spread);
    CHECK(spread < 1.5);

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
