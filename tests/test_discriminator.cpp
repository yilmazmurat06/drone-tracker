// Discriminator doğrulama testi.
//
// Üç şeyi ölçer:
//   A) Hedef tespitinin drone_score'u anlamlı derecede yüksek (>0.6).
//   B) Hedef skoru, yanlış pozitiflerin skorundan yüksek (ayırt edicilik).
//   C) Hedef skoru zamansal kararlılıkla artar (son skor > ortalama).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/detection/stability_discriminator.hpp"
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

int main() {
    std::printf("test_discriminator\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0, false);
    io::SyntheticImuSource imu(cfg, 1000.0, t0);
    cam.open();
    imu.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    stabilization::KltGyroStabilizer stab(scfg);
    detection::MogDetector det;
    detection::StabilityDiscriminator disc;

    const int kFrames = 150;
    const int kWarmup = 20;
    double sum_target = 0, sum_fp_near = 0, sum_fp_far = 0;
    int n_target = 0, n_fp_near = 0, n_fp_far = 0;
    double last_target_score = 0;

    for (int i = 0; i < kFrames; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        auto sf = stab.stabilize(*f, imu_batch);
        auto dets = det.detect(sf);
        disc.score(dets);

        if (i < kWarmup) continue;
        const io::Vec2 gt = cam.last_target_observed_px();

        double best_d = 1e9;
        int best_idx = -1;
        for (size_t di = 0; di < dets.size(); ++di) {
            const double e = std::hypot(dets[di].centroid.x - gt.x,
                                        dets[di].centroid.y - gt.y);
            if (e < best_d) { best_d = e; best_idx = (int)di; }
        }

        if (best_idx >= 0 && best_d < 4.0) {
            sum_target += dets[best_idx].drone_score;
            ++n_target;
            last_target_score = dets[best_idx].drone_score;
            for (size_t di = 0; di < dets.size(); ++di) {
                if ((int)di == best_idx) continue;
                const double ed = std::hypot(dets[di].centroid.x - gt.x,
                                             dets[di].centroid.y - gt.y);
                if (ed < 30.0) {
                    sum_fp_near += dets[di].drone_score;
                    ++n_fp_near;
                } else {
                    sum_fp_far += dets[di].drone_score;
                    ++n_fp_far;
                }
            }
        } else if (!dets.empty()) {
            for (const auto& d : dets) {
                const double ed = std::hypot(d.centroid.x - gt.x,
                                             d.centroid.y - gt.y);
                if (ed < 30.0) {
                    sum_fp_near += d.drone_score;
                    ++n_fp_near;
                } else {
                    sum_fp_far += d.drone_score;
                    ++n_fp_far;
                }
            }
        }
    }

    const double avg_target = n_target > 0 ? sum_target / n_target : 0;
    const double avg_fp_near = n_fp_near > 0 ? sum_fp_near / n_fp_near : 0;
    const double avg_fp_far = n_fp_far > 0 ? sum_fp_far / n_fp_far : 0;

    std::printf("   kare=%d  hedef=%.3f (n=%d)  fp_yakin=%.3f (n=%d)  "
                "fp_uzak=%.3f (n=%d)  son_hedef=%.3f\n",
                kFrames - kWarmup, avg_target, n_target,
                avg_fp_near, n_fp_near, avg_fp_far, n_fp_far, last_target_score);

    CHECK(n_target > 30);                        // çoğu karede hedef bulunmalı
    CHECK(avg_target > 0.65);                     // hedef skoru yüksek
    CHECK(last_target_score > 0.7);              // zamansal kararlılık oturmuş
    CHECK(avg_target > avg_fp_near + 0.1);       // hedef yakın FP'den anlamlı yüksek
    if (n_fp_far > 3) CHECK(avg_target > avg_fp_far * 1.2);  // hedef uzak FP'den %20 yüksek

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
