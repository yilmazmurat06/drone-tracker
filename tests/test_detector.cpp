// Detektör doğrulama testi (OpenCV gerekir).
//
// Sentetik sahnede hedefin GÖZLEMLENEN yer-gerçeğini bildiğimiz için tespit
// kalitesini sayısal ölçeriz: recall (yakalama oranı), konum hatası, yanlış
// pozitif/kare. Tam pipeline: kamera -> stabilize -> detect.

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/detection/mog_detector.hpp"
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
    std::printf("test_detector_recall_precision\n");
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

    const int kFrames = 220;
    const int kWarmup = 130;         // MOG2 arka planı öğrensin
    const double kMatchRadius = 5.0;

    int eval_frames = 0, hits = 0, false_pos = 0;
    double sum_err = 0;
    for (int i = 0; i < kFrames; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        auto sf = stab.stabilize(*f, imu_batch);
        auto dets = det.detect(sf);

        if (i < kWarmup) continue;
        ++eval_frames;

        const io::Vec2 gt = cam.last_target_observed_px();
        double best = std::numeric_limits<double>::max();
        for (const auto& d : dets) {
            const double e = std::hypot(d.centroid.x - gt.x, d.centroid.y - gt.y);
            best = std::min(best, e);
            if (e > kMatchRadius) ++false_pos;  // hedefle eşleşmeyen = yanlış pozitif
        }
        if (best <= kMatchRadius) {
            ++hits;
            sum_err += best;
        }
    }

    const double recall = static_cast<double>(hits) / eval_frames;
    const double mean_err = hits ? sum_err / hits : -1;
    const double fp_per_frame = static_cast<double>(false_pos) / eval_frames;
    std::printf("   degerlendirilen kare=%d  recall=%.2f  konum hatasi=%.2f px  "
                "yanlis pozitif/kare=%.2f  ref sifirlama=%d\n",
                eval_frames, recall, mean_err, fp_per_frame, det.reference_resets());

    CHECK(eval_frames > 50);
    CHECK(recall > 0.85);          // kesintisiz tespit hedefi
    CHECK(mean_err < 2.5);         // alt-piksel yakınlık
    CHECK(fp_per_frame < 3.0);     // makul yanlış pozitif

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
