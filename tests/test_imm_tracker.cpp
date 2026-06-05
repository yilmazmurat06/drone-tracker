// IMM tracker doğrulama testi (OpenCV gerekir).
//
// ImmTracker (iki α-β modelli hafif IMM), α-β tek-filtreyle AYNI çıtayı tutturmalı:
//   A) Kilit sürekliliği (kesintisiz takip).
//   B) Konum doğruluğu + kimlik kararlılığı.
//   C) Boşluk köprüleme (tespit kesilse coast edip kilidi korur, sonra iyileşir).
//
// IMM'in asıl üstünlüğü (keskin/ani manevra) sentetik düz-sinüs sahnesinde
// görünmez; bu test IMM'in en azından α-β kadar SAĞLAM olduğunu garanti eder.

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/imm_tracker.hpp"

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
    std::printf("test_imm_tracker\n");
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
    tracking::ImmTracker tracker;

    const int kFrames = 240;
    const int kWarmup = 50;
    const double kMatchRadius = 6.0;
    const int kGapStart = 185, kGapEnd = 189;  // 5 kare tespit boşluğu

    int eval = 0, locked = 0, gap_locked = 0, gap_eval = 0;
    double sum_err = 0;
    std::set<std::uint32_t> matched_ids;
    for (int i = 0; i < kFrames; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        auto sf = stab.stabilize(*f, imu_batch);
        auto dets = det.detect(sf);

        const bool in_gap = (i >= kGapStart && i <= kGapEnd);
        if (in_gap) dets.clear();

        auto tracks = tracker.update(dets, (*f)->stamp);

        if (i < kWarmup) continue;
        ++eval;
        const io::Vec2 gt = cam.last_target_observed_px();

        double best = 1e9;
        std::uint32_t best_id = 0;
        for (const auto& t : tracks) {
            const double e = std::hypot(t.position.x - gt.x, t.position.y - gt.y);
            if (e < best) { best = e; best_id = t.id; }
        }
        if (best <= kMatchRadius) {
            ++locked;
            sum_err += best;
            matched_ids.insert(best_id);
            if (in_gap) ++gap_locked;
        }
        if (in_gap) ++gap_eval;
    }

    const double continuity = static_cast<double>(locked) / eval;
    const double mean_err = locked ? sum_err / locked : -1;
    std::printf("   kare=%d  kilit surekliligi=%.2f  konum hatasi=%.2f px  "
                "ID sayisi=%zu  bosluk-kilit=%d/%d\n",
                eval, continuity, mean_err, matched_ids.size(), gap_locked, gap_eval);

    CHECK(eval > 50);
    CHECK(continuity > 0.95);
    CHECK(mean_err < 2.5);
    CHECK(matched_ids.size() <= 2);
    CHECK(gap_locked > 0);

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
