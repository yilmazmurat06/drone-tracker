// AlphaBetaTracker (YEDEK tracker) doğrulama testi (OpenCV gerekir).
//
// Varsayılan artık KalmanTracker (bkz. test_tracker.cpp). Bu test, yedek α-β
// filtresinin AYNI çıtayı tuttuğunu garanti eder; Kalman ile karşılaştırma temeli.
//
// Tam pipeline (kamera -> stabilize -> detect -> track) üzerinde:
//   A) Kilit sürekliliği  B) Konum hatası + kimlik kararlılığı  C) Boşluk köprüleme.

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/alpha_beta_tracker.hpp"

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
    std::printf("test_alpha_beta_lock_continuity (yedek tracker)\n");
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
    tracking::TrackerConfig tcfg;
    tcfg.alpha = 0.55;
    tcfg.beta  = 0.12;
    tcfg.gate_px = 12.0;
    tcfg.gate_expand_per_frame = 1.5;
    tcfg.confirm_hits = 3;
    tcfg.confirm_window = 8;
    tcfg.max_coast = 22;
    tcfg.tentative_max_miss = 2;
    tracking::AlphaBetaTracker tracker(tcfg);

    const int kFrames = 240;
    const int kWarmup = 50;
    const double kMatchRadius = 6.0;
    const int kGapStart = 185, kGapEnd = 189;  // 5 kare boşluk

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
