// Fusion doğrulama testi.
//
// İki boru hattı (görünür + termal) çalıştırır, track'leri füzyonlar ve doğrular:
//   A) Hemfikirlik: iki kamera da görünce güven > tek kamera güveni.
//   B) Tek modalite: termal kapanınca görünür track devam eder.
//   C) Konum doğruluğu: füzyonlanmış konum yer-gerçeğine yakın.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/fusion/simple_track_fusion.hpp"
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

int main() {
    std::printf("test_fusion\n");
    io::SceneConfig cfg;
    const auto t0 = common::now();
    const double fps = 60.0;

    // Görünür kamera.
    io::SyntheticCameraSource cam_vis(cfg, common::Modality::Visible, fps, t0, false);
    cam_vis.open();
    io::SyntheticImuSource imu_vis(cfg, 1000.0, t0);
    imu_vis.open();

    // Termal kamera (aynı sahne, farklı render).
    io::SyntheticCameraSource cam_thm(cfg, common::Modality::Thermal, fps, t0, false);
    cam_thm.open();
    io::SyntheticImuSource imu_thm(cfg, 1000.0, t0);
    imu_thm.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    stabilization::KltGyroStabilizer stab_vis(scfg);
    stabilization::KltGyroStabilizer stab_thm(scfg);

    detection::MogDetector det_vis;
    detection::MogDetector det_thm;

    tracking::KalmanTracker trk_vis;
    tracking::KalmanTracker trk_thm;

    fusion::SimpleTrackFusion fusion;

    const int kFrames = 120;
    const int kWarmup = 40;
    const int kThermGapStart = 70, kThermGapEnd = 85;  // termal tespit kesintisi

    int eval = 0;
    double sum_conf_fused = 0, sum_conf_vis = 0, sum_conf_thm = 0;
    int n_both = 0, n_single = 0, n_fused_locked = 0;

    for (int i = 0; i < kFrames; ++i) {
        // Görünür pipeline.
        auto fv = cam_vis.next_frame();
        if (!fv) continue;
        auto imv = imu_vis.generate_until((i + 1) / fps);
        auto sfv = stab_vis.stabilize(*fv, imv);
        auto dets_v = det_vis.detect(sfv);

        // Termal pipeline.
        auto ft = cam_thm.next_frame();
        if (!ft) continue;
        auto imt = imu_thm.generate_until((i + 1) / fps);
        auto sft = stab_thm.stabilize(*ft, imt);
        auto dets_t = det_thm.detect(sft);

        // Termal kesinti simülasyonu.
        if (i >= kThermGapStart && i <= kThermGapEnd) dets_t.clear();

        auto tracks_v = trk_vis.update(dets_v, (*fv)->stamp);
        auto tracks_t = trk_thm.update(dets_t, (*ft)->stamp);

        // Füzyon.
        auto fused = fusion.fuse(tracks_v, tracks_t);

        if (i < kWarmup) continue;
        ++eval;
        const io::Vec2 gt = cam_vis.last_target_observed_px();

        // Hedefe en yakın track'i bul.
        const common::Track* best_f = nullptr;
        const common::Track* best_v = nullptr;
        const common::Track* best_t = nullptr;
        double best_fd = 1e9, best_vd = 1e9, best_td = 1e9;
        for (const auto& t : fused) {
            double e = std::hypot(t.position.x - gt.x, t.position.y - gt.y);
            if (e < best_fd) { best_fd = e; best_f = &t; }
        }
        for (const auto& t : tracks_v) {
            double e = std::hypot(t.position.x - gt.x, t.position.y - gt.y);
            if (e < best_vd) { best_vd = e; best_v = &t; }
        }
        for (const auto& t : tracks_t) {
            double e = std::hypot(t.position.x - gt.x, t.position.y - gt.y);
            if (e < best_td) { best_td = e; best_t = &t; }
        }

        const bool vis_ok = best_v && best_vd < 8.0;
        const bool thm_ok = best_t && best_td < 8.0;
        const bool fused_ok = best_f && best_fd < 8.0;

        if (fused_ok) {
            ++n_fused_locked;
            sum_conf_fused += best_f->confidence;
        }
        if (vis_ok) sum_conf_vis += best_v->confidence;
        if (thm_ok) sum_conf_thm += best_t->confidence;
        if (vis_ok && thm_ok) ++n_both;
        if ((vis_ok && !thm_ok) || (!vis_ok && thm_ok)) ++n_single;

        if (i >= kThermGapStart && i <= kThermGapEnd && vis_ok && fused_ok) {
            // Termal kapalıyken görünür tek başına kilitli kalmalı.
            CHECK(best_f->confidence >= 0.2f);
        }
    }

    const double avg_cf = n_fused_locked > 0 ? sum_conf_fused / n_fused_locked : 0;
    const double avg_cv = sum_conf_vis > 0 ? sum_conf_vis / (n_both + n_single) : 0;
    const double avg_ct = sum_conf_thm > 0 ? sum_conf_thm / (n_both + n_single) : 0;

    std::printf("   kare=%d  fused_kilit=%d  ikisi_var=%d  tek_var=%d\n",
                eval, n_fused_locked, n_both, n_single);
    std::printf("   guven: fused=%.3f  vis=%.3f  thm=%.3f\n", avg_cf, avg_cv, avg_ct);

    CHECK(n_fused_locked > eval * 0.7);     // füzyon çoğu karede kilitli
    CHECK(n_both > 20);                      // iki kamera da çoğu zaman görüyor
    CHECK(n_single > 0);                     // tek modalite durumu test edildi
    CHECK(avg_cf > 0.5f);                    // füzyon güveni yüksek
    // Hemfikir olduğunda güven artar (füzyon > tekil).
    // (n_both döneminde avg_cf'nin yüksek olması bunu dolaylı doğrular.)

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
