// Pipeline benchmark: farklı sahne parametrelerinde uçtan uca performans ölçümü.
//
// Her senaryoda 180 kare (3sn @60fps) çalışır ve şunları ölçer:
//   - Detection: recall, precision, RMS konum hatası
//   - Tracking:  kilit sürekliliği, ID atlaması, coast süresi, konum hatası
//   - Timing:    toplam süre, ortalama kare süresi
//
// Rapor: her senaryo için tek satır + en kötü senaryoların özeti.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "dtrack/common/cue.hpp"
#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"

using namespace dtrack;
using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string name;
    // Detection
    double det_recall{0};        // yer-gerçeğine yakın tespit oranı
    double det_precision{0};     // tespitlerin ne kadarı hedefe yakın
    double det_rms_err{0};       // konum hatası RMS (px)
    double det_fp_per_frame{0};  // kare başına yanlış pozitif
    // Tracking
    double trk_continuity{0};    // kilitli kare / toplam
    double trk_rms_err{0};       // track konum hatası RMS (px)
    int trk_id_count{0};         // benzersiz track ID sayısı
    double trk_coast_frac{0};    // coast karelerinin oranı
    int trk_lost_frames{0};      // hiç track olmayan kare sayısı
    // Timing
    double total_s{0};
    double ms_per_frame{0};
    int total_frames{0};
};

static double dist(const io::Vec2& a, const cv::Point2f& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

static BenchResult run_bench(const io::SceneConfig& cfg, std::string name,
                             bool use_cue = true) {
    const auto t0 = common::now();
    io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0, false);
    io::SyntheticImuSource imu(cfg, 1000.0, t0);
    cam.open();
    imu.open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    stabilization::KltGyroStabilizer stab(scfg);
    detection::MogDetector det;
    detection::DetectorConfig dcfg;
    // Hedef boyutuna göre alan aralığını ayarla.
    float target_area = 3.14159f * cfg.target_sigma_px * cfg.target_sigma_px * 2.0f;
    dcfg.min_area = std::max(1.5, (double)target_area * 0.4);
    dcfg.max_area = std::max(50.0, (double)target_area * 6.0);
    dcfg.use_cue_recovery = use_cue;  // kapalı-döngü A/B karşılaştırması için
    det = detection::MogDetector(dcfg);

    tracking::KalmanTracker tracker;

    const int kFrames = 180;
    const int kWarmup = 30;

    BenchResult r;
    r.name = std::move(name);

    int n_det_hit = 0, n_fp = 0;
    double sum_det_err2 = 0;  // kare hata toplamı (RMS için)
    int n_trk_locked = 0, n_trk_lost = 0, n_trk_coast = 0;
    double sum_trk_err2 = 0;
    std::set<std::uint32_t> seen_ids;
    int eval_frames = 0;
    int n_target_present = 0;

    auto t_start = Clock::now();

    for (int i = 0; i < kFrames; ++i) {
        auto f = cam.next_frame();
        if (!f) continue;
        auto imu_batch = imu.generate_until((i + 1) / 60.0);
        auto sf = stab.stabilize(*f, imu_batch);
        auto dets = det.detect(sf);
        auto tracks = tracker.update(dets, (*f)->stamp);
        // Kapalı-döngü: bu karenin iz tahminini bir sonraki kare tespitine geri besle.
        if (use_cue) det.set_cue(common::make_cue(tracks));

        if (i < kWarmup) continue;
        ++eval_frames;
        const io::Vec2 gt = cam.last_target_observed_px();

        // Hedef kare sınırları içinde mi?
        bool target_visible = (gt.x >= 0 && gt.y >= 0 &&
                               gt.x < cfg.width && gt.y < cfg.height);
        if (target_visible) ++n_target_present;

        // --- Detection metrikleri ---
        double best_det_d = 1e9;
        int best_det_idx = -1;
        for (size_t di = 0; di < dets.size(); ++di) {
            const double d = dist(gt, dets[di].centroid);
            if (d < best_det_d) { best_det_d = d; best_det_idx = (int)di; }
        }
        if (target_visible) {
            if (best_det_idx >= 0 && best_det_d < 8.0) {
                ++n_det_hit;
                sum_det_err2 += best_det_d * best_det_d;
                n_fp += (int)dets.size() - 1;
            } else {
                n_fp += (int)dets.size();
            }
        } else {
            n_fp += (int)dets.size();
        }

        // --- Tracking metrikleri ---
        double best_trk_d = 1e9;
        std::uint32_t best_trk_id = 0;
        bool any_confirmed = false, any_coasting = false;
        for (const auto& t : tracks) {
            const double d = dist(gt, t.position);
            if (d < best_trk_d) { best_trk_d = d; best_trk_id = t.id; }
            if (t.status == common::TrackStatus::Confirmed) any_confirmed = true;
            if (t.status == common::TrackStatus::Coasting) any_coasting = true;
        }
        if (tracks.empty()) {
            ++n_trk_lost;
        } else {
            if (best_trk_d < 10.0) {
                ++n_trk_locked;
                sum_trk_err2 += best_trk_d * best_trk_d;
                seen_ids.insert(best_trk_id);
            }
            if (any_coasting && !any_confirmed) ++n_trk_coast;
        }
    }

    auto t_end = Clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    r.total_s = elapsed.count();
    r.ms_per_frame = eval_frames > 0 ? (elapsed.count() * 1000.0 / eval_frames) : 0;
    r.total_frames = eval_frames;

    r.det_recall = n_target_present > 0 ? (double)n_det_hit / n_target_present : 1.0;
    r.det_precision = (n_det_hit + n_fp) > 0 ? (double)n_det_hit / (n_det_hit + n_fp) : 0;
    r.det_rms_err = n_det_hit > 0 ? std::sqrt(sum_det_err2 / n_det_hit) : 0;
    r.det_fp_per_frame = eval_frames > 0 ? (double)n_fp / eval_frames : 0;

    r.trk_continuity = eval_frames > 0 ? (double)n_trk_locked / eval_frames : 0;
    r.trk_rms_err = n_trk_locked > 0 ? std::sqrt(sum_trk_err2 / n_trk_locked) : 0;
    r.trk_id_count = (int)seen_ids.size();
    r.trk_coast_frac = eval_frames > 0 ? (double)n_trk_coast / eval_frames : 0;
    r.trk_lost_frames = n_trk_lost;

    return r;
}

int main() {
    std::printf("%-28s %6s %6s %7s %6s %6s %7s %4s %5s %6s %5s\n",
                "senaryo", "recall", "prec", "det_rms", "fp/k", "cont", "trk_rms",
                "ID#", "coast", "ms/k", "kayip");
    std::printf("%s\n", std::string(100, '-').c_str());

    io::SceneConfig base;
    std::vector<BenchResult> results;

    // --- Senaryo kümesi ---
    struct Vary {
        const char* label;
        float target_sigma_px;
        float target_vel_x;
        float maneuver_amp;
        float ego_amp_yaw;
        float target_intensity;
        float bg_texture;
    };

    std::vector<Vary> scenarios = {
        {"baseline",           1.1f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"tiny_2px",           0.7f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"large_6px",          1.8f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"medium_15px",        2.5f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"close_25px",         4.0f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"slow_target",        1.1f, 20.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"fast_target",        1.1f, 100.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"no_maneuver",        1.1f, 55.0f,  0.0f, 0.012f, 90.0f, 18.0f},
        {"heavy_maneuver",     1.1f, 55.0f, 80.0f, 0.012f, 90.0f, 18.0f},
        {"calm_ego",           1.1f, 55.0f, 40.0f, 0.006f, 90.0f, 18.0f},
        {"heavy_ego",          1.1f, 55.0f, 40.0f, 0.024f, 90.0f, 18.0f},
        {"dim_target",         1.1f, 55.0f, 40.0f, 0.012f, 50.0f, 18.0f},
        {"bright_target",      1.1f, 55.0f, 40.0f, 0.012f, 130.0f, 18.0f},
        {"noisy_bg",           1.1f, 55.0f, 40.0f, 0.012f, 90.0f, 30.0f},
        {"clean_bg",           1.1f, 55.0f, 40.0f, 0.012f, 90.0f, 10.0f},
        {"worst_dim_tiny_fast", 0.7f, 100.0f, 80.0f, 0.024f, 50.0f, 30.0f},
        {"best_bright_large_slow", 1.8f, 20.0f, 0.0f, 0.006f, 130.0f, 10.0f},
    };

    for (const auto& s : scenarios) {
        io::SceneConfig cfg = base;
        cfg.target_sigma_px = s.target_sigma_px;
        cfg.target_vel.x = s.target_vel_x;
        cfg.target_maneuver_amp = s.maneuver_amp;
        cfg.ego_amp_yaw = s.ego_amp_yaw;
        cfg.target_intensity = s.target_intensity;
        cfg.bg_texture = s.bg_texture;

        auto r = run_bench(cfg, s.label);
        results.push_back(r);

        std::printf("%-28s %5.2f %5.2f %6.2f %5.2f %5.2f %6.2f %3d %4.1f%% %5.2f %4d\n",
                    r.name.c_str(),
                    r.det_recall, r.det_precision, r.det_rms_err,
                    r.det_fp_per_frame,
                    r.trk_continuity, r.trk_rms_err, r.trk_id_count,
                    r.trk_coast_frac * 100,
                    r.ms_per_frame, r.trk_lost_frames);
    }

    // --- A/B: kapalı-döngü cued recovery KAPALI vs AÇIK (zor senaryolarda) ---
    // Faydanın dürüst kanıtı: aynı senaryoyu cue kapalı ve açık koşturup recall/
    // coast/continuity farkını göster. (Kolay senaryolarda zaten ~1.0 -> fark yok.)
    std::printf("\n--- A/B: cued recovery (geri besleme) KAPALI -> ACIK ---\n");
    std::printf("%-22s %12s %12s %12s\n", "senaryo", "recall", "coast%", "trk_rms");
    struct AbCase { const char* label; float sigma, vel, man, ego, inten, tex; };
    std::vector<AbCase> ab = {
        {"worst_dim_tiny_fast", 0.7f, 100.0f, 80.0f, 0.024f, 50.0f, 30.0f},
        {"noisy_bg",            1.1f, 55.0f, 40.0f, 0.012f, 90.0f, 30.0f},
        {"tiny_2px",            0.7f, 55.0f, 40.0f, 0.012f, 90.0f, 18.0f},
        {"dim_target",          1.1f, 55.0f, 40.0f, 0.012f, 50.0f, 18.0f},
    };
    for (const auto& c : ab) {
        io::SceneConfig cfg = base;
        cfg.target_sigma_px = c.sigma; cfg.target_vel.x = c.vel;
        cfg.target_maneuver_amp = c.man; cfg.ego_amp_yaw = c.ego;
        cfg.target_intensity = c.inten; cfg.bg_texture = c.tex;
        auto off = run_bench(cfg, c.label, /*use_cue=*/false);
        auto on  = run_bench(cfg, c.label, /*use_cue=*/true);
        std::printf("%-22s %5.2f -> %5.2f %5.1f -> %5.1f %5.2f -> %5.2f\n", c.label,
                    off.det_recall, on.det_recall,
                    off.trk_coast_frac * 100, on.trk_coast_frac * 100,
                    off.trk_rms_err, on.trk_rms_err);
    }

    // --- Özet: En zayıf metrikler ---
    std::printf("\n--- ZAYIF NOKTALAR (iyilestirme gereken) ---\n");
    double min_recall = 2.0, min_cont = 2.0;
    int worst_recall = -1, worst_cont = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].det_recall < min_recall) {
            min_recall = results[i].det_recall; worst_recall = (int)i;
        }
        if (results[i].trk_continuity < min_cont) {
            min_cont = results[i].trk_continuity; worst_cont = (int)i;
        }
    }
    if (worst_recall >= 0)
        std::printf("  En dusuk recall   : %s (%.2f)\n",
                    results[worst_recall].name.c_str(), results[worst_recall].det_recall);
    if (worst_cont >= 0)
        std::printf("  En dusuk continuity: %s (%.2f, %d kayip kare)\n",
                    results[worst_cont].name.c_str(), results[worst_cont].trk_continuity,
                    results[worst_cont].trk_lost_frames);

    // Global skor: recall * continuity (0-1 arası, 1=mükemmel)
    double worst_global = 2.0;
    int worst_idx = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        double g = results[i].det_recall * results[i].trk_continuity;
        if (g < worst_global) { worst_global = g; worst_idx = (int)i; }
    }
    if (worst_idx >= 0)
        std::printf("  En kotu global     : %s (%.3f)\n",
                    results[worst_idx].name.c_str(), worst_global);

    // Başarısızlık kontrolü: hiçbir senaryoda recall < 0.3 veya continuity < 0.3 olmamalı
    bool ok = true;
    for (const auto& r : results) {
        if (r.det_recall < 0.3) {
            std::printf("  FAIL: %s recall cok dusuk (%.2f)\n", r.name.c_str(), r.det_recall);
            ok = false;
        }
        if (r.trk_continuity < 0.3 && r.total_frames > 50) {
            std::printf("  FAIL: %s continuity cok dusuk (%.2f)\n", r.name.c_str(),
                        r.trk_continuity);
            ok = false;
        }
    }

    if (ok) {
        std::printf("\nTUM SENARYOLAR KABUL EDILEBILIR SINIRLARDA\n");
        return 0;
    }
    std::printf("\nIYILESTIRME GEREKIYOR\n");
    return 1;
}
