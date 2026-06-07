// test_cue_recovery: kapalı-döngü "cued" ROI kurtarma (track-before-detect-lite).
//
// Kurtarma, hedefe KİLİTLENDİKTEN sonra global tespit kaçırsa bile tracker'ın
// tahmini etrafındaki ROI'de düşük eşikle hedefi yakalayıp kesintisiz takibi
// sürdürmek içindir (bkz. common/cue.hpp, detection/mog_detector.cpp adım 7).
//
// Doğrulanan özellikler:
//   A) KURTARIR: global eşik düşük-SNR hedefi kaçıracak kadar yüksekken, geçerli
//      bir cue verilirse detektör tahmin ROI'sinde hedefi bulur (from_cue=true).
//      Cue YOKKEN aynı detektör hiçbir şey bulamaz (kurtarma valid cue'ya bağlı).
//   B) SAHTE ÜRETMEZ: cue boş bir arka plan bölgesini gösterirse (hedef yok),
//      lokal SNR tabanı sayesinde detektör sahte tespit üretmez.
//   C) ENTEGRASYON: zorlu sentetik sahnede uçtan uca, kapalı-döngü recall'ı
//      düşürmez, artırır (≥ açık-döngü ve > 0.95).
//
// Kontrollü görüntüler (A,B) deterministiktir (sabit RNG tohumu) -> OpenCV gerekir
// ama dış veriye bağlı değildir.

#include <cmath>
#include <cstdio>

#include <opencv2/core.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"

using namespace dtrack;

static int g_fail = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__); \
            ++g_fail;                                                \
        }                                                            \
    } while (0)

// Düz arka plan (40) + gürültü + (opsiyonel) alt-piksel Gaussian hedef.
static cv::Mat make_image(int W, int H, bool with_target, cv::Point2f tpos, float amp,
                          float sigma, unsigned seed) {
    cv::Mat f(H, W, CV_32FC1, cv::Scalar(40.0f));
    cv::RNG rng(seed);
    cv::Mat noise(H, W, CV_32FC1);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, 1.0);  // düşük gürültü -> deterministik eşik
    f += noise;
    if (with_target) {
        const int r = std::max(2, static_cast<int>(std::ceil(sigma * 3)));
        const float s2 = 2.0f * sigma * sigma;
        for (int dy = -r; dy <= r; ++dy) {
            const int py = static_cast<int>(std::lround(tpos.y)) + dy;
            if (py < 0 || py >= H) continue;
            float* row = f.ptr<float>(py);
            for (int dx = -r; dx <= r; ++dx) {
                const int px = static_cast<int>(std::lround(tpos.x)) + dx;
                if (px < 0 || px >= W) continue;
                const float fx = px - tpos.x, fy = py - tpos.y;
                row[px] += amp * std::exp(-(fx * fx + fy * fy) / s2);
            }
        }
    }
    cv::Mat img8;
    f.convertTo(img8, CV_8UC1);
    return img8;
}

static common::StabilizedFrame make_sf(const cv::Mat& img) {
    auto frame = std::make_shared<common::Frame>();
    frame->index = 0;
    frame->stamp = common::now();
    frame->modality = common::Modality::Visible;
    frame->image = img;
    common::StabilizedFrame sf;
    sf.frame = common::FramePtr(frame);
    sf.ego.homography = cv::Matx33f::eye();
    sf.ego.valid = true;
    return sf;
}

// Global tespiti İMKANSIZ kılan config (eşik tavanı) ama kurtarma açık.
static detection::DetectorConfig blind_global_cfg() {
    detection::DetectorConfig c;
    c.use_dog = false;          // sadeleştir: top-hat yolu
    c.use_lcm = false;
    c.thresh_k = 100.0;         // global eşik imkansız yüksek
    c.thresh_min_abs = 250.0;
    c.min_peak_tophat = 250.0;  // global tepe tabanı imkansız -> global hiçbir şey bulmaz
    c.use_cue_recovery = true;
    c.cue_thresh_k = 3.0f;
    c.cue_min_peak_tophat = 8.0f;
    c.cue_meas_std = 3.0f;
    c.cue_min_radius = 6;
    c.cue_max_radius = 40;
    return c;
}

static int count_from_cue(const std::vector<common::Detection>& d) {
    int n = 0;
    for (const auto& x : d) if (x.from_cue) ++n;
    return n;
}

int main() {
    std::printf("test_cue_recovery\n");
    const int W = 200, H = 200;
    const cv::Point2f tpos(100.0f, 100.0f);

    // --- A) Düşük-SNR hedefi global kaçırır; cue ile kurtarılır. ---
    {
        // amp=30: global tabanı (250) altında -> global kör; cue tabanı (8) üstünde.
        cv::Mat img = make_image(W, H, /*with_target=*/true, tpos, 30.0f, 1.2f, 7u);

        // A1: cue YOK -> kurtarma tetiklenmez -> hiçbir tespit yok.
        detection::MogDetector det_nocue(blind_global_cfg());
        auto sf1 = make_sf(img);
        auto d_nocue = det_nocue.detect(sf1);
        CHECK(d_nocue.empty());  // global kör + cue yok -> 0

        // A2: cue hedefi gösteriyor -> ROI kurtarma hedefi bulur.
        detection::MogDetector det_cue(blind_global_cfg());
        common::TargetCue cue;
        cue.valid = true;
        cue.predicted = tpos;
        cue.gate_radius = 15.0f;
        det_cue.set_cue(cue);
        auto sf2 = make_sf(img);
        auto d_cue = det_cue.detect(sf2);
        CHECK(count_from_cue(d_cue) >= 1);  // kurtarma üretti
        double best = 1e9;
        bool best_from_cue = false;
        for (const auto& x : d_cue) {
            const double e = std::hypot(x.centroid.x - tpos.x, x.centroid.y - tpos.y);
            if (e < best) { best = e; best_from_cue = x.from_cue; }
        }
        CHECK(best < 2.0);          // kurtarılan ölçüm hedefe alt-piksel yakın
        CHECK(best_from_cue);       // en yakın tespit kurtarma kaynaklı
        std::printf("   A) kurtarma hatasi=%.2f px, from_cue=%d\n", best, (int)best_from_cue);
    }

    // --- B) Cue boş bölgeyi gösterirse sahte tespit üretilmez. ---
    {
        cv::Mat img = make_image(W, H, /*with_target=*/false, tpos, 0.0f, 1.2f, 11u);
        detection::MogDetector det(blind_global_cfg());
        common::TargetCue cue;
        cue.valid = true;
        cue.predicted = cv::Point2f(100.0f, 100.0f);  // boş (hedef yok)
        cue.gate_radius = 20.0f;
        det.set_cue(cue);
        auto sf = make_sf(img);
        auto d = det.detect(sf);
        CHECK(count_from_cue(d) == 0);  // lokal SNR tabanı sahteyi engeller
        std::printf("   B) bos cue -> from_cue tespit sayisi=%d (beklenen 0)\n",
                    count_from_cue(d));
    }

    // --- C) Entegrasyon: zorlu sahnede kapalı-döngü recall'ı artırır. ---
    {
        io::SceneConfig cfg;
        cfg.target_sigma_px = 0.7f;   // küçük
        cfg.target_intensity = 50.0f; // sönük
        cfg.target_vel = {100.0f, -18.0f};
        cfg.target_maneuver_amp = 80.0f;
        cfg.ego_amp_yaw = 0.024f;     // ağır ego
        cfg.bg_texture = 30.0f;       // gürültülü arka plan

        auto run = [&](bool use_cue) {
            const auto t0 = common::now();
            io::SyntheticCameraSource cam(cfg, common::Modality::Visible, 60.0, t0, false);
            io::SyntheticImuSource imu(cfg, 1000.0, t0);
            cam.open();
            imu.open();
            stabilization::StabilizerConfig scfg;
            scfg.focal_px = cfg.focal_px;
            stabilization::KltGyroStabilizer stab(scfg);
            detection::DetectorConfig dcfg;
            float ta = 3.14159f * cfg.target_sigma_px * cfg.target_sigma_px * 2.0f;
            dcfg.min_area = std::max(1.5, (double)ta * 0.4);
            dcfg.max_area = std::max(50.0, (double)ta * 6.0);
            dcfg.use_cue_recovery = use_cue;
            detection::MogDetector det(dcfg);
            tracking::KalmanTracker tracker;

            int present = 0, hit = 0;
            for (int i = 0; i < 180; ++i) {
                auto f = cam.next_frame();
                if (!f) continue;
                auto imu_b = imu.generate_until((i + 1) / 60.0);
                auto sf = stab.stabilize(*f, imu_b);
                auto dets = det.detect(sf);
                auto tracks = tracker.update(dets, (*f)->stamp);
                if (use_cue) det.set_cue(common::make_cue(tracks));
                if (i < 30) continue;
                const io::Vec2 gt = cam.last_target_observed_px();
                if (gt.x < 0 || gt.y < 0 || gt.x >= cfg.width || gt.y >= cfg.height) continue;
                ++present;
                double best = 1e9;
                for (const auto& d : dets)
                    best = std::min<double>(best, std::hypot(d.centroid.x - gt.x, d.centroid.y - gt.y));
                if (best < 8.0) ++hit;
            }
            return present > 0 ? (double)hit / present : 0.0;
        };

        const double recall_off = run(false);
        const double recall_on = run(true);
        std::printf("   C) recall: kapali=%.3f -> acik=%.3f\n", recall_off, recall_on);
        CHECK(recall_on >= recall_off);  // kapalı-döngü recall'ı düşürmez
        CHECK(recall_on > 0.95);         // zorlu sahnede bile neredeyse kesintisiz
    }

    if (g_fail == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_fail);
    return 1;
}
