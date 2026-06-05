// drone-tracker uygulama girişi (çift kamera pipeline'ı + geç füzyon + görselleştirme).
//
// Pipeline:
//   [cam_vis] -> [stab] -> [detect+score] -> [track] ─┐
//   [cam_thm] -> [stab] -> [detect+score] -> [track] ─┤
//                                                       ├─ [fusion] -> [viz]
//
// İki kamera (görünür + termal) bağımsız track üretir; füzyon bunları
// track seviyesinde birleştirir. Viz kutusu: mor=hemfikir (iki kamera),
// yeşil=görünür-only, mavi=termal-only.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/detect_stage.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/detection/stability_discriminator.hpp"
#include "dtrack/fusion/simple_track_fusion.hpp"
#include "dtrack/io/camera_stage.hpp"
#include "dtrack/io/synthetic_camera_source.hpp"
#include "dtrack/io/synthetic_imu_source.hpp"
#include "dtrack/pipeline/pipeline.hpp"
#include "dtrack/pipeline/stage.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/stabilization/stabilize_stage.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"
#include "dtrack/tracking/track_stage.hpp"

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

class VizWriter {
public:
    VizWriter(std::string out_dir, double fps)
        : out_dir_(std::move(out_dir)), fps_(fps) {}

    void write(const dtrack::common::FramePtr& frame,
               const std::vector<dtrack::common::Track>& tracks) {
        if (!frame || frame->image.empty()) return;
        const auto n = count_++;

        cv::Mat vis;
        cv::cvtColor(frame->image, vis, cv::COLOR_GRAY2BGR);

        if (!writer_.isOpened()) {
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer_.open(out_dir_ + "/track_fused.mp4", fourcc, fps_, vis.size(), true);
        }

        bool any_lock = false;
        for (const auto& t : tracks) {
            using dtrack::common::TrackStatus;
            cv::Scalar color;
            if (t.confidence > 0.7f)
                color = cv::Scalar(255, 0, 255);   // mor = hemfikir (yüksek güven)
            else if (t.status == TrackStatus::Confirmed)
                color = cv::Scalar(0, 255, 0);      // yeşil = görünür onaylı
            else
                color = cv::Scalar(0, 215, 255);    // sarı = coast

            if (t.status == TrackStatus::Confirmed)
                any_lock = true;

            const float half = std::max(9.0f, t.scale * 4.0f);
            const cv::Point2f c(t.position.x, t.position.y);
            cv::rectangle(vis, cv::Point2f(c.x - half, c.y - half),
                          cv::Point2f(c.x + half, c.y + half), color, 1, cv::LINE_AA);

            char label[64];
            std::snprintf(label, sizeof(label), "ID%u %.0f%%", t.id, t.confidence * 100);
            cv::putText(vis, label, cv::Point2f(c.x - half, c.y - half - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);

            cv::circle(vis, cv::Point2f(t.predicted.x, t.predicted.y), 2,
                       cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

            auto& tr = trails_[t.id];
            tr.push_back(c);
            if (tr.size() > 40) tr.pop_front();
            for (size_t i = 1; i < tr.size(); ++i)
                cv::line(vis, tr[i - 1], tr[i], color, 1, cv::LINE_AA);
        }
        if (any_lock) locked_++;

        if (writer_.isOpened()) writer_.write(vis);
        if (saved_ < 6 && any_lock) {
            char path[256];
            std::snprintf(path, sizeof(path), "%s/fused_%03llu.png", out_dir_.c_str(),
                          static_cast<unsigned long long>(n));
            cv::imwrite(path, vis);
            ++saved_;
        }
    }

    void finalize() { if (writer_.isOpened()) writer_.release(); }
    std::uint64_t count() const { return count_; }
    std::uint64_t locked() const { return locked_; }

private:
    std::string out_dir_;
    double fps_;
    cv::VideoWriter writer_;
    std::map<std::uint32_t, std::deque<cv::Point2f>> trails_;
    std::uint64_t count_{0};
    std::uint64_t locked_{0};
    int saved_{0};
};
}  // namespace

int main() {
    using namespace dtrack;
    std::signal(SIGINT, on_signal);

    const std::string out_dir = "samples";
    std::system("mkdir -p samples");

    const auto t0 = common::now();
    const double fps = 60.0;
    io::SceneConfig cfg;

    // ============================================================
    // GÖRÜNÜR PIPELINE
    // ============================================================
    auto cam_vis = std::make_shared<io::SyntheticCameraSource>(
        cfg, common::Modality::Visible, fps, t0, true);
    auto imu_vis = std::make_shared<io::SyntheticImuSource>(cfg, 1000.0, t0);
    cam_vis->open();
    imu_vis->open();

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = cfg.focal_px;
    auto stab_vis = std::make_shared<stabilization::KltGyroStabilizer>(scfg);
    auto det_vis = std::make_shared<detection::MogDetector>();
    auto disc_vis = std::make_shared<detection::StabilityDiscriminator>();
    auto trk_vis = std::make_shared<tracking::AlphaBetaTracker>();

    auto vfq = std::make_shared<common::SpscRingBuffer<common::FramePtr>>(8);
    auto vsq = std::make_shared<common::SpscRingBuffer<common::StabilizedFrame>>(8);
    auto vdq = std::make_shared<common::SpscRingBuffer<common::FrameDetections>>(8);
    auto vtq = std::make_shared<common::SpscRingBuffer<common::FrameTracks>>(8);

    auto cam_s = std::make_shared<io::CameraStage>(cam_vis);
    auto stab_s = std::make_shared<stabilization::StabilizeStage>(stab_vis, imu_vis);
    auto det_s = std::make_shared<detection::DetectStage>(det_vis, disc_vis);
    auto trk_s = std::make_shared<tracking::TrackStage>(trk_vis);

    cam_s->connect(nullptr, vfq);
    stab_s->connect(vfq, vsq);
    det_s->connect(vsq, vdq);
    trk_s->connect(vdq, vtq);

    pipeline::Pipeline vis_pipeline;
    vis_pipeline.add(cam_s);
    vis_pipeline.add(stab_s);
    vis_pipeline.add(det_s);
    vis_pipeline.add(trk_s);

    // ============================================================
    // TERMAL PIPELINE
    // ============================================================
    auto cam_thm = std::make_shared<io::SyntheticCameraSource>(
        cfg, common::Modality::Thermal, fps, t0, true);
    auto imu_thm = std::make_shared<io::SyntheticImuSource>(cfg, 1000.0, t0);
    cam_thm->open();
    imu_thm->open();

    auto stab_thm = std::make_shared<stabilization::KltGyroStabilizer>(scfg);
    auto det_thm = std::make_shared<detection::MogDetector>();
    auto disc_thm = std::make_shared<detection::StabilityDiscriminator>();
    auto trk_thm = std::make_shared<tracking::AlphaBetaTracker>();

    auto tfq = std::make_shared<common::SpscRingBuffer<common::FramePtr>>(8);
    auto tsq = std::make_shared<common::SpscRingBuffer<common::StabilizedFrame>>(8);
    auto tdq = std::make_shared<common::SpscRingBuffer<common::FrameDetections>>(8);
    auto ttq = std::make_shared<common::SpscRingBuffer<common::FrameTracks>>(8);

    auto cam_ts = std::make_shared<io::CameraStage>(cam_thm);
    auto stab_ts = std::make_shared<stabilization::StabilizeStage>(stab_thm, imu_thm);
    auto det_ts = std::make_shared<detection::DetectStage>(det_thm, disc_thm);
    auto trk_ts = std::make_shared<tracking::TrackStage>(trk_thm);

    cam_ts->connect(nullptr, tfq);
    stab_ts->connect(tfq, tsq);
    det_ts->connect(tsq, tdq);
    trk_ts->connect(tdq, ttq);

    pipeline::Pipeline thm_pipeline;
    thm_pipeline.add(cam_ts);
    thm_pipeline.add(stab_ts);
    thm_pipeline.add(det_ts);
    thm_pipeline.add(trk_ts);

    // ============================================================
    // FÜZYON + VIZ
    // ============================================================
    fusion::SimpleTrackFusion fuser;
    VizWriter viz(out_dir, fps);

    std::printf("vis(cam->stab->det->trk) + thm(cam->stab->det->trk) -> fusion -> viz\n");
    std::printf("~%.0f fps, ~5 sn. Renk: mor=hemfikir, yesil=gorunur, sari=coast\n", fps);

    vis_pipeline.start();
    thm_pipeline.start();

    const auto run_until = t0 + std::chrono::seconds(5);
    uint64_t fused_count = 0, locked_count = 0;
    while (!g_stop && common::now() < run_until) {
        // Her iki track kuyruğundan da oku.
        bool had_any = false;
        common::FramePtr last_frame;

        auto vt = vtq->pop();
        auto tt = ttq->pop();

        std::vector<common::Track> vis_tracks, thm_tracks;
        if (vt) {
            vis_tracks = std::move(vt->tracks);
            last_frame = vt->frame;
        }
        if (tt) {
            thm_tracks = std::move(tt->tracks);
            if (!last_frame) last_frame = tt->frame;
        }

        if (vt || tt) {
            had_any = true;
            auto fused = fuser.fuse(vis_tracks, thm_tracks);
            viz.write(last_frame, fused);
            ++fused_count;
            if (!fused.empty()) ++locked_count;
        }

        if (!had_any) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    vis_pipeline.stop();
    thm_pipeline.stop();
    viz.finalize();

    std::printf("\n--- OZET ---\n");
    std::printf("Fusion kare    : %llu\n", static_cast<unsigned long long>(fused_count));
    std::printf("Kilitli kare   : %llu / %llu\n",
                static_cast<unsigned long long>(locked_count),
                static_cast<unsigned long long>(fused_count));
    std::printf("Video          : %s/track_fused.mp4  (+ fused_*.png)\n", out_dir.c_str());
    std::printf("Durduruldu.\n");
    return 0;
}
