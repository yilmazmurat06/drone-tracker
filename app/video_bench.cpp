// Video benchmark: MP4 dosyasını pipeline'dan geçirir, CANLI görselleştirme yapar.
//
// Kullanım: ./video_bench <video.mp4> [cikti_dizini]
//
// IMU olmadığı için stabilizasyon saf optical flow ile çalışır.
// Canlı olarak gösterir: track kutusu, tespit noktaları, sağ-üst panel (metrikler).
// Çıktı: <cikti_dizini>/<video_adi>_tracked.mp4 + PNG örnekleri.
// 'q' veya ESC ile erken çıkış.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/io/video_camera_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"

using namespace dtrack;
using Clock = std::chrono::high_resolution_clock;

static void draw_metrics_panel(cv::Mat& vis, int frame_n, int total, int n_det,
                                int n_trk, int n_conf, int n_coast,
                                const common::Track* best,
                                double ms, double fps) {
    const int panel_w = 310;
    const int panel_h = 220;
    const int x0 = vis.cols - panel_w - 10;
    const int y0 = 10;

    // Yarı saydam arka plan.
    cv::Mat roi = vis(cv::Rect(x0, y0, panel_w, panel_h));
    cv::Mat overlay;
    roi.copyTo(overlay);
    cv::addWeighted(overlay, 0.25, cv::Mat(panel_h, panel_w, CV_8UC3, cv::Scalar(0, 0, 0)),
                    0.75, 0, roi);

    auto put = [&](int row, const char* label, const char* fmt, ...) {
        char buf[128];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        cv::putText(vis, buf, cv::Point(x0 + 8, y0 + 20 + row * 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1,
                    cv::LINE_AA);
    };

    int r = 0;
    put(r++, "", "KARE  %d / %d", frame_n, total);
    put(r++, "", "Tespit  %d  |  Track %d (C:%d c:%d)",
        n_det, n_trk, n_conf, n_coast);
    put(r++, "", "Sure  %.1f ms  |  FPS %.0f", ms, fps);

    if (best) {
        const char* st = best->status == common::TrackStatus::Confirmed ? "KILITLI" :
                         best->status == common::TrackStatus::Coasting ? "COAST" : "...";
        const cv::Scalar sc = best->status == common::TrackStatus::Confirmed
                                  ? cv::Scalar(0, 255, 0)
                                  : cv::Scalar(0, 215, 255);
        put(r++, "", "Durum  %s  |  Guven %.0f%%", st, best->confidence * 100);
        put(r++, "", "Poz  (%.1f, %.1f)", best->position.x, best->position.y);
        put(r++, "", "Hiz  (%.0f, %.0f) px/s", best->velocity.x, best->velocity.y);
        put(r++, "", "Olcek %.1f  |  Hit %d", best->scale, best->hits);
        // Mini durum çubuğu.
        const int bar_w = 200;
        const int bar_h = 6;
        const int bx = x0 + 8, by = y0 + 8 + r * 20 + 10;
        cv::rectangle(vis, cv::Rect(bx, by, bar_w, bar_h), cv::Scalar(60, 60, 60), -1);
        cv::rectangle(vis, cv::Rect(bx, by, (int)(bar_w * best->confidence), bar_h),
                      sc, -1);
    } else {
        put(r++, "", "Durum  TAKIP YOK");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Kullanim: %s <video.mp4> [cikti_dizini]\n", argv[0]);
        std::printf("  'q' veya ESC: erken cikis\n");
        return 1;
    }
    const std::string video_path = argv[1];
    const std::string out_dir = argc >= 3 ? argv[2] : "samples";

    std::system(("mkdir -p " + out_dir).c_str());
    size_t slash = video_path.find_last_of("/\\");
    std::string base = (slash != std::string::npos) ? video_path.substr(slash + 1) : video_path;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    const std::string out_video = out_dir + "/" + base + "_tracked.mp4";

    // --- Pipeline kurulumu ---
    io::VideoCameraSource cam(video_path, common::Modality::Visible);
    if (!cam.open()) {
        std::fprintf(stderr, "HATA: Video acilamadi: %s\n", video_path.c_str());
        return 1;
    }

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = 700.0f;
    stabilization::KltGyroStabilizer stab(scfg);

    detection::MogDetector det;
    tracking::AlphaBetaTracker tracker;

    // Pencere.
    const std::string win = "Drone Tracker - " + base;
    cv::namedWindow(win, cv::WINDOW_NORMAL);

    cv::VideoWriter writer;
    int saved_png = 0;

    std::uint64_t total_frames = 0, locked_frames = 0;
    double total_ms = 0;
    std::deque<double> ms_history;  // FPS için kayan pencere

    std::printf("=== Video Benchmark (CANLI) ===\n");
    std::printf("  Video : %s\n", video_path.c_str());
    std::printf("  Cikti : %s\n", out_video.c_str());
    std::printf("  'q'/ESC ile cik, 'p' ile durdur/devam\n\n");

    bool paused = false;

    while (true) {
        if (!paused) {
            auto frame_opt = cam.next_frame();
            if (!frame_opt) break;
            auto frame = *frame_opt;
            if (!frame || frame->image.empty()) continue;

            auto t0 = Clock::now();

            auto sf = stab.stabilize(frame, {});
            auto dets = det.detect(sf);
            auto tracks = tracker.update(dets, frame->stamp);

            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
            ++total_frames;

            ms_history.push_back(ms);
            if (ms_history.size() > 30) ms_history.pop_front();
            double avg_ms_30 = 0;
            for (double v : ms_history) avg_ms_30 += v;
            avg_ms_30 /= ms_history.size();
            double fps = ms_history.size() > 0 ? 1000.0 / avg_ms_30 : 0;

            // Kilit / track istatistikleri.
            bool has_confirmed = false;
            int n_conf = 0, n_coast = 0;
            for (const auto& t : tracks) {
                if (t.status == common::TrackStatus::Confirmed) {
                    has_confirmed = true;
                    ++n_conf;
                } else if (t.status == common::TrackStatus::Coasting) {
                    ++n_coast;
                }
            }
            if (has_confirmed) ++locked_frames;

            // En iyi track (en yüksek güvenli).
            const common::Track* best = nullptr;
            float best_conf = -1;
            for (const auto& t : tracks) {
                if (t.confidence > best_conf) {
                    best_conf = t.confidence;
                    best = &t;
                }
            }

            // Görselleştirme.
            cv::Mat vis;
            if (frame->image.channels() == 1)
                cv::cvtColor(frame->image, vis, cv::COLOR_GRAY2BGR);
            else
                frame->image.copyTo(vis);

            // Tespit noktaları (küçük mavi).
            for (const auto& d : dets)
                cv::circle(vis, cv::Point2f(d.centroid.x, d.centroid.y), 2,
                           cv::Scalar(255, 150, 0), -1, cv::LINE_AA);

            // Track kutuları.
            for (const auto& t : tracks) {
                const bool conf = t.status == common::TrackStatus::Confirmed;
                const cv::Scalar color = conf ? cv::Scalar(0, 255, 0)
                                              : cv::Scalar(0, 215, 255);
                const float half = std::max(12.0f, t.scale * 5.0f);
                const cv::Point2f c(t.position.x, t.position.y);
                cv::rectangle(vis, cv::Point2f(c.x - half, c.y - half),
                              cv::Point2f(c.x + half, c.y + half), color, 2, cv::LINE_AA);

                char label[64];
                std::snprintf(label, sizeof(label), "ID%u", t.id);
                cv::putText(vis, label, cv::Point2f(c.x - half, c.y - half - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);

                if (conf) {
                    cv::circle(vis, cv::Point2f(t.predicted.x, t.predicted.y), 3,
                               cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
            }

            // Sağ-üst metrik paneli.
            draw_metrics_panel(vis, (int)total_frames, 0 /*total unknown*/,
                               (int)dets.size(), (int)tracks.size(), n_conf, n_coast,
                               best, ms, fps);

            // Canlı gösterim.
            cv::imshow(win, vis);

            if (!writer.isOpened()) {
                const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open(out_video, fourcc, 30.0, vis.size(), true);
            }
            if (writer.isOpened()) writer.write(vis);

            if (saved_png < 6 && has_confirmed) {
                char path[256];
                std::snprintf(path, sizeof(path), "%s/%s_track_%03d.png",
                              out_dir.c_str(), base.c_str(), saved_png);
                cv::imwrite(path, vis);
                ++saved_png;
            }
        }

        // Klavye kontrolü.
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {  // q veya ESC
            std::printf("\nKullanici cikisi.\n");
            break;
        }
        if (key == 'p') {
            paused = !paused;
            if (paused) std::printf("\nDURAKLATILDI. 'p' ile devam.\n");
            else std::printf("DEVAM...\n");
        }
        if (key == ' ') {
            // Boşluk: tek kare ilerlet (pause modunda).
            paused = false;
            // bir sonraki döngüde bir kare işler, sonra tekrar kontrol ederiz
            // basitlik için: bir kare işleyip pause'a dön
        }
    }

    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();

    // --- Rapor ---
    double avg_ms = total_frames > 0 ? total_ms / total_frames : 0;
    double lock_pct = total_frames > 0 ? 100.0 * locked_frames / total_frames : 0;

    std::printf("\n--- SONUC ---\n");
    std::printf("  Toplam kare        : %llu\n", static_cast<unsigned long long>(total_frames));
    std::printf("  Kilitli kare       : %llu (%.1f%%)\n",
                static_cast<unsigned long long>(locked_frames), lock_pct);
    std::printf("  Ortalama sure      : %.2f ms/kare\n", avg_ms);
    std::printf("  Toplam sure        : %.2f s\n", total_ms / 1000.0);
    std::printf("  Cikti videosu      : %s\n", out_video.c_str());
    std::printf("  PNG ornekleri      : %s/%s_track_*.png\n", out_dir.c_str(), base.c_str());

    return 0;
}
