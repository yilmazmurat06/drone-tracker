// video_eval: GERÇEK (veya kaydedilmiş) video üzerinde pipeline'ı HEADLESS koşturup
// bench_pipeline ile AYNI metrikleri üretir. Sentetik sahne dışında doğrulamanın
// (Det-Fly / Anti-UAV / ARD-MAV gibi açık verisetleri ya da uçuş klipleri) kapısıdır.
//
// Kullanim:
//   ./video_eval <video> [--gt <annotations.csv>] [--match <px>] [--track <px>]
//                        [--warmup <kare>] [--focal <px>] [--csv <out.csv>]
//
// Yer-gerçeği (GT) CSV biçimleri (otomatik algılanır, '#' yorum satırı):
//   - 3 sütun : frame,cx,cy                      (merkez noktası)
//   - >=6 sütun (MOT): frame,id,left,top,w,h,... (merkez = left+w/2, top+h/2)
// frame 0-tabanlı kare numarası. Birden çok hedef desteklenir (aynı frame için
// çok satır); recall "kare içinde EN AZ bir GT yakalandı" üzerinden hesaplanır.
//
// GT yoksa: tespit/track sayıları + onaylı-track kare oranı + zamanlama raporlanır
// (recall/precision GT olmadan ölçülemez).
//
// NOT: Gerçek videoda IMU yoktur -> stabilizasyon saf optical-flow ile çalışır.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/common/time.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/mog_detector.hpp"
#include "dtrack/detection/sky_region_detector.hpp"
#include "dtrack/io/video_camera_source.hpp"
#include "dtrack/stabilization/klt_gyro_stabilizer.hpp"
#include "dtrack/tracking/kalman_tracker.hpp"

using namespace dtrack;
using Clock = std::chrono::high_resolution_clock;

struct GtPoint { float x, y; };

// frame_no -> o karedeki GT merkez(ler)i.
static std::map<int, std::vector<GtPoint>> load_gt(const std::string& path, bool& ok) {
    std::map<int, std::vector<GtPoint>> gt;
    std::ifstream in(path);
    ok = in.is_open();
    if (!ok) return gt;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line);
        std::vector<double> v;
        double x;
        while (ss >> x) v.push_back(x);
        if (v.size() < 3) continue;
        const int frame = static_cast<int>(v[0]);
        if (v.size() >= 6) {  // MOT: frame,id,left,top,w,h
            const float cx = static_cast<float>(v[2] + v[4] * 0.5);
            const float cy = static_cast<float>(v[3] + v[5] * 0.5);
            gt[frame].push_back({cx, cy});
        } else {  // frame,cx,cy
            gt[frame].push_back({static_cast<float>(v[1]), static_cast<float>(v[2])});
        }
    }
    return gt;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Kullanim: %s <video> [--gt <csv>] [--match <px>] [--track <px>]"
                    " [--warmup <kare>] [--focal <px>] [--csv <out.csv>]\n"
                    "          [--detector mog|sky] [--save <out.mp4>]\n", argv[0]);
        return 1;
    }
    const std::string video_path = argv[1];
    std::string gt_path, csv_out, save_path, detector = "mog";
    double match_px = 8.0, track_px = 10.0, focal = 700.0;
    int warmup = 30;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gt") && i + 1 < argc) gt_path = argv[++i];
        else if (!std::strcmp(argv[i], "--match") && i + 1 < argc) match_px = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--track") && i + 1 < argc) track_px = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--focal") && i + 1 < argc) focal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_out = argv[++i];
        else if (!std::strcmp(argv[i], "--detector") && i + 1 < argc) detector = argv[++i];
        else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) save_path = argv[++i];
    }

    bool have_gt = false;
    std::map<int, std::vector<GtPoint>> gt;
    if (!gt_path.empty()) {
        gt = load_gt(gt_path, have_gt);
        if (!have_gt) std::fprintf(stderr, "UYARI: GT acilamadi: %s\n", gt_path.c_str());
    }

    io::VideoCameraSource cam(video_path, common::Modality::Visible);
    if (!cam.open()) {
        std::fprintf(stderr, "HATA: Video acilamadi: %s\n", video_path.c_str());
        return 1;
    }

    stabilization::StabilizerConfig scfg;
    scfg.focal_px = static_cast<float>(focal);
    stabilization::KltGyroStabilizer stab(scfg);
    std::unique_ptr<detection::IDetector> det;
    if (detector == "sky") det = std::make_unique<detection::SkyRegionDetector>();
    else det = std::make_unique<detection::MogDetector>();
    tracking::KalmanTracker tracker;
    std::printf("  detektor        : %s\n", detector.c_str());

    cv::VideoWriter writer;  // --save verilirse annotasyonlu cikti

    // GT olmasa da çalışan sayaçlar.
    int frame_no = 0, eval_frames = 0;
    int n_det_total = 0, n_trk_total = 0, n_conf_frames = 0;
    double total_ms = 0;

    // GT'li metrikler.
    int n_gt_frames = 0, n_det_hit = 0, n_fp = 0;
    double sum_det_err2 = 0;
    int n_trk_locked = 0;
    double sum_trk_err2 = 0;
    std::set<std::uint32_t> seen_ids;

    std::ofstream csv;
    if (!csv_out.empty()) {
        csv.open(csv_out);
        csv << "frame,n_det,n_trk,best_trk_id,best_trk_x,best_trk_y,gt_x,gt_y,det_err,trk_err\n";
    }

    while (true) {
        auto fopt = cam.next_frame();
        if (!fopt) break;
        auto frame = *fopt;
        if (!frame || frame->image.empty()) { ++frame_no; continue; }

        auto t0 = Clock::now();
        auto sf = stab.stabilize(frame, {});            // IMU yok -> saf OF
        auto dets = det->detect(sf);
        auto tracks = tracker.update(dets, frame->stamp);
        auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- Annotasyonlu cikti (--save) ---
        if (!save_path.empty()) {
            cv::Mat vis;
            if (frame->image.channels() == 1) cv::cvtColor(frame->image, vis, cv::COLOR_GRAY2BGR);
            else frame->image.copyTo(vis);
            for (const auto& d : dets)  // tespitler: kucuk mavi nokta
                cv::circle(vis, d.centroid, 3, cv::Scalar(255, 150, 0), -1, cv::LINE_AA);
            for (const auto& t : tracks) {  // track kutusu: yesil=confirmed, sari=coasting
                const bool cf = t.status == common::TrackStatus::Confirmed;
                const cv::Scalar c = cf ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 215, 255);
                const float half = std::max(14.0f, t.scale * 3.0f);
                cv::rectangle(vis, cv::Point2f(t.position.x - half, t.position.y - half),
                              cv::Point2f(t.position.x + half, t.position.y + half), c, 2);
                char lb[48];
                std::snprintf(lb, sizeof(lb), "ID%u h%d", t.id, t.hits);
                cv::putText(vis, lb, cv::Point2f(t.position.x - half, t.position.y - half - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv::LINE_AA);
            }
            if (!writer.isOpened())
                writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0,
                            vis.size(), true);
            if (writer.isOpened()) writer.write(vis);
        }

        const int idx = frame_no++;
        if (idx < warmup) continue;
        ++eval_frames;
        total_ms += ms;
        n_det_total += static_cast<int>(dets.size());
        n_trk_total += static_cast<int>(tracks.size());

        bool any_conf = false;
        for (const auto& t : tracks)
            if (t.status == common::TrackStatus::Confirmed) any_conf = true;
        if (any_conf) ++n_conf_frames;

        // En iyi (en yakın) track ve GT eşlemesi (GT varsa).
        float gx = -1, gy = -1;
        double best_det = 1e9, best_trk = 1e9;
        std::uint32_t best_trk_id = 0;

        auto it = gt.find(idx);
        const bool gt_present = have_gt && it != gt.end() && !it->second.empty();
        if (gt_present) {
            ++n_gt_frames;
            // En yakın GT noktasına olan en iyi tespit/track mesafesi.
            for (const auto& g : it->second) {
                for (const auto& d : dets) {
                    const double e = std::hypot(d.centroid.x - g.x, d.centroid.y - g.y);
                    if (e < best_det) { best_det = e; gx = g.x; gy = g.y; }
                }
                for (const auto& t : tracks) {
                    const double e = std::hypot(t.position.x - g.x, t.position.y - g.y);
                    if (e < best_trk) { best_trk = e; best_trk_id = t.id; }
                }
            }
            // recall/precision
            if (best_det <= match_px) {
                ++n_det_hit;
                sum_det_err2 += best_det * best_det;
                n_fp += static_cast<int>(dets.size()) - 1;  // hedefe en yakın 1 hariç
            } else {
                n_fp += static_cast<int>(dets.size());
            }
            // continuity
            if (best_trk <= track_px) {
                ++n_trk_locked;
                sum_trk_err2 += best_trk * best_trk;
                seen_ids.insert(best_trk_id);
            }
        } else if (have_gt) {
            // Bu karede GT yok -> tüm tespitler FP sayılır.
            n_fp += static_cast<int>(dets.size());
        }

        if (csv.is_open()) {
            csv << idx << ',' << dets.size() << ',' << tracks.size() << ','
                << best_trk_id << ',';
            if (!tracks.empty()) {
                const auto& t = tracks.front();
                csv << t.position.x << ',' << t.position.y << ',';
            } else {
                csv << ",,";
            }
            csv << gx << ',' << gy << ',';
            if (best_det < 1e8) csv << best_det; csv << ',';
            if (best_trk < 1e8) csv << best_trk; csv << '\n';
        }
    }

    // --- Rapor ---
    std::printf("=== video_eval ===\n");
    std::printf("  video           : %s\n", video_path.c_str());
    std::printf("  degerlendirilen : %d kare (warmup %d)\n", eval_frames, warmup);
    std::printf("  zamanlama       : %.2f ms/kare  (~%.0f FPS)\n",
                eval_frames ? total_ms / eval_frames : 0.0,
                (eval_frames && total_ms > 0) ? eval_frames * 1000.0 / total_ms : 0.0);
    std::printf("  tespit/kare     : %.2f\n",
                eval_frames ? (double)n_det_total / eval_frames : 0.0);
    std::printf("  track/kare      : %.2f\n",
                eval_frames ? (double)n_trk_total / eval_frames : 0.0);
    std::printf("  onayli-track kare: %d / %d (%.1f%%)\n", n_conf_frames, eval_frames,
                eval_frames ? 100.0 * n_conf_frames / eval_frames : 0.0);

    if (have_gt && n_gt_frames > 0) {
        const double recall = (double)n_det_hit / n_gt_frames;
        const double prec = (n_det_hit + n_fp) > 0 ? (double)n_det_hit / (n_det_hit + n_fp) : 0;
        const double det_rms = n_det_hit ? std::sqrt(sum_det_err2 / n_det_hit) : 0;
        const double cont = (double)n_trk_locked / n_gt_frames;
        const double trk_rms = n_trk_locked ? std::sqrt(sum_trk_err2 / n_trk_locked) : 0;
        const double fp_per_frame = eval_frames ? (double)n_fp / eval_frames : 0;
        std::printf("  --- GT metrikleri (%d GT karesi, match=%.1fpx) ---\n",
                    n_gt_frames, match_px);
        std::printf("  recall          : %.3f\n", recall);
        std::printf("  precision       : %.3f\n", prec);
        std::printf("  det RMS         : %.2f px\n", det_rms);
        std::printf("  FP/kare         : %.2f\n", fp_per_frame);
        std::printf("  continuity      : %.3f  (track<%.1fpx)\n", cont, track_px);
        std::printf("  trk RMS         : %.2f px\n", trk_rms);
        std::printf("  benzersiz ID#   : %zu\n", seen_ids.size());
    } else {
        std::printf("  (GT yok -> recall/precision/continuity hesaplanamadi)\n");
    }
    if (csv.is_open()) std::printf("  ayrinti CSV     : %s\n", csv_out.c_str());
    return 0;
}
