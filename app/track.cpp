// ============================================================================
//  dtrack_track — TAM ZİNCİR: P1 stabilize → P2 tespit → P4 takip.
//
//  AMAÇ: P2'nin gürültülü adaylarını, M-of-N temporal tutarlılığıyla temizlemek.
//  Yalnız ONAYLANMIŞ (Confirmed) izler çizilir → paralaks pırıltısı ekrana
//  gelmez. Gerçek hedef (zeplin, uçak) tutarlı iz oluşturduğu için kalır.
//
//  Çizim:
//    yeşil kutu + id + hız oku → Confirmed iz
//    --show-tentative : sarı nokta → henüz onaylanmamış (Tentative) izler
//
//  Kullanım:
//    dtrack_track [prefix] [--save out.mp4] [--dump N] [--max-frames N]
//                 [--speed S] [--show-tentative] [--roi x,y,w,h]
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/detection/clutter_discriminator.hpp"
#include "dtrack/detection/moving_target_detector.hpp"
#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"
#include "dtrack/tracking/multi_target_tracker.hpp"

namespace {

struct Args {
    std::string prefix = "data/flight_01_084727";
    std::string save_path;
    int dump_n = 0, max_frames = 0;
    double speed = 1.0;
    bool show_tentative = false;
    cv::Rect roi;
    float score_thresh = 0.50f;  // P3: bu eşiğin altındaki adaylar tracker'a gitmez
};

Args parse_args(int argc, char** argv) {
    Args a; bool pset = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--save" && i + 1 < argc) a.save_path = argv[++i];
        else if (s == "--dump" && i + 1 < argc) a.dump_n = std::stoi(argv[++i]);
        else if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--speed" && i + 1 < argc) a.speed = std::stod(argv[++i]);
        else if (s == "--show-tentative") a.show_tentative = true;
        else if (s == "--roi" && i + 1 < argc) {
            int x, y, w, h;
            if (std::sscanf(argv[++i], "%d,%d,%d,%d", &x, &y, &w, &h) == 4)
                a.roi = cv::Rect(x, y, w, h);
        }
        else if (s == "--score-thresh" && i + 1 < argc) a.score_thresh = std::stof(argv[++i]);
        else if (!s.empty() && s[0] != '-' && !pset) { a.prefix = s; pset = true; }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    dtrack::RecordedFrameSource video(args.prefix + ".mp4");
    if (!video.is_open()) { std::cerr << "HATA: video acilamadi\n"; return 1; }
    dtrack::TelemetryLog telem;
    const bool have_telem = telem.load(args.prefix + ".telemetry.csv");

    dtrack::GyroFlowStabilizer stab;
    dtrack::MovingTargetDetector det;
    dtrack::ClutterDiscriminator disc;
    dtrack::MultiTargetTracker trk;
    if (!args.roi.empty()) det.set_roi(args.roi);

    const double fps = video.fps();
    const bool dumping = args.dump_n > 0;
    const bool saving = !args.save_path.empty();
    const bool live = !dumping && !saving;

    if (dumping) std::printf("%6s %8s %10s %10s\n", "kare", "aday", "tentative", "confirmed");
    if (live) cv::namedWindow("dtrack track", cv::WINDOW_NORMAL);

    cv::VideoWriter writer;
    const int delay_ms = std::max(1, static_cast<int>(1000.0 / (fps * args.speed)));

    double prev_t = 0; bool have_prev = false;
    int shown = 0;

    dtrack::Frame frame, warped;
    while (video.next(frame)) {
        const double t = static_cast<double>(frame.id) / fps;
        std::vector<dtrack::Telemetry> window;
        if (have_telem && have_prev) window = {telem.at(prev_t), telem.at(t)};

        cv::Matx33f H;
        stab.stabilize(frame, window, warped, H);

        std::vector<dtrack::Detection> dets;
        det.detect(warped, dets);

        // P3: her adaya skor ver, eşiğin altını ele (clutter reddi).
        for (auto& d : dets) d.score = disc.score(d);
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                       [&](const dtrack::Detection& d){ return d.score < args.score_thresh; }),
                   dets.end());

        std::vector<dtrack::Track> tracks;
        trk.update(dets, tracks);

        prev_t = t; have_prev = true;
        ++shown;

        int n_tent = 0, n_conf = 0;
        for (const auto& tr : tracks) {
            if (tr.status == dtrack::Track::Status::Confirmed) ++n_conf;
            else if (tr.status == dtrack::Track::Status::Tentative) ++n_tent;
        }

        if (dumping) {
            std::printf("%6lld %8zu %10d %10d\n",
                        (long long)frame.id, dets.size(), n_tent, n_conf);
        } else {
            cv::Mat vis = warped.image.clone();
            for (const auto& tr : tracks) {
                const bool conf = tr.status == dtrack::Track::Status::Confirmed;
                if (!conf && !args.show_tentative) continue;
                const cv::Point c(cvRound(tr.pos.x), cvRound(tr.pos.y));
                if (conf) {
                    // Ölçülen silüet kutusu (yoksa küçük varsayılan), 4px pay ile.
                    cv::Rect b = tr.bbox.width > 0
                        ? (tr.bbox + cv::Size(8, 8)) - cv::Point(4, 4)
                        : cv::Rect(c.x - 14, c.y - 14, 28, 28);
                    cv::rectangle(vis, b, cv::Scalar(0, 255, 0), 2);
                    cv::putText(vis, "#" + std::to_string(tr.id), {b.x, b.y - 6},
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    // hız oku (5x büyütülmüş)
                    cv::arrowedLine(vis, c,
                                    {c.x + cvRound(tr.vel.x * 5), c.y + cvRound(tr.vel.y * 5)},
                                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.3);
                } else {
                    cv::circle(vis, c, 3, cv::Scalar(0, 200, 255), -1);  // tentative
                }
            }
            cv::putText(vis, "confirmed: " + std::to_string(n_conf) +
                        "   (aday: " + std::to_string(dets.size()) + ")",
                        {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::Mat outimg = vis;
            if (outimg.cols > 1920) cv::resize(outimg, outimg, {}, 0.5, 0.5);
            if (saving) {
                if (!writer.isOpened())
                    writer.open(args.save_path, cv::VideoWriter::fourcc('m','p','4','v'),
                                fps, outimg.size());
                writer.write(outimg);
            } else {
                cv::imshow("dtrack track", outimg);
                const int k = cv::waitKey(delay_ms);
                if (k == 'q' || k == 27) break;
            }
        }

        if (args.max_frames > 0 && shown >= args.max_frames) break;
        if (args.dump_n > 0 && shown >= args.dump_n) break;
    }

    if (saving) std::cout << "kaydedildi -> " << args.save_path << "\n";
    return 0;
}
