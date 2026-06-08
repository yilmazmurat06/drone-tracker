// ============================================================================
//  dtrack_detect — P2 hibrit tespiti gerçek footage üzerinde gösterir.
//
//  AKIŞ: kare → P1 stabilizasyon (GyroFlowStabilizer) → P2 tespit
//  (MovingTargetDetector, kare-farkı + MOG2) → aday kutuları çiz.
//
//  Liftoff hava trafiği (zeplin, küçük uçaklar) hareketli hedeflerdir; gökyüzü
//  önünde net sıyrılmaları beklenir. Zemin/ağaç paralaksı bir miktar yanlış-
//  pozitif üretir (P3/cue ile sonra elenecek).
//
//  Kullanım:
//    dtrack_detect [prefix] [--save out.mp4] [--dump N] [--max-frames N]
//                  [--speed S] [--mask]
// ============================================================================
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"
#include "dtrack/detection/moving_target_detector.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"

namespace {

struct Args {
    std::string prefix = "data/flight_01_084727";
    std::string save_path;
    int dump_n = 0, max_frames = 0;
    double speed = 1.0;
    bool show_mask = false;   // --mask: sağda birleşik tespit maskesini de göster
    cv::Rect roi;             // --roi x,y,w,h: cue-odaklı arama bölgesi (boş = tüm kare)
    // top-hat (yerel kontrast) dalı — hızlı deneme için CLI'dan ayarlanabilir
    bool th_set = false;
    bool tophat = true;
    int  tophat_thresh = 20, tophat_ksize = 13, tophat_mode = 0;
};

Args parse_args(int argc, char** argv) {
    Args a; bool pset = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--save" && i + 1 < argc) a.save_path = argv[++i];
        else if (s == "--dump" && i + 1 < argc) a.dump_n = std::stoi(argv[++i]);
        else if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--speed" && i + 1 < argc) a.speed = std::stod(argv[++i]);
        else if (s == "--mask") a.show_mask = true;
        else if (s == "--no-tophat") { a.tophat = false; a.th_set = true; }
        else if (s == "--tophat-thresh" && i + 1 < argc) { a.tophat_thresh = std::stoi(argv[++i]); a.th_set = true; }
        else if (s == "--tophat-ksize"  && i + 1 < argc) { a.tophat_ksize  = std::stoi(argv[++i]); a.th_set = true; }
        else if (s == "--tophat-mode"   && i + 1 < argc) { a.tophat_mode   = std::stoi(argv[++i]); a.th_set = true; }
        else if (s == "--roi" && i + 1 < argc) {
            int x, y, w, h;
            if (std::sscanf(argv[++i], "%d,%d,%d,%d", &x, &y, &w, &h) == 4)
                a.roi = cv::Rect(x, y, w, h);
        }
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
    dtrack::MovingTargetDetector::Params det_p;
    if (args.th_set) {
        det_p.tophat        = args.tophat;
        det_p.tophat_thresh = args.tophat_thresh;
        det_p.tophat_ksize  = args.tophat_ksize;
        det_p.tophat_mode   = args.tophat_mode;
        std::cerr << "TOPHAT: on=" << det_p.tophat << " thresh=" << det_p.tophat_thresh
                  << " ksize=" << det_p.tophat_ksize << " mode=" << det_p.tophat_mode << "\n";
    }
    dtrack::MovingTargetDetector det(det_p);
    if (!args.roi.empty()) {
        det.set_roi(args.roi);
        std::cerr << "CUE: arama bolgesi " << args.roi << " ile sinirli\n";
    }

    const double fps = video.fps();
    const bool dumping = args.dump_n > 0;
    const bool saving = !args.save_path.empty();
    const bool live = !dumping && !saving;

    if (dumping) std::printf("%6s %8s %10s\n", "kare", "aday", "ornek_bbox");
    if (live) cv::namedWindow("dtrack detect", cv::WINDOW_NORMAL);

    cv::VideoWriter writer;
    const int delay_ms = std::max(1, static_cast<int>(1000.0 / (fps * args.speed)));

    double prev_t = 0; bool have_prev = false;
    int shown = 0;
    long total_dets = 0;

    dtrack::Frame frame, warped;
    while (video.next(frame)) {
        const double t = static_cast<double>(frame.id) / fps;
        std::vector<dtrack::Telemetry> window;
        if (have_telem && have_prev) window = {telem.at(prev_t), telem.at(t)};

        cv::Matx33f H;
        stab.stabilize(frame, window, warped, H);

        std::vector<dtrack::Detection> dets;
        const bool ok = det.detect(warped, dets);
        total_dets += dets.size();

        prev_t = t; have_prev = true;
        ++shown;

        if (dumping) {
            std::string ex;
            if (!dets.empty()) {
                const auto& b = dets.front().bbox;
                ex = std::to_string(b.x) + "," + std::to_string(b.y) + " " +
                     std::to_string(b.width) + "x" + std::to_string(b.height);
            }
            std::printf("%6lld %8zu %10s\n", (long long)frame.id, dets.size(), ex.c_str());
        } else {
            cv::Mat vis = warped.image.clone();
            if (!args.roi.empty())  // cue bölgesini sarı çiz
                cv::rectangle(vis, args.roi, cv::Scalar(0, 200, 255), 2);
            for (const auto& d : dets) {
                // Küçük hedefi görünür kılmak için kutuyu biraz şişir.
                cv::Rect b = d.bbox;
                b.x -= 6; b.y -= 6; b.width += 12; b.height += 12;
                cv::rectangle(vis, b, cv::Scalar(0, 255, 0), 2);
            }
            cv::putText(vis, "aday: " + std::to_string(dets.size()) +
                        (ok ? "" : "  (model isiniyor)"),
                        {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::Mat outimg = vis;
            if (args.show_mask) {
                cv::Mat m; cv::cvtColor(det.last_mask(), m, cv::COLOR_GRAY2BGR);
                cv::hconcat(vis, m, outimg);
            }
            if (outimg.cols > 1920) cv::resize(outimg, outimg, {}, 0.5, 0.5);

            if (saving) {
                if (!writer.isOpened())
                    writer.open(args.save_path, cv::VideoWriter::fourcc('m','p','4','v'),
                                fps, outimg.size());
                writer.write(outimg);
            } else {
                cv::imshow("dtrack detect", outimg);
                const int k = cv::waitKey(delay_ms);
                if (k == 'q' || k == 27) break;
            }
        }

        if (args.max_frames > 0 && shown >= args.max_frames) break;
        if (args.dump_n > 0 && shown >= args.dump_n) break;
    }

    std::printf("\nTOPLAM aday tespit: %ld  (%d kare)\n", total_dets, shown);
    if (saving) std::cout << "kaydedildi -> " << args.save_path << "\n";
    return 0;
}
