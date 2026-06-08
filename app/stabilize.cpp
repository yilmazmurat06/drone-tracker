// ============================================================================
//  dtrack_stabilize — P1 (3a+3b) stabilizasyonu gösterir/ölçer.
//
//  FİKİR: Stabilizasyon işe yarıyorsa, ardışık iki kare HİZALANDIKTAN sonra
//  aralarındaki fark ≈ 0 olmalı (arka plan sabit); yalnız gerçek hareketli
//  nesneler kalır. Stabilizasyon olmadan ise EGO-MOTION yüzünden her şey kayar
//  ve fark görüntüsü "dolu" olur.
//
//  3b: artık GyroFlowStabilizer. Telemetri (.telemetry.csv) yüklenir; her kare
//  için [önceki, geçerli] kare zamanları arasındaki gyro penceresi beslenir.
//  Optik akış güvenilirse Flow, çökerse Gyro yoluna düşülür. Etikette hangi
//  yolun seçildiği yazar.
//
//  Görsel: solda HAM kare-farkı, sağda STABİLİZE kare-farkı.
//  Sayısal: merkez bölgenin ortalama mutlak farkı (raw vs stab) ve iyileşme oranı.
//
//  Kullanım:
//    dtrack_stabilize [prefix] [--save out.mp4] [--dump N] [--max-frames N] [--speed S]
// ============================================================================
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/core/time.hpp"
#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"

namespace {

cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g; cv::cvtColor(img, g, cv::COLOR_BGR2GRAY); return g;
}

// Kenar (warp'tan gelen siyah bant) farkı şişirmesin diye merkez bölge.
cv::Rect center_roi(const cv::Size& s, double margin = 0.12) {
    int mx = static_cast<int>(s.width * margin);
    int my = static_cast<int>(s.height * margin);
    return cv::Rect(mx, my, s.width - 2 * mx, s.height - 2 * my);
}

void label(cv::Mat& img, const std::string& s, cv::Point org, cv::Scalar col) {
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv::LINE_AA);
}

const char* mode_name(dtrack::GyroFlowStabilizer::Mode m) {
    switch (m) {
        case dtrack::GyroFlowStabilizer::Mode::Flow: return "FLOW";
        case dtrack::GyroFlowStabilizer::Mode::Gyro: return "GYRO";
        default:                                     return "YOK";
    }
}

struct Args {
    std::string prefix = "data/flight_01_084727";
    std::string save_path;
    int dump_n = 0, max_frames = 0;
    double speed = 1.0;
    bool force_gyro = false;  // optik akışı kapat → tüm videoyu SAF GYRO ile stabilize et
};

Args parse_args(int argc, char** argv) {
    Args a; bool pset = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--save" && i + 1 < argc) a.save_path = argv[++i];
        else if (s == "--dump" && i + 1 < argc) a.dump_n = std::stoi(argv[++i]);
        else if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--speed" && i + 1 < argc) a.speed = std::stod(argv[++i]);
        else if (s == "--force-gyro") a.force_gyro = true;
        else if (!s.empty() && s[0] != '-' && !pset) { a.prefix = s; pset = true; }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const std::string mp4 = args.prefix + ".mp4";

    dtrack::RecordedFrameSource video(mp4);
    if (!video.is_open()) { std::cerr << "HATA: video acilamadi: " << mp4 << "\n"; return 1; }

    // Telemetri (gyro) — yüklenemezse gyro yolu devre dışı kalır, OF-only çalışır.
    dtrack::TelemetryLog telem;
    const bool have_telem = telem.load(args.prefix + ".telemetry.csv");
    if (!have_telem)
        std::cerr << "UYARI: telemetri yok (" << args.prefix
                  << ".telemetry.csv) → yalnız optik akış.\n";

    // --force-gyro: OF güven eşiğini ulaşılamaz yap → her kare GYRO yoluna düşer.
    // Saf gyro stabilizasyonunu izlemek + AxisMap'in doğru olup olmadığını gözle
    // doğrulamak için (yanlış eksen → sahne TERS kayar, kötüleşir).
    dtrack::GyroFlowStabilizer::Params sp;
    if (args.force_gyro) sp.of.min_inliers = 1000000000;
    dtrack::GyroFlowStabilizer stab(sp);
    if (args.force_gyro)
        std::cerr << "MOD: --force-gyro (optik akis kapali, saf gyro)\n";
    const double fps = video.fps();
    const bool dumping = args.dump_n > 0;
    const bool saving = !args.save_path.empty();
    const bool live = !dumping && !saving;

    if (dumping)
        std::printf("%6s %6s %8s %8s %8s %8s %8s\n",
                    "kare", "yol", "izlenen", "inlier", "ham", "stab", "iyilesme");
    if (live) cv::namedWindow("dtrack stabilize", cv::WINDOW_NORMAL);

    cv::VideoWriter writer;
    const int delay_ms = std::max(1, static_cast<int>(1000.0 / (fps * args.speed)));

    cv::Mat prev_bgr;
    double sum_raw = 0, sum_stab = 0;
    int counted = 0, shown = 0;
    double prev_t_rel = 0.0;
    bool   have_prev_t = false;

    dtrack::Frame frame;
    while (video.next(frame)) {
        // Kare zamanı: t = i / fps (telemetri t_rel ile aynı eksen).
        const double t_rel = static_cast<double>(frame.id) / fps;

        // Gyro penceresi: [önceki kare, geçerli kare] zaman aralığını interpolasyonla
        // ören iki uç örnek. integrate_rotation bu ω'yu trapezle θ'ya çevirir.
        std::vector<dtrack::Telemetry> window;
        if (have_telem && have_prev_t)
            window = {telem.at(prev_t_rel), telem.at(t_rel)};

        dtrack::Frame warped;
        cv::Matx33f H;
        const bool ok = stab.stabilize(frame, window, warped, H);

        if (!prev_bgr.empty()) {
            const cv::Mat gp = to_gray(prev_bgr);
            const cv::Mat gc = to_gray(frame.image);
            const cv::Mat gw = to_gray(warped.image);
            const cv::Rect roi = center_roi(gp.size());

            cv::Mat d_raw, d_stab;
            cv::absdiff(gc(roi), gp(roi), d_raw);
            cv::absdiff(gw(roi), gp(roi), d_stab);
            const double m_raw = cv::mean(d_raw)[0];
            const double m_stab = cv::mean(d_stab)[0];
            sum_raw += m_raw; sum_stab += m_stab; ++counted;

            if (dumping) {
                std::printf("%6lld %6s %8d %8d %8.2f %8.2f %7.2fx\n",
                            (long long)frame.id, mode_name(stab.last_mode()),
                            stab.last_tracked(), stab.last_inliers(),
                            m_raw, m_stab, m_stab > 1e-6 ? m_raw / m_stab : 0.0);
            } else {
                // Görsel: tam-kare fark görüntüleri, kontrast için x3.
                cv::Mat vr, vs;
                cv::absdiff(gc, gp, vr);
                cv::absdiff(gw, gp, vs);
                vr *= 3; vs *= 3;
                cv::cvtColor(vr, vr, cv::COLOR_GRAY2BGR);
                cv::cvtColor(vs, vs, cv::COLOR_GRAY2BGR);
                label(vr, "HAM fark (ego-motion)  ort=" + std::to_string((int)m_raw),
                      {20, 40}, {0, 200, 255});
                label(vs, std::string("STAB[") + mode_name(stab.last_mode()) + "]" +
                          (ok ? "" : "(dusuk guven)") +
                          " fark  ort=" + std::to_string((int)m_stab),
                      {20, 40}, {0, 255, 0});
                cv::Mat sideby;
                cv::hconcat(vr, vs, sideby);
                if (sideby.cols > 1920) cv::resize(sideby, sideby, {}, 0.5, 0.5);

                if (saving) {
                    if (!writer.isOpened())
                        writer.open(args.save_path, cv::VideoWriter::fourcc('m','p','4','v'),
                                    fps, sideby.size());
                    writer.write(sideby);
                } else {
                    cv::imshow("dtrack stabilize", sideby);
                    const int k = cv::waitKey(delay_ms);
                    if (k == 'q' || k == 27) break;
                }
            }
        }

        prev_bgr = frame.image.clone();
        prev_t_rel = t_rel;
        have_prev_t = true;
        ++shown;
        if (args.max_frames > 0 && shown >= args.max_frames) break;
        if (args.dump_n > 0 && shown >= args.dump_n) break;
    }

    if (counted > 0) {
        const double ar = sum_raw / counted, as = sum_stab / counted;
        std::printf("\nORTALAMA: ham=%.2f  stab=%.2f  iyilesme=%.2fx  (%d kare)\n",
                    ar, as, as > 1e-6 ? ar / as : 0.0, counted);
    }
    if (saving) std::cout << "kaydedildi -> " << args.save_path << "\n";
    return 0;
}
