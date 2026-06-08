// ============================================================================
//  dtrack_replay — kayıtlı uçuşu (mp4) telemetri (csv) overlay'i ile oynatır.
//
//  AMAÇ (Adım 2): video ile telemetriyi SENKRONLU göstererek t_rel↔kare
//  hizalamasının doğru olduğunu gözle kanıtlamak. Kamera sağa/sola döndüğünde
//  gyro_yaw, yukarı/aşağı baktığında gyro_pitch çubukları buna uygun oynamalı.
//
//  Kullanım:
//    dtrack_replay [prefix] [--save out.mp4] [--dump N] [--max-frames N] [--speed S]
//      prefix : data/flight_01_084727 gibi (uzantısız). Vars: data/flight_01_084727
//      --save : ekran yerine annotasyonlu mp4 yazar (ekransız sunucu için)
//      --dump : ilk N kare için kare→telemetri eşlemesini yazdırır (GUI yok)
//      --speed: oynatma hızı çarpanı (1.0 = gerçek zaman)
// ============================================================================
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/core/time.hpp"
#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"

namespace {

constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;

// Quaternion (x,y,z,w) → Tait-Bryan euler açıları (radyan): roll, pitch, yaw.
// NOT: Liftoff'un eksen/işaret konvansiyonu farklı olabilir; ufuk çizgisi bu
// yüzden "deneysel". Asıl güvenilir sinyal etiketli gyro değerleridir.
void quat_to_euler(const dtrack::Telemetry& t, double& roll, double& pitch, double& yaw) {
    const double x = t.att_x, y = t.att_y, z = t.att_z, w = t.att_w;
    roll  = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    double sinp = 2.0 * (w * y - z * x);
    sinp = std::max(-1.0, std::min(1.0, sinp));     // asin alanına kırp
    pitch = std::asin(sinp);
    yaw   = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

void put_line(cv::Mat& img, const std::string& s, int row) {
    const cv::Point org(14, 30 + row * 26);
    // okunurluk için siyah kontur + beyaz yazı
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}

// Bir gyro ekseni için merkezden iki yana açılan yatay çubuk.
void gyro_bar(cv::Mat& img, const std::string& label, double val_deg, int row) {
    const int y = img.rows - 90 + row * 26;
    const int cx = 150;             // sıfır ekseni
    const int half = 120;           // maksimum yarı uzunluk (piksel)
    const double scale = half / 300.0;  // 300 deg/s → tam çubuk
    int len = static_cast<int>(std::max(-(double)half, std::min((double)half, val_deg * scale)));
    cv::line(img, {cx, y}, {cx, y - 16}, cv::Scalar(180, 180, 180), 1);  // sıfır işareti
    cv::rectangle(img, cv::Rect(std::min(cx, cx + len), y - 12,
                                std::abs(len), 12),
                  cv::Scalar(0, 200, 255), cv::FILLED);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%-10s %+7.1f deg/s", label.c_str(), val_deg);
    cv::putText(img, buf, {cx + half + 12, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, buf, {cx + half + 12, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void draw_overlay(cv::Mat& img, int64_t frame_id, double t_rel,
                  const dtrack::Telemetry& tel) {
    double roll, pitch, yaw;
    quat_to_euler(tel, roll, pitch, yaw);

    char buf[128];
    std::snprintf(buf, sizeof(buf), "kare %lld   t_rel %.3f s", (long long)frame_id, t_rel);
    put_line(img, buf, 0);
    std::snprintf(buf, sizeof(buf), "attitude (quat->euler): roll %+6.1f  pitch %+6.1f  yaw %+6.1f deg",
                  roll * kRad2Deg, pitch * kRad2Deg, yaw * kRad2Deg);
    put_line(img, buf, 1);
    std::snprintf(buf, sizeof(buf), "hiz: %.1f m/s   pos(%.0f, %.0f, %.0f)",
                  std::sqrt(tel.vel_x * tel.vel_x + tel.vel_y * tel.vel_y + tel.vel_z * tel.vel_z),
                  tel.pos_x, tel.pos_y, tel.pos_z);
    put_line(img, buf, 2);

    gyro_bar(img, "gyro pitch", tel.gyro_pitch * kRad2Deg, 0);
    gyro_bar(img, "gyro roll",  tel.gyro_roll  * kRad2Deg, 1);
    gyro_bar(img, "gyro yaw",   tel.gyro_yaw   * kRad2Deg, 2);

    // Deneysel yapay ufuk: roll ile döndür, pitch ile dikey kaydır.
    const cv::Point2f c(img.cols * 0.5f, img.rows * 0.5f - static_cast<float>(pitch * 600.0));
    const float L = img.cols * 0.35f;
    const cv::Point2f d(std::cos(roll), std::sin(roll));
    cv::line(img, c - L * d, c + L * d, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
}

struct Args {
    std::string prefix = "data/flight_01_084727";
    std::string save_path;
    int  dump_n = 0;
    int  max_frames = 0;     // 0 = sınırsız
    double speed = 1.0;
};

Args parse_args(int argc, char** argv) {
    Args a;
    bool prefix_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--save" && i + 1 < argc)            a.save_path = argv[++i];
        else if (s == "--dump" && i + 1 < argc)       a.dump_n = std::stoi(argv[++i]);
        else if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--speed" && i + 1 < argc)      a.speed = std::stod(argv[++i]);
        else if (!s.empty() && s[0] != '-' && !prefix_set) { a.prefix = s; prefix_set = true; }
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const std::string mp4 = args.prefix + ".mp4";
    const std::string csv = args.prefix + ".telemetry.csv";

    dtrack::RecordedFrameSource video(mp4);
    if (!video.is_open()) {
        std::cerr << "HATA: video acilamadi: " << mp4 << "\n";
        return 1;
    }
    dtrack::TelemetryLog log;
    if (!log.load(csv)) {
        std::cerr << "HATA: telemetri yuklenemedi: " << csv << "\n";
        return 1;
    }

    const double fps = video.fps();
    std::cout << "video: " << mp4 << "  fps=" << fps
              << "  kare=" << video.frame_count() << "\n";
    std::cout << "telemetri: " << log.size() << " ornek, t_rel ["
              << log.first_t_rel() << " .. " << log.last_t_rel() << "]\n";

    // --- DUMP modu: GUI yok, sadece senkronu sayısal yazdır ---
    if (args.dump_n > 0) {
        std::printf("%6s %10s %10s %10s %10s\n",
                    "kare", "t_rel", "gyroP", "gyroR", "gyroY");
        for (int i = 0; i < args.dump_n; ++i) {
            const double t_rel = i / fps;
            const dtrack::Telemetry t = log.at(t_rel);
            std::printf("%6d %10.4f %10.4f %10.4f %10.4f\n",
                        i, t_rel, t.gyro_pitch, t.gyro_roll, t.gyro_yaw);
        }
        return 0;
    }

    // --- SAVE veya LIVE ---
    cv::VideoWriter writer;
    const bool saving = !args.save_path.empty();
    const bool live = !saving;
    if (live) cv::namedWindow("dtrack replay", cv::WINDOW_NORMAL);

    const int delay_ms = std::max(1, static_cast<int>(1000.0 / (fps * args.speed)));
    bool paused = false;

    dtrack::Frame frame;
    int64_t shown = 0;
    while (video.next(frame)) {
        const double t_rel = dtrack::ns_to_seconds(frame.t);  // = id/fps
        const dtrack::Telemetry tel = log.at(t_rel);
        draw_overlay(frame.image, frame.id, t_rel, tel);

        if (saving) {
            if (!writer.isOpened()) {
                writer.open(args.save_path,
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps, frame.image.size());
                if (!writer.isOpened()) {
                    std::cerr << "HATA: cikti yazilamadi: " << args.save_path << "\n";
                    return 1;
                }
            }
            writer.write(frame.image);
        } else {
            cv::imshow("dtrack replay", frame.image);
            const int key = cv::waitKey(paused ? 0 : delay_ms);
            if (key == 'q' || key == 27) break;     // q / ESC
            if (key == ' ') paused = !paused;        // boşluk: duraklat
        }

        ++shown;
        if (args.max_frames > 0 && shown >= args.max_frames) break;
        if (saving && (shown % 60 == 0))
            std::cout << "\rkaydedilen kare: " << shown << std::flush;
    }

    if (saving) std::cout << "\rkaydedildi: " << shown << " kare -> " << args.save_path << "\n";
    std::cout << "bitti.\n";
    return 0;
}
