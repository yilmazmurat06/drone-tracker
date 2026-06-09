// ============================================================================
//  dtrack_horizon_calib — TELEMETRİ UFKU konvansiyon kalibrasyonu (#14).
//
//  AMAÇ: attitude quaternion'dan ufuk çizgisini doğru çizebilmek için "kameradaki
//  yerçekimi yönü" konvansiyonunu (dünya-yukarı ekseni + body→kamera işaretli
//  permütasyonu) BULMAK. Tıpkı gyro AxisMap kalibrasyonu (dtrack_calibrate) gibi:
//  birçok karede GÖRÜNTÜ ufkunu (gök/zemin sınırı) referans alır, 144 konvansiyonu
//  bu referansa karşı puanlar, en düşük piksel-hatalıyı seçer.
//
//  GÖRÜNTÜ UFKU (referans): HSV gök maskesi (parlak ∧ (soluk ∨ mavi)); her sütunda
//  gök→zemin geçiş satırı; sağlam doğru (Huber) uydur. Yalnız NET ufuklu kareler.
//
//  TELEMETRİ UFKU: g_cam·(u−cx, v−cy, f)=0. g_cam = işaretli-perm( −R[axis] ),
//  R = quaternion'dan body→world. Eksen + perm + işaret = konvansiyon.
//
//  Kullanım: dtrack_horizon_calib [prefix] [--max-frames N] [--step K] [--fov D]
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"

namespace {

struct Args {
    std::string prefix = "data/flight_01_084727";
    int    max_frames = 600;   // kalibrasyon için ilk N kare (net ufuklar burada)
    int    step       = 20;    // her K karede bir örnekle
    float  fov_deg    = 128.f;
};

Args parse_args(int argc, char** argv) {
    Args a; bool pset = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--step"  && i + 1 < argc) a.step = std::stoi(argv[++i]);
        else if (s == "--fov"   && i + 1 < argc) a.fov_deg = std::stof(argv[++i]);
        else if (!s.empty() && s[0] != '-' && !pset) { a.prefix = s; pset = true; }
    }
    return a;
}

// HSV gök maskesi (detektördeki tanımla aynı): parlak ∧ (soluk ∨ mavi).
cv::Mat sky_mask(const cv::Mat& bgr) {
    cv::Mat hsv, ch[3];
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, ch);
    cv::Mat bright = ch[2] > 140;
    cv::Mat pale   = ch[1] < 70;
    cv::Mat blue   = (ch[0] > 95) & (ch[0] < 135);
    cv::Mat sky;
    cv::bitwise_or(pale, blue, sky);
    cv::bitwise_and(sky, bright, sky);
    return sky;  // 0/255
}

// Görüntü ufku: her sütunda gök(üst)→zemin(alt) geçişi; Huber doğru. valid=false →
// güvenilmez (tüm-gök, tüm-zemin veya az nokta). Dönen: v = m·u + c.
struct Line { double m = 0, c = 0; bool valid = false; };

Line image_horizon(const cv::Mat& bgr) {
    cv::Mat sky = sky_mask(bgr);
    // morfolojik kapat: küçük gök-deliklerini (ince bulut kenarı) doldur → run kararlı
    cv::morphologyEx(sky, sky, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    const int W = sky.cols, H = sky.rows;
    const int need = std::max(6, H / 36);   // gök/zemin "uzun run" eşiği
    std::vector<cv::Point2f> pts;
    for (int u = 0; u < W; u += std::max(1, W / 160)) {
        // üstten aşağı: ilk yeterince-uzun ZEMIN (¬sky) run'ının başı = ufuk
        int run = 0, horizon = -1;
        for (int v = 0; v < H; ++v) {
            if (sky.at<uchar>(v, u) == 0) { if (++run >= need) { horizon = v - need + 1; break; } }
            else run = 0;
        }
        if (horizon > 2 && horizon < H - 2) pts.push_back({(float)u, (float)horizon});
    }
    Line ln;
    if (pts.size() < 12) return ln;          // az nokta → güvenilmez
    cv::Vec4f l;
    cv::fitLine(pts, l, cv::DIST_HUBER, 0, 0.01, 0.01);  // (vx,vy,x0,y0)
    if (std::fabs(l[0]) < 1e-3) return ln;
    ln.m = l[1] / l[0];
    ln.c = l[3] - ln.m * l[2];
    // artık (residual) kontrolü: çoğu nokta çizgiye yakın olmalı
    double res = 0; int inl = 0;
    for (auto& p : pts) {
        double d = std::fabs(p.y - (ln.m * p.x + ln.c));
        if (d < 25) ++inl;
        res += d;
    }
    if (inl < (int)pts.size() * 0.5) return ln;  // çok dağınık → güvenilmez
    ln.valid = true;
    return ln;
}

// quaternion → R (body→world)
void quat_R(double x, double y, double z, double w, double R[3][3]) {
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - z * w);     R[0][2] = 2 * (x * z + y * w);
    R[1][0] = 2 * (x * y + z * w);     R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - x * w);
    R[2][0] = 2 * (x * z - y * w);     R[2][1] = 2 * (y * z + x * w);     R[2][2] = 1 - 2 * (x * x + y * y);
}

struct Conv { int axis; std::array<int, 3> perm; std::array<int, 3> sign; };

std::vector<Conv> all_convs() {
    const int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    std::vector<Conv> out;
    for (int a = 0; a < 3; ++a)
        for (auto& p : perms)
            for (int s = 0; s < 8; ++s)
                out.push_back({a, {p[0], p[1], p[2]},
                               {(s & 1) ? 1 : -1, (s & 2) ? 1 : -1, (s & 4) ? 1 : -1}});
    return out;
}

// Konvansiyon + quaternion → g_cam (kameradaki yerçekimi yönü).
cv::Vec3d g_cam_of(const Conv& c, const dtrack::Telemetry& t) {
    double R[3][3];
    quat_R(t.att_x, t.att_y, t.att_z, t.att_w, R);
    const cv::Vec3d v(-R[c.axis][0], -R[c.axis][1], -R[c.axis][2]);  // world_down (body)
    return {c.sign[0] * v[c.perm[0]], c.sign[1] * v[c.perm[1]], c.sign[2] * v[c.perm[2]]};
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    dtrack::RecordedFrameSource video(args.prefix + ".mp4");
    if (!video.is_open()) { std::cerr << "HATA: video acilamadi\n"; return 1; }
    dtrack::TelemetryLog telem;
    if (!telem.load(args.prefix + ".telemetry.csv")) {
        std::cerr << "HATA: telemetri yuklenemedi\n"; return 1;
    }
    const double fps = video.fps();

    // --- 1) NET ufuklu kareleri topla: (telemetri, görüntü-ufku doğrusu) ----------
    struct Sample { dtrack::Telemetry tm; Line img; int id; };
    std::vector<Sample> samples;
    int W = 0, H = 0, sampled = 0;
    dtrack::Frame frame;
    while (video.next(frame)) {
        if (frame.id >= args.max_frames) break;
        if (frame.id % args.step != 0) continue;
        ++sampled;
        W = frame.image.cols; H = frame.image.rows;
        const Line li = image_horizon(frame.image);
        if (!li.valid) continue;
        const double t = static_cast<double>(frame.id) / fps;
        samples.push_back({telem.at(t), li, (int)frame.id});
    }
    std::printf("Ornek kare: %d, net-ufuklu: %zu  (cozunurluk %dx%d)\n",
                sampled, samples.size(), W, H);
    if (samples.size() < 5) { std::cerr << "Yetersiz ornek.\n"; return 1; }

    const double cx = W * 0.5, cy = H * 0.5;
    const double f = (W * 0.5) / std::tan(0.5 * args.fov_deg * CV_PI / 180.0);
    const double us[3] = {W * 0.25, W * 0.5, W * 0.75};  // karşılaştırma noktaları

    // --- 2) 144 konvansiyonu puanla (ortalama |Δv| piksel) ------------------------
    struct Score { Conv c; double err; int used; };
    std::vector<Score> scores;
    for (const Conv& c : all_convs()) {
        double sum = 0; int used = 0;
        for (const Sample& s : samples) {
            const cv::Vec3d g = g_cam_of(c, s.tm);
            if (std::fabs(g[1]) < 1e-3) continue;        // ~dikey → ufuk tanımsız
            double e = 0;
            for (double u : us) {
                const double v_tel = cy - (g[0] * (u - cx) + g[2] * f) / g[1];
                const double v_img = s.img.m * u + s.img.c;
                e += std::fabs(v_tel - v_img);
            }
            sum += e / 3.0; ++used;
        }
        if (used >= (int)samples.size() / 2)
            scores.push_back({c, sum / std::max(1, used), used});
    }
    std::sort(scores.begin(), scores.end(),
              [](const Score& a, const Score& b) { return a.err < b.err; });

    // --- 3) En iyi konvansiyonları yazdır -----------------------------------------
    std::printf("\nEn iyi konvansiyonlar (ort. |Δv| piksel; küçük = iyi):\n");
    std::printf("%-5s %-8s %-14s %-14s %-8s\n", "sira", "axis", "perm", "sign", "err_px");
    for (int i = 0; i < std::min<int>(8, scores.size()); ++i) {
        const Score& s = scores[i];
        std::printf("%-5d %-8d [%d %d %d]        (%+d %+d %+d)    %.1f\n",
                    i + 1, s.c.axis, s.c.perm[0], s.c.perm[1], s.c.perm[2],
                    s.c.sign[0], s.c.sign[1], s.c.sign[2], s.err);
    }
    const Score& best = scores.front();
    std::printf("\nSECILEN: axis=%d perm={%d,%d,%d} sign={%+d,%+d,%+d}  hata=%.1f px\n",
                best.c.axis, best.c.perm[0], best.c.perm[1], best.c.perm[2],
                best.c.sign[0], best.c.sign[1], best.c.sign[2], best.err);
    return 0;
}
