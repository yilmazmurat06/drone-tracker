// ============================================================================
//  dtrack_calibrate — P1 (3b-ii): gyro→kamera eksen eşlemesi (AxisMap) + FOV
//  kalibrasyonu.
//
//  PROBLEM: GyroFlowStabilizer'ın gövde gyro'sunu (pitch/roll/yaw) kamera dönme
//  eksenlerine hangi SIRA ve İŞARETLE eşleyeceğini bilmiyoruz; ayrıca FOV (odak)
//  belirsiz. Yanlış eşleme → gyro kareyi ters yöne warp eder (--force-gyro'da
//  gördüğümüz bozulma).
//
//  ÇÖZÜM — optik akışı "doğru cevap" (ground truth) olarak kullan:
//    Dokulu karelerde OF güvenilir bir cur→prev homografisi H_of verir.
//    1) H_of'tan kamera dönme vektörünü çıkar:  R ≈ K⁻¹·H_of·K  (en yakın
//       dönmeye SVD ile izdüşür) → t = Rodrigues(R)   [HEDEF, kamera çerçevesi]
//    2) Aynı kare aralığında ham gövde gyro'sunu integre et → ω_body  [GİRDİ]
//    3) 48 işaretli-permütasyon (6 sıra × 2³ işaret) × bir FOV ızgarası tara;
//       Σ‖M·ω_body − t‖² artığını en küçük yapan (M, FOV) çiftini seç.
//    M → AxisMap (src[k]=sütun, sign[k]=işaret).
//
//  ÇAPRAZ-DOĞRULAMA: att_* quaternion (drift'siz) bağımsız bir referans. Gyro'yu
//  integre edince elde edilen dönme AÇISI, quaternion'dan gelen açıya eşit olmalı
//  (eksen eşlemesinden BAĞIMSIZ → gyro'nun birim/integrasyonunu doğrular).
//
//  Kullanım:
//    dtrack_calibrate [prefix] [--max-frames N] [--min-inliers K] [--motion-deg D]
// ============================================================================
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>   // Rodrigues, SVD
#include <opencv2/core.hpp>

#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"
#include "dtrack/stabilization/optical_flow_stabilizer.hpp"

namespace {

struct Args {
    std::string prefix = "data/flight_01_084727";
    int   max_frames = 1500;
    int   min_inliers = 40;     // OF bu kadar inlier vermezse kareyi atla
    double motion_deg = 0.2;    // bu açıdan küçük dönmeler gürültü → atla
};

Args parse_args(int argc, char** argv) {
    Args a; bool pset = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--max-frames" && i + 1 < argc) a.max_frames = std::stoi(argv[++i]);
        else if (s == "--min-inliers" && i + 1 < argc) a.min_inliers = std::stoi(argv[++i]);
        else if (s == "--motion-deg" && i + 1 < argc) a.motion_deg = std::stod(argv[++i]);
        else if (!s.empty() && s[0] != '-' && !pset) { a.prefix = s; pset = true; }
    }
    return a;
}

// Bir kareye yakın gövde gyro'sunun [prev_t, t] aralığındaki trapez integrali
// (ham, AxisMap UYGULANMADAN). → ω_body dönme vektörü (rad).
cv::Vec3d integrate_body(const dtrack::Telemetry& a, const dtrack::Telemetry& b) {
    const double dt = b.t_rel - a.t_rel;
    if (dt <= 0) return {0, 0, 0};
    const double d2r = CV_PI / 180.0;  // gyro_* DERECE/s → rad/s
    return {0.5 * (a.gyro_pitch + b.gyro_pitch) * dt * d2r,
            0.5 * (a.gyro_roll + b.gyro_roll) * dt * d2r,
            0.5 * (a.gyro_yaw + b.gyro_yaw) * dt * d2r};
}

// 3x3 matrisi en yakın dönmeye izdüşür (orthogonal Procrustes): R = U·Vᵀ.
cv::Matx33d nearest_rotation(const cv::Matx33d& M) {
    cv::Mat w, u, vt;
    cv::SVD::compute(cv::Mat(M), w, u, vt);
    cv::Mat R = u * vt;
    if (cv::determinant(R) < 0) {            // yansımayı (det=-1) engelle
        u.col(2) *= -1;
        R = u * vt;
    }
    return cv::Matx33d(R.ptr<double>());
}

cv::Vec3d rodrigues_vec(const cv::Matx33d& R) {
    cv::Mat rv;
    cv::Rodrigues(cv::Mat(R), rv);
    return {rv.at<double>(0), rv.at<double>(1), rv.at<double>(2)};
}

// İki birim quaternion (x,y,z,w) arası bağıl dönmenin AÇISI (rad).
// q_rel = q0⁻¹ ⊗ q1 ; açı = 2·acos(|w_rel|).
double quat_rel_angle(const dtrack::Telemetry& a, const dtrack::Telemetry& b) {
    // q0⁻¹ = konjuge (birim varsayımı): (-x,-y,-z, w)
    const double x0 = -a.att_x, y0 = -a.att_y, z0 = -a.att_z, w0 = a.att_w;
    const double x1 = b.att_x, y1 = b.att_y, z1 = b.att_z, w1 = b.att_w;
    // Hamilton çarpımı q_rel = q0⁻¹ ⊗ q1, yalnız w bileşeni yeterli:
    const double w_rel = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
    return 2.0 * std::acos(std::min(1.0, std::fabs(w_rel)));
}

// 48 işaretli permütasyon: 6 sıra-permütasyonu × 2³ işaret.
struct Candidate { std::array<int, 3> src; std::array<int, 3> sign; };

std::vector<Candidate> all_candidates() {
    const int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    std::vector<Candidate> out;
    for (auto& p : perms)
        for (int s = 0; s < 8; ++s)
            out.push_back({{p[0], p[1], p[2]},
                           {(s & 1) ? 1 : -1, (s & 2) ? 1 : -1, (s & 4) ? 1 : -1}});
    return out;
}

cv::Vec3d apply_candidate(const Candidate& c, const cv::Vec3d& omega) {
    return {c.sign[0] * omega[c.src[0]],
            c.sign[1] * omega[c.src[1]],
            c.sign[2] * omega[c.src[2]]};
}

const char* axis_name(int i) { return i == 0 ? "pitch" : i == 1 ? "roll" : "yaw"; }

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
    dtrack::OpticalFlowStabilizer of;
    dtrack::GyroFlowStabilizer geo;  // yalnız intrinsics(K) yardımcısı için

    // --- Örnek topla: OF güvenilir + belirgin hareketli kareler ---------------
    std::vector<cv::Vec3d>   omegas;   // ham gövde integralleri
    std::vector<cv::Matx33d> hofs;     // OF homografileri (cur→prev)
    std::vector<double>      gyro_ang, quat_ang;  // çapraz-doğrulama açıları
    cv::Size img_size;

    dtrack::Frame frame, warped;
    cv::Matx33f H;
    double prev_t = 0; bool have_prev = false;
    int seen = 0;

    while (video.next(frame) && seen < args.max_frames) {
        ++seen;
        const double t = static_cast<double>(frame.id) / fps;
        const bool ok = of.stabilize(frame, {}, warped, H);

        if (have_prev && ok && of.last_inliers() >= args.min_inliers) {
            const dtrack::Telemetry a = telem.at(prev_t), b = telem.at(t);
            const cv::Vec3d omega = integrate_body(a, b);
            const cv::Matx33d Hof(H(0,0),H(0,1),H(0,2),
                                  H(1,0),H(1,1),H(1,2),
                                  H(2,0),H(2,1),H(2,2));
            omegas.push_back(omega);
            hofs.push_back(Hof);
            gyro_ang.push_back(cv::norm(omega));         // gyro dönme açısı (rad)
            quat_ang.push_back(quat_rel_angle(a, b));    // quaternion açısı (rad)
            img_size = frame.image.size();
        }
        prev_t = t; have_prev = true;
    }

    if (omegas.size() < 20) {
        std::cerr << "HATA: yeterli ornek yok (" << omegas.size() << ")\n"; return 1;
    }

    // --- Çapraz-doğrulama: gyro açısı vs quaternion açısı --------------------
    double sg = 0, sq = 0, sgq = 0, sgg = 0;
    for (size_t i = 0; i < gyro_ang.size(); ++i) {
        sg += gyro_ang[i]; sq += quat_ang[i];
        sgq += gyro_ang[i] * quat_ang[i]; sgg += gyro_ang[i] * gyro_ang[i];
    }
    const double scale = sgg > 1e-12 ? sgq / sgg : 0.0;  // quat ≈ scale·gyro
    std::printf("== Cross-check (gyro vs quaternion donme acisi) ==\n");
    std::printf("  ort gyro=%.4f rad  ort quat=%.4f rad  oran(quat/gyro)=%.3f\n",
                sg / gyro_ang.size(), sq / quat_ang.size(), scale);
    std::printf("  (oran ~1 ise gyro birim/integrasyon dogru)\n\n");

    // --- FOV × 48 aday taraması ---------------------------------------------
    const auto cands = all_candidates();
    const double deg2rad = CV_PI / 180.0;
    const double motion_min = args.motion_deg * deg2rad;

    double best_sse = 1e300; int best_ci = -1; double best_fov = 0;
    // kimlik (kalibrasyonsuz) referans artığı:
    double ident_sse = 0; bool ident_done = false;

    for (double fov = 60; fov <= 150; fov += 2.0) {
        const cv::Matx33f Kf = geo.intrinsics(img_size, fov);
        const cv::Matx33d K(Kf(0,0),Kf(0,1),Kf(0,2), Kf(1,0),Kf(1,1),Kf(1,2),
                            Kf(2,0),Kf(2,1),Kf(2,2));
        const cv::Matx33d Ki = K.inv();

        // Bu FOV için hedef vektörleri t_i'yi önceden hesapla.
        std::vector<cv::Vec3d> targets; targets.reserve(hofs.size());
        std::vector<int> used; used.reserve(hofs.size());
        for (size_t i = 0; i < hofs.size(); ++i) {
            const cv::Matx33d R = nearest_rotation(Ki * hofs[i] * K);
            const cv::Vec3d t = rodrigues_vec(R);
            if (cv::norm(t) < motion_min) continue;   // gürültülü/durağan kareyi at
            targets.push_back(t);
            used.push_back(static_cast<int>(i));
        }
        if (targets.size() < 20) continue;

        for (size_t ci = 0; ci < cands.size(); ++ci) {
            double sse = 0;
            for (size_t j = 0; j < targets.size(); ++j) {
                const cv::Vec3d pred = apply_candidate(cands[ci], omegas[used[j]]);
                sse += cv::norm(pred - targets[j], cv::NORM_L2SQR);
            }
            if (sse < best_sse) { best_sse = sse; best_ci = static_cast<int>(ci); best_fov = fov; }
            // kimlik adayını (src=0,1,2 sign=+,+,+) FOV=120'de referans al:
            if (!ident_done && std::abs(fov - 120) < 1e-6 &&
                cands[ci].src == std::array<int,3>{0,1,2} &&
                cands[ci].sign == std::array<int,3>{1,1,1}) {
                ident_sse = sse; ident_done = true;
            }
        }
    }

    if (best_ci < 0) { std::cerr << "HATA: kalibrasyon basarisiz\n"; return 1; }

    // Kazananın bu FOV'daki örnek sayısıyla RMS açı hatasını ver.
    const Candidate& bc = cands[best_ci];
    // En iyi FOV'da kullanılan örnek sayısını yeniden say (RMS için):
    int n_best = 0; {
        const cv::Matx33f Kf = geo.intrinsics(img_size, best_fov);
        const cv::Matx33d K(Kf(0,0),Kf(0,1),Kf(0,2), Kf(1,0),Kf(1,1),Kf(1,2),
                            Kf(2,0),Kf(2,1),Kf(2,2));
        const cv::Matx33d Ki = K.inv();
        for (size_t i = 0; i < hofs.size(); ++i)
            if (cv::norm(rodrigues_vec(nearest_rotation(Ki * hofs[i] * K))) >= motion_min) ++n_best;
    }
    const double rms_deg = std::sqrt(best_sse / std::max(1, n_best)) / deg2rad;
    const double ident_rms = ident_done ? std::sqrt(ident_sse / std::max(1, n_best)) / deg2rad : -1;

    std::printf("== Kalibrasyon sonucu (%zu ornek) ==\n", omegas.size());
    std::printf("  En iyi FOV (yatay) : %.0f derece\n", best_fov);
    std::printf("  AxisMap:\n");
    for (int k = 0; k < 3; ++k)
        std::printf("    cam_%c <- %+d * %-5s   (src[%d]=%d, sign[%d]=%+d)\n",
                    "xyz"[k], bc.sign[k], axis_name(bc.src[k]),
                    k, bc.src[k], k, bc.sign[k]);
    std::printf("  RMS aci hatasi     : %.3f derece/kare", rms_deg);
    if (ident_rms >= 0) std::printf("   (kimlik/kalibrasyonsuz: %.3f)", ident_rms);
    std::printf("\n\n");

    std::printf("Params'a yaz:\n");
    std::printf("  p.fov_h_deg = %.0f;\n", best_fov);
    for (int k = 0; k < 3; ++k)
        std::printf("  p.axes.src[%d] = %d;  p.axes.sign[%d] = %+d;\n",
                    k, bc.src[k], k, bc.sign[k]);
    return 0;
}
