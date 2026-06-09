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
//                 [--speed S] [--show-tentative] [--roi x,y,w,h] [--lock FRAC]
//
//  --lock FRAC : manuel kilit modu. Karenin en-boy oranıyla orantılı, ortalı bir
//    nişangah kutusu (genişlik/yükseklik × FRAC). Arama yalnız bu kutuda yapılır →
//    pilot hedefi merkeze alıp kilitler (otomatik cue + rescan kapanır). HUD nişangahı
//    çizilir; içeride confirmed iz varsa "LOCK", yoksa "SEARCH".
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
    float score_thresh  = 0.70f;  // P3: bu eşiğin altındaki adaylar tracker'a gitmez
    float large_admit   = 3000.f; // (#12) bbox alanı (px²) bundan büyük bloblar skordan muaf (zeplin gibi)
    // Manuel kilit (boresight) ROI: karenin EN-BOY ORANIYLA orantılı, ortalı arama kutusu.
    // Pilot hedefi merkeze alır → sistem yalnız bu kutudakine kilitlenir (manuel cue).
    // 0=kapalı. Örn. 0.4 → genişliğin ve yüksekliğin %40'ı (oran korunur).
    // VARSAYILAN AÇIK (0.5): nişangah/lock çerçevesi kalıcı (kullanıcı isteği).
    // Kapatmak için --lock 0.
    float lock_frac     = 0.5f;
    // Lock modunda P3 şekil-eşiği: kutu göğe nişanlandığı için YÜKSEK RECALL → tüm
    // sky-gate'li adayları (her boy/şekilde hava aracı) kabul et. 0=hepsini al.
    float lock_score    = 0.f;
    // (#14) Lock'ta doku kapısı: halkanın asgari pürüzsüz oranı. <0 → detektör varsayılanı (0.65).
    float lock_smooth   = -1.f;
    // (#14) Ufuk doğrulama: --draw-horizon ile telemetri ufkunu çiz; --hconv N konvansiyon.
    bool  draw_horizon  = false;
    int   hconv         = 0;
    // (#14) Ufuk KAPISI: ufuk altı (zemin) adayları ele. VARSAYILAN KAPALI (opt-in
    // --horizon-gate): kalibre ufuk şu an ~60px hatalı ve kararsız → gate güvenilmez.
    // Düzgün çalışması FOV/distorsiyon/offset kalibrasyonu ister (gelecek iş). 0/1.
    int   horizon_gate  = 0;
    int   horizon_margin = 90;
    // Tracker parametreleri (CLI'dan ayarlanabilir; varsayılanlar MultiTargetTracker::Params ile eşleşmeli)
    double gate_dist    = 25.0;
    int    confirm_hits = 5;
    double min_travel   = 25.0;
    // Kapalı-döngü cue
    bool   closed_loop  = true;   // confirmed izlerden ROI geri besle (--no-cue ile kapat)
    int    cue_margin   = 200;    // ROI padding (px): confirmed hedefin etrafındaki arama alanı
    int    rescan_every = 5;      // (#11) her N karede bir ROI'yi aç → tam-kare yeniden tara.
                                  // Cue confirmed (çoğu kez zemin) ize kilitlenip ROI'yi daraltır;
                                  // periyodik tam-kare tarama gökteki gerçek hedefi (zeplin) bulur,
                                  // #12 fix'leri sayesinde coasting'le tutunup confirmed olur.
                                  // ÖNEMLİ: rescan_every ≤ max_misses(5) olmalı; aksi halde iz iki
                                  // rescan arasında coasting'de ölür, hits biriktiremez (8 çalışmaz).
                                  // 0=kapalı. Küçük=daha çevik ama daha çok paralaks FP.
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
        else if (s == "--score-thresh"  && i + 1 < argc) a.score_thresh  = std::stof(argv[++i]);
        else if (s == "--large-admit"   && i + 1 < argc) a.large_admit   = std::stof(argv[++i]);
        else if (s == "--lock"          && i + 1 < argc) a.lock_frac     = std::stof(argv[++i]);
        else if (s == "--lock-score"    && i + 1 < argc) a.lock_score    = std::stof(argv[++i]);
        else if (s == "--lock-smooth"   && i + 1 < argc) a.lock_smooth   = std::stof(argv[++i]);
        else if (s == "--draw-horizon")                  a.draw_horizon  = true;
        else if (s == "--hconv"         && i + 1 < argc) a.hconv         = std::stoi(argv[++i]);
        else if (s == "--horizon-gate")                  a.horizon_gate  = 1;
        else if (s == "--no-horizon")                    a.horizon_gate  = 0;
        else if (s == "--horizon-margin" && i + 1 < argc) a.horizon_margin = std::stoi(argv[++i]);
        else if (s == "--gate-dist"     && i + 1 < argc) a.gate_dist     = std::stod(argv[++i]);
        else if (s == "--confirm-hits"  && i + 1 < argc) a.confirm_hits  = std::stoi(argv[++i]);
        else if (s == "--min-travel"    && i + 1 < argc) a.min_travel    = std::stod(argv[++i]);
        else if (s == "--no-cue")                        a.closed_loop   = false;
        else if (s == "--cue-margin"    && i + 1 < argc) a.cue_margin    = std::stoi(argv[++i]);
        else if (s == "--rescan-every"  && i + 1 < argc) a.rescan_every  = std::stoi(argv[++i]);
        else if (!s.empty() && s[0] != '-' && !pset) { a.prefix = s; pset = true; }
    }
    return a;
}

// attitude quaternion'dan KAMERA çerçevesinde yerçekimi yönü (gx,gy,gz).
// hconv: konvansiyon seçimi (doğrulama). 0..5 → world_down = ±e_axis (R^T·e = R'nin satırı).
cv::Vec3d gravity_cam(const dtrack::Telemetry& t, int hconv) {
    const double x = t.att_x, y = t.att_y, z = t.att_z, w = t.att_w;
    const double R[3][3] = {
        {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)},
        {2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
        {2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)}};
    // KALİBRE konvansiyon (dtrack_horizon_calib): axis=1(Y), perm={2,1,0}, sign={+,+,−}.
    // world_down BODY = −R[1] satırı; g_cam = işaretli-permütasyon.
    const cv::Vec3d v(-R[1][0], -R[1][1], -R[1][2]);
    (void)hconv;
    return {+v[2], +v[1], -v[0]};
}

// Ufuk çizgisini çiz (DOĞRULAMA — henüz kapı değil). g·(u−cx, v−cy, f)=0.
void draw_horizon(cv::Mat& vis, const cv::Vec3d& g, float fov_deg, int hconv) {
    const int W = vis.cols, H = vis.rows;
    const double cx = W * 0.5, cy = H * 0.5;
    const double f = (W * 0.5) / std::tan(0.5 * fov_deg * CV_PI / 180.0);
    if (std::fabs(g[1]) < 1e-6) return;        // dikey çizgi → atla
    auto v_at = [&](double u) { return cy - (g[0] * (u - cx) + g[2] * f) / g[1]; };
    const cv::Point p0(0, cvRound(v_at(0))), p1(W - 1, cvRound(v_at(W - 1)));
    cv::line(vis, p0, p1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);   // kırmızı ufuk
    cv::putText(vis, "HORIZON hconv=" + std::to_string(hconv), {20, H - 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
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
    if (args.lock_frac > 0.f && args.lock_smooth >= 0.f) {
        // (#14, OPT-IN) Doku kapısı: çevresi dokulu (zemin) adayları ele. Recall'ı
        // feda edebilir (bulut önündeki büyük uçak da elenebilir) → varsayılan KAPALI;
        // yalnız --lock-smooth verilince açılır. Asıl precision = telemetri ufku (#14).
        det_p.texture_gate = true;
        det_p.texture_smooth_min = args.lock_smooth;
    }
    dtrack::MovingTargetDetector det(det_p);
    dtrack::ClutterDiscriminator disc;
    dtrack::MultiTargetTracker::Params trk_p;
    trk_p.gate_dist    = args.gate_dist;
    trk_p.confirm_hits = args.confirm_hits;
    trk_p.min_travel   = args.min_travel;
    dtrack::MultiTargetTracker trk(trk_p);
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
    cv::Rect last_cue;  // görselleştirme için son aktif ROI

    dtrack::Frame frame, warped;
    while (video.next(frame)) {
        const double t = static_cast<double>(frame.id) / fps;
        std::vector<dtrack::Telemetry> window;
        if (have_telem && have_prev) window = {telem.at(prev_t), telem.at(t)};

        cv::Matx33f H;
        stab.stabilize(frame, window, warped, H);

        // Manuel kilit (boresight) ROI: karenin en-boy oranıyla orantılı, ortalı kutu.
        // Aramayı bu kutuya sabitler → pilot hedefi merkeze alıp kilitler (otomatik cue
        // ve rescan devre dışı; aşağıdaki cue bloğu lock_frac>0 iken atlanır).
        cv::Rect lock_roi;
        if (args.lock_frac > 0.f) {
            const cv::Size fsz = warped.image.size();
            const int lw = cvRound(fsz.width  * args.lock_frac);
            const int lh = cvRound(fsz.height * args.lock_frac);
            lock_roi = cv::Rect((fsz.width - lw) / 2, (fsz.height - lh) / 2, lw, lh);
            det.set_roi(lock_roi);
        }

        std::vector<dtrack::Detection> dets;
        det.detect(warped, dets);

        // (#14) Ufuk kapısı: telemetri ufkunun ALTINDAKİ (zemin) adayları ele.
        // Pürüzsüz uzak zemin/haze'i de keser (renk/doku elemiyordu — geometrik ayraç).
        // Lock'ta otomatik açık. Kalibre ufuk ~60px hatalı → marj (horizon_margin) ver.
        const bool use_hgate = (args.horizon_gate == 1);   // opt-in (#14 kalibrasyon WIP)
        if (use_hgate && have_telem) {
            const cv::Vec3d g = gravity_cam(telem.at(t), 0);
            const cv::Size fsz = warped.image.size();
            const double cx = fsz.width * 0.5, cy = fsz.height * 0.5;
            const double f = (fsz.width * 0.5) / std::tan(0.5 * 128.0 * CV_PI / 180.0);
            if (std::fabs(g[1]) > 1e-6) {
                dets.erase(std::remove_if(dets.begin(), dets.end(),
                    [&](const dtrack::Detection& d) {
                        const double vh = cy - (g[0] * (d.centroid.x - cx) + g[2] * f) / g[1];
                        return d.centroid.y > vh + args.horizon_margin;  // ufuk altı = zemin
                    }), dets.end());
            }
        }

        // P3: her adaya skor ver, eşiğin altını ele (clutter reddi).
        // (#12) İSTİSNA: alanı large_admit'ten BÜYÜK bloblar şekil-skorundan MUAF tutulur.
        // NEDEN: P3 küçük kompakt drone'a göre ayarlı; büyük+uzun zeplin düşük skor alır
        // (compactness tavanı). Ring-gate (#13) zemini zaten eledi → büyük in-sky blob
        // nadir ve büyük olasılıkla gerçek hedef. Kararı tracker'ın zamansal tutarlılığına
        // (M-of-N + vel-consistency) bırak; clutter ise titreyeceği için orada elenir.
        for (auto& d : dets) d.score = disc.score(d);
        // Lock modunda YÜKSEK RECALL: kutu operatör tarafından göğe nişanlandığı için
        // P3 şekil-eşiğini düşür (lock_score, vars. 0 = tümünü al) → her boy/şekilde hava
        // aracı (küçük planör, büyük FPV dron, zeplin) yakalanır. Clutter'ı ring-gate
        // (zemin) + tracker zamansal tutarlılığı (titreyen bulut benekleri) eler.
        const float eff_thresh = args.lock_frac > 0.f ? args.lock_score : args.score_thresh;
        // NOT: ölçüt d.area DEĞİL d.bbox alanı: top-hat yalnız koyu gondol/afiş
        // piksellerini sayar (area küçük), ama silüet kutusu (expand_to_silhouette)
        // tüm zeplini kaplar → büyük cismi bbox alanı yakalar.
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                       [&](const dtrack::Detection& d){
                           return d.score < eff_thresh &&
                                  d.bbox.area() < args.large_admit;
                       }),
                   dets.end());

        std::vector<dtrack::Track> tracks;
        trk.update(dets, tracks);

        // Kapalı-döngü cue: confirmed izlerin bir-kare-sonraki tahmin konumlarından
        // ROI oluştur, detektöre geri besle. Confirmed iz yoksa → tam kare (fallback).
        // NEDEN: ROI daraldıkça paralaks yanlış-pozitiflerin büyük çoğunluğu dışarıda kalır.
        if (args.closed_loop && args.roi.empty() && args.lock_frac <= 0.f) {
            cv::Rect cue;
            const cv::Size fsz = warped.image.size();
            const cv::Rect frame_rect(0, 0, fsz.width, fsz.height);
            const int m = args.cue_margin;
            for (const auto& tr : tracks) {
                if (tr.status != dtrack::Track::Status::Confirmed) continue;
                // Bir kare ilerisi tahmini: pos + vel (Kalman'ın ilerlemesi)
                const cv::Point2f next = tr.pos + tr.vel;
                cv::Rect r(cvRound(next.x) - m, cvRound(next.y) - m, 2 * m, 2 * m);
                r &= frame_rect;                   // kare sınırına kırp
                if (r.area() > 0)
                    cue = cue.area() == 0 ? r : (cue | r);
            }
            if (cue.area() > 0) { det.set_roi(cue); last_cue = cue; }
            else                 { det.clear_roi(); last_cue = cv::Rect(); }

            // (#11) Periyodik tam-kare yeniden tarama (OPSİYONEL, --rescan-every ile aç):
            // cue paralaks gibi YANLIŞ bir hedefe kilitlenip ROI'yi oraya daraltmış olabilir
            // → gökteki gerçek hedef hiç aranmaz. Her rescan_every karede bir ROI'yi aç.
            // NOT: Tek başına confirmed kazandırmaz (bkz. #12 tespit tutarsızlığı); rescan'da
            // görülen hedef diğer karelerde yine ROI dışında kaldığı için M-of-N dolmaz.
            // Bu yüzden varsayılan KAPALI; #12 çözülünce anlamlı olacak.
            if (args.rescan_every > 0 && (shown % args.rescan_every == 0)) {
                det.clear_roi(); last_cue = cv::Rect();
            }
        }

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
            // (#14) Ufuk doğrulama çizimi (telemetri attitude → ufuk çizgisi).
            if (args.draw_horizon && have_telem)
                draw_horizon(vis, gravity_cam(telem.at(t), args.hconv), 128.f, args.hconv);
            // Aktif cue ROI: soluk mavi dikdörtgen
            if (args.closed_loop && args.lock_frac <= 0.f && last_cue.area() > 0)
                cv::rectangle(vis, last_cue, cv::Scalar(180, 80, 0), 1);

            // Manuel kilit nişangahı (HUD): köşe braketleri + merkez artı.
            // İçeride confirmed iz varsa "LOCK" (sarı), yoksa "SEARCH" (beyaz).
            if (args.lock_frac > 0.f && lock_roi.area() > 0) {
                const bool locked = n_conf > 0;
                const cv::Scalar col = locked ? cv::Scalar(0, 255, 0) : cv::Scalar(230, 230, 230);
                const int L = std::max(14, lock_roi.width / 10);  // köşe çentik uzunluğu
                const cv::Point tl = lock_roi.tl(), br = lock_roi.br();
                const cv::Point tr(br.x, tl.y), bl(tl.x, br.y);
                cv::line(vis, tl, tl + cv::Point(L, 0), col, 2); cv::line(vis, tl, tl + cv::Point(0, L), col, 2);
                cv::line(vis, tr, tr + cv::Point(-L, 0), col, 2); cv::line(vis, tr, tr + cv::Point(0, L), col, 2);
                cv::line(vis, bl, bl + cv::Point(L, 0), col, 2); cv::line(vis, bl, bl + cv::Point(0, -L), col, 2);
                cv::line(vis, br, br + cv::Point(-L, 0), col, 2); cv::line(vis, br, br + cv::Point(0, -L), col, 2);
                const cv::Point ctr((tl.x + br.x) / 2, (tl.y + br.y) / 2);
                cv::line(vis, ctr - cv::Point(12, 0), ctr + cv::Point(12, 0), col, 1);
                cv::line(vis, ctr - cv::Point(0, 12), ctr + cv::Point(0, 12), col, 1);
                cv::putText(vis, locked ? "LOCK" : "SEARCH", {tl.x, tl.y - 10},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv::LINE_AA);
            }
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
