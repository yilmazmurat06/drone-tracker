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
//  --lock FRAC : manuel kilit modu (VARSAYILAN AÇIK). Karenin en-boy oranıyla
//    orantılı, ortalı bir nişangah kutusu. Arama yalnız bu kutuda yapılır.
//
//  GÜDÜM KATMANI (lock modunda): detector OTOMATİK KİLİTLEMEZ — yalnız PİLOTA
//  numaralı ADAY sunar. Pilot bir drone'a kilitlenir → tek-hedef görsel tracker
//  (Siamese sınıfı; şimdilik NCC stub) KESİNTİSİZ sürer. Takip güveni eşik altına
//  düşerse sistem şüpheye düşüp detection'a dönerek hedefi YENİDEN DOĞRULAR.
//    Klavye: '1'..'9' = adayı kilitle,  'u' = kilidi bırak,  'q'/ESC = çık.
//    Fare    : tıklanan en yakın adaya kilitlen.
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtrack/detection/clutter_discriminator.hpp"
#include "dtrack/detection/moving_target_detector.hpp"
#include "dtrack/guidance/guidance_controller.hpp"
#include "dtrack/guidance/nano_siamese_tracker.hpp"
#include "dtrack/guidance/ncc_template_tracker.hpp"
#include "dtrack/guidance/yolo_roi_tracker.hpp"
#include "dtrack/io/recorded_frame_source.hpp"
#include "dtrack/io/telemetry_log.hpp"
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"
#include "dtrack/tracking/multi_target_tracker.hpp"

namespace {

struct Args {
    std::string prefix = "data/flight_01_084727";
    std::string save_path;
    bool dump = false;
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
    // FAZ 1 spike: kilit-sonrası tek-hedef tracker seçimi.
    //   varsayılan = NCC stub. --siamese → hazır NanoTrack (cv::TrackerNano) adaptörü.
    bool   use_siamese  = false;
    std::string nano_backbone = "models/nanotrack_backbone_sim.onnx";
    std::string nano_head     = "models/nanotrack_head_sim.onnx";
    // FAZ 4: --yolo → eğitilmiş YOLO-in-ROI tracker (tracking-by-detection).
    // Güveni "drone-luk" ölçer (NanoTrack'in doygun benzerlik skorunun aksine).
    bool   use_yolo     = false;
    std::string yolo_model    = "models/yolo11_drone.onnx";
    // Headless A/B değerlendirmesi için "pilot" simülasyonu: bu kareden itibaren,
    // SEARCH'te aday varsa nişangah MERKEZİNE en yakın adayı otomatik kilitle.
    // -1 = kapalı (GUI'de manuel seçim). dump modunda spike ölçümü için kullanılır.
    int    auto_lock_at = -1;
    // Kontrollü deney: auto-lock merkeze değil bu piksele en yakın adayı kilitler
    // (örn. kasıtlı YER kilidi atıp sky-ekseninin ateşlediğini kanıtlamak için).
    cv::Point auto_lock_px{-1, -1};
    int    start_frame  = 0;   // video bu kareden başlat (seek)
    // LOCK-INTEGRITY: kilit-sonrası geometrik bütünlük bekçisi (sky-ring+boyut+hareket).
    // Varsayılan AÇIK. --no-integrity ile kapat (A/B ölçümü; yalnız-güven eski davranış).
    int    use_integrity = 1;
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
        else if (s == "--dump") {
            // --dump          → boolean (tüm kareler)
            // --dump N        → eski compat: her N karede bir (≤0 ise tüm kareler)
            a.dump = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                a.dump_n = std::stoi(argv[++i]);
            }
        }
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
        else if (s == "--siamese")                       a.use_siamese   = true;
        else if (s == "--nano-backbone" && i + 1 < argc) a.nano_backbone = argv[++i];
        else if (s == "--nano-head"     && i + 1 < argc) a.nano_head     = argv[++i];
        else if (s == "--yolo")                          a.use_yolo      = true;
        else if (s == "--yolo-model"    && i + 1 < argc) { a.use_yolo = true; a.yolo_model = argv[++i]; }
        else if (s == "--auto-lock"     && i + 1 < argc) a.auto_lock_at  = std::stoi(argv[++i]);
        else if (s == "--auto-lock-px"  && i + 1 < argc) {
            int x, y;
            if (std::sscanf(argv[++i], "%d,%d", &x, &y) == 2) a.auto_lock_px = {x, y};
        }
        else if (s == "--no-integrity")                  a.use_integrity = 0;
        else if (s == "--start"         && i + 1 < argc) a.start_frame   = std::stoi(argv[++i]);
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

// Fare seçim durumu: en son sunulan adaylar + güdüm denetleyicisi. Tıklamada
// (sol tuş) tıklanan noktaya en yakın aday merkezi kilitlenir. setMouseCallback
// userdata olarak geçer. scale = görüntü, ekrana >1920 ise 0.5 küçültülmüş olabilir.
struct MouseState {
    dtrack::GuidanceController* ctrl = nullptr;
    std::vector<dtrack::Detection> cands;
    double scale = 1.0;   // ekran→görüntü ölçek düzeltmesi (1/scale ile çarpılır)
};

void on_mouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* ms = static_cast<MouseState*>(userdata);
    if (!ms || !ms->ctrl || ms->cands.empty()) return;
    const float fx = static_cast<float>(x / ms->scale), fy = static_cast<float>(y / ms->scale);
    int best = -1; float best_d2 = 1e18f;
    for (int i = 0; i < static_cast<int>(ms->cands.size()); ++i) {
        const cv::Point2f c = ms->cands[i].centroid;
        const float d2 = (c.x - fx) * (c.x - fx) + (c.y - fy) * (c.y - fy);
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    if (best >= 0) ms->ctrl->select(best);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    dtrack::RecordedFrameSource video(args.prefix + ".mp4");
    if (!video.is_open()) { std::cerr << "HATA: video acilamadi\n"; return 1; }
    if (args.start_frame > 0) video.seek(args.start_frame);
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

    // --- Güdüm katmanı (yalnız lock modunda aktif) ---
    // Tek-hedef tracker NPU-hedefli arayüz; şimdilik NCC CPU stub. Doğrulayıcı =
    // mevcut P3 ClutterDiscriminator (re-acquire'da drone-luk skoru). Pilot kilidi
    // klavye/fare ile verir; otomatik kilit YOK.
    const bool guided = args.lock_frac > 0.f;
    // Tracker seçimi (FAZ 1 spike): --siamese → hazır NanoTrack; yoksa NCC stub.
    // Polimorfik: GuidanceController yalnız ISingleTargetTracker& görür (değişmez).
    std::unique_ptr<dtrack::ISingleTargetTracker> st_tracker;
    if (args.use_yolo) {
        // FAZ 4: eğitilmiş YOLO-in-ROI. Öncelik YOLO'da (--yolo --siamese birlikte
        // verilirse YOLO kazanır); kurulamazsa aşağıdaki zincire (Nano/NCC) düşer.
        try {
            st_tracker = std::make_unique<dtrack::YoloRoiTracker>(
                dtrack::YoloRoiTracker::Params{args.yolo_model});
            std::cout << "tracker: YOLO-in-ROI — " << args.yolo_model << "\n";
        } catch (const std::exception& e) {
            std::cerr << "UYARI: YOLO kurulamadi (" << e.what()
                      << ") → siradaki tracker'a donuluyor\n";
        }
    }
    if (!st_tracker && args.use_siamese) {
        try {
            st_tracker = std::make_unique<dtrack::NanoSiameseTracker>(
                dtrack::NanoSiameseTracker::Params{args.nano_backbone, args.nano_head});
            std::cout << "tracker: NanoTrack (Siamese) — " << args.nano_backbone << "\n";
        } catch (const std::exception& e) {
            std::cerr << "UYARI: NanoTrack kurulamadi (" << e.what()
                      << ") → NCC stub'a donuluyor\n";
        }
    }
    if (!st_tracker) {
        st_tracker = std::make_unique<dtrack::NccTemplateTracker>();
        if (!args.use_siamese) std::cout << "tracker: NCC template (stub)\n";
    }
    dtrack::GuidanceController::Params gp;
    gp.use_integrity = (args.use_integrity != 0);
    if (args.use_yolo && st_tracker) {
        // YOLO güven SEMANTİĞİ NanoTrack'ten farklı: Nano doygun ~0.8 üretir
        // (0.3 altı = gerçek kopma); YOLO ise kare-kare doğal dalgalanır (0.28
        // dipleri normal, ölçüldü) ve TESPİT YOKSA zaten tam 0 döner — ayrım net.
        // Eşikler buna göre: tek-kare dip SUSPECT tetiklemesin; sayaçlı eşik
        // (suspect_conf×3 kare) görevde kalır. verify_score düşürülür: YOLO'nun
        // kendisi her kare appearance doğruladığı için P3 ikincil kapı.
        gp.lost_conf    = 0.05f;
        gp.suspect_conf = 0.35f;
        gp.verify_score = 0.35f;
        // Re-acquire SABRI: detector SUSPECT'te zaten her kare tam görevde (aday
        // üretimi bedava) ve kilit TEK ATIMLIK (düşerse pilot yeniden seçene dek
        // hedef YOK). 8 karelik bütçe, kutu kötü re-seed'le çapadan uzaklaştığında
        // yetmiyor (ölçüldü: hedef ringde, çapa 190px ötede). Sabırlı ol.
        gp.reacquire_frames = 30;
    }
    dtrack::GuidanceController guide(*st_tracker, disc, gp);
    MouseState mouse; mouse.ctrl = &guide;

    const double fps = video.fps();
    const bool dumping = args.dump || args.dump_n > 0;
    const bool saving = !args.save_path.empty();
    const bool live = !dumping && !saving;

    if (dumping) std::printf("%6s %8s %10s %10s\n", "kare", "aday", "tentative", "confirmed");
    if (live) {
        cv::namedWindow("dtrack track", cv::WINDOW_NORMAL);
        if (guided) cv::setMouseCallback("dtrack track", on_mouse, &mouse);
    }

    cv::VideoWriter writer;
    const int delay_ms = std::max(1, static_cast<int>(1000.0 / (fps * args.speed)));

    double prev_t = 0; bool have_prev = false;
    int shown = 0;
    cv::Rect last_cue;  // görselleştirme için son aktif ROI
    // GÜDÜM dar-ROI: kilit sonrası (TRACK/SUSPECT) controller'ın ürettiği hedef-takipli
    // pencere. Cue deseni gibi BİR KARE GECİKMELİ uygulanır (detect, on_frame'den önce
    // koşar): bu kare hesaplanan ROI bir sonraki karenin detector'ına verilir.
    cv::Rect guided_roi;
    bool auto_lock_done = false;  // tek-atımlık pilot simülasyonu (bir kez kilitle)

    dtrack::Frame frame, warped;
    while (video.next(frame)) {
        const double t = static_cast<double>(frame.id) / fps;
        std::vector<dtrack::Telemetry> window;
        if (have_telem && have_prev) window = {telem.at(prev_t), telem.at(t)};

        cv::Matx33f H;
        stab.stabilize(frame, window, warped, H);
        // NOT: ego-hareket telafisi (guide.compensate(H⁻¹)) ÖLÇÜLDÜ ve BAĞLANMADI:
        // OF homografisi YER özelliklerine oturur (gökte köşe yok) → dönme + yer-
        // paralaksı içerir; paralaks GÖK hedefi için sahtedir ve ardışık warped
        // karelerde zaten yaklaşık sadeleşir. Telafi sadeleşen terimi geri enjekte
        // eder (400 kare A/B: TRACK 111→99). Gelecek iş: gyro'dan SAF-DÖNME
        // homografisiyle telafi (paralaks içermez → gökte geçerli).

        // Manuel kilit (boresight) ROI: karenin en-boy oranıyla orantılı, ortalı kutu.
        // Aramayı bu kutuya sabitler → pilot hedefi merkeze alıp kilitler (otomatik cue
        // ve rescan devre dışı; aşağıdaki cue bloğu lock_frac>0 iken atlanır).
        cv::Rect lock_roi;
        if (args.lock_frac > 0.f) {
            const cv::Size fsz = warped.image.size();
            const int lw = cvRound(fsz.width  * args.lock_frac);
            const int lh = cvRound(fsz.height * args.lock_frac);
            lock_roi = cv::Rect((fsz.width - lw) / 2, (fsz.height - lh) / 2, lw, lh);
            // Kilit aktifken (TRACK/SUSPECT) işlemi hedef-takipli DAR pencereye daralt;
            // aksi halde (SEARCH) pilotun aday görmesi için merkezi nişangah kutusu.
            det.set_roi(guided_roi.area() > 0 ? guided_roi : lock_roi);
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
        dtrack::GuidanceController::Out gout;
        if (guided) {
            // GÜDÜM: detector adaylarını pilota sun / tek-hedef takibi sür. Çok-hedef
            // tracker devre dışı (kilit-sonrası görsel takip tek hedefe odaklanır).
            guide.on_frame(warped.image, dets, gout);
            mouse.cands = gout.candidates;   // fare seçimi en son adaylara bakar
            // "Pilot" simülasyonu (headless A/B): kareden itibaren SEARCH'te aday varsa
            // hedef noktaya (vars. nişangah merkezi; --auto-lock-px ile özel piksel) en
            // yakın adayı otomatik kilitle. TEK ATIMLIK: pilot bir kez seçer — kilit
            // sonradan düşerse sim YENİDEN kilitlemez (önceki davranış release sonrası
            // boş-gök gürültüsüne kilitlenip ölçümü kirletiyordu).
            if (args.auto_lock_at >= 0 && !auto_lock_done &&
                (int)frame.id >= args.auto_lock_at &&
                gout.state == dtrack::GuidanceController::State::Search &&
                !gout.candidates.empty()) {
                const cv::Size fsz = warped.image.size();
                const cv::Point2f ctr = (args.auto_lock_px.x >= 0)
                    ? cv::Point2f(args.auto_lock_px)
                    : cv::Point2f(fsz.width * 0.5f, fsz.height * 0.5f);
                int best = 0; float best_d2 = 1e18f;
                for (int i = 0; i < (int)gout.candidates.size(); ++i) {
                    const cv::Point2f c = gout.candidates[i].centroid;
                    const float d2 = (c.x-ctr.x)*(c.x-ctr.x) + (c.y-ctr.y)*(c.y-ctr.y);
                    if (d2 < best_d2) { best_d2 = d2; best = i; }
                }
                guide.select(best);
                auto_lock_done = true;    // tek atımlık: bayrak tüketildi
            }
            // Bir sonraki kare için dar-ROI: TRACK/SUSPECT'te hedef-takipli pencere,
            // SEARCH'te boş (→ merkezi nişangaha geri dön).
            guided_roi = gout.roi;
        } else {
            trk.update(dets, tracks);
        }

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
            if (guided) {
                // GÜDÜM dump: durum + güven + hedef kutusu (spike A/B ölçümü).
                using GState = dtrack::GuidanceController::State;
                const char* st = gout.state == GState::Track ? "TRACK"
                               : gout.state == GState::Suspect ? "SUSPECT" : "SEARCH";
                std::printf("%6lld  %-7s conf=%.3f  sky=%.2f%-7s aday=%2zu  box=[%d,%d,%d,%d]\n",
                            (long long)frame.id, st, gout.confidence, gout.sky,
                            (gout.integrity_reason[0] ? (std::string("(")+gout.integrity_reason+")").c_str() : ""),
                            gout.candidates.size(), gout.target.x, gout.target.y,
                            gout.target.width, gout.target.height);
            } else {
                std::printf("%6lld %8zu %10d %10d\n",
                            (long long)frame.id, dets.size(), n_tent, n_conf);
            }
        } else {
            cv::Mat vis = warped.image.clone();
            // (#14) Ufuk doğrulama çizimi (telemetri attitude → ufuk çizgisi).
            if (args.draw_horizon && have_telem)
                draw_horizon(vis, gravity_cam(telem.at(t), args.hconv), 128.f, args.hconv);
            // Aktif cue ROI: soluk mavi dikdörtgen
            if (args.closed_loop && args.lock_frac <= 0.f && last_cue.area() > 0)
                cv::rectangle(vis, last_cue, cv::Scalar(180, 80, 0), 1);

            // Manuel kilit nişangahı (HUD): köşe braketleri + merkez artı.
            // Renk/etiket GÜDÜM durumundan gelir: SEARCH (beyaz), TRACK (yeşil),
            // REACQUIRE (kırmızı, yanıp söner). Otomatik kilit YOK.
            using GState = dtrack::GuidanceController::State;
            if (args.lock_frac > 0.f && lock_roi.area() > 0) {
                cv::Scalar col(230, 230, 230); std::string label = "SEARCH";
                if (guided) {
                    if (gout.state == GState::Track) { col = {0, 255, 0}; label = "TRACK"; }
                    else if (gout.state == GState::Suspect) {
                        // yanıp sönen kırmızı: "doğru şeyi mi takip ediyorum?"
                        col = (shown % 2) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 140, 255);
                        label = "REACQUIRE";
                    }
                }
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
                cv::putText(vis, label, {tl.x, tl.y - 10},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv::LINE_AA);
            }

            if (guided) {
                // SEARCH/SUSPECT: adayları NUMARALI çiz (pilot '1'..'9' / tık ile seçer).
                if (gout.state != GState::Track) {
                    for (int i = 0; i < static_cast<int>(gout.candidates.size()); ++i) {
                        const cv::Rect b = gout.candidates[i].bbox;
                        cv::rectangle(vis, b, cv::Scalar(0, 200, 255), 1);
                        cv::putText(vis, std::to_string(i + 1), {b.x, b.y - 4},
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 2);
                    }
                }
                // Dar işlem penceresi (ROI): kilit sonrası görüntü işlemenin yoğunlaştığı
                // hedef-takipli bölge — soluk camgöbeği. STM32N6'da işlenen asıl alan budur.
                if (gout.roi.area() > 0)
                    cv::rectangle(vis, gout.roi, cv::Scalar(160, 160, 0), 1, cv::LINE_AA);
                // TRACK/SUSPECT: kilitli hedef kutusu + güven çubuğu.
                if (gout.has_target && gout.target.area() > 0) {
                    const cv::Scalar tc = (gout.state == GState::Track)
                        ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 140, 255);
                    cv::rectangle(vis, gout.target, tc, 2);
                    const cv::Point bp(gout.target.x, gout.target.y - 8);
                    const int bw = gout.target.width;
                    cv::rectangle(vis, {bp.x, bp.y - 5, bw, 5}, cv::Scalar(60, 60, 60), -1);
                    cv::rectangle(vis, {bp.x, bp.y - 5, cvRound(bw * gout.confidence), 5}, tc, -1);
                }
                const std::string st = gout.state == GState::Track ? "TRACK"
                                     : gout.state == GState::Suspect ? "REACQUIRE" : "SEARCH";
                cv::putText(vis, st + "  conf: " + cv::format("%.2f", gout.confidence) +
                            "   (aday: " + std::to_string(gout.candidates.size()) + ")",
                            {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                // Kilit-bütünlüğü satırı: gök-çevre oranı + (düştüyse) hangi eksen.
                if (gp.use_integrity && gout.has_target) {
                    const bool bad = gout.integrity_reason[0] != '\0';
                    std::string ig = "integrity: sky=" + cv::format("%.2f", gout.sky);
                    if (bad) ig += "  FAIL(" + std::string(gout.integrity_reason) + ")";
                    cv::putText(vis, ig, {20, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                bad ? cv::Scalar(0, 0, 255) : cv::Scalar(180, 220, 180),
                                2, cv::LINE_AA);
                }
            } else {
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
            }

            cv::Mat outimg = vis;
            double disp_scale = 1.0;
            if (outimg.cols > 1920) { cv::resize(outimg, outimg, {}, 0.5, 0.5); disp_scale = 0.5; }
            mouse.scale = disp_scale;   // fare tıklamasını görüntü koordinatına çevir
            if (saving) {
                if (!writer.isOpened())
                    writer.open(args.save_path, cv::VideoWriter::fourcc('m','p','4','v'),
                                fps, outimg.size());
                writer.write(outimg);
            } else {
                cv::imshow("dtrack track", outimg);
                const int k = cv::waitKey(delay_ms);
                if (k == 'q' || k == 27) break;
                // GÜDÜM klavyesi: '1'..'9' adayı kilitle, 'u' kilidi bırak.
                else if (guided && k >= '1' && k <= '9') guide.select(k - '1');
                else if (guided && k == 'u') guide.release();
            }
        }

        if (args.max_frames > 0 && shown >= args.max_frames) break;
        if (!args.dump && args.dump_n > 0 && shown >= args.dump_n) break;
    }

    if (saving) std::cout << "kaydedildi -> " << args.save_path << "\n";
    return 0;
}
