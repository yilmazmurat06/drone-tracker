// live_track: CANLI, ETKİLEŞİMLİ "cued" takip uygulaması.
//
// Kullanıcı (operatör) ekranda oynayan video/simülatör akışı üzerinde FARE ile
// hedefi bir kutuyla işaretler; sistem o hedefe KİLİTLENİR ve canlı izler. Bu,
// projenin "sistem hedefe CUED gelir, problem takiptir" çerçevesine birebir uyar.
//
// TAKİP MOTORU (CuedTracker = hibrit):
//   1) CSRT korelasyon takibi  -> sağlam merkez takibi (clutter/arazi önünde de).
//   2) GÖKYÜZÜ-vs-NESNE segmentasyonu (kutu çevresi ROI'de) -> kutuyu nesnenin
//      TAMAMINA sıkıca oturtur (CSRT'nin ölçek adaptasyonu zayıf; kutu nesneyi
//      kısmen kaplıyordu). Ölçek değişimini (yaklaşan/uzaklaşan/manevra) yakalar.
//   3) HIZ TAHMİNİ -> arama ROI'sini bir sonraki konuma kaydırır. Videolar gerçekte
//      yavaşlatılmış; gerçek hedef hızlı hareket eder. Genişletilmiş, hız-merkezli
//      ROI büyük kare-arası yer değişimine tolerans sağlar.
//   4) Periyodik CSRT yeniden-init (rafine kutudan) -> ölçek+drift düzeltme.
//
// AYNI kod hem kayıtlı videoda hem SİMÜLATÖRDE çalışır (kaynak argümanı):
//   ./live_track ucus.mp4              (kayıtlı klip)
//   ./live_track "udp://@:5600"        (AirSim/Gazebo UDP)
//   ./live_track "rtsp://host:554/s"   (RTSP)
//   ./live_track 0                     (kamera/donanım)
//
// FARE: SOL TUŞ basılı sürükle = hedef kutusu çiz, bırak = kilitle.
// TUŞLAR: SPACE dur/devam | r kilidi sıfırla | [ / ] yavaş/hızlı | s kare-kaydet |
//         q veya ESC çık.   --save <out.mp4> verilirse oturum kaydedilir.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

using Clock = std::chrono::high_resolution_clock;

// ===================== Fare durumu (kutu çizimi) =====================
struct MouseState {
    bool dragging = false;
    bool ready = false;
    cv::Point p0, p1;
};
static MouseState g_mouse;

static void on_mouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_mouse.dragging = true;
        g_mouse.ready = false;
        g_mouse.p0 = g_mouse.p1 = {x, y};
    } else if (event == cv::EVENT_MOUSEMOVE && g_mouse.dragging) {
        g_mouse.p1 = {x, y};
    } else if (event == cv::EVENT_LBUTTONUP) {
        g_mouse.dragging = false;
        g_mouse.p1 = {x, y};
        g_mouse.ready = true;
    }
}

static cv::Rect rect_from_points(cv::Point a, cv::Point b) {
    return cv::Rect(cv::Point(std::min(a.x, b.x), std::min(a.y, b.y)),
                    cv::Point(std::max(a.x, b.x), std::max(a.y, b.y)));
}

static cv::Ptr<cv::Tracker> make_tracker(const std::string& kind) {
    if (kind == "kcf") return cv::TrackerKCF::create();
    return cv::TrackerCSRT::create();
}

// ===================== Hibrit cued takipçi =====================
// Hibrit "cued" takipçi.
// İLKE: CSRT'ye init'ten SONRA HİÇ DOKUNMA (yeniden-init YOK) -> tam plain-CSRT
// kararlılığı (operatörün "kilitliyor" dediği davranış). Segmentasyon SADECE
// ÇİZİLEN kutuyu nesneye oturtmak için (track'i ETKİLEMEZ). Böylece en kötü
// ihtimalle = düz CSRT; asla daha kötü değil. Segmentasyon güvenilir değilse
// (terrain/düşük kontrast) CSRT kutusu çizilir.
struct CuedTracker {
    cv::Ptr<cv::Tracker> csrt;
    std::string kind = "csrt";
    cv::Rect draw_box;          // ÇİZİLECEK kutu (hareket blob'u ya da CSRT kutusu)
    cv::Point2f center{0, 0};
    cv::Point2f vel{0, 0};      // px/kare (hız tahmini -> arama penceresi kayması)
    bool active = false, lost = false, refine = true;  // refine: hareket-tabanlı takip açık
    int weak_streak = 0;        // hareket blob'u bulunamayan ardışık kare

    cv::Mat prev_gray_;         // bir önceki kare (gri) -> hareket farkı için
    int since_reinit = 0;

    // Ayarlar.
    float roi_expand = 2.4f;    // arama ROI = kutu * bu (hız-merkezli; hızlı harekete tolerans)
    int motion_thr = 16;        // hizalanmış arka plan farkı eşiği (gri seviye)
    int min_blob_area = 20;     // en küçük hareketli blob alanı
    int reinit_every = 12;      // CSRT'yi (yedek) hareket-kutusundan bu kadar karede bir hizala
    // Segmentasyon (yalnız nişangah kilidinde kutu boyutu için).
    int diff_thr = 16;
    int edge_thr = 40;

    cv::Rect box_rect() const { return draw_box; }

    static cv::Mat to_gray(const cv::Mat& f) {
        cv::Mat g;
        if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
        else g = f.clone();
        return g;
    }

    void init(const cv::Mat& frame, cv::Rect b, const std::string& k, bool ref) {
        kind = k;
        refine = ref;
        draw_box = b;
        center = {b.x + b.width / 2.0f, b.y + b.height / 2.0f};
        vel = {0, 0};
        active = true;
        lost = false;
        weak_streak = 0;
        since_reinit = 0;
        prev_gray_ = to_gray(frame);
        csrt = make_tracker(kind);
        csrt->init(frame, b);
    }

    // Tahmini konum çevresindeki ROI'de HAREKETLİ blob'u bul (arka plan telafili).
    // phaseCorrelate ile yerel arka plan kaymasını kestir -> önceki kareyi hizala ->
    // fark al -> hedef (farklı hareket eden) parlar. Tam koord. kutu döner; yoksa boş.
    cv::Rect motion_blob(const cv::Mat& cur_gray, cv::Rect roi, cv::Point2f predicted) const {
        roi &= cv::Rect(0, 0, cur_gray.cols, cur_gray.rows);
        if (roi.width < 16 || roi.height < 16 || prev_gray_.empty()) return {};
        cv::Mat pr = prev_gray_(roi), cr = cur_gray(roi);
        cv::Mat prf, crf;
        pr.convertTo(prf, CV_32F);
        cr.convertTo(crf, CV_32F);
        cv::Mat hann;
        cv::createHanningWindow(hann, roi.size(), CV_32F);
        const cv::Point2d shift = cv::phaseCorrelate(prf, crf, hann);  // arka plan kayması prev->cur
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
        cv::Mat pr_al;
        cv::warpAffine(pr, pr_al, M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::Mat diff;
        cv::absdiff(cr, pr_al, diff);
        cv::Mat mask;
        cv::threshold(diff, mask, motion_thr, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
        // Hizalama/warp kenar artığını ele (ROI kenar şeridini sıfırla).
        const int m = 4;
        cv::rectangle(mask, cv::Rect(0, 0, mask.cols, mask.rows), 0, 2 * m);

        cv::Mat lab, st, ct;
        const int n = cv::connectedComponentsWithStats(mask, lab, st, ct, 8);
        const cv::Point2f pin(predicted.x - roi.x, predicted.y - roi.y);
        const float maxd = 0.5f * std::max(roi.width, roi.height);
        int best = -1;
        float best_d = 1e9f;
        for (int i = 1; i < n; ++i) {
            const int a = st.at<int>(i, cv::CC_STAT_AREA);
            if (a < min_blob_area) continue;
            const int w = st.at<int>(i, cv::CC_STAT_WIDTH), h = st.at<int>(i, cv::CC_STAT_HEIGHT);
            if (w > 0.8 * roi.width || h > 0.8 * roi.height) continue;  // arka plan değil, kompakt
            const float d = std::hypot((float)ct.at<double>(i, 0) - pin.x,
                                       (float)ct.at<double>(i, 1) - pin.y);
            if (d < best_d && d < maxd) { best_d = d; best = i; }
        }
        if (best < 0) return {};
        cv::Rect bb(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                    st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
        return bb + roi.tl();
    }

    // ROI içinde gökyüzüne karşı baskın KOMPAKT nesneyi segmentle -> tam koord. kutu.
    cv::Rect segment(const cv::Mat& frame, cv::Rect roi) const {
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width < 8 || roi.height < 8) return {};
        cv::Mat g;
        if (frame.channels() == 3) cv::cvtColor(frame(roi), g, cv::COLOR_BGR2GRAY);
        else g = frame(roi).clone();

        // Gökyüzü referansı = ROI KENAR şeridinin ortalaması (kenar ~ arka plan).
        const int bw = std::max(2, roi.width / 8), bh = std::max(2, roi.height / 8);
        cv::Mat border = cv::Mat::zeros(g.size(), CV_8U);
        cv::rectangle(border, cv::Rect(0, 0, g.cols, g.rows), 255, -1);
        cv::rectangle(border, cv::Rect(bw, bh, g.cols - 2 * bw, g.rows - 2 * bh), 0, -1);
        const double sky = cv::mean(g, border)[0];

        cv::Mat fdiff;
        cv::absdiff(g, cv::Scalar(sky), fdiff);
        cv::Mat obj = fdiff > diff_thr;
        cv::Mat gx, gy, gxa, gya, grad;
        cv::Sobel(g, gx, CV_16S, 1, 0);
        cv::Sobel(g, gy, CV_16S, 0, 1);
        cv::convertScaleAbs(gx, gxa);
        cv::convertScaleAbs(gy, gya);
        cv::addWeighted(gxa, 0.5, gya, 0.5, 0, grad);
        cv::bitwise_or(obj, grad > edge_thr, obj);
        cv::morphologyEx(obj, obj, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
        cv::morphologyEx(obj, obj, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

        cv::Mat lab, st, ct;
        const int n = cv::connectedComponentsWithStats(obj, lab, st, ct, 8);
        const cv::Point2f c0(g.cols / 2.0f, g.rows / 2.0f);
        int best = -1;
        double best_score = -1;
        for (int i = 1; i < n; ++i) {
            const int a = st.at<int>(i, cv::CC_STAT_AREA);
            if (a < 25) continue;
            const double dist = std::hypot(ct.at<double>(i, 0) - c0.x, ct.at<double>(i, 1) - c0.y);
            const double score = a - dist * 6.0;  // büyük + merkeze yakın
            if (score > best_score) { best_score = score; best = i; }
        }
        if (best < 0) return {};
        cv::Rect bb(st.at<int>(best, cv::CC_STAT_LEFT), st.at<int>(best, cv::CC_STAT_TOP),
                    st.at<int>(best, cv::CC_STAT_WIDTH), st.at<int>(best, cv::CC_STAT_HEIGHT));
        return bb + roi.tl();
    }

    void update(const cv::Mat& frame) {
        if (!active) return;
        const cv::Point2f prev_c = center;
        const cv::Rect frame_r(0, 0, frame.cols, frame.rows);
        const cv::Mat cur_gray = to_gray(frame);

        bool located = false;

        if (refine) {
            // --- PRIMER: hareket-tabanlı. Hız-merkezli arama penceresi. ---
            const cv::Point2f pred(center.x + vel.x, center.y + vel.y);
            const float sw = std::max(40.0f, draw_box.width * roi_expand);
            const float sh = std::max(40.0f, draw_box.height * roi_expand);
            const cv::Rect roi(cvRound(pred.x - sw / 2), cvRound(pred.y - sh / 2),
                               cvRound(sw), cvRound(sh));
            const cv::Rect mb = motion_blob(cur_gray, roi, pred);
            if (mb.width > 4 && mb.height > 4) {
                draw_box = mb & frame_r;
                center = {mb.x + mb.width / 2.0f, mb.y + mb.height / 2.0f};
                lost = false;
                weak_streak = 0;
                located = true;
                // Yedek CSRT'yi hareket-kutusundan periyodik hizala (drift'siz kalsın).
                if (++since_reinit >= reinit_every) {
                    cv::Rect b = draw_box & frame_r;
                    if (b.width > 6 && b.height > 6) { csrt = make_tracker(kind); csrt->init(frame, b); }
                    since_reinit = 0;
                }
            }
        }

        if (!located) {
            // --- YEDEK: hareket yok (hedef durdu/belirsiz) -> CSRT. ---
            cv::Rect cb;
            const bool ok = csrt->update(frame, cb);
            if (ok && (cb & frame_r).area() > 0) {
                draw_box = cb & frame_r;
                center = {cb.x + cb.width / 2.0f, cb.y + cb.height / 2.0f};
                lost = false;
            } else {
                lost = true;
            }
            ++weak_streak;
        }

        vel = 0.6f * vel + 0.4f * cv::Point2f(center.x - prev_c.x, center.y - prev_c.y);
        prev_gray_ = cur_gray;
    }
};

// Ekran merkezindeki nişangah (crosshair / boresight). Gerçek pilot, hedefi buraya
// alıp tetikleyerek kilit atar (fare yok). Görece büyük, merkez boşluklu reticle.
static void draw_crosshair(cv::Mat& vis) {
    const cv::Point c(vis.cols / 2, vis.rows / 2);
    const int R = std::max(28, vis.rows / 10);  // görece büyük
    const int gap = R / 3;
    const cv::Scalar col(0, 255, 0);
    cv::circle(vis, c, R, col, 1, cv::LINE_AA);
    cv::circle(vis, c, 2, col, -1, cv::LINE_AA);
    cv::line(vis, {c.x - R - 12, c.y}, {c.x - gap, c.y}, col, 1, cv::LINE_AA);
    cv::line(vis, {c.x + gap, c.y}, {c.x + R + 12, c.y}, col, 1, cv::LINE_AA);
    cv::line(vis, {c.x, c.y - R - 12}, {c.x, c.y - gap}, col, 1, cv::LINE_AA);
    cv::line(vis, {c.x, c.y + gap}, {c.x, c.y + R + 12}, col, 1, cv::LINE_AA);
}

// Nişangah konumundaki nesneyi segmentasyonla bul -> kutu. Bulamazsa varsayılan
// boyutta merkez kutusu döner (yine de kilitlenir, sonra rafine olur).
static cv::Rect lock_box_at(const CuedTracker& seg, const cv::Mat& frame, cv::Point ctr) {
    const int s = std::max(64, frame.rows / 8);
    cv::Rect roi(ctr.x - s / 2, ctr.y - s / 2, s, s);
    cv::Rect b = seg.segment(frame, roi);
    const bool good = b.width > 6 && b.height > 6 &&
                      b.width < 0.85f * roi.width && b.height < 0.85f * roi.height;
    if (good) return b;
    const int d = std::max(44, frame.cols / 18);  // segmentasyon başarısız -> varsayılan kutu
    return cv::Rect(ctr.x - d / 2, ctr.y - d / 2, d, d) & cv::Rect(0, 0, frame.cols, frame.rows);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "Kullanim: %s <video|udp://@:5600|rtsp://...|kamera_idx>\n"
            "          [--tracker csrt|kcf] [--width <px>] [--start <kare>]\n"
            "          [--save <out.mp4>] [--no-loop] [--no-refine]\n"
            "KILIT: hedefi NISANGAHA al + 'f' veya ENTER (pilot/boresight) ya da fare ile kutu ciz.\n"
            "TUS  : SPACE dur/devam | r yeniden sec | [ ] hiz | s kare-kaydet | q/ESC cik\n",
            argv[0]);
        return 1;
    }
    const std::string source = argv[1];
    std::string tracker_kind = "csrt", save_path, cue_str;
    int display_w = 1280, start_frame = 0, cue_frame = 0, max_frames = 0;
    bool loop = true, refine = true, headless = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tracker") && i + 1 < argc) tracker_kind = argv[++i];
        else if (!std::strcmp(argv[i], "--width") && i + 1 < argc) display_w = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--start") && i + 1 < argc) start_frame = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) save_path = argv[++i];
        else if (!std::strcmp(argv[i], "--no-loop")) loop = false;
        else if (!std::strcmp(argv[i], "--no-refine")) refine = false;
        // Headless test/otomasyon: fare yerine sabit cue kutusu (proc/display koord.).
        else if (!std::strcmp(argv[i], "--cue") && i + 1 < argc) cue_str = argv[++i];  // "x,y,w,h"
        else if (!std::strcmp(argv[i], "--cue-frame") && i + 1 < argc) cue_frame = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--headless")) headless = true;
        else if (!std::strcmp(argv[i], "--max-frames") && i + 1 < argc) max_frames = std::atoi(argv[++i]);
    }
    cv::Rect cue_box;
    bool center_lock = false;  // headless: nişangah (merkez) kilidi testi
    if (!cue_str.empty()) {
        if (cue_str == "center") center_lock = true;
        else {
            int x, y, w, h;
            if (std::sscanf(cue_str.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) == 4)
                cue_box = cv::Rect(x, y, w, h);
        }
        loop = false;  // otomasyonda döngü kapalı
    }

    cv::VideoCapture cap;
    const bool is_index = !source.empty() &&
                          source.find_first_not_of("0123456789") == std::string::npos;
    if (is_index) cap.open(std::stoi(source));
    else cap.open(source);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "HATA: kaynak acilamadi: %s\n", source.c_str());
        return 1;
    }
    const bool is_file = !is_index && source.rfind("rtsp://", 0) != 0 &&
                         source.rfind("udp://", 0) != 0 && source.rfind("http", 0) != 0;
    if (is_file && start_frame > 0) cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);

    const std::string win = "live_track  (fare ile hedefi isaretle)";
    if (!headless) {
        cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(win, on_mouse);
    }

    CuedTracker trk;
    bool paused = false;
    cv::Mat raw, proc;
    int wait_ms = 25, snap = 0;
    cv::VideoWriter writer;
    std::deque<double> ms_hist;
    std::uint64_t fno = 0;

    std::printf("=== live_track (hibrit: CSRT + segment-rafine + hiz-tahmini) ===\n"
                "  kaynak: %s | takipci: %s | rafine: %s\n"
                "  FARE ile hedefi kutuyla isaretle. SPACE=dur, r=yeniden sec, q=cik.\n",
                source.c_str(), tracker_kind.c_str(), refine ? "acik" : "kapali");

    while (true) {
        if (!paused) {
            if (!cap.read(raw) || raw.empty()) {
                if (is_file && loop) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); fno = 0; continue; }
                break;
            }
            ++fno;
            const double scale = static_cast<double>(display_w) / raw.cols;
            if (scale > 0 && scale != 1.0)
                cv::resize(raw, proc, cv::Size(), scale, scale, cv::INTER_AREA);
            else
                raw.copyTo(proc);

            // Otomatik cue (headless test): sabit kutu ya da nişangah (merkez) kilidi.
            if ((cue_box.area() > 0 || center_lock) && !trk.active &&
                fno >= static_cast<std::uint64_t>(std::max(1, cue_frame))) {
                cv::Rect r = center_lock
                                 ? lock_box_at(trk, proc, {proc.cols / 2, proc.rows / 2})
                                 : (cue_box & cv::Rect(0, 0, proc.cols, proc.rows));
                if (r.width > 6 && r.height > 6) {
                    trk.init(proc, r, tracker_kind, refine);
                    std::printf("  [auto-cue%s] kare %llu: (%d,%d %dx%d)\n",
                                center_lock ? "/center" : "", (unsigned long long)fno, r.x, r.y,
                                r.width, r.height);
                }
            }
        }
        if (max_frames > 0 && fno >= static_cast<std::uint64_t>(max_frames)) break;

        // Yeni fare seçimi -> kilitle (cue).
        if (g_mouse.ready) {
            cv::Rect r = rect_from_points(g_mouse.p0, g_mouse.p1) &
                         cv::Rect(0, 0, proc.cols, proc.rows);
            g_mouse.ready = false;
            if (r.width > 6 && r.height > 6) {
                trk.init(proc, r, tracker_kind, refine);
                std::printf("  [cue] kilit: (%d,%d %dx%d)\n", r.x, r.y, r.width, r.height);
            }
        }

        // Takip güncelle.
        double ms = 0;
        if (trk.active && !paused && !trk.lost) {
            auto t0 = Clock::now();
            trk.update(proc);  // refine içeride; CSRT'ye dokunulmaz
            ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ms_hist.push_back(ms);
            if (ms_hist.size() > 30) ms_hist.pop_front();
        }

        // Çizim.
        cv::Mat vis = proc.clone();
        if (g_mouse.dragging)
            cv::rectangle(vis, rect_from_points(g_mouse.p0, g_mouse.p1),
                          cv::Scalar(0, 255, 255), 2);
        if (trk.active) {
            // yeşil=kilitli (CSRT izliyor), kırmızı=kayıp (CSRT başarısız).
            const cv::Scalar c = trk.lost ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::Rect b = trk.box_rect();
            cv::rectangle(vis, b, c, 2);
            const cv::Point ctr(cvRound(trk.center.x), cvRound(trk.center.y));
            cv::drawMarker(vis, ctr, c, cv::MARKER_CROSS, 18, 1);
            if (!trk.lost)
                cv::arrowedLine(vis, ctr,
                                {ctr.x + cvRound(trk.vel.x * 6), ctr.y + cvRound(trk.vel.y * 6)},
                                cv::Scalar(255, 200, 0), 2, cv::LINE_AA, 0, 0.3);
            cv::putText(vis, trk.lost ? "KAYIP - yeniden sec (r veya fare)" : "KILITLI",
                        {b.x, std::max(0, b.y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
                        cv::LINE_AA);
        }

        double avg_ms = 0;
        for (double v : ms_hist) avg_ms += v;
        avg_ms = ms_hist.empty() ? 0 : avg_ms / ms_hist.size();
        char l1[200], l2[200];
        std::snprintf(l1, sizeof(l1),
                      "kare %llu | takip %.1f ms (~%.0f FPS) | %s%s | kutu %dx%d | hiz %.0f px/k",
                      static_cast<unsigned long long>(fno), avg_ms,
                      avg_ms > 0 ? 1000.0 / avg_ms : 0.0, tracker_kind.c_str(),
                      paused ? " [DURDU]" : "", trk.draw_box.width, trk.draw_box.height,
                      std::hypot(trk.vel.x, trk.vel.y));
        std::snprintf(l2, sizeof(l2),
                      trk.active ? "FARE: yeni kutu | r: sifirla | SPACE: dur | [ ] hiz | q: cik"
                                 : "FARE ile hedefi kutuyla isaretle | SPACE: dur | q: cik");
        cv::rectangle(vis, {0, 0}, {vis.cols, 52}, cv::Scalar(0, 0, 0), -1);
        cv::putText(vis, l1, {10, 21}, cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220, 220, 220),
                    1, cv::LINE_AA);
        cv::putText(vis, l2, {10, 43}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(180, 220, 180),
                    1, cv::LINE_AA);

        // Nişangah (her zaman, kutunun üstünde) — pilot bununla nişan alır.
        draw_crosshair(vis);

        if (!save_path.empty()) {
            if (!writer.isOpened())
                writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0,
                            vis.size(), true);
            if (writer.isOpened()) writer.write(vis);
        }

        if (headless) continue;  // pencere yok: sadece işle + (varsa) kaydet

        cv::imshow(win, vis);
        const int key = cv::waitKey(paused ? 20 : wait_ms) & 0xFF;
        if (key == 'q' || key == 27) break;
        else if (key == ' ') paused = !paused;
        else if (key == 'f' || key == 13) {  // NİŞANGAH KİLİDİ (pilot tetiği / ENTER)
            const cv::Point cc(proc.cols / 2, proc.rows / 2);
            const cv::Rect b = lock_box_at(trk, proc, cc);
            if (b.width > 6 && b.height > 6) {
                trk.init(proc, b, tracker_kind, refine);
                std::printf("  [nisangah-kilit] (%d,%d %dx%d)\n", b.x, b.y, b.width, b.height);
            }
        }
        else if (key == 'r') { trk.active = false; trk.lost = false; trk.csrt.release(); ms_hist.clear(); }
        else if (key == '[') wait_ms = std::min(200, wait_ms + 5);
        else if (key == ']') wait_ms = std::max(1, wait_ms - 5);
        else if (key == 's') {
            char p[64];
            std::snprintf(p, sizeof(p), "live_snap_%03d.png", snap++);
            cv::imwrite(p, vis);
            std::printf("  [kayit] %s\n", p);
        }
    }

    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();
    if (!save_path.empty()) std::printf("  kayit: %s\n", save_path.c_str());
    return 0;
}
