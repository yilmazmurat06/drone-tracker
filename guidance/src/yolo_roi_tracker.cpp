// ============================================================================
//  YoloRoiTracker implementasyonu — bkz. başlık dosyası.
//
//  ONNX çıktı düzeni (ultralytics export, decode dahil): (1, 5, N) tensörü.
//  Satırlar: cx, cy, w, h (GİRDİ-piksel koordinatı, 0..input) + sınıf güveni.
//  N = anchor sayısı (128² için 336: 16²+8²+4² hücre × stride 8/16/32).
// ============================================================================
#include "dtrack/guidance/yolo_roi_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace dtrack {

namespace {
cv::Mat to_bgr(const cv::Mat& img) {
    if (img.channels() == 3) return img;
    cv::Mat c;
    cv::cvtColor(img, c, cv::COLOR_GRAY2BGR);
    return c;
}
}  // namespace

YoloRoiTracker::YoloRoiTracker(Params p) : p_(std::move(p)) {
    // Kurucuda yükle → dosya yoksa BURADA fırlat; çağıran stub'a düşebilir
    // (NanoSiameseTracker ile aynı fail-fast deseni).
    net_ = cv::dnn::readNetFromONNX(p_.model_path);
}

cv::Rect YoloRoiTracker::init(const cv::Mat& frame, const cv::Rect& bbox,
                              bool refine) {
    const cv::Rect b = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (b.area() <= 0) { ready_ = false; return b; }
    last_box_   = b;
    ready_      = true;
    miss_count_ = 0;
    if (!refine) {
        // ZATEN sıkı kutu (re-acquire adayı) → snap YOK, aynen benimse. EMA da
        // HEMEN devrede (fresh_init_=false): adopte edilen kutu güvenilir boyut
        // referansıdır; ilk track'te EMA atlanırsa ham YOLO yine büyük hipoteze
        // atlar (ölçüldü, kare 3186: 184×116 → 326×197).
        fresh_init_ = false;
        return b;
    }
    fresh_init_ = true;   // ilk track'te EMA yok (gevşek tohum boyutu karışmasın)
    // SNAP: tohum kutusu detector blobu — gevşek/küçük olabilir. YOLO'yu hemen
    // bu karede koştur ve kutuyu modelin SIKI kutusuna oturt; etkin kutu DÖNER,
    // çağıran boyut/hareket referanslarını buna kurar (arayüz sözleşmesi).
    const STResult r = track(frame);
    if (r.confidence < p_.conf_min) last_box_ = b;  // model görmediyse tohumda kal
    return last_box_;
}

void YoloRoiTracker::apply_motion(const cv::Matx33f& M) {
    if (!ready_ || last_box_.area() <= 0) return;
    // Yalnız MERKEZ taşınır (perspektif bölmeli); boyut korunur — kare-arası
    // ego dönmesi yerel ölçeği ihmal edilir düzeyde değiştirir, w/h'a dokunmak
    // boyut-EMA'yı ve büyüme referansını gereksiz oynatır.
    const cv::Vec3f c(last_box_.x + last_box_.width * 0.5f,
                      last_box_.y + last_box_.height * 0.5f, 1.f);
    const cv::Vec3f m = M * c;
    if (std::fabs(m[2]) < 1e-6f) return;
    last_box_.x = cvRound(m[0] / m[2] - last_box_.width * 0.5f);
    last_box_.y = cvRound(m[1] / m[2] - last_box_.height * 0.5f);
}

STResult YoloRoiTracker::track(const cv::Mat& frame) {
    STResult r;
    r.bbox = last_box_;
    if (!ready_) return r;

    const cv::Mat img = to_bgr(frame);

    // 1) Arama penceresi: son kutunun merkezi etrafında KARE pencere.
    //    Küçük hedef → native 128 (piksel kaybı yok); büyük hedef → 2.5×kenar,
    //    sonra 128'e küçülür (büyük hedef downscale'i tolere eder — PHASE0).
    const int   side_t = std::max(last_box_.width, last_box_.height);
    // Kaçırma genişletmesi: ardışık tespitsiz karelerde pencere kademeli büyür
    // (bkz. Params::miss_expand) — hedef pencereden çıktıysa geri yakalanır.
    const float expand = 1.f + p_.miss_expand *
                         static_cast<float>(std::min(miss_count_, p_.miss_expand_max));
    const int   side   = std::min({std::max(p_.input,
                                   static_cast<int>(side_t * p_.search_scale * expand)),
                                   img.cols, img.rows});
    const cv::Point2f ctr(last_box_.x + last_box_.width * 0.5f,
                          last_box_.y + last_box_.height * 0.5f);
    int x0 = static_cast<int>(std::lround(ctr.x - side * 0.5f));
    int y0 = static_cast<int>(std::lround(ctr.y - side * 0.5f));
    x0 = std::clamp(x0, 0, img.cols - side);
    y0 = std::clamp(y0, 0, img.rows - side);
    const cv::Rect win(x0, y0, side, side);

    // 2) Pencereyi ağa ver. blobFromImage: BGR→RGB (swapRB), [0,1] ölçek —
    //    ultralytics'in eğitimde kullandığı normalizasyonla birebir aynı olmalı.
    cv::Mat blob = cv::dnn::blobFromImage(img(win), 1.0 / 255.0,
                                          cv::Size(p_.input, p_.input),
                                          cv::Scalar(), /*swapRB=*/true);
    net_.setInput(blob);
    cv::Mat out = net_.forward();                       // (1, 5, N)
    const int N = out.size[2];
    const float* d = out.ptr<float>();                  // satır-major: 5 satır × N

    // 3) Aday seçimi: güven eşiği + merkeze-yakınlık cezası.
    //    Pencerede İKİNCİ bir hava aracı olabilir; kimlik = uzamsal süreklilik,
    //    bu yüzden çıplak max-conf DEĞİL mesafe-ağırlıklı skor kullanılır.
    const float scale = static_cast<float>(side) / p_.input;  // girdi→pencere
    const float prev_area = static_cast<float>(last_box_.area());
    int   best = -1;
    float best_score = -1e9f, best_conf = 0.f;
    for (int i = 0; i < N; ++i) {
        const float conf = d[4 * N + i];
        if (conf < p_.conf_min) continue;
        const float bx = d[0 * N + i], by = d[1 * N + i];
        const float dx = bx * scale + win.x - ctr.x;
        const float dy = by * scale + win.y - ctr.y;
        const float dist = std::sqrt(dx * dx + dy * dy) / side;  // [0,~0.7]
        // Boyut tutarlılığı: önceki kutuyla alan oranının |log|'u (bkz. başlık).
        const float area = (d[2 * N + i] * scale) * (d[3 * N + i] * scale);
        const float size_pen = (prev_area > 0.f && area > 0.f)
            ? std::fabs(std::log(area / prev_area)) : 0.f;
        const float score = conf - p_.dist_weight * dist - p_.size_weight * size_pen;
        if (score > best_score) { best_score = score; best = i; best_conf = conf; }
    }

    if (best < 0) {
        // Tespit yok: kutuyu OLDUĞU YERDE bırak, güven 0 → controller SUSPECT'e
        // geçip re-acquire başlatır (coasting kararı controller'ın işi, bizim değil).
        ++miss_count_;     // sonraki karede pencere genişler
        r.confidence = 0.f;
        return r;
    }
    miss_count_ = 0;

    // 4) Girdi-koordinatlarını tam kareye geri çevir. Boyut-EMA: w,h düzleştirilir,
    //    merkez HAM kalır (lag yaratmamak için — bkz. başlık).
    const float bx = d[0 * N + best] * scale + win.x;
    const float by = d[1 * N + best] * scale + win.y;
    float bw = d[2 * N + best] * scale;
    float bh = d[3 * N + best] * scale;
    if (last_box_.area() > 0 && !fresh_init_) {
        bw = p_.size_ema * bw + (1.f - p_.size_ema) * last_box_.width;
        bh = p_.size_ema * bh + (1.f - p_.size_ema) * last_box_.height;
    }
    fresh_init_ = false;
    cv::Rect box(static_cast<int>(std::lround(bx - bw / 2)),
                 static_cast<int>(std::lround(by - bh / 2)),
                 static_cast<int>(std::lround(bw)),
                 static_cast<int>(std::lround(bh)));
    box &= cv::Rect(0, 0, img.cols, img.rows);
    if (box.area() <= 0) { r.confidence = 0.f; return r; }

    last_box_    = box;
    r.bbox       = box;
    r.confidence = best_conf;   // drone-luk güveni (doygun DEĞİL — NCC'nin aksine)
    return r;
}

} // namespace dtrack
