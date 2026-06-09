// ============================================================================
//  NanoSiameseTracker implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/nano_siamese_tracker.hpp"

#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace dtrack {

namespace {
// cv::TrackerNano 3-kanal (BGR) görüntü bekler; gri gelirse BGR'ye çevir.
cv::Mat to_bgr(const cv::Mat& img) {
    if (img.channels() == 3) return img;
    cv::Mat c;
    cv::cvtColor(img, c, cv::COLOR_GRAY2BGR);
    return c;
}
}  // namespace

NanoSiameseTracker::NanoSiameseTracker(Params p) : p_(std::move(p)) {
    cv::TrackerNano::Params tp;
    tp.backbone = p_.backbone;
    tp.neckhead = p_.neckhead;
    // Modelleri kurucuda yükle → dosya yoksa BURADA fırlat (fail-fast); çağıran
    // (track.cpp) yakalayıp NCC stub'a döner. Pilot seçiminde patlamaz.
    trk_ = cv::TrackerNano::create(tp);
}

void NanoSiameseTracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
    const cv::Mat img = to_bgr(frame);
    cv::Rect b = bbox & cv::Rect(0, 0, img.cols, img.rows);
    if (b.area() <= 0) { ready_ = false; return; }
    trk_->init(img, b);
    last_box_ = b;
    ready_    = true;
}

STResult NanoSiameseTracker::track(const cv::Mat& frame) {
    STResult r;
    r.bbox = last_box_;
    if (!ready_ || !trk_) return r;

    const cv::Mat img = to_bgr(frame);
    cv::Rect box = last_box_;
    const bool ok = trk_->update(img, box);
    const float score = trk_->getTrackingScore();   // [0,1] benzerlik tepe (= precision)

    if (ok) { last_box_ = box; r.bbox = box; }
    // update() hedefi kaybettiğini bildirirse güveni sıfırla (GuidanceController
    // SUSPECT'e geçsin). Aksi halde ağın kendi skoru.
    r.confidence = ok ? score : 0.f;
    return r;
}

} // namespace dtrack
