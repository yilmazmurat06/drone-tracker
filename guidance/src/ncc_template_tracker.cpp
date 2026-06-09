// ============================================================================
//  NccTemplateTracker implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/ncc_template_tracker.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace dtrack {

namespace {
// Görüntüyü tek kanal griye indir (zaten griyse dokunma).
cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}
}  // namespace

void NccTemplateTracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
    const cv::Mat gray = to_gray(frame);
    cv::Rect b = bbox & cv::Rect(0, 0, gray.cols, gray.rows);
    if (b.area() <= 0) { have_templ_ = false; return; }
    templ_      = gray(b).clone();   // şablonu (z) sakla
    last_box_   = b;
    have_templ_ = true;
}

STResult NccTemplateTracker::track(const cv::Mat& frame) {
    STResult r;
    r.bbox = last_box_;
    if (!have_templ_) return r;

    const cv::Mat gray = to_gray(frame);

    // Arama penceresi: son kutuyu merkezleyerek search_scale kat genişlet.
    const int sw = std::max(p_.min_search,
                            cvRound(last_box_.width  * p_.search_scale));
    const int sh = std::max(p_.min_search,
                            cvRound(last_box_.height * p_.search_scale));
    const cv::Point c(last_box_.x + last_box_.width / 2,
                      last_box_.y + last_box_.height / 2);
    cv::Rect search(c.x - sw / 2, c.y - sh / 2, sw, sh);
    search &= cv::Rect(0, 0, gray.cols, gray.rows);

    // Şablon arama penceresine sığmıyorsa eşleştirme yapılamaz → düşük güven.
    if (search.width < templ_.cols || search.height < templ_.rows) {
        r.confidence = 0.f;
        return r;
    }

    cv::Mat resp;  // korelasyon yanıt haritası (Siamese'de cross-correlation tepesi)
    cv::matchTemplate(gray(search), templ_, resp, cv::TM_CCOEFF_NORMED);

    double maxv = 0.0; cv::Point maxloc;
    cv::minMaxLoc(resp, nullptr, &maxv, nullptr, &maxloc);

    // Tepe konumu → yeni kutu (arama penceresi ofsetiyle global koordinata taşı).
    const cv::Rect new_box(search.x + maxloc.x, search.y + maxloc.y,
                           templ_.cols, templ_.rows);
    last_box_ = new_box;

    r.bbox = new_box;
    // TM_CCOEFF_NORMED ∈ [-1,1]; güveni [0,1]'e taşı (negatif korelasyon = eşleşme yok).
    r.confidence = static_cast<float>(std::clamp(maxv, 0.0, 1.0));
    return r;
}

} // namespace dtrack
