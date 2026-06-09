// ============================================================================
//  GuidanceController implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/guidance_controller.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace dtrack {

namespace {
// İki kutunun kesişim/birleşim oranı (IoU) ∈ [0,1].
float iou(const cv::Rect& a, const cv::Rect& b) {
    const float inter = static_cast<float>((a & b).area());
    const float uni   = static_cast<float>(a.area() + b.area()) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}
}  // namespace

GuidanceController::GuidanceController(ISingleTargetTracker& tracker,
                                      IDiscriminator& verifier,
                                      Params p)
    : tracker_(tracker), verifier_(verifier), p_(std::move(p)) {}

void GuidanceController::release() {
    state_          = State::Search;
    target_         = cv::Rect();
    confidence_     = 0.f;
    low_conf_count_ = 0;
    suspect_count_  = 0;
}

void GuidanceController::select(int candidate_index) {
    if (candidate_index < 0 ||
        candidate_index >= static_cast<int>(last_cands_.size()))
        return;
    // Pilot bir adaya kilitlendi: tracker bir SONRAKİ on_frame'de init edilir.
    target_         = last_cands_[candidate_index].bbox;
    state_          = State::Track;
    confidence_     = 1.f;
    low_conf_count_ = 0;
    suspect_count_  = 0;
    needs_init_     = true;
}

bool GuidanceController::verify_target(const std::vector<Detection>& cands,
                                       Detection* matched) const {
    // Takip kutusuyla en çok örtüşen, drone-skoru yeterli adayı bul.
    float best_iou = 0.f; const Detection* best = nullptr;
    for (const auto& d : cands) {
        const float ov = iou(d.bbox, target_);
        if (ov > best_iou) { best_iou = ov; best = &d; }
    }
    if (!best || best_iou <= 0.f) return false;
    if (best->score < p_.verify_score) return false;
    if (matched) *matched = *best;
    return true;
}

void GuidanceController::on_frame(const cv::Mat& frame,
                                  const std::vector<Detection>& cands,
                                  Out& out) {
    last_cands_ = cands;

    switch (state_) {
    case State::Search:
        // Kilit yok: detector adayları pilota sunulur, OTOMATİK KİLİT YOK.
        break;

    case State::Track: {
        // İlk TRACK karesinde şablonu çıkar (select() sonrası).
        if (needs_init_) { tracker_.init(frame, target_); needs_init_ = false; }
        const STResult r = tracker_.track(frame);
        target_     = r.bbox;
        confidence_ = r.confidence;

        if (r.confidence < p_.lost_conf) {
            // Ani kayıp: doğrudan şüpheye geç (kesinti).
            state_ = State::Suspect; suspect_count_ = 0; low_conf_count_ = 0;
        } else if (r.confidence < p_.suspect_conf) {
            if (++low_conf_count_ >= p_.suspect_frames) {
                state_ = State::Suspect; suspect_count_ = 0;
            }
        } else {
            low_conf_count_ = 0;  // güven geri geldi
        }
        break;
    }

    case State::Suspect: {
        // "Doğru şeyi mi takip ediyorum?" Detector tam görevde; takip bölgesini
        // adaylara karşı drone-doğrula. Tracker'ı da koştur (konumu güncel tut).
        const STResult r = tracker_.track(frame);
        target_     = r.bbox;
        confidence_ = r.confidence;

        Detection matched;
        if (verify_target(cands, &matched)) {
            // Doğrulandı → tracker'ı taze kutuyla yeniden tohumla → TRACK.
            target_ = matched.bbox;
            tracker_.init(frame, target_);
            state_ = State::Track;
            low_conf_count_ = 0; suspect_count_ = 0;
        } else if (r.confidence < p_.lost_conf ||
                   ++suspect_count_ >= p_.reacquire_frames) {
            // Doğrulanamadı / güven çok düştü / süre doldu → kilidi bırak.
            release();
        }
        break;
    }
    }

    out.state      = state_;
    out.candidates = (state_ == State::Track) ? std::vector<Detection>{} : cands;
    out.target     = target_;
    out.confidence = confidence_;
    out.has_target = (state_ != State::Search);
}

} // namespace dtrack
