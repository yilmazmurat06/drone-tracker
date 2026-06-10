// ============================================================================
//  GuidanceController implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/guidance_controller.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace dtrack {

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
    prev_box_       = cv::Rect();
    lock_box0_      = cv::Rect();
    last_good_box_  = cv::Rect();
}

// Hedefi izleyen dar işlem penceresi: kutuyu merkez alıp roi_margin× büyüt, kareye kırp.
cv::Rect GuidanceController::dynamic_roi(const cv::Mat& frame, const cv::Rect& box) const {
    if (box.area() <= 0) return cv::Rect();
    const int side = std::max(1, static_cast<int>(std::lround(
        p_.roi_margin * std::max(box.width, box.height))));
    const int cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    return cv::Rect(cx - side / 2, cy - side / 2, side, side)
           & cv::Rect(0, 0, frame.cols, frame.rows);
}

// Üç geometrik eksen: boyut (patlama) → hareket (ışınlanma) → gök-çevre (yere sürüklenme).
// Gök ekseni yalnız RENKLİ (BGR) karede ölçülebilir; gri/içeriksiz karede atlanır.
IntegrityResult GuidanceController::check_integrity(const cv::Mat& frame,
                                                    const cv::Rect& box) const {
    IntegrityResult r;
    if (!lock_integrity::size_sane(box, frame.size(), lock_box0_,
                                   p_.max_area_frac, p_.max_growth)) {
        r.ok = false; r.reason = "size"; return r;
    }
    const float roi_side = p_.roi_margin * std::max(box.width, box.height);
    if (!lock_integrity::motion_sane(box, prev_box_, p_.max_jump_frac * roi_side)) {
        r.ok = false; r.reason = "motion"; return r;
    }
    if (frame.channels() == 3) {
        r.sky = lock_integrity::sky_ring(frame, box);
        if (r.sky < p_.sky_ring_min) { r.ok = false; r.reason = "sky"; }
    }
    return r;
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
    lock_box0_      = target_;       // büyüme referansı = kilit anındaki kutu
    prev_box_       = cv::Rect();    // hareket referansı yok (ilk kare atlanır)
    last_good_box_  = target_;       // re-acquire çapası kilit noktasından başlar
}

bool GuidanceController::verify_target(const cv::Mat& frame,
                                       const std::vector<Detection>& cands,
                                       Detection* matched) const {
    // ÇAPALI re-acquire: sürüklenmiş target_ ile örtüşme ARANMAZ (tracker zaten yanlış
    // yerde olabilir). Arama "en son SAĞLAM gördüğüm" noktaya (last_good_box_) çapalanır.
    const cv::Rect& anchor = last_good_box_.area() > 0 ? last_good_box_ : target_;
    if (anchor.area() <= 0) return false;
    const cv::Point2f ac(anchor.x + anchor.width * 0.5f, anchor.y + anchor.height * 0.5f);
    const float radius = p_.roi_margin * std::max(anchor.width, anchor.height);

    // Boyut bandı: aday, kilit kutusunun [1/max_growth, max_growth] alan bandında olmalı
    // → ne dev zeplin/sahne bloğu ne minik gürültü blobu yeniden tohumlanır.
    const double area0 = static_cast<double>(lock_box0_.area());

    float best_d = radius; const Detection* best = nullptr;
    for (const auto& d : cands) {
        if (d.score < p_.verify_score) continue;                       // P3 drone-luk
        const cv::Point2f c(d.bbox.x + d.bbox.width * 0.5f,
                            d.bbox.y + d.bbox.height * 0.5f);
        const float dist = static_cast<float>(cv::norm(c - ac));
        if (dist > radius || dist >= best_d) continue;                 // çapa yarıçapı + en yakın
        if (p_.use_integrity) {
            const double a = static_cast<double>(d.bbox.area());
            if (area0 > 0 && (a > p_.max_growth * area0 ||
                              a * p_.max_growth < area0)) continue;    // boyut bandı dışı
            if (frame.channels() == 3 &&
                lock_integrity::sky_ring(frame, d.bbox) < p_.sky_ring_min)
                continue;                                              // gök-çevresiz (yer) aday
            if (lock_integrity::edge_density(frame, d.bbox) < p_.verify_edge_min)
                continue;                                              // yumuşak (bulut) aday
        }
        best_d = dist; best = &d;
    }
    if (!best) return false;
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

        // BEKÇİ 1: kilit-bütünlüğü. Güven yüksek olsa BİLE geometri bozulursa SUSPECT.
        const IntegrityResult ig = p_.use_integrity ? check_integrity(frame, target_)
                                                     : IntegrityResult{};
        last_sky_ = ig.sky; last_reason_ = ig.reason;

        if (ig.ok) last_good_box_ = target_;  // re-acquire çapası: son SAĞLAM konum

        if (!ig.ok) {
            // Geometrik sürüklenme/patlama/ışınlanma → tracker'a güvenme, şüpheye geç.
            state_ = State::Suspect; suspect_count_ = 0; low_conf_count_ = 0;
        } else if (r.confidence < p_.lost_conf) {
            // BEKÇİ 2: ani güven kaybı (kesinti) → doğrudan şüphe.
            state_ = State::Suspect; suspect_count_ = 0; low_conf_count_ = 0;
        } else if (r.confidence < p_.suspect_conf) {
            if (++low_conf_count_ >= p_.suspect_frames) {
                state_ = State::Suspect; suspect_count_ = 0;
            }
        } else {
            low_conf_count_ = 0;  // güven geri geldi
        }
        prev_box_ = target_;      // hareket referansını güncelle
        break;
    }

    case State::Suspect: {
        // "Doğru şeyi mi takip ediyorum?" Detector tam görevde; takip bölgesini
        // adaylara karşı drone-doğrula. Tracker'ı da koştur (konumu güncel tut).
        const STResult r = tracker_.track(frame);
        target_     = r.bbox;
        confidence_ = r.confidence;
        if (p_.use_integrity && frame.channels() == 3)
            last_sky_ = lock_integrity::sky_ring(frame, target_);

        Detection matched;
        if (verify_target(frame, cands, &matched)) {
            // Doğrulandı (P3 + gök-çevre) → tracker'ı taze kutuyla yeniden tohumla → TRACK.
            target_ = matched.bbox;
            tracker_.init(frame, target_);
            state_ = State::Track;
            low_conf_count_ = 0; suspect_count_ = 0;
            lock_box0_     = target_;      // büyüme referansını yeniden-tohuma sıfırla
            last_good_box_ = target_;      // çapa da taze doğrulanmış konuma taşınır
            last_reason_ = "";
        } else if (r.confidence < p_.lost_conf ||
                   ++suspect_count_ >= p_.reacquire_frames) {
            // Doğrulanamadı / güven çok düştü / süre doldu → kilidi bırak.
            release();
        }
        prev_box_ = target_;
        break;
    }
    }

    out.state      = state_;
    out.candidates = (state_ == State::Track) ? std::vector<Detection>{} : cands;
    out.target     = target_;
    out.roi        = (state_ == State::Search) ? cv::Rect() : dynamic_roi(frame, target_);
    out.confidence = confidence_;
    out.sky        = last_sky_;
    out.integrity_reason = last_reason_;
    out.has_target = (state_ != State::Search);
}

} // namespace dtrack
