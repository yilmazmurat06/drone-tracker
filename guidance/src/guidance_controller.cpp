// ============================================================================
//  GuidanceController implementasyonu — bkz. başlık dosyası.
// ============================================================================
#include "dtrack/guidance/guidance_controller.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <vector>

namespace dtrack {

GuidanceController::GuidanceController(ISingleTargetTracker& tracker,
                                      IDiscriminator& verifier,
                                      Params p)
    : tracker_(tracker), verifier_(verifier), p_(std::move(p)) {}

void GuidanceController::release() {
    state_           = State::Search;
    target_          = cv::Rect();
    confidence_      = 0.f;
    low_conf_count_  = 0;
    lost_count_      = 0;
    size_fail_count_ = 0;
    suspect_count_   = 0;
    prev_box_       = cv::Rect();
    lock_box0_      = cv::Rect();
    last_good_box_  = cv::Rect();
}

void GuidanceController::compensate(const cv::Matx33f& M) {
    if (state_ == State::Search) return;   // taşınacak kutu durumu yok
    // Kutunun merkezini M ile taşı (perspektif bölmeli); boyut korunur.
    const auto warp_box = [&M](cv::Rect& b) {
        if (b.area() <= 0) return;
        const cv::Vec3f c(b.x + b.width * 0.5f, b.y + b.height * 0.5f, 1.f);
        const cv::Vec3f m = M * c;
        if (std::fabs(m[2]) < 1e-6f) return;
        b.x = cvRound(m[0] / m[2] - b.width * 0.5f);
        b.y = cvRound(m[1] / m[2] - b.height * 0.5f);
    };
    warp_box(target_);
    warp_box(prev_box_);       // taşınmazsa telafinin kendisi "motion" FAIL üretir
    warp_box(last_good_box_);  // re-acquire çapası da aynı çerçevede kalmalı
    // lock_box0_ taşınMAZ: salt boyut referansı, konumu kullanılmıyor.
    tracker_.apply_motion(M);
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
    state_           = State::Track;
    confidence_      = 1.f;
    low_conf_count_  = 0;
    lost_count_      = 0;
    size_fail_count_ = 0;
    suspect_count_   = 0;
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
    // Yarıçap SUSPECT süresiyle kademeli büyür (3 kata kadar): hedef biz ararken
    // hareket eder; sabit yarıçap, kötü re-seed sonrası çapadan 1-2 kutu uzaklıkta
    // duran GERÇEK hedefi kıl payı dışarıda bırakıyordu (ölçüldü: çapa-hedef ~190px,
    // yarıçap 180px → 40 kare boyunca re-acquire başarısız).
    const float grow   = std::min(3.f, 1.f + 0.25f * static_cast<float>(suspect_count_));
    const float radius = p_.roi_margin * std::max(anchor.width, anchor.height) * grow;

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
        // İlk TRACK karesinde şablonu çıkar (select() sonrası). Tracker tohumu
        // RAFİNE EDEBİLİR (YOLO snap) → boyut/çapa referansları DÖNEN kutuya
        // kurulur; tohum (gevşek detector blobu) referans olursa integrity
        // sonsuz FAIL(size) döngüsüne girer (ölçüldü, kare 3082–3089).
        if (needs_init_) {
            target_        = tracker_.init(frame, target_);
            lock_box0_     = target_;
            last_good_box_ = target_;
            needs_init_    = false;
            // Bu karede track() ÇAĞRILMAZ: init zaten inference koştu (YOLO snap).
            // İkinci koşu (1) kare başına çift NPU maliyeti, (2) iki koşunun kutu
            // farkı lock_box0_ ile target_'ı ayrıştırıp sahte FAIL(size) üretir
            // (ölçüldü: kilit 3090'da düştü). Takip bir SONRAKİ kareden başlar.
            confidence_ = 1.f;
            prev_box_   = target_;
            break;
        }
        const STResult r = tracker_.track(frame);
        target_     = r.bbox;
        confidence_ = r.confidence;

        // BEKÇİ 1: kilit-bütünlüğü. Güven yüksek olsa BİLE geometri bozulursa SUSPECT.
        const IntegrityResult ig = p_.use_integrity ? check_integrity(frame, target_)
                                                     : IntegrityResult{};
        last_sky_ = ig.sky; last_reason_ = ig.reason;

        if (ig.ok) {
            last_good_box_ = target_;  // re-acquire çapası: son SAĞLAM konum
            // Büyüme referansını YAVAŞÇA mevcut kutuya yaklaştır (bkz. Params).
            // Yalnız integrity geçince: bozuk kutu referansı kirletemez.
            if (p_.growth_adapt > 0.f && lock_box0_.area() > 0) {
                const float a = p_.growth_adapt;
                lock_box0_.width  = cvRound((1.f - a) * lock_box0_.width  + a * target_.width);
                lock_box0_.height = cvRound((1.f - a) * lock_box0_.height + a * target_.height);
            }
        }

        // SIZE ihlali sayaçlı (gürültülü eksen, bkz. Params::size_fail_frames);
        // sky/motion ihlali ANLIK (kesin sinyaller). ok ise sayaç sıfırlanır.
        bool integrity_trip = false;
        if (!ig.ok) {
            if (std::string_view(ig.reason) == "size")
                integrity_trip = (++size_fail_count_ >= p_.size_fail_frames);
            else
                integrity_trip = true;
        } else {
            size_fail_count_ = 0;
        }

        if (integrity_trip) {
            // Geometrik sürüklenme/patlama/ışınlanma → tracker'a güvenme, şüpheye geç.
            state_ = State::Suspect; suspect_count_ = 0; low_conf_count_ = 0;
            size_fail_count_ = 0; lost_count_ = 0;
        } else if (r.confidence < p_.lost_conf) {
            // BEKÇİ 2: güven kaybı (kesinti). COAST: YOLO-ROI'de conf=0 tek karelik
            // kaçırma olabilir → lost_frames ardışık kare sürerse şüpheye geç.
            if (++lost_count_ >= p_.lost_frames) {
                state_ = State::Suspect; suspect_count_ = 0; low_conf_count_ = 0;
                lost_count_ = 0;
            }
        } else if (r.confidence < p_.suspect_conf) {
            lost_count_ = 0;
            if (++low_conf_count_ >= p_.suspect_frames) {
                state_ = State::Suspect; suspect_count_ = 0;
            }
        } else {
            low_conf_count_ = 0;  // güven geri geldi
            lost_count_     = 0;
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
            // refine=false: matched ZATEN sıkı bir YOLO adayı; yeniden snap kutuyu
            // büyük hipoteze patlatır (ölçüldü, kare 3185) → adayı aynen benimse.
            target_ = tracker_.init(frame, matched.bbox, /*refine=*/false);
            state_ = State::Track;
            low_conf_count_ = 0; suspect_count_ = 0; lost_count_ = 0;
            lock_box0_     = target_;      // büyüme referansını yeniden-tohuma sıfırla
            last_good_box_ = target_;      // çapa da taze doğrulanmış konuma taşınır
            last_reason_ = "";
        } else if (++suspect_count_ >= p_.reacquire_frames) {
            // Doğrulanamadı, süre doldu → kilidi bırak. NOT: düşük güven burada
            // ANLIK bırakmaz — YOLO-ROI tracker'da conf=0 "bu karede tespit yok"
            // demek (geçici kaçırma); tam re-acquire bütçesi kullanılır.
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
