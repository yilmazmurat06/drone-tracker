#pragma once
// ============================================================================
//  GuidanceController — Output/güdüm katmanının BEYNİ (durum makinesi).
//
//  PİLOT İŞ AKIŞI (kullanıcı gereksinimi):
//    1. Detector YARDIMCIDIR, otomatik kilitlemez. Lock çerçevesindeki tespitler
//       pilota NUMARALI ADAY olarak gösterilir. Çerçevede birden fazla drone
//       varsa hangisine kilitleneceğine PİLOT karar verir (select()).
//    2. Kilit sonrası KESİNTİSİZ tek-hedef takip: ISingleTargetTracker (Siamese
//       sınıfı, NPU) sürer; çok-hedef tracker DEVRE DIŞI.
//    3. ŞÜPHE → hızlı detection'a dönüş: takip güveni (confidence/"precision")
//       eşik altına düşerse VEYA kesinti olursa, sistem detection'a geri döner ve
//       takip edilen bölgenin GERÇEKTEN DRONE olup olmadığını yeniden doğrular.
//
//  DURUMLAR:
//    SEARCH  : kilit yok. Adaylar pilota sunulur. select(i) → TRACK.
//    TRACK   : tek-hedef tracker sürer. confidence < suspect_conf (suspect_frames
//              kare boyunca) → SUSPECT. release() → SEARCH.
//    SUSPECT : "doğru şeyi mi takip ediyorum?" Detector tam görevde; takip edilen
//              bölgede DRONE-DOĞRULAMA (ClutterDiscriminator yer tutucu, ileride
//              int8 sınıflandırma kafası). Doğrulandı → tracker'ı yeniden tohumla
//              → TRACK. confidence < lost_conf veya doğrulanamaz → SEARCH.
//
//  DONANIM: tracker NPU-hedefli (ISingleTargetTracker enjekte edilir); detector
//  ve discriminator CPU'da kalır (hibrit). Controller donanımdan bağımsızdır.
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/detection/discriminator.hpp"
#include "dtrack/guidance/single_target_tracker.hpp"

namespace dtrack {

class GuidanceController {
public:
    enum class State { Search, Track, Suspect };

    struct Params {
        float suspect_conf   = 0.50f; // bu güvenin altı → şüphe sayacı işler
        float lost_conf      = 0.30f; // bu güvenin altı → doğrudan hedef kaybı (SEARCH)
        int   suspect_frames = 3;     // ardışık kaç düşük-güven kare → SUSPECT
        float verify_score   = 0.50f; // SUSPECT'te drone-doğrulama (P3 skoru) eşiği
        int   reacquire_frames = 8;   // SUSPECT'te kaç kare doğrulama denenir → sonra SEARCH
    };

    // Tracker (NPU-hedefli) ve discriminator (CPU) dışarıdan enjekte edilir.
    GuidanceController(ISingleTargetTracker& tracker, IDiscriminator& verifier)
        : GuidanceController(tracker, verifier, Params{}) {}
    GuidanceController(ISingleTargetTracker& tracker,
                       IDiscriminator& verifier,
                       Params p);

    // Güdümün bir karelik dışa açık çıktısı (track.cpp HUD'u bunu çizer).
    struct Out {
        State                   state = State::Search;
        std::vector<Detection>  candidates;        // SEARCH/SUSPECT: numaralı adaylar
        cv::Rect                target;            // TRACK/SUSPECT: kilitli hedef kutusu
        float                   confidence = 0.f;  // tek-hedef tracker güveni
        bool                    has_target = false;
    };

    // Bir kareyi işle. cands = lock çerçevesindeki (skorlanmış) detector adayları.
    void on_frame(const cv::Mat& frame,
                  const std::vector<Detection>& cands,
                  Out& out);

    // PİLOT seçimi: en son sunulan adaylardan index'i kilitle (SEARCH → TRACK).
    // Geçersiz index yok sayılır. SEARCH dışındaki durumda da çağrılabilir (önce
    // kilidi bırakıp yeniden kilitler).
    void select(int candidate_index);

    // Kilidi bırak → SEARCH.
    void release();

    State state() const { return state_; }

private:
    ISingleTargetTracker& tracker_;
    IDiscriminator&       verifier_;
    Params                p_;

    State                 state_ = State::Search;
    std::vector<Detection> last_cands_;  // son sunulan adaylar (select() bunlara bakar)
    cv::Rect              target_;
    float                 confidence_ = 0.f;
    int                   low_conf_count_ = 0;  // ardışık düşük-güven kare sayacı
    int                   suspect_count_  = 0;  // SUSPECT'te geçen kare sayısı
    bool                  needs_init_ = false;  // select() sonrası tracker.init() bekliyor

    // SUSPECT'te takip bölgesinin gerçekten drone olup olmadığını doğrula.
    bool verify_target(const std::vector<Detection>& cands, Detection* matched) const;
};

} // namespace dtrack
