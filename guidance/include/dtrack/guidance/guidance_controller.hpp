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
//    TRACK   : tek-hedef tracker sürer. İKİ bağımsız bekçi SUSPECT'e atar:
//              (1) confidence < suspect_conf (suspect_frames kare) — tracker'ın
//                  KENDİ güveni; (2) LOCK-INTEGRITY — güven yüksek olsa bile kutu
//                  geometrisi bozulursa (çevre artık gök değil / kutu patladı /
//                  ışınlandı) ANINDA SUSPECT. Çünkü görsel tracker yere/buluta
//                  kilitlenince de kendinden emindir; güven tek başına yetmez.
//    SUSPECT : "doğru şeyi mi takip ediyorum?" Detector tam görevde; takip edilen
//              bölge DRONE-DOĞRULANIR (P3 skoru + gök-çevre AND koşulu → yerdeki
//              clutter'a yeniden kilitlenmeyi engeller). Doğrulandı → tracker'ı
//              yeniden tohumla → TRACK. confidence < lost_conf veya doğrulanamaz
//              + süre dolar → SEARCH.
//
//  DAR ROI: kilit sonrası işlem hedefi izleyen dar pencerede (roi_margin×kutu)
//  yoğunlaşır (bkz. Out::roi) — STM32N6'da az piksel = az M55 yükü + az clutter.
//
//  DONANIM: tracker NPU-hedefli (ISingleTargetTracker enjekte edilir); detector,
//  discriminator ve LockIntegrity M55 CPU'da kalır (hibrit). Controller donanım-bağımsız.
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/detection/discriminator.hpp"
#include "dtrack/guidance/lock_integrity.hpp"
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

        // --- LOCK-INTEGRITY (kilit-bütünlüğü geometrik bekçisi) ---
        bool  use_integrity  = true;  // false → tam eski davranış (yalnız güven)
        float sky_ring_min   = 0.55f; // kutu çevre halkası gök oranı bunun altı → yere sürüklenme
        float max_area_frac  = 0.08f; // kutu kare alanının bu kadarından büyükse → patlama
        float max_growth     = 2.5f;  // kilit kutusuna göre >bu kat büyüme → patlama
                                      // (4'tü; ölçümde sürüklenme zeplin afişine ATLADIKTAN
                                      // sonra yakalanıyordu → erken yakalama için sıkıldı)
        float max_jump_frac  = 0.5f;  // merkez sıçraması > bu × ROI-yan → ışınlanma
        float roi_margin     = 2.0f;  // dar işlem penceresi = bu × max(kutu en/boy)
        float verify_edge_min = 0.10f;// reseed adayı kenar yoğunluğu eşiği — gök-çevre +
                                      // boyut bandını GEÇEN bulut tutamını eler (bulut
                                      // yumuşak, drone keskin; yalnız re-acquire'da)
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
        cv::Rect                roi;               // TRACK/SUSPECT: dar işlem penceresi (boş=SEARCH)
        float                   confidence = 0.f;  // tek-hedef tracker güveni
        float                   sky = 1.f;         // ölçülen gök-çevre oranı (HUD)
        const char*             integrity_reason = ""; // bütünlük düştüyse eksen: sky/size/motion
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

    cv::Rect              prev_box_;            // bir önceki kutu (hareket sağlığı için)
    cv::Rect              lock_box0_;           // kilit/yeniden-tohum anındaki kutu (büyüme referansı)
    cv::Rect              last_good_box_;       // integrity GEÇEN son kutu — re-acquire ÇAPASI.
                                                // Sürüklenmiş target_ değil, "en son nerede
                                                // sağlam gördüm" noktası aranır.
    float                 last_sky_ = 1.f;      // son ölçülen gök-çevre oranı (HUD)
    const char*           last_reason_ = "";    // son bütünlük-düşüş ekseni (HUD)

    // SUSPECT'te takip bölgesinin gerçekten drone olup olmadığını doğrula (+ gök-çevre koşulu).
    bool verify_target(const cv::Mat& frame, const std::vector<Detection>& cands,
                       Detection* matched) const;

    // Kilit-bütünlüğü: kutu geometrisi hâlâ "havada gerçek hedef" mi? (boyut+hareket+gök).
    IntegrityResult check_integrity(const cv::Mat& frame, const cv::Rect& box) const;

    // Hedefi izleyen dar işlem penceresi (roi_margin × max(kutu en/boy), kareye kırpılı).
    cv::Rect dynamic_roi(const cv::Mat& frame, const cv::Rect& box) const;
};

} // namespace dtrack
