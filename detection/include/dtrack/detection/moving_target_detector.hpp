#pragma once
// ============================================================================
//  MovingTargetDetector — P2, Adım 4: HİBRİT hareketli hedef tespiti.
//
//  NEDEN HİBRİT?  Sahnede iki tür hedef var (Liftoff hava trafiği):
//    • küçük/hızlı uçak-noktaları (2–6 px) → ardışık kareler arası KARE-FARKI
//      bunları net yakalar (küçük ama hızlı → büyük göreli yer değiştirme).
//    • büyük/yavaş zeplin → kare-farkı yalnız KENARINI görür (iç pikseller
//      değişmez); MOG2 arka plan modeli yavaş/büyük cismi dolu yakalar.
//  İki maskeyi BİRLEŞTİRİP (union) connected-component ile blob'lara çeviririz.
//
//  GİRDİ: P1'den geçmiş (stabilize) kare. Stabilizasyon arka planı sabitler →
//  geriye yalnız GERÇEKTEN hareket eden cisimler kalır. (Uyarı: warp KENAR
//  bandı siyahtır ve her kare kayar → border_margin ile yok sayılır; ayrıca
//  3B zemin paralaksı yanlış-pozitif üretebilir → alan filtresi + ileride P3/cue.)
//
//  ÇIKTI: ham `Detection` adayları (centroid/bbox/area/aspect). "Drone mu?"
//  SORULMAZ — o P3'ün (IDiscriminator) işi.
//
//  DURUM (iç): önceki gri kare (fark için) + MOG2 modeli (arka plan için).
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/detection/detector.hpp"

namespace dtrack {

class MovingTargetDetector : public IDetector {
public:
    struct Params {
        // --- kare-farkı dalı ---
        int    diff_thresh   = 22;     // gri-fark eşiği [0-255] (üstü = hareket)
        // --- MOG2 dalı ---
        int    mog_history   = 250;    // arka plan öğrenme penceresi (kare)
        double mog_var_thresh = 25.0;  // Mahalanobis eşik² (küçük = hassas)
        // --- ortak son-işlem ---
        int    open_ksize    = 3;      // morfolojik açma: tek-piksel gürültüyü sil
        int    close_ksize   = 3;      // kapama: blob içini birleştir
        double min_area      = 3.0;    // min blob alanı (px) — gürültü reddi
        double max_area      = 40000;  // max blob alanı (px) — kocaman zemin reddi
        int    merge_gap     = 10;     // bu kadar yakın (px) bloblar tek hedefe birleşir
                                       // (büyük cismin parça parça blob'larını toparlar). 0=kapalı
        int    border_margin = 24;     // warp kenar bandını yok say (px)
        int    warmup        = 8;      // ilk N kare: model ısınıyor → detect=false
        // --- gökyüzü kapısı (sky gate): hava-hava → hedef gökyüzü önünde ---
        // Zemin/ağaç paralaksı yanlış-pozitiflerini eler. HSV'de gökyüzü =
        // parlak ∧ (soluk ∨ mavi). Yeşil/doygun zemin reddedilir.
        bool   sky_gate      = true;
        int    sky_v_min     = 140;    // asgari parlaklık (V)
        int    sky_s_max     = 70;     // azami doygunluk (S) → "soluk/bulut"
        int    sky_dilate    = 7;      // maskeyi genişlet: küçük koyu hedef gök içinde sayılsın
                                       // (büyük tutarsan ağaç-gök sınırı sızar)
        // --- silüet kutusu (tüm cismi kaplayan bbox) ---
        // Hareket maskesi cismin yalnız kenarını görür. Gök önünde silüet =
        // ¬sky'ın, hedef merkezini içeren bağlı bileşeni → tam sınırlayıcı kutu.
        bool   tight_box     = true;
        double max_sil_frac  = 0.10;   // silüet bileşeni kareden büyükse (zemine düşmüş) iptal
    };

    MovingTargetDetector();
    explicit MovingTargetDetector(Params params);

    // IDetector: stabilize kareden ham aday tespitleri çıkarır.
    // return false → model henüz ısınmadı (warmup) → çıktı güvenilmez.
    //        true  → çalıştı (out boş olabilir = bu karede hedef yok).
    bool detect(const Frame& stabilized, std::vector<Detection>& out) override;

    void reset();

    // Cue-odaklı arama: yalnız bu dikdörtgende ara (boş = tüm kare, border'lı).
    // Aramayı daraltmak hız + yanlış-pozitif kazandırır (initial.txt: "cued").
    void set_roi(const cv::Rect& roi) { roi_ = roi; }
    void clear_roi() { roi_ = cv::Rect(); }

    // Teşhis/görselleştirme için son birleşik maske (tüm kare boyutu).
    const cv::Mat& last_mask() const { return last_mask_; }
    int last_count() const { return last_count_; }

private:
    Params  p_;
    cv::Mat prev_gray_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog_;
    cv::Mat last_mask_;
    cv::Rect roi_;          // boş → tüm kare (border_margin uygulanır)
    int     seen_ = 0;
    int     last_count_ = 0;
};

} // namespace dtrack
