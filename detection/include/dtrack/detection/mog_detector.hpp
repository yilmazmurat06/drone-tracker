#pragma once
//
// MogDetector: küçük (2-6 px) hava hedefi için tespit (IDetector, Problem 2).
//
// İki tamamlayıcı sinyali birleştirir (araştırmadaki "iki yöntemi çapraz-doğrula"):
//   - TOP-HAT (uzamsal): beyaz top-hat = görüntü - açma(opening). Küçük parlak
//     noktayı vurgular, yavaş değişen arka planı bastırır. Hareketsiz hedefi bile
//     yakalar AMA statik parlak çöpü (yıldız/işaret) de işaretler.
//   - MOG2 (zamansal): Mixture of Gaussians arka plan modeli. Ego-telafisinden
//     sonra statik çöp sabit -> arka plan; yalnız HAREKETLİ hedef foreground.
//   - top-hat ∧ MOG2 => statik parlak çöp elenir, hareketli hedef kalır.
//
// Kümülatif referans kaydı (registration): hareketli kamerada MOG2'nin sabit bir
// model görmesi için her kareyi bir REFERANS karenin koordinatına warp ederiz.
// Kare-arası ego dönüşümleri çarparak kümülatif dönüşüm (güncel->referans) tutulur.
// Kamera çok kayınca (kümülatif öteleme eşiği aşınca) referans sıfırlanır ->
// kareden taşma ve drift birikimi önlenir (sürekli çalışma için kritik).
//
// Çıktı Detection'lar GÜNCEL kare koordinatındadır (uçuş kontrolcüsü canlı kareyi
// referans alır); referans-koord centroid'i kümülatif dönüşümün tersiyle geri eşlenir.

#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>

#include <vector>

#include "dtrack/common/types.hpp"
#include "dtrack/detection/detector.hpp"

namespace dtrack::detection {

struct DetectorConfig {
    // Top-hat yapısal eleman boyutu (hedeften biraz büyük, tek sayı).
    // 3×3=2-3px, 5×5=3-5px, 9×9=5-10px, 15×15=10-20px hedef.
    // Çoklu ölçek tümünü çalıştırır ve maksimum yanıtı alır.
    int tophat_ksize = 5;
    bool multi_scale_tophat = true;  // 3×3 + 5×5 + 9×9 + 15×15
    // DoG (Difference of Gaussians) — iki bant: küçük (uzak) ve orta (yakın) hedef.
    bool use_dog = true;
    float dog_sigma1 = 0.6f;    // küçük bant sigma dar
    float dog_sigma2 = 1.8f;    // küçük bant sigma geniş
    float dog_sigma1b = 1.5f;   // orta bant sigma dar (5-15 px hedef)
    float dog_sigma2b = 5.0f;   // orta bant sigma geniş

    // MOG2
    int mog_history = 100;
    double mog_var_threshold = 16.0;
    double mog_learning_rate = -1.0;  // -1 = otomatik (1/history civarı)

    // Top-hat eşiği: thr = max(mean + k*std, min_abs) (valid bölge üzerinden).
    double thresh_k = 6.0;
    double thresh_min_abs = 10.0;

    // MOG2 foreground'u kayıt hatasına tolerans için genişlet (yarıçap px).
    int fg_dilate = 2;

    // Geometrik ön eleme (blob).
    double min_area = 2.0;
    double max_area = 400.0;     // yakın hedefi de kapsar (25 px çapa kadar)
    double max_aspect = 4.0;     // çok uzun = kenar/çizgi artığı
    // Bileşen başına en yüksek top-hat yanıtı bunun altındaysa zayıf gürültü çöpü
    // say ve ele. Gerçek hedef parlaktır (yüksek top-hat tepe); recall'u bozmaz.
    double min_peak_tophat = 25.0;

    // Referans sıfırlama: kümülatif öteleme min(W,H)*frac'i aşınca.
    double ref_reset_frac = 0.25;

    // --- Yönlü yerel kontrast (LCM/WLDM) — eş-doğrusal kenar reddi ---
    // IRST literatüründeki çok-yönlü Local Contrast Measure fikri. Aday merkezinden
    // GEÇEN bir kenar/çizgi, KARŞILIKLI iki sektörü birden parlatır (yapı her iki
    // yöne uzanır). Kompakt hedef (sönük olsa da) tüm yönlerde arka planla
    // çevrilidir; tek-piksel gürültü yalnız BİR sektörü parlatır. Bir karşıt çiftin
    // HER İKİ ucu da iç parlaklığın lcm_line_ratio katından parlaksa -> içinden çizgi
    // geçiyor -> ele. reg (warp'lı gri) üzerinde, top-hat ∧ MOG2'den GEÇMİŞ adaylara.
    //
    // ÖNEMLİ — recall önceliği: varsayılan değerler RECALL-GÜVENLİdir (worst-case
    // recall/continuity'yi koruyacak şekilde muhafazakâr). Sentetik sahnenin FP'leri
    // İZOTROPİK doku tepeleridir (kenar değil) -> kontrast ölçüsüyle dim hedeften
    // ayrılamazlar; bu yüzden sentetik benchmark'ta LCM neredeyse no-op'tur (bilinçli).
    // Asıl faydası GERÇEK yapısal arka planda (bulut/ufuk/yer kenarları) görülür ve
    // video_eval ile gerçek veride doğrulanmalıdır. Daha agresif ayar (lr↓, prom↓)
    // precision'ı artırır AMA worst-case recall'dan verir (recall-önceliğine aykırı).
    bool use_lcm = true;
    int lcm_gap_px = 3;             // iç bölge ile arka plan halkası arası boşluk (px)
    int lcm_bg_patch = 1;           // arka plan örneği yarı-pencere (1 -> 3×3)
    float lcm_line_ratio = 0.90f;   // karşıt çift uçları iç_ort'un bu kadarından parlaksa = çizgi
    // Belirginlik koruması: iç_ort komşu ortalamasından bu kadar (gri seviye) FAZLA
    // değilse LCM hiç uygulanmaz (dim/düşük-SNR'da kontrast güvenilmez -> recall korunur).
    float lcm_min_prominence = 10.0f;

    // --- Kapalı-döngü cued ROI kurtarma (track-before-detect-lite, bkz. cue.hpp) ---
    // Tracker kilitliyken (cue geçerli) VE global tespit kapı içinde aday bulamazsa,
    // tahmin etrafındaki küçük ROI'de DÜŞÜK eşikle, MOG2-AND şartı OLMADAN tek en iyi
    // top-hat tepesi kurtarılır. MOG2-AND'siz olması KRİTİK: hover / düşük-SNR hedef
    // zamansal hareket maskesini tetiklemese bile (global pipeline'da görünmez olur)
    // kapalı-döngü ROI onu yakalar -> KESİNTİSİZ TAKİP (IRST kapalı-döngü ROI yaklaşımı).
    // Lokal SNR tabanı (cue_min_peak_tophat) saf gürültünün kurtarma üretmesini sınırlar.
    bool use_cue_recovery = true;
    float cue_thresh_k = 3.0f;          // ROI lokal eşik katsayısı (global thresh_k=6'dan düşük)
    float cue_min_peak_tophat = 12.0f;  // ROI tepe top-hat tabanı (global min_peak=25'ten düşük, 0 değil)
    float cue_meas_std = 3.0f;          // kurtarma ölçümünün σ_r ipucu (px); normalden büyük -> az güven
    int cue_max_radius = 40;            // ROI yarıçap üst sınırı (px)
    int cue_min_radius = 6;             // ROI yarıçap alt sınırı (px)
};

class MogDetector : public IDetector {
public:
    explicit MogDetector(DetectorConfig cfg = {});

    std::vector<common::Detection> detect(const common::StabilizedFrame& sf) override;
    void reset() override;

    // Kapalı-döngü geri besleme: tracker'ın bir sonraki kare tahmini. Bir SONRAKİ
    // detect()'te global tespit kapıyı dolduramazsa bu ROI'de kurtarma yapılır.
    void set_cue(const common::TargetCue& cue) override { cue_ = cue; }

    // Son karenin güncel->referans (kümülatif) dönüşümü. Tracking pürüzsüz
    // referans koordinatında takip etmek için kullanır.
    cv::Matx33f image_to_reference() const override { return h_cum_; }

    // Teşhis: o ana kadarki referans sıfırlama sayısı.
    int reference_resets() const { return resets_; }

private:
    void make_reference(const cv::Mat& gray);

    DetectorConfig cfg_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog_;
    cv::Mat tophat_kernel_;
    cv::Mat tophat_kernel_small_;   // 3×3
    cv::Mat tophat_kernel_med_;     // 9×9
    cv::Mat tophat_kernel_large_;   // 15×15
    cv::Mat fg_kernel_;

    cv::Matx33f h_cum_ = cv::Matx33f::eye();  // güncel -> referans
    bool has_ref_{false};
    cv::Size ref_size_{};
    int resets_{0};

    common::TargetCue cue_{};  // tracker'dan gelen son geri besleme (kapalı-döngü)
};

}  // namespace dtrack::detection
