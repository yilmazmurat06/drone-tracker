#pragma once
// ============================================================================
//  YoloRoiTracker — YOLO-in-ROI tek-hedef tracker (Faz 4).
//
//  NEDEN: NanoTrack şablon benzerliği ölçer; kutu yanlış şeye kayarsa bile
//  güven yüksek kalır (ölçüldü: 0.77–0.81 doygun). Bizim eğittiğimiz YOLO ise
//  "bu kesitte DRONE var mı?" sorusunu yanıtlar → hedef drone değilse güven
//  düşer, GuidanceController'ın SUSPECT eşiği gerçekten anlam kazanır.
//
//  ÇALIŞMA ŞEKLİ (tracking-by-detection):
//    init  : kutuyu kaydet (şablon YOK — kimlik = uzamsal süreklilik).
//    track : son kutunun çevresinden kare arama penceresi kes (ROI≤128 native,
//            >128 ise 128'e küçült — PHASE0 kuralı), YOLO koş, adaylardan
//            önceki merkeze EN YAKIN + güvenli olanı seç, tam-kare koordinata
//            geri çevir. confidence = YOLO sınıf güveni (drone-luk).
//
//  DONANIM: bu sınıf cv::dnn/CPU referansı; N6'da aynı ağ NPU int8 koşacak,
//  decode + pencere mantığı (bu dosya) M55'te kalacak. Arayüz değişmez.
// ============================================================================
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "dtrack/guidance/single_target_tracker.hpp"

namespace dtrack {

class YoloRoiTracker : public ISingleTargetTracker {
public:
    struct Params {
        std::string model_path = "models/yolo11_drone.onnx";
        int   input    = 128;   // ağın girdi kenarı (eğitimle aynı)
        float conf_min = 0.25f; // bu güvenin altındaki YOLO kutuları yok sayılır
        // Arama penceresi: max(input, search_scale × hedefin büyük kenarı).
        // Küçük hedefte native 128 (downscale yok); büyük hedefte 128'e küçültülür.
        // 2.5 İDİ → 1.4'e DÜŞÜRÜLDÜ: model crops128 ile (hedef kareyi DOLDURUR,
        // ölçüldü %74 "devasa") eğitildi; 2.5× pencerede 150px hedef 128'de ~51px
        // kalıp eğitim dağılımının DIŞINA düşüyor → conf çöküyor. GT-merkezli
        // ölçüm (3150-3300): ölçek 2.5→1.4 ort. conf 0.378→0.66 (≈2×). Yavaş
        // hedef (~1.4px/kare) için 1.4× margin yeter; hızlı sıçramayı miss_expand
        // açar. >91px hedefte etkili (altı zaten native 128).
        float search_scale = 1.4f;
        // Aday seçimi: skor = conf − dist_weight × (merkeze uzaklık / pencere)
        //                          − size_weight × |log(alan / önceki alan)|.
        // Mesafe terimi: penceredeki BAŞKA drone'a atlama önlemi. Boyut terimi:
        // model "yalnız gövde" ↔ "gövde+pervane bulanıklığı" hipotezleri arasında
        // SALINIYOR (ölçüldü: 88↔164px, alan oranı 3.5 → integrity FAIL); önceki
        // kutuyla boyut tutarlılığı skora girince model tutarlı hipotezde kalır.
        // log: 2× büyüme ile 2× küçülme aynı cezayı alsın (oran simetrisi).
        float dist_weight = 0.5f;
        float size_weight = 0.3f;
        // Boyut-EMA: çıktı kutusunun w,h'ı düzleştirilir (yeni = α×yolo + (1−α)×önceki).
        // MERKEZ HAM KALIR: dron hızlı manevra yapar, merkez yumuşatması kutuyu
        // hedefin gerisinde sürükler (lag). Salınım boyutta — tedavi de boyutta.
        float size_ema = 0.6f;
        // KAÇIRMA GENİŞLETMESİ: tespit gelmeyen her karede arama penceresi
        // kenarı (1 + miss_expand × kaçırma) katına çıkar (miss_expand_max'a
        // kadar). NEDEN: pencere son kutuda SABİT kalırsa hızlı hedef/kamera
        // hareketinde hedef pencereden çıkar ve conf=0'a SAPLANIR (ölçüldü,
        // 3185–3224: 40 kare conf=0). Bedel: büyük pencere 128'e küçülür →
        // küçük hedef incelir; bu yüzden sınırlı ve kademeli.
        float miss_expand     = 0.5f;
        int   miss_expand_max = 4;
    };

    // model yüklenemezse fırlatır (fail-fast). NOT: varsayılan argüman ("= {}")
    // BİLEREK yok — sınıf içinde Params'ın kendi varsayılanları henüz tam
    // ayrıştırılmadığı için C++ buna izin vermez (clang hatası).
    explicit YoloRoiTracker(Params p);

    cv::Rect init(const cv::Mat& frame, const cv::Rect& bbox,
                  bool refine = true) override;
    STResult track(const cv::Mat& frame) override;
    void apply_motion(const cv::Matx33f& M) override;

private:
    Params       p_;
    cv::dnn::Net net_;
    cv::Rect     last_box_;
    bool         ready_ = false;
    // init sonrası İLK track'te boyut-EMA atlanır: tohum (detector blobu) boyutu
    // gevşektir, EMA ile modelin sıkı kutusuna bulaşıp büyüme referansını kirletir.
    bool         fresh_init_ = false;
    int          miss_count_ = 0;  // ardışık tespitsiz kare (pencere genişletme)
};

} // namespace dtrack
