#pragma once
// ============================================================================
//  NccTemplateTracker — ISingleTargetTracker'ın ÖĞRETİCİ CPU YER TUTUCUSU.
//
//  AMAÇ: Gerçek int8 Siamese ağ NPU'da koşana kadar, güdüm durum makinesini
//  uçtan uca çalıştırabilmek ve mantığı görselleştirebilmek için bir referans
//  implementasyon. Siamese ile AYNI ARAYÜZÜ ve AYNI SEMANTİĞİ (template-vs-search,
//  [0,1] güven) modeller → GuidanceController hiç değişmeden SiameseTracker'a
//  takas edilir.
//
//  YÖNTEM (klasik, ama Siamese'le birebir aynı şema):
//    init  : bbox bölgesini gri patch olarak SAKLA (= şablon z). Siamese'de bu
//            şablonun CNN gömmesidir; burada ham piksel.
//    track : son konum etrafında bir ARAMA penceresi (search) aç, şablonu
//            cv::matchTemplate(TM_CCOEFF_NORMED) ile kaydır, tepe konumu = yeni
//            merkez, tepe değeri ([-1,1]→[0,1]) = confidence. Siamese'de bu,
//            korelasyon/cross-correlation yanıt haritasının tepesidir.
//
//  NOT: Bu stub APPEARANCE öğrenmez (ölçek/dönme/aydınlatmaya kırılgan); amacı
//  doğruluk değil, ARAYÜZ ve GÜDÜM AKIŞINI doğrulamak. Gerçek dayanıklılık
//  Siamese ağdan gelecek.
// ============================================================================
#include "dtrack/guidance/single_target_tracker.hpp"

namespace dtrack {

class NccTemplateTracker : public ISingleTargetTracker {
public:
    struct Params {
        // Arama penceresi, şablon kutusunun kaç katı büyüklüğünde olsun?
        // Büyük = hızlı hedefi yakalar ama daha pahalı/daha çok dikkat dağıtıcı.
        float search_scale = 2.5f;
        int   min_search   = 48;   // arama penceresi kenarı için taban (px)
    };

    NccTemplateTracker() = default;
    explicit NccTemplateTracker(Params p) : p_(p) {}

    cv::Rect init(const cv::Mat& frame, const cv::Rect& bbox,
                  bool refine = true) override;
    STResult track(const cv::Mat& frame) override;

private:
    Params   p_;
    cv::Mat  templ_;             // gri şablon patch (z)
    cv::Rect last_box_;          // son bilinen hedef kutusu
    bool     have_templ_ = false;
};

} // namespace dtrack
