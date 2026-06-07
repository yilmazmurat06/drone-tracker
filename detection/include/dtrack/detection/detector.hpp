#pragma once
//
// IDetector: tespit aşamasının soyut arayüzü.
//
// Görev (bkz. Problem 2): stabilize karede arka plandan sıyrılan soluk piksel
// kümelerini bul. Tipik implementasyon: MOG2 (Mixture of Gaussians arka plan
// modeli) -> foreground maskesi -> connected-component ile blob'lar -> her blob
// için centroid/alan/aspect ile ilk eleme. Çıktı ham Detection listesi.
//
// Not: "drone mu?" kararı burada KESİNLEŞMEZ. Detektör aday üretir; çoklu-özellik
// skorlama (Problem 3) ve kinematik kapı (Problem 4) sonraki aşamalarda devreye girer.

#include "dtrack/common/cue.hpp"
#include "dtrack/common/types.hpp"

#include <vector>

namespace dtrack::detection {

class IDetector {
public:
    virtual ~IDetector() = default;

    // Stabilize kareden ham tespit adaylarını çıkar. Detection.centroid GÜNCEL
    // (canlı) görüntü koordinatındadır (uçuş kontrolcüsü canlı kareyi referans alır).
    virtual std::vector<common::Detection> detect(const common::StabilizedFrame& sf) = 0;

    // Kapalı-döngü geri besleme: tracker'ın bir sonraki kare için hedef tahmini
    // (bkz. common/cue.hpp). Bir SONRAKİ detect() çağrısında, global tespit kapı
    // içinde aday bulamazsa bu ROI'de düşük eşikli kurtarma yapılır. Cue'yu
    // kullanmayan detektörler için no-op (varsayılan).
    virtual void set_cue(const common::TargetCue& /*cue*/) {}

    // Son işlenen kareyi, detektörün KARARLI (referans) koordinatına eşleyen 3x3
    // dönüşüm. Tracking bunu kullanarak ego-jitter'dan arınmış, pürüzsüz koordinatta
    // takip eder. Ego-telafisi yapmayan detektörler için kimlik (varsayılan).
    virtual cv::Matx33f image_to_reference() const { return cv::Matx33f::eye(); }

    // Arka plan modelini sıfırla (örn. sahne/aydınlanma sert değişti, track restart).
    virtual void reset() = 0;
};

}  // namespace dtrack::detection
