#pragma once
//
// IDiscriminator: "bu aday drone mu, yoksa kuş/bulut/gürültü mü?" sorusuna
// olasılık skoru üreten soyut arayüz (bkz. Problem 3).
//
// Hiçbir tek özellik kesinlik vermez; birkaç zayıf ayırt edici (termal imza,
// blob şekli kararlılığı, hareket tutarlılığı, ikinci kamera teyidi) birleştirilir
// ve [0,1] bir skora indirgenir. Eşik menzile göre değişebilir (uzakta düşük,
// yakında yüksek). Bu, tracker'ın innovation gate'inden AYRI, ondan önceki bir
// yumuşak filtredir.

#include "dtrack/common/types.hpp"

#include <vector>

namespace dtrack::detection {

class IDiscriminator {
public:
    virtual ~IDiscriminator() = default;

    // Her tespitin drone_score alanını [0,1] aralığında doldurur.
    // Geçmişe (şekil/hareket kararlılığı) ihtiyaç duyduğu için iç durum tutabilir.
    virtual void score(std::vector<common::Detection>& detections) = 0;

    virtual void reset() = 0;
};

}  // namespace dtrack::detection
