#pragma once
//
// ITracker: takip aşamasının soyut arayüzü (bkz. Problem 4).
//
// Görev: tespitleri zaman içinde track'lere bağla, her track'in konum+hızını
// kestir ve bir sonraki anı ileri tahmin et (interceptor kesişim noktasına
// yönelsin, bayat konumu kovalamasın). Tipik implementasyon: kare başına
// predict + data association (gating) + update; hareket modeli olarak IMM
// (Interacting Multiple Model: sabit hız / sabit ivme / koordineli dönüş'ü
// paralel çalıştırıp harmanlar). Track yaşam döngüsü M-of-N ile yönetilir.

#include "dtrack/common/types.hpp"

#include <vector>

namespace dtrack::tracking {

class ITracker {
public:
    virtual ~ITracker() = default;

    // Bir karenin (skorlanmış) tespitlerini işler:
    //   1) tüm track'leri bu zaman damgasına predict et
    //   2) tespitleri track'lere ata (innovation gate ile fiziksel imkânsızı ele)
    //   3) eşleşenleri update et, eşleşmeyenlerden yeni (tentative) track aç
    //   4) yaşam döngüsünü güncelle (Tentative/Confirmed/Coasting/Lost)
    // Dönüş: o anki aktif track'lerin anlık görüntüsü.
    virtual std::vector<common::Track> update(
        const std::vector<common::Detection>& detections,
        common::Timestamp stamp) = 0;

    virtual void reset() = 0;
};

}  // namespace dtrack::tracking
