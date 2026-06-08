#pragma once
// ============================================================================
//  IDetector — P2: sub-pixel tespit arayüzü.
//
//  FİKİR: Hedef sadece 2-6 piksel → tanınacak şekil/doku yok. Bu bir
//  sinyal-gürültü problemi: arka plan modelinden (MOG2) sapan soluk piksel
//  kümelerini bul, connected-component ile blob'la, centroid/alan/oran ile
//  ham aday Detection'lar üret. "Drone mu?" sorusu burada SORULMAZ (bkz.
//  IDiscriminator).
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM. İmplementasyon Adım 4'te (örn. Mog2Detector).
// ============================================================================
#include <vector>
#include "dtrack/core/types.hpp"

namespace dtrack {

class IDetector {
public:
    virtual ~IDetector() = default;

    // Stabilize edilmiş kareden ham aday tespitleri çıkarır.
    //   stabilized : (tercihen) P1'den geçmiş kare
    //   out        : bulunan aday blob'lar (boş olabilir)
    // return false → tespit yapılamadı (örn. arka plan modeli henüz öğrenmedi)
    virtual bool detect(const Frame& stabilized, std::vector<Detection>& out) = 0;
};

} // namespace dtrack
