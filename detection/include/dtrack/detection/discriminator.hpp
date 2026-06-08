#pragma once
// ============================================================================
//  IDiscriminator — P3: bir tespitin "drone olma" olasılığını skorlar.
//
//  FİKİR: Hız tek başına yetmez (drone hızındaki kuş?). Birkaç zayıf ayırt
//  ediciyi birleştir: şekil kararlılığı (drone sabit geometri, kuş kanat
//  çırpar), hareket tutarlılığı (motor tahriki düzgün vs periyodik çırpış).
//  Termal ayraç ŞİMDİLİK YOK (Liftoff termal vermiyor — ertelendi).
//
//  Skor [0,1] döner; eşik menzile göre ayarlanır (uzakta düşük, yakında yüksek).
//
//  NOT: Bazı ayraçlar geçmiş kareleri ister (örn. aspect-ratio varyansı). Bu
//  durumda implementasyon kendi iç durumunu (state) tutar; arayüz sade kalır.
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM. İmplementasyon Adım 5'te.
// ============================================================================
#include "dtrack/core/types.hpp"

namespace dtrack {

class IDiscriminator {
public:
    virtual ~IDiscriminator() = default;

    // Tek bir tespite "drone olma" skoru verir, [0,1].
    virtual float score(const Detection& detection) = 0;
};

} // namespace dtrack
