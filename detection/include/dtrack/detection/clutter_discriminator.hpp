#pragma once
// ============================================================================
//  ClutterDiscriminator — P3: klasik özellik tabanlı clutter eleyici.
//
//  FIKIR: Gökyüzü önündeki blob'ları üç zayıf ipucuyla skor:
//    1. DOLULUK (compactness): area / (bbox.w × bbox.h)
//       → Drone: kompakt, dolu. Kenar parçacıkları/streak: kutu dolu değil.
//    2. EN-BOY ORANI (aspect ratio): AR=1'e yakın → yüksek skor (log-Gauss).
//       → Drone: kübike yakın. Warp artefaktı / çizgi: aşırı uzun.
//    3. ALAN BANDI (area): [area_min, area_max] arasında tam skor,
//       dışında yumuşak düşüş.
//       → 1px gürültü veya kocaman bulut düşük skor alır.
//
//  Sonuç: score = w_compact·c + w_ar·ar + w_area·a  ∈ [0,1]
//  track.cpp'de score < threshold olanlar tracker'a iletilmez.
//
//  GELECEKTEKİ GENIŞLEME (Issue #6): score() imzası aynı kalacak;
//  NNDiscriminator aynı IDiscriminator'dan türeyip yalnız patch alacak.
// ============================================================================
#include "dtrack/detection/discriminator.hpp"

namespace dtrack {

class ClutterDiscriminator : public IDiscriminator {
public:
    struct Params {
        // --- ağırlıklar (toplamı 1 olmalı) ---
        float w_compact = 0.50f;   // doluluk ağırlığı
        float w_ar      = 0.30f;   // aspect-ratio ağırlığı
        float w_area    = 0.20f;   // alan-bant ağırlığı

        // --- aspect-ratio Gauss genişliği (log ölçeği) ---
        // ar_sigma=0.7 → AR=2 için ~skor 0.58, AR=5 için ~0.14
        float ar_sigma  = 0.70f;

        // --- alan bant sınırları (piksel²) ---
        float area_min  =   2.f;   // altı: gürültü — yumuşak ceza
        float area_max  = 800.f;   // üstü: büyük nesne — yumuşak ceza
    };

    ClutterDiscriminator() = default;
    explicit ClutterDiscriminator(Params p) : p_(p) {}

    // IDiscriminator: tespite [0,1] skor ver (yüksek = drone'a benziyor).
    float score(const Detection& d) override;

private:
    Params p_;
};

} // namespace dtrack
