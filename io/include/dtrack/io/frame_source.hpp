#pragma once
// ============================================================================
//  IFrameSource — kare kaynağı arayüzü (soyut/abstract sınıf).
//
//  ÖĞREN: "arayüz" = saf sanal (pure virtual) metotlardan oluşan, gövdesi
//  olmayan bir sözleşme. Pipeline yalnızca bu sözleşmeyi bilir; karelerin
//  nereden geldiğini (kayıtlı .mp4, sentetik, canlı ekran) UMURSAMAZ.
//  Bu sayede aynı pipeline'a farklı kaynaklar takılabilir (bağımlılığın
//  tersine çevrilmesi / dependency inversion).
// ============================================================================
#include "dtrack/core/types.hpp"

namespace dtrack {

class IFrameSource {
public:
    virtual ~IFrameSource() = default;

    // Sıradaki kareyi 'out'a yazar.
    // return true  → kare alındı
    //        false → akış bitti (dosya sonu) veya kare yok
    virtual bool next(Frame& out) = 0;
};

} // namespace dtrack
