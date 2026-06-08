#pragma once
// ============================================================================
//  IStage<In, Out> — genel pipeline aşaması soyutlaması.
//
//  FİKİR: Her aşama bir girdiyi (In) bir çıktıya (Out) dönüştürür. Aynı aşama
//  HEM tek-thread'li runner'da (deterministik, debug kolay) HEM çok-thread'li
//  runner'da (ring buffer'larla, performans) kullanılabilir — sadece onları
//  ÇALIŞTIRAN kod değişir. (Plan kararı 5: hibrit yaklaşım.)
//
//  Şu an pipeline.hpp doğrudan modül arayüzlerini (IDetector vb.) çağıran sade
//  bir runner kullanıyor. IStage, çok-thread'li sürüme (Adım 7) geçerken
//  aşamaları tek tip bir şekilde sarmalamak için hazır duruyor.
// ============================================================================

namespace dtrack {

template <typename In, typename Out>
class IStage {
public:
    virtual ~IStage() = default;

    // 'in'i işleyip 'out' üretir.
    // return true  → 'out' geçerli bir çıktı taşıyor
    //        false → bu adımda çıktı yok (örn. arka plan hâlâ öğreniyor)
    virtual bool process(const In& in, Out& out) = 0;
};

} // namespace dtrack
