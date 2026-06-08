#pragma once
// ============================================================================
//  ITracker — P4: takip arayüzü.
//
//  FİKİR: Tespit "hedef var mı?", takip "nerede ve BİR SONRAKİ AN nerede
//  olacak?" sorusunu cevaplar. Bu ileri tahmin sayesinde interceptor, hedefin
//  bayat konumunu kovalamak yerine bir KESİŞİM noktasına yönelir.
//
//  Yöntem (Adım 6): Kalman state'i = 2B konum + hız. Manevra için IMM
//  (Interacting Multiple Model): sabit hız + sabit ivme + koordineli dönüş
//  modellerini paralel koşup harmanlar. Track başlatma: M-of-N kuralı
//  (örn. 5 karede 3 tespit) gürültüyü bastırır.
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM. İmplementasyon Adım 6'da.
// ============================================================================
#include <vector>
#include "dtrack/core/types.hpp"

namespace dtrack {

class ITracker {
public:
    virtual ~ITracker() = default;

    // Bu karenin tespitleriyle dahili track'leri günceller (predict + correct +
    // veri ilişkilendirme + track yönetimi) ve aktif track'leri döndürür.
    //   detections : bu karenin (skorlanmış) tespitleri
    //   out_tracks : güncel aktif track'ler (Confirmed/Tentative)
    virtual void update(const std::vector<Detection>& detections,
                        std::vector<Track>& out_tracks) = 0;
};

} // namespace dtrack
