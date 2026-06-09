#pragma once
// ============================================================================
//  ISingleTargetTracker — Output/güdüm katmanı: TEK hedef görsel tracker.
//
//  NEDEN AYRI BİR ARAYÜZ?  P4 MultiTargetTracker çok-hedef veri-ilişkilendirme
//  yapar (Kalman + gate + M-of-N); gürültüyü temporal tutarlılıkla eler ve PİLOTA
//  ADAY sunar. Pilot bir drone'a kilitlendikten SONRA ise sorun değişir: artık
//  "hangi blob hangi iz?" değil, "BU hedef görsel olarak bir sonraki karede
//  NEREDE?" sorusu vardır. Bunu çözmenin doğru yolu görsel-özellik (appearance)
//  eşlemesidir: bir ŞABLON (template) çıkar, sonraki karede arama bölgesinde
//  en benzer yeri bul. Siamese ağların (SiamFC/NanoTrack) yaptığı tam budur.
//
//  DONANIM HEDEFİ: bu arayüzün gerçek implementasyonu NPU üzerinde int8 koşan
//  küçük bir Siamese ağ olacak (bkz. yol haritası: 2.5MB / ~0.6 TOPS bütçe).
//  Arayüz donanımdan BAĞIMSIZDIR: GuidanceController yalnız bu sözleşmeyi bilir,
//  bu yüzden NccTemplateTracker (CPU stub) → SiameseTracker (NPU) takası
//  GuidanceController'ı hiç değiştirmeden yapılabilir.
//
//  SÖZLEŞME:
//    init(frame, bbox) : kilitli hedefin şablonunu (z) çıkar — referans gömme.
//    track(frame)      : hedefin son konumu çevresindeki arama bölgesinde (x)
//                        en yüksek benzerliği bul. confidence ∈ [0,1] = normalize
//                        tepe skoru. BU confidence, güdümün "precision" metriğidir:
//                        eşiğin altına düşerse GuidanceController şüpheye düşüp
//                        detection'a geri döner (re-acquire).
// ============================================================================
#include <opencv2/core.hpp>

namespace dtrack {

// Tek-hedef tracker'ın bir karelik çıktısı.
struct STResult {
    cv::Rect bbox;            // bu karedeki tahmini hedef kutusu
    float    confidence = 0.f;// benzerlik tepe skoru [0,1] (1 = mükemmel eşleşme)
};

class ISingleTargetTracker {
public:
    virtual ~ISingleTargetTracker() = default;

    // Kilitli hedef kutusundan şablonu çıkar (yeniden tohumlama için de çağrılır).
    virtual void init(const cv::Mat& frame, const cv::Rect& bbox) = 0;

    // Bu karede hedefi ara; bbox + güven döndür. init() çağrılmadan çağrılmamalı.
    virtual STResult track(const cv::Mat& frame) = 0;
};

} // namespace dtrack
