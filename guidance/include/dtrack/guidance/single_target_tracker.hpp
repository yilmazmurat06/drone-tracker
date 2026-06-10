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
    // DÖNÜŞ: etkin kilit kutusu. Tracker tohumu RAFİNE EDEBİLİR (örn. YOLO tohum
    // blobunu kendi sıkı tespitine oturtur); çağıran (GuidanceController) boyut/
    // hareket referanslarını DÖNEN kutuya kurmalı — tohum detector blobu gevşek
    // olabilir, referans yanlış kurulursa integrity sonsuz FAIL(size) döngüsüne
    // girer (ölçüldü). Rafine etmeyen tracker bbox'ı (kareye kırpıp) aynen döndürür.
    //
    // refine: bbox GEVŞEK bir detector blobu mu (true → tracker kendi sıkı kutusuna
    // oturtabilir) yoksa ZATEN SIKI bir kutu mu (false → olduğu gibi benimse, yeniden
    // snap YOK). İlk pilot-pikseli kilidi gevşek blob verir (refine=true); re-acquire
    // zaten doğrulanmış sıkı YOLO adayı verir (refine=false) — yeniden snap, kararlı
    // kılmaya çalıştığımız hipotez seçimini yeniden açıp kutuyu patlatır (ölçüldü,
    // kare 3185: 128×94 → 320×181). Şablon-tabanlı tracker'lar için anlamsız (yok say).
    virtual cv::Rect init(const cv::Mat& frame, const cv::Rect& bbox,
                          bool refine = true) = 0;

    // Bu karede hedefi ara; bbox + güven döndür. init() çağrılmadan çağrılmamalı.
    virtual STResult track(const cv::Mat& frame) = 0;

    // EGO-HAREKET TELAFİSİ: iç hedef konumunu M homografisiyle yeni karenin
    // koordinat çerçevesine taşı (track()'ten ÖNCE çağrılır). NEDEN: stabilizer
    // her kareyi bir önceki HAM kareye hizalar → ardışık warped kareler AYNI
    // çerçevede DEĞİL; kamera sarsıntısında arama penceresi hedefin yanına
    // düşer (ölçüldü: conf=0 koşuları). Varsayılan no-op (şablon tracker'ların
    // arama bölgesi göreli, controller kutuları zaten taşır).
    virtual void apply_motion(const cv::Matx33f& /*M*/) {}
};

} // namespace dtrack
