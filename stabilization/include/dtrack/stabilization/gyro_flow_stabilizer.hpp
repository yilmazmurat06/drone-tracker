#pragma once
// ============================================================================
//  GyroFlowStabilizer — P1, alt-adım 3b: optik akış + GYRO füzyonu.
//
//  NEDEN GEREKLİ?  3a'daki optik akış, takip edilecek köşe olduğu sürece harika
//  çalışır. Ama hedef (ve kamera) gökyüzüne baktığında sahne ÖZELLİKSİZdir:
//  goodFeaturesToTrack köşe bulamaz, stabilize(...) false döner ve kareyi olduğu
//  gibi bırakırız → ego-motion geri gelir. İşte tam bu anda telemetrideki GYRO
//  (açısal hız) devreye girmeli: köşeye ihtiyaç duymadan, kameranın DÖNMESİNİ
//  doğrudan ölçer.
//
//  TEMEL FİKİR — "saf dönme homografisi":
//    Kamera yalnızca DÖNERSE (öteleme yok), iki kare arasındaki görüntü dönüşümü
//    bir homografidir:   H = K · R · K⁻¹
//    Burada K kamera iç matrisi (odak + asal nokta), R ise kameranın iki kare
//    arası 3B dönmesi. Gyro bize açısal hızı (ω, rad/s) verir; kareler arası
//    Δt ile çarpıp integre edersek dönme vektörünü (θ = ∫ω dt) buluruz, ardından
//    Rodrigues ile θ → R. Öteleme (paralaks) bu modelde YOK — ama uzak arka plan
//    (gökyüzü, ufuk) için paralaks ihmal edilebilir, yani tam da OF'un çöktüğü
//    yerde gyro modeli en doğru olduğu yerdedir. İki yöntem birbirini tamamlar.
//
//  FÜZYON STRATEJİSİ (bu adımda basit "seç-ya da-düş"):
//    1) Optik akışı dene. Yeterli inlier'lı güvenilir homografi bulursa ONU kullan
//       (paralaksı da kapsar, en doğrusu). Mode = Flow.
//    2) Bulamazsa gyro homografisine düş. Mode = Gyro.
//    3) İkisi de yoksa (ilk kare / telemetri yok) kimlik. Mode = None.
//    İleride: iki kestirimi güvene göre AĞIRLIKLI harmanlama.
//
//  EKSEN KALİBRASYONU (AxisMap):
//    Gyro eksenleri drone GÖVDE çerçevesinde (pitch/roll/yaw); homografi ise
//    GÖRÜNTÜ/kamera çerçevesi ister. Bu ikisi arasındaki işaret/sıra eşlemesi
//    simülatöre göre değişir ve baştan bilinmez. AxisMap bu eşlemeyi (hangi gövde
//    bileşeni → hangi kamera ekseni, hangi işaretle) parametreleştirir; gerçek
//    veride att_* quaternion'a veya OF'a karşı kalibre edilir (3b'nin 2. yarısı).
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/stabilization/optical_flow_stabilizer.hpp"
#include "dtrack/stabilization/stabilizer.hpp"

namespace dtrack {

class GyroFlowStabilizer : public IStabilizer {
public:
    // Gövde gyro'su (pitch=0, roll=1, yaw=2) → kamera dönme ekseni eşlemesi.
    // Kamera ekseni k, gövde bileşeni src[k]'den sign[k] işaretiyle beslenir.
    // VARSAYILAN = Liftoff verisinden kalibre edildi (Adım 3b-ii, dtrack_calibrate;
    // 3 uçuşta da tutarlı): cam_x=−pitch, cam_y=+yaw, cam_z=−roll.
    struct AxisMap {
        int   src[3]  = {0, 2, 1};
        float sign[3] = {-1.f, +1.f, -1.f};
    };

    struct Params {
        float   fov_h_deg = 128.f;   // yatay görüş açısı → odak f (kalibre: ~124–132°)
        AxisMap axes;                // gyro→kamera eksen eşlemesi (kalibrasyon)
        OpticalFlowStabilizer::Params of;  // gömülü optik akış parametreleri
    };

    // Hangi yolla stabilize edildi? (teşhis/görselleştirme için)
    enum class Mode { None, Flow, Gyro };

    GyroFlowStabilizer();
    explicit GyroFlowStabilizer(Params params);

    // IStabilizer sözleşmesi.
    //   telemetry_window : ÖNCEKİ ve GEÇERLİ kare zamanları arasını kapsayan,
    //                      t_rel'e göre ARTAN sıralı gyro örnekleri. ω bu pencere
    //                      üzerinde trapez integralle θ (kareler-arası dönme) verir.
    //   return true  → kare stabilize edildi (Flow ya da Gyro)
    //          false → stabilize edilemedi (ilk kare / telemetri yok → kimlik)
    bool stabilize(const Frame& in,
                   const std::vector<Telemetry>& telemetry_window,
                   Frame& out,
                   cv::Matx33f& homography) override;

    void reset();

    Mode last_mode()    const { return last_mode_; }
    int  last_tracked() const { return of_.last_tracked(); }
    int  last_inliers() const { return of_.last_inliers(); }

    // --- Açık (testlenebilir) yardımcılar -----------------------------------
    // Verilen kare boyutu için iç matris K (asal nokta = merkez).
    // Tek argümanlı sürüm p_.fov_h_deg kullanır; iki argümanlı sürüm (kalibrasyon
    // taraması için) FOV'u dışarıdan alır.
    cv::Matx33f intrinsics(const cv::Size& image_size) const;
    cv::Matx33f intrinsics(const cv::Size& image_size, float fov_h_deg) const;

    // Saf dönme homografisi: H = K · Rodrigues(rotvec) · K⁻¹.
    static cv::Matx33f rotation_homography(const cv::Matx33f& K, const cv::Vec3f& rotvec);

    // telemetry_window üzerindeki ω'yu (AxisMap uygulanmış, kamera çerçevesinde)
    // trapez integralle topla → ÖNCEKİ→GEÇERLİ kare dönme vektörü θ_cam (radyan).
    cv::Vec3f integrate_rotation(const std::vector<Telemetry>& window) const;

private:
    Params                p_;
    OpticalFlowStabilizer of_;        // gömülü 3a stabilizatörü (prev_gray'i o tutar)
    bool                  has_prev_ = false;
    Mode                  last_mode_ = Mode::None;
};

} // namespace dtrack
