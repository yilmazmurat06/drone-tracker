#pragma once
// ============================================================================
//  OpticalFlowStabilizer — P1, alt-adım 3a: optik akış tabanlı stabilizasyon.
//
//  YÖNTEM:
//    1) Önceki karede iyi köşe noktaları bul (goodFeaturesToTrack).
//    2) Bu noktaları geçerli kareye KLT optical flow ile takip et
//       (calcOpticalFlowPyrLK).
//    3) Nokta çiftlerinden cur→prev homografisini RANSAC ile kestir
//       (findHomography) — RANSAC, gerçek hareketli nesnelerin/yanlış
//       takiplerin (aykırı değer) sonucu bozmasını engeller.
//    4) Geçerli kareyi bu homografiyle warp et → önceki kareyle hizalanır.
//
//  Köşe yoksa (özelliksiz gökyüzü) güvenli geri dönüş: kimlik dönüşümü +
//  stabilize(...)=false. İleride (3b) bu durumda gyro devralacak.
//
//  Durum bilgisi: önceki gri kareyi içinde tutar (ardışık kareler arası çalışır).
// ============================================================================
#include <opencv2/core.hpp>

#include "dtrack/stabilization/stabilizer.hpp"

namespace dtrack {

class OpticalFlowStabilizer : public IStabilizer {
public:
    struct Params {
        int    max_corners   = 300;    // takip edilecek azami köşe
        double quality_level = 0.01;   // köşe kalite eşiği (goodFeaturesToTrack)
        double min_distance  = 12.0;   // köşeler arası asgari mesafe (piksel)
        int    lk_window     = 21;     // KLT pencere boyu
        double ransac_reproj = 3.0;    // RANSAC yeniden-izdüşüm eşiği (piksel)
        int    min_inliers   = 15;     // bu kadar inlier yoksa güvenme
    };

    OpticalFlowStabilizer();                       // varsayılan parametreler
    explicit OpticalFlowStabilizer(Params params);

    // IStabilizer: 'in'i önceki kareyle hizalar. telemetry bu alt-adımda kullanılmaz.
    // return true  → güvenilir homografi bulundu ve uygulandı
    //        false → ilk kare ya da yetersiz köşe (out = kopya, H = kimlik)
    bool stabilize(const Frame& in,
                   const std::vector<Telemetry>& telemetry_window,
                   Frame& out,
                   cv::Matx33f& homography) override;

    void reset();  // dahili önceki-kare durumunu temizler

    // Teşhis/demo için son kare istatistikleri:
    int last_tracked() const { return last_tracked_; }
    int last_inliers() const { return last_inliers_; }

private:
    Params  p_;
    cv::Mat prev_gray_;
    bool    has_prev_ = false;
    int     last_tracked_ = 0;
    int     last_inliers_ = 0;
};

} // namespace dtrack
