#pragma once
//
// KltGyroStabilizer: gyro + sparse optical flow füzyonuyla ego-motion telafisi
// (IStabilizer implementasyonu, bkz. Problem 1).
//
// KLT = Kanade-Lucas-Tomasi (köşe tespiti + piramidal optical flow takibi).
//
// Akış (her kare):
//   1) PREDICT  : IMU gyro'yu kare aralığında entegre et -> tahmini kare-arası
//                 kamera dönüşü -> piksel kayma = focal * dtheta. Bias çıkarılır.
//   2) MEASURE  : önceki karenin köşelerini LK ile bu kareye takip et (ileri-geri
//                 hata kontrolü), RANSAC similarity ile arka plan hareketini
//                 robust kestir (hareketli hedef = outlier).
//   3) CORRECT  : OF geçerliyse ölçümle gyro bias'ını güncelle (online kalibrasyon);
//                 dönüşüm için OF (drift'siz) kullanılır. OF geçersizse (gökyüzü)
//                 bias-düzeltilmiş gyro'ya düşülür ve ego.valid=false işaretlenir.
//   4) WARP     : kestirilen kare-arası dönüşümle güncel kareyi önceki karenin
//                 koordinatına hizala -> arka plan sabit.
//
// Çıktı EgoMotion.homography: GÜNCEL kareyi ÖNCEKİ kare koordinatına eşleyen
// kare-arası dönüşüm (3x3). Bir sonraki (detection) aşaması arka plan modelini
// bununla hizalı tutar.

#include <opencv2/core.hpp>

#include <vector>

#include "dtrack/common/types.hpp"
#include "dtrack/stabilization/stabilizer.hpp"

namespace dtrack::stabilization
{

    struct StabilizerConfig
    {
        // Kamera içsel parametresi: odak uzaklığı (piksel). Sentetikte 700; gerçekte
        // kalibrasyondan gelir. Gyro açısı <-> piksel kayma dönüşümünde kullanılır.
        float focal_px = 700.0f;

        // Köşe (feature) tespiti — Shi-Tomasi (goodFeaturesToTrack).
        int max_features = 200;
        double feature_quality = 0.01;
        double min_feature_distance = 8.0;
        int redetect_below = 60; // takip edilen köşe bu sayının altına düşünce yeniden tespit

        // Lucas-Kanade takibi
        int lk_window = 21;       // arama penceresi (px)
        int lk_pyramid = 3;       // piramit seviyesi
        double fb_error_px = 1.0; // ileri-geri (forward-backward) tutarlılık eşiği

        // RANSAC robust similarity kestirimi
        double ransac_thresh_px = 3.0;
        int min_inliers = 12; // bunun altında OF geçersiz -> gyro fallback

        // Füzyon kazançları (sabit = kararlı-durum Kalman kazancı / tümleyici filtre).
        float k_bias = 0.05f; // gyro bias düzeltme kazancı (OF innovation'dan)
    };

    class KltGyroStabilizer : public IStabilizer
    {
    public:
        explicit KltGyroStabilizer(StabilizerConfig cfg = {}) : cfg_(cfg) {}

        common::StabilizedFrame stabilize(
            const common::FramePtr &frame,
            const std::vector<common::ImuSample> &imu_samples) override;

        void reset() override;

        // Online kestirilen gyro bias [rad/s] (x=yatay/yaw, y=dikey/pitch). Test/teşhis için.
        cv::Vec2f estimated_bias() const { return bias_; }

    private:
        // imu_samples'ı [prev_stamp, frame.stamp] aralığında entegre eder.
        // Dönüş: bias-düzeltilmiş tahmini kare-arası piksel kayma (arka plan, prev->curr).
        cv::Vec2f integrate_gyro(const std::vector<common::ImuSample> &imu_samples,
                                 common::Timestamp frame_stamp, double &dt_out) const;

        StabilizerConfig cfg_;

        cv::Mat prev_gray_;
        std::vector<cv::Point2f> prev_pts_;
        common::Timestamp prev_stamp_{};
        bool has_prev_{false};

        cv::Vec2f bias_{0.0f, 0.0f}; // online gyro bias kestirimi [rad/s]
    };

} // namespace dtrack::stabilization
