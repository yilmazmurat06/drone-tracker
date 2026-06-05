#pragma once
//
// SyntheticImuSource: SceneModel'den GERÇEK açısal hızı okuyup üstüne sensör
// kusurları ekleyen sentetik IMU.
//
// Eklenen kusurlar (Problem 1'deki drift'i gerçekçi kılmak için):
//   - bias: yavaşça gezinen (random walk) sabit-benzeri ofset -> entegre edilince
//     drift üretir; stabilizasyonun optical flow ile bunu düzeltmesi beklenir.
//   - white noise: her örnekte bağımsız gürültü.
//
// IMU kamera kare hızından çok daha hızlı örneklenir (imu_rate, örn. 1000 Hz).
// drain() son okumadan bu yana biriken tüm örnekleri zaman sırasıyla verir;
// böylece stabilizasyon stage'i bir kareyi denk gelen IMU penceresiyle eşler.

#include <random>

#include "dtrack/common/types.hpp"
#include "dtrack/io/imu_source.hpp"
#include "dtrack/io/synthetic_scene.hpp"

namespace dtrack::io {

class SyntheticImuSource : public IImuSource {
public:
    SyntheticImuSource(SceneConfig cfg, double imu_rate_hz, common::Timestamp t0,
                       double gyro_noise_std = 0.004,   // rad/s
                       double gyro_bias_walk = 0.0008); // rad/s per sqrt(s)

    bool open() override;
    void close() override;
    bool is_open() const override { return open_; }

    std::vector<common::ImuSample> drain() override;

    // Deterministik üretim: t0'dan up_to_sec saniyesine kadar henüz üretilmemiş
    // tüm örnekleri verir. drain() bunu wall-clock "now" ile çağırır; testler
    // sabit bir süre vererek deterministik doğrulama yapabilir.
    std::vector<common::ImuSample> generate_until(double up_to_sec);

private:
    SceneModel scene_;
    double rate_;
    common::Timestamp t0_;
    double noise_std_;
    double bias_walk_;

    bool open_{false};
    std::uint64_t next_sample_{0};  // üretilecek sonraki örneğin indeksi
    cv::Vec3f bias_{0, 0, 0};       // o anki gyro bias (gezinir)
    std::mt19937 rng_;
    std::normal_distribution<double> ndist_{0.0, 1.0};
};

}  // namespace dtrack::io
