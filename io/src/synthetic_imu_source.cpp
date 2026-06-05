#include "dtrack/io/synthetic_imu_source.hpp"

#include <cmath>

#include "dtrack/common/time.hpp"

namespace dtrack::io {

SyntheticImuSource::SyntheticImuSource(SceneConfig cfg, double imu_rate_hz,
                                       common::Timestamp t0, double gyro_noise_std,
                                       double gyro_bias_walk)
    : scene_(cfg),
      rate_(imu_rate_hz),
      t0_(t0),
      noise_std_(gyro_noise_std),
      bias_walk_(gyro_bias_walk),
      rng_(cfg.seed ^ 0x9e3779b9u) {}

bool SyntheticImuSource::open() {
    open_ = true;
    next_sample_ = 0;
    bias_ = {0, 0, 0};
    return true;
}

void SyntheticImuSource::close() { open_ = false; }

std::vector<common::ImuSample> SyntheticImuSource::drain() {
    if (!open_) return {};
    // Şu ana (wall-clock) kadar geçen süreye karşılık gelen örnekleri üret.
    const double now_t = common::millis_between(t0_, common::now()) / 1000.0;
    return generate_until(now_t);
}

std::vector<common::ImuSample> SyntheticImuSource::generate_until(double up_to_sec) {
    std::vector<common::ImuSample> out;
    if (!open_) return out;

    const auto target_count = static_cast<std::uint64_t>(up_to_sec * rate_);
    const double dt = 1.0 / rate_;
    // Random-walk adımı: ölçek sqrt(dt) ile (Brownian).
    const double walk_step = bias_walk_ * std::sqrt(dt);

    out.reserve(target_count > next_sample_ ? target_count - next_sample_ : 0);
    for (; next_sample_ < target_count; ++next_sample_) {
        const double t = next_sample_ * dt;

        // Bias'ı gezdir (her örnekte küçük rastgele adım) -> entegrede drift.
        bias_[0] += static_cast<float>(walk_step * ndist_(rng_));
        bias_[1] += static_cast<float>(walk_step * ndist_(rng_));

        const Vec2 true_rate = scene_.ego_rate(t);

        common::ImuSample s;
        s.stamp = t0_ + std::chrono::duration_cast<common::Duration>(
                            std::chrono::duration<double>(t));
        s.angular_velocity = {
            true_rate.x + bias_[0] + static_cast<float>(noise_std_ * ndist_(rng_)),
            true_rate.y + bias_[1] + static_cast<float>(noise_std_ * ndist_(rng_)),
            0.0f + static_cast<float>(noise_std_ * ndist_(rng_))};
        // İvmeölçeri şimdilik modellemiyoruz (stabilizasyon yalnız gyro kullanıyor).
        s.acceleration = {0, 0, 9.81f};
        out.push_back(s);
    }
    return out;
}

}  // namespace dtrack::io
