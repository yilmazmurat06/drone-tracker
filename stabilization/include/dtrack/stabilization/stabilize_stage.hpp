#pragma once
//
// StabilizeStage: stabilizasyonu pipeline'a sarar.
//   girdi:  FramePtr (kameradan)
//   çıktı:  StabilizedFrame (stabilize kare + EgoMotion)
//
// IMU pipeline'da ayrı bir stage DEĞİL: bu stage IImuSource'u tutar ve her karede
// drain() ile o kareye denk gelen gyro örneklerini çeker. Böylece kare ile IMU
// zaman damgası üzerinden eşlenir (kamera ve IMU aynı t0'ı paylaşır).

#include <memory>
#include <optional>

#include "dtrack/common/types.hpp"
#include "dtrack/io/imu_source.hpp"
#include "dtrack/pipeline/stage.hpp"
#include "dtrack/stabilization/stabilizer.hpp"

namespace dtrack::stabilization {

class StabilizeStage
    : public pipeline::Stage<common::FramePtr, common::StabilizedFrame> {
public:
    StabilizeStage(std::shared_ptr<IStabilizer> stab,
                   std::shared_ptr<io::IImuSource> imu)
        : Stage("stabilize"), stab_(std::move(stab)), imu_(std::move(imu)) {
        if (imu_ && !imu_->is_open()) imu_->open();
    }

protected:
    std::optional<common::StabilizedFrame> process(common::FramePtr&& frame) override {
        if (!frame) return std::nullopt;
        // Bu ana kadar biriken gyro örneklerini al (stabilizer zaman aralığına göre eler).
        std::vector<common::ImuSample> imu;
        if (imu_) imu = imu_->drain();
        return stab_->stabilize(frame, imu);
    }

private:
    std::shared_ptr<IStabilizer> stab_;
    std::shared_ptr<io::IImuSource> imu_;
};

}  // namespace dtrack::stabilization
