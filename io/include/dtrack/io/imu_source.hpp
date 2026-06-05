#pragma once
//
// IImuSource: IMU (Inertial Measurement Unit = atalet ölçüm birimi; gyroscope +
// accelerometer) kaynağının soyut arayüzü.
//
// IMU kamera kare hızından (örn. ~60 Hz) çok daha hızlı örneklenir (örn. ~1 kHz).
// Stabilizasyonun "predict" adımını gyro besler (bkz. Problem 1). Bu yüzden tek
// örnek değil, "şu ana kadar biriken tüm örnekler" toplu alınabilmeli.

#include <vector>

#include "dtrack/common/types.hpp"

namespace dtrack::io {

class IImuSource {
public:
    virtual ~IImuSource() = default;

    virtual bool open() = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;

    // Son okumadan bu yana biriken tüm IMU örneklerini zaman sırasıyla döndürür
    // (ve iç tamponu boşaltır). Kare ile hizalamayı stabilizasyon stage'i yapar.
    virtual std::vector<common::ImuSample> drain() = 0;
};

}  // namespace dtrack::io
