#pragma once
//
// IStabilizer: ego-motion (kendi platform hareketi) telafisinin soyut arayüzü.
//
// Görev (bkz. Problem 1): kameranın kendi dönme/titreme hareketini, gyro örnekleri
// ile sparse optical flow füzyonundan kestirip bir homography (3x3) olarak ver;
// kareyi warp ederek arka planı sabitle. Geriye yalnızca hedefin GERÇEK hareketi
// kalsın ki detektör onu sahte hareketten ayırt edebilsin.

#include "dtrack/common/types.hpp"

#include <vector>

namespace dtrack::stabilization {

class IStabilizer {
public:
    virtual ~IStabilizer() = default;

    // Bir kareyi, o kareye zamanca denk gelen IMU örnekleriyle birlikte alır;
    // ego-motion'u kestirir ve stabilize (warp edilmiş) kareyi döndürür.
    // imu_samples boş olabilir (özelliksiz gökyüzü / IMU yok) -> ego.valid=false
    // ve mümkünse saf gyro'ya / birim matrise düşülür.
    virtual common::StabilizedFrame stabilize(
        const common::FramePtr& frame,
        const std::vector<common::ImuSample>& imu_samples) = 0;

    // İç durumu (önceki kare, izlenen köşeler, Kalman) sıfırla -> track yeniden başlatma.
    virtual void reset() = 0;
};

}  // namespace dtrack::stabilization
