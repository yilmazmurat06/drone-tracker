#pragma once
//
// Tek bir saat tanımı. IMU ve kamera zaman damgalarının (timestamp) AYNI saatten
// gelmesi kritik: track-seviyesi füzyon ve gyro+flow hizalaması ms hassasiyetinde
// zaman eşleştirmesine bağlı (bkz. Problem 1 ve Problem 5).
//
// steady_clock kullanırız çünkü monotondur (geri gitmez, NTP/saat ayarından
// etkilenmez) — gecikme ölçümü ve süre farkı için doğru seçim.

#include <chrono>
#include <cstdint>

namespace dtrack::common {

using Clock = std::chrono::steady_clock;
using Timestamp = Clock::time_point;
using Duration = Clock::duration;

inline Timestamp now() { return Clock::now(); }

// İki zaman damgası arası farkı milisaniye (double) olarak verir.
inline double millis_between(Timestamp a, Timestamp b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace dtrack::common
