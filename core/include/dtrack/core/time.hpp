#pragma once
// ============================================================================
//  Zaman yardımcıları.
//  ÖĞREN: "steady_clock" geri gitmeyen, sadece ileri akan bir saattir —
//  süre ölçmek için doğru seçim. (system_clock duvar saatidir, NTP ile geri
//  zıplayabilir; gecikme ölçümünde kullanılmaz.)
// ============================================================================
#include <chrono>
#include "dtrack/core/types.hpp"

namespace dtrack {

// Şu an, nanosaniye olarak (monotonik).
inline Timestamp now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Saniye ↔ nanosaniye dönüşümleri (telemetri t_rel'i saniye cinsinden).
inline Timestamp seconds_to_ns(double s) { return static_cast<Timestamp>(s * 1e9); }
inline double    ns_to_seconds(Timestamp ns) { return static_cast<double>(ns) * 1e-9; }

} // namespace dtrack
