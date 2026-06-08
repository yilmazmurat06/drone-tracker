#pragma once
// ============================================================================
//  ITelemetrySource — telemetri (gyro/attitude/...) kaynağı arayüzü.
//  IFrameSource'un telemetri eşdeğeri. Kaynak: kayıtlı .csv (Adım 2),
//  ileride canlı UDP (gerçek Liftoff).
// ============================================================================
#include "dtrack/core/types.hpp"

namespace dtrack {

class ITelemetrySource {
public:
    virtual ~ITelemetrySource() = default;

    // Sıradaki telemetri örneğini 'out'a yazar.
    // return false → akış bitti.
    virtual bool next(Telemetry& out) = 0;
};

} // namespace dtrack
