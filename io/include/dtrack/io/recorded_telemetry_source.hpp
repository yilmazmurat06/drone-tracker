#pragma once
// ============================================================================
//  RecordedTelemetrySource — kayıtlı .csv telemetri dosyasından örnek okur.
//  (data/flight_XX.telemetry.csv → 28 sütun, ~100 Hz)
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM. Implementasyon Adım 2'de.
//         O zaman t_rel↔video senkronu ve interpolasyon da burada/komşuda olacak.
// ============================================================================
#include <memory>
#include <string>
#include "dtrack/io/telemetry_source.hpp"

namespace dtrack {

class RecordedTelemetrySource : public ITelemetrySource {
public:
    explicit RecordedTelemetrySource(std::string path);
    ~RecordedTelemetrySource() override;

    bool next(Telemetry& out) override;  // impl Adım 2

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dtrack
