// ============================================================================
//  RecordedTelemetrySource implementasyonu — .csv'yi TelemetryLog ile yükler ve
//  örnekleri SIRAYLA (akış olarak) verir. (Senkron/interpolasyon için doğrudan
//  TelemetryLog.at() kullanılır; bu sınıf ileride pipeline'ın akış ihtiyacı için.)
// ============================================================================
#include "dtrack/io/recorded_telemetry_source.hpp"

#include "dtrack/io/telemetry_log.hpp"

namespace dtrack {

struct RecordedTelemetrySource::Impl {
    TelemetryLog log;
    std::size_t  i = 0;
};

RecordedTelemetrySource::RecordedTelemetrySource(std::string path)
    : impl_(std::make_unique<Impl>()) {
    impl_->log.load(path);
}

RecordedTelemetrySource::~RecordedTelemetrySource() = default;

bool RecordedTelemetrySource::next(Telemetry& out) {
    if (impl_->i >= impl_->log.size()) return false;
    out = impl_->log.samples()[impl_->i++];
    return true;
}

} // namespace dtrack
