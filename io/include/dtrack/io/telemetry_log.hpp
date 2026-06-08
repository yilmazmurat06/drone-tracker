#pragma once
// ============================================================================
//  TelemetryLog — bir .csv telemetri dosyasının tamamını belleğe yükler ve
//  herhangi bir t_rel anına INTERPOLASYONLU erişim sağlar.
//
//  NEDEN bütün dosyayı belleğe alıyoruz?
//  Video 60 fps, telemetri ~100 Hz → birebir hizalı DEĞİL. Kare i'nin zamanı
//  t = i/60 sn; o ana en yakın iki telemetri örneği arasında interpolasyon
//  yaparız. Bu "zamana göre arama" sıralı akıştan (next-next-next) çok daha
//  uygun olur; dosyalar küçük (~10k satır, birkaç MB) → tamamı belleğe sığar.
//
//  ÖĞREN: ITelemetrySource (akış) ile TelemetryLog (rastgele erişim) FARKLI
//  ihtiyaçlardır. Replay senkronu rastgele erişim ister; pipeline ileride akış
//  isteyebilir. İkisini de sağlıyoruz.
// ============================================================================
#include <cstddef>
#include <string>
#include <vector>

#include "dtrack/core/types.hpp"

namespace dtrack {

// İki örnek arasında (veya uçlarda kırpılmış) t_rel anının interpolasyonu.
//  - skaler alanlar (gyro/pos/vel): lineer
//  - quaternion (att_*): en kısa yoldan nlerp + normalize
// 'samples' t_rel'e göre ARTAN sıralı olmalı.
Telemetry interpolate_telemetry(const std::vector<Telemetry>& samples, double t_rel);

class TelemetryLog {
public:
    // 28 sütunlu telemetri csv'sini yükler. Başarılıysa true.
    bool load(const std::string& csv_path);

    // t_rel anındaki (interpolasyonlu) telemetri.
    Telemetry at(double t_rel) const { return interpolate_telemetry(samples_, t_rel); }

    const std::vector<Telemetry>& samples() const { return samples_; }
    bool        empty() const { return samples_.empty(); }
    std::size_t size()  const { return samples_.size(); }

    // İlk/son t_rel (yüklü değilse 0).
    double first_t_rel() const { return samples_.empty() ? 0.0 : samples_.front().t_rel; }
    double last_t_rel()  const { return samples_.empty() ? 0.0 : samples_.back().t_rel; }

private:
    std::vector<Telemetry> samples_;  // t_rel'e göre sıralı
};

} // namespace dtrack
