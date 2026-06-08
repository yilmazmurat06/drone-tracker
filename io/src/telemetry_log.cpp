// ============================================================================
//  TelemetryLog + interpolate_telemetry implementasyonu.
// ============================================================================
#include "dtrack/io/telemetry_log.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace dtrack {

namespace {

// Bir satırı virgülden böler (sayısal csv; tırnaklı alan beklemiyoruz).
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

// Güvenli double ayrıştırma: boş/bozuksa 0 döner.
double parse_d(const std::vector<std::string>& f, int idx) {
    if (idx < 0 || idx >= static_cast<int>(f.size())) return 0.0;
    try { return std::stod(f[idx]); } catch (...) { return 0.0; }
}

double lerp(double a, double b, double t) { return a + (b - a) * t; }

} // namespace

// ----------------------------------------------------------------------------
//  Zamana göre interpolasyon.
// ----------------------------------------------------------------------------
Telemetry interpolate_telemetry(const std::vector<Telemetry>& s, double t_rel) {
    if (s.empty()) return Telemetry{};
    if (t_rel <= s.front().t_rel) return s.front();   // uçta kırp
    if (t_rel >= s.back().t_rel)  return s.back();

    // İkili arama: t_rel >= hedef olan ilk örnek (üst komşu).
    auto it = std::lower_bound(
        s.begin(), s.end(), t_rel,
        [](const Telemetry& a, double v) { return a.t_rel < v; });
    const Telemetry& hi = *it;
    const Telemetry& lo = *(it - 1);

    const double span = hi.t_rel - lo.t_rel;
    const double a = (span > 1e-12) ? (t_rel - lo.t_rel) / span : 0.0;  // [0,1]

    Telemetry r;
    r.t_rel = t_rel;
    // Skalerler → lineer.
    r.gyro_pitch = lerp(lo.gyro_pitch, hi.gyro_pitch, a);
    r.gyro_roll  = lerp(lo.gyro_roll,  hi.gyro_roll,  a);
    r.gyro_yaw   = lerp(lo.gyro_yaw,   hi.gyro_yaw,   a);
    r.pos_x = lerp(lo.pos_x, hi.pos_x, a);
    r.pos_y = lerp(lo.pos_y, hi.pos_y, a);
    r.pos_z = lerp(lo.pos_z, hi.pos_z, a);
    r.vel_x = lerp(lo.vel_x, hi.vel_x, a);
    r.vel_y = lerp(lo.vel_y, hi.vel_y, a);
    r.vel_z = lerp(lo.vel_z, hi.vel_z, a);

    // Quaternion → nlerp (en kısa yol). 100Hz örnekler arası açı küçük olduğu
    // için nlerp ≈ slerp; ucuz ve birim quaternion'u korur.
    double dot = lo.att_x * hi.att_x + lo.att_y * hi.att_y +
                 lo.att_z * hi.att_z + lo.att_w * hi.att_w;
    const double sgn = (dot < 0.0) ? -1.0 : 1.0;  // en kısa yolu seç
    double qx = lerp(lo.att_x, sgn * hi.att_x, a);
    double qy = lerp(lo.att_y, sgn * hi.att_y, a);
    double qz = lerp(lo.att_z, sgn * hi.att_z, a);
    double qw = lerp(lo.att_w, sgn * hi.att_w, a);
    const double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (n > 1e-12) { qx /= n; qy /= n; qz /= n; qw /= n; }
    r.att_x = qx; r.att_y = qy; r.att_z = qz; r.att_w = qw;

    return r;
}

// ----------------------------------------------------------------------------
//  CSV yükleme (sütunları İSME göre eşleştirir → sütun sırasına bağımlı değil).
// ----------------------------------------------------------------------------
bool TelemetryLog::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string header;
    if (!std::getline(f, header)) return false;

    std::unordered_map<std::string, int> idx;
    {
        auto cols = split_csv(header);
        for (int i = 0; i < static_cast<int>(cols.size()); ++i) idx[cols[i]] = i;
    }
    auto col = [&](const char* name) -> int {
        auto it = idx.find(name);
        return (it == idx.end()) ? -1 : it->second;
    };

    const int c_trel = col("t_rel");
    const int c_gp = col("gyro_pitch"), c_gr = col("gyro_roll"), c_gy = col("gyro_yaw");
    const int c_ax = col("att_x"), c_ay = col("att_y"), c_az = col("att_z"), c_aw = col("att_w");
    const int c_px = col("pos_x"), c_py = col("pos_y"), c_pz = col("pos_z");
    const int c_vx = col("vel_x"), c_vy = col("vel_y"), c_vz = col("vel_z");
    if (c_trel < 0) return false;  // t_rel olmadan senkron yapamayız

    samples_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto fields = split_csv(line);
        Telemetry t;
        t.t_rel = parse_d(fields, c_trel);
        t.gyro_pitch = parse_d(fields, c_gp);
        t.gyro_roll  = parse_d(fields, c_gr);
        t.gyro_yaw   = parse_d(fields, c_gy);
        t.att_x = parse_d(fields, c_ax);
        t.att_y = parse_d(fields, c_ay);
        t.att_z = parse_d(fields, c_az);
        t.att_w = parse_d(fields, c_aw);
        t.pos_x = parse_d(fields, c_px);
        t.pos_y = parse_d(fields, c_py);
        t.pos_z = parse_d(fields, c_pz);
        t.vel_x = parse_d(fields, c_vx);
        t.vel_y = parse_d(fields, c_vy);
        t.vel_z = parse_d(fields, c_vz);
        samples_.push_back(t);
    }

    // Güvence: t_rel'e göre sıralı olsun (interpolasyon ve ikili arama için şart).
    std::sort(samples_.begin(), samples_.end(),
              [](const Telemetry& a, const Telemetry& b) { return a.t_rel < b.t_rel; });

    return !samples_.empty();
}

} // namespace dtrack
