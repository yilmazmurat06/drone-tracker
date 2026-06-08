// ============================================================================
//  interpolate_telemetry birim testleri (dosya gerektirmez — saf mantık).
// ============================================================================
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dtrack/io/telemetry_log.hpp"

using dtrack::Telemetry;
using dtrack::interpolate_telemetry;

static std::vector<Telemetry> two_samples() {
    Telemetry a, b;
    a.t_rel = 0.0; a.gyro_pitch = 0.0;  a.pos_x = 0.0;   a.att_w = 1.0;
    b.t_rel = 1.0; b.gyro_pitch = 10.0; b.pos_x = 100.0; b.att_w = 1.0;
    return {a, b};
}

// Tam ortada → değerlerin ortalaması (lineer).
TEST(TelemetryInterp, MidpointIsLinear) {
    auto s = two_samples();
    Telemetry m = interpolate_telemetry(s, 0.5);
    EXPECT_NEAR(m.gyro_pitch, 5.0, 1e-9);
    EXPECT_NEAR(m.pos_x, 50.0, 1e-9);
    EXPECT_NEAR(m.t_rel, 0.5, 1e-9);
}

// %25 noktası.
TEST(TelemetryInterp, QuarterPoint) {
    auto s = two_samples();
    Telemetry m = interpolate_telemetry(s, 0.25);
    EXPECT_NEAR(m.gyro_pitch, 2.5, 1e-9);
    EXPECT_NEAR(m.pos_x, 25.0, 1e-9);
}

// Aralık dışı → uçlara kırpılır (ekstrapolasyon yok).
TEST(TelemetryInterp, ClampsOutsideRange) {
    auto s = two_samples();
    EXPECT_NEAR(interpolate_telemetry(s, -5.0).gyro_pitch, 0.0, 1e-9);
    EXPECT_NEAR(interpolate_telemetry(s, 99.0).gyro_pitch, 10.0, 1e-9);
}

// Quaternion interpolasyonu birim normda kalmalı.
TEST(TelemetryInterp, QuaternionStaysUnit) {
    std::vector<Telemetry> s(2);
    s[0].t_rel = 0.0; s[0].att_x = 0; s[0].att_y = 0; s[0].att_z = 0;            s[0].att_w = 1;            // 0°
    s[1].t_rel = 1.0; s[1].att_x = 0; s[1].att_y = 0; s[1].att_z = 0.7071067811; s[1].att_w = 0.7071067811; // 90° yaw
    Telemetry m = interpolate_telemetry(s, 0.5);
    double n = std::sqrt(m.att_x * m.att_x + m.att_y * m.att_y +
                         m.att_z * m.att_z + m.att_w * m.att_w);
    EXPECT_NEAR(n, 1.0, 1e-9);
}

// Boş girdi → çökmeden varsayılan döner.
TEST(TelemetryInterp, EmptyIsSafe) {
    std::vector<Telemetry> s;
    Telemetry m = interpolate_telemetry(s, 0.5);
    EXPECT_EQ(m.t_rel, 0.0);
}
