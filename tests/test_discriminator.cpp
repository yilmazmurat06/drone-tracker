// ============================================================================
//  ClutterDiscriminator birim testi.
//
//  STRATEJİ: Bilinen özellikte sentetik Detection nesneleri oluştur; skorların
//  beklenen sıralamasını (drone > streak > gürültü > dev cisim) doğrula.
//  Gerçek görüntü gerekmez — özellik vektörü doğrudan hesaplanıyor.
// ============================================================================
#include <gtest/gtest.h>
#include "dtrack/detection/clutter_discriminator.hpp"

using namespace dtrack;

namespace {

Detection make_det(float area, int bw, int bh) {
    Detection d;
    d.area         = area;
    d.bbox         = cv::Rect(0, 0, bw, bh);
    d.aspect_ratio = bh > 0 ? static_cast<float>(bw) / bh : 1.f;
    d.centroid     = {bw / 2.f, bh / 2.f};
    return d;
}

} // namespace

// Kompakt kare blob (5×5, dolu) → yüksek skor
TEST(ClutterDiscriminator, CompactBlob_HighScore) {
    ClutterDiscriminator disc;
    auto d = make_det(22.f, 5, 5);  // %88 dolu, AR=1
    EXPECT_GT(disc.score(d), 0.70f);
}

// Çok ince yatay streak (60×2 kutu, 8px alan) → düşük skor
TEST(ClutterDiscriminator, ThinStreak_LowScore) {
    ClutterDiscriminator disc;
    auto d = make_det(8.f, 60, 2);  // doluluk≈0.07, AR=30 → aşırı uzun
    EXPECT_LT(disc.score(d), 0.40f);
}

// Dev cisim (500×400 kutu, 50000px alan) → alan cezası → düşük skor
TEST(ClutterDiscriminator, HugeBlob_LowScore) {
    ClutterDiscriminator disc;
    auto d = make_det(50000.f, 500, 400);  // alan >> area_max(800)
    EXPECT_LT(disc.score(d), 0.50f);
}

// Tek piksel gürültü → alan cezası var ama şekli mükemmel (doluluk=1, AR=1).
// Discriminator şekle bakar; MovingTargetDetector min_area=3 ile bunu zaten
// elediği için discriminator aşırı ceza uygulamaz — skor drone'dan biraz düşük.
TEST(ClutterDiscriminator, SinglePixel_LowerThanCompactDrone) {
    ClutterDiscriminator disc;
    auto tiny  = make_det(1.f, 1, 1);   // area=1 < area_min=2
    auto drone = make_det(22.f, 5, 5);  // iyi drone
    EXPECT_LT(disc.score(tiny), disc.score(drone));
}

// Drone > streak: kompakt blob daha yüksek skor almalı
TEST(ClutterDiscriminator, DroneBeatsStreak) {
    ClutterDiscriminator disc;
    auto drone  = make_det(22.f, 5, 5);
    auto streak = make_det(8.f, 60, 2);
    EXPECT_GT(disc.score(drone), disc.score(streak));
}

// Bbox boyutsuz giriş → sıfır döner, crash olmaz
TEST(ClutterDiscriminator, ZeroBbox_ReturnsZero) {
    ClutterDiscriminator disc;
    auto d = make_det(0.f, 0, 0);
    EXPECT_FLOAT_EQ(disc.score(d), 0.f);
}
