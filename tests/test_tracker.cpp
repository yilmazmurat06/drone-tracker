// ============================================================================
//  MultiTargetTracker birim testi.
//  - Düzgün hareket eden hedef → Confirmed iz, doğru konum/hız.
//  - Rastgele tutarsız gürültü → hiç Confirmed iz (M-of-N gürültüyü bastırır).
//  - Tespit boşluğunda → iz coasting ile yaşar (Kalman tahmini).
// ============================================================================
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/tracking/multi_target_tracker.hpp"

using namespace dtrack;

namespace {

Detection det_at(float x, float y, int64_t t = 0) {
    Detection d;
    d.centroid = {x, y};
    d.bbox = cv::Rect(static_cast<int>(x) - 3, static_cast<int>(y) - 3, 6, 6);
    d.area = 30.f;
    d.t = t;
    return d;
}

int count_confirmed(const std::vector<Track>& ts) {
    int n = 0;
    for (const auto& t : ts)
        if (t.status == Track::Status::Confirmed) ++n;
    return n;
}

}  // namespace

// Sabit hızla hareket eden hedef → onaylanmalı, konum/hız doğru kestirilmeli.
TEST(Tracker, ConfirmsConsistentTarget) {
    MultiTargetTracker trk;
    std::vector<Track> out;

    const float x0 = 100, y0 = 120, vx = 5, vy = 3;
    for (int k = 0; k < 8; ++k)
        trk.update({det_at(x0 + vx * k, y0 + vy * k, k)}, out);

    ASSERT_EQ(count_confirmed(out), 1);
    Track c;
    for (const auto& t : out) if (t.status == Track::Status::Confirmed) c = t;

    // Son kare k=7 → konum (135,141) civarı.
    EXPECT_NEAR(c.pos.x, x0 + vx * 7, 6.0);
    EXPECT_NEAR(c.pos.y, y0 + vy * 7, 6.0);
    // Kalman hızı gerçek hıza yakınsamalı.
    EXPECT_NEAR(c.vel.x, vx, 1.5);
    EXPECT_NEAR(c.vel.y, vy, 1.5);
}

// Rastgele, tutarsız tespitler → hiçbir Confirmed iz oluşmamalı.
TEST(Tracker, RejectsRandomNoise) {
    MultiTargetTracker trk;
    std::vector<Track> out;
    cv::RNG rng(777);

    for (int k = 0; k < 40; ++k) {
        std::vector<Detection> dets;
        for (int i = 0; i < 2; ++i)  // her kare 2 rastgele "pırıltı"
            dets.push_back(det_at(rng.uniform(0.f, 1280.f), rng.uniform(0.f, 720.f), k));
        trk.update(dets, out);
    }
    EXPECT_EQ(count_confirmed(out), 0);
}

// Tespit boşluğunda iz coasting ile yaşamalı, boşluk sonrası hâlâ Confirmed.
TEST(Tracker, CoastsThroughGap) {
    MultiTargetTracker trk;
    std::vector<Track> out;
    const float x0 = 200, y0 = 200, vx = 6, vy = 0;  // min_travel'ı geçecek hız

    int k = 0;
    for (; k < 6; ++k) trk.update({det_at(x0 + vx * k, y0, k)}, out);  // onayla
    ASSERT_EQ(count_confirmed(out), 1);

    // 3 kare tespitsiz (boşluk) → coasting.
    for (int g = 0; g < 3; ++g, ++k) trk.update({}, out);
    EXPECT_EQ(count_confirmed(out), 1) << "coasting basarisiz, iz kayboldu";

    // Tahmin ileri taşımış olmalı (x ~ x0 + vx*k).
    Track c;
    for (const auto& t : out) if (t.status == Track::Status::Confirmed) c = t;
    EXPECT_GT(c.pos.x, x0 + vx * 6 - 5);
}
