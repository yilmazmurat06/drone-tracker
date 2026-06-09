// ============================================================================
//  test_lock_integrity — kilit-bütünlüğü geometrik denetçilerinin birim testleri.
//
//  Sentetik patch'lerle (gerçek video gerekmez) üç ekseni ayrı ayrı doğrularız:
//  gök-çevre oranı, boyut sağlığı, hareket sağlığı.
// ============================================================================
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "dtrack/guidance/lock_integrity.hpp"

using namespace dtrack;
using namespace dtrack::lock_integrity;

namespace {
// BGR sahne: tüm kareyi tek renge boya, sonra merkeze koyu bir "hedef" kutusu çiz.
cv::Mat make_scene(cv::Scalar bg, cv::Rect target, cv::Scalar target_color) {
    cv::Mat img(480, 640, CV_8UC3, bg);
    cv::rectangle(img, target, target_color, cv::FILLED);
    return img;
}
const cv::Scalar kSky(235, 170, 90);    // BGR: mavi-baskın + parlak gök
const cv::Scalar kGrass(40, 130, 60);   // BGR: yeşil-baskın çimen (B<G, koyu)
const cv::Scalar kDrone(35, 35, 35);    // koyu drone silüeti
}  // namespace

// Gök önündeki hedef: çevre halkası ≈ tamamen gök → oran ~1.
TEST(LockIntegrity, SkyRingHighForAirborne) {
    const cv::Rect box(300, 220, 40, 40);
    const cv::Mat scene = make_scene(kSky, box, kDrone);
    EXPECT_GT(sky_ring(scene, box), 0.9f);
}

// Çimen önündeki (yere sürüklenmiş) hedef: çevre yeşil-baskın+koyu → oran ~0.
TEST(LockIntegrity, SkyRingLowForGroundLock) {
    const cv::Rect box(300, 220, 40, 40);
    const cv::Mat scene = make_scene(kGrass, box, kDrone);
    EXPECT_LT(sky_ring(scene, box), 0.1f);
}

// İç kutu halkadan dışlanır: hedefin kendisi koyu olsa bile gök halka oranı yüksek kalır.
TEST(LockIntegrity, SkyRingExcludesInnerBox) {
    const cv::Rect box(300, 220, 60, 60);
    const cv::Mat scene = make_scene(kSky, box, kDrone);  // büyük koyu iç kutu
    EXPECT_GT(sky_ring(scene, box), 0.9f);                // halka hâlâ gök
}

// Boyut: normal kutu geçer; tüm kareyi kaplayan kutu (patlama) elenir.
TEST(LockIntegrity, SizeSaneRejectsBlowup) {
    const cv::Size frame(640, 480);
    const cv::Rect lock0(300, 220, 40, 40);
    EXPECT_TRUE(size_sane({300, 220, 40, 40}, frame, lock0, 0.08f, 4.0f));
    EXPECT_FALSE(size_sane({0, 0, 640, 480}, frame, lock0, 0.08f, 4.0f));  // tüm kare
}

// Boyut: kilit kutusuna göre aşırı büyüme (>4×) elenir.
TEST(LockIntegrity, SizeSaneRejectsGrowth) {
    const cv::Size frame(1920, 1080);
    const cv::Rect lock0(900, 500, 40, 40);
    EXPECT_TRUE(size_sane({900, 500, 70, 70}, frame, lock0, 0.08f, 4.0f));   // ~3×, ok
    EXPECT_FALSE(size_sane({900, 500, 120, 120}, frame, lock0, 0.08f, 4.0f));// >9×, patlama
}

// Hareket: küçük kayma geçer; büyük sıçrama (ışınlanma) elenir.
TEST(LockIntegrity, MotionSaneRejectsJump) {
    const cv::Rect prev(300, 220, 40, 40);
    EXPECT_TRUE(motion_sane({308, 225, 40, 40}, prev, 60.f));   // ~9px, ok
    EXPECT_FALSE(motion_sane({500, 400, 40, 40}, prev, 60.f));  // ~265px, ışınlanma
}

// Hareket: referans yoksa (ilk kare) daima geçer.
TEST(LockIntegrity, MotionSaneNoReference) {
    EXPECT_TRUE(motion_sane({300, 220, 40, 40}, cv::Rect(), 60.f));
}
