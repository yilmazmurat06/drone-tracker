// ============================================================================
//  test_guidance — GuidanceController durum makinesi geçişleri.
//
//  Gerçek tracker/discriminator yerine KONTROL EDİLEBİLİR sahteler enjekte edilir
//  → geçişleri deterministik test ederiz (görüntü içeriğinden bağımsız).
// ============================================================================
#include <gtest/gtest.h>

#include "dtrack/guidance/guidance_controller.hpp"

using namespace dtrack;

namespace {

// Güveni ve kutuyu testin dışarıdan dayattığı sahte tek-hedef tracker.
struct FakeTracker : ISingleTargetTracker {
    float    next_conf = 1.0f;
    cv::Rect next_box{10, 10, 20, 20};
    int      init_calls = 0;
    void init(const cv::Mat&, const cv::Rect& b) override { ++init_calls; next_box = b; }
    STResult track(const cv::Mat&) override { return {next_box, next_conf}; }
};

// Sabit skor döndüren sahte doğrulayıcı.
struct FakeVerifier : IDiscriminator {
    float val = 1.0f;
    float score(const Detection&) override { return val; }
};

Detection make_cand(cv::Rect box, float score) {
    Detection d;
    d.bbox = box;
    d.centroid = {box.x + box.width / 2.f, box.y + box.height / 2.f};
    d.score = score;
    return d;
}

const cv::Mat kFrame = cv::Mat::zeros(480, 640, CV_8UC1);

}  // namespace

// SEARCH: aday var ama OTOMATİK KİLİT YOK — durum Search kalır, adaylar sunulur.
TEST(Guidance, SearchDoesNotAutoLock) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    std::vector<Detection> cands{make_cand({100, 100, 20, 20}, 0.9f)};

    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);

    EXPECT_EQ(out.state, GuidanceController::State::Search);
    EXPECT_FALSE(out.has_target);
    EXPECT_EQ(out.candidates.size(), 1u);
    EXPECT_EQ(trk.init_calls, 0);
}

// Pilot seçimi → TRACK: tracker init edilir, hedef sürülür.
TEST(Guidance, SelectStartsTracking) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    std::vector<Detection> cands{make_cand({100, 100, 20, 20}, 0.9f)};

    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);   // adayları sun
    ctrl.select(0);                       // pilot 0. adayı kilitler
    EXPECT_EQ(ctrl.state(), GuidanceController::State::Track);

    trk.next_conf = 0.9f;
    ctrl.on_frame(kFrame, {}, out);       // ilk TRACK karesi → init
    EXPECT_EQ(out.state, GuidanceController::State::Track);
    EXPECT_TRUE(out.has_target);
    EXPECT_EQ(trk.init_calls, 1);
    EXPECT_TRUE(out.candidates.empty());  // TRACK'te aday sunulmaz
}

// Geçersiz seçim yok sayılır.
TEST(Guidance, InvalidSelectIgnored) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    std::vector<Detection> cands{make_cand({100, 100, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);
    ctrl.select(5);
    EXPECT_EQ(ctrl.state(), GuidanceController::State::Search);
}

// TRACK → düşük güven N kare → SUSPECT.
TEST(Guidance, LowConfidenceTriggersSuspect) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController::Params p; p.suspect_conf = 0.5f; p.suspect_frames = 3;
    GuidanceController ctrl(trk, ver, p);
    std::vector<Detection> cands{make_cand({100, 100, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.4f;  // suspect_conf altı ama lost_conf üstü
    for (int i = 0; i < 2; ++i) { ctrl.on_frame(kFrame, {}, out); }
    EXPECT_EQ(out.state, GuidanceController::State::Track);   // henüz eşik dolmadı
    ctrl.on_frame(kFrame, {}, out);                            // 3. düşük kare
    EXPECT_EQ(out.state, GuidanceController::State::Suspect);
}

// SUSPECT → doğrulama BAŞARILI → tracker yeniden tohumlanır → TRACK.
TEST(Guidance, SuspectReacquiresOnVerify) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController::Params p; p.lost_conf = 0.3f; p.suspect_conf = 0.5f;
    p.suspect_frames = 1; p.verify_score = 0.5f;
    GuidanceController ctrl(trk, ver, p);
    std::vector<Detection> cands{make_cand({10, 10, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.4f;  trk.next_box = {10, 10, 20, 20};
    ctrl.on_frame(kFrame, {}, out);             // → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);

    const int before = trk.init_calls;
    // Takip kutusuyla örtüşen, yüksek skorlu aday → doğrulanır.
    ctrl.on_frame(kFrame, cands, out);
    EXPECT_EQ(out.state, GuidanceController::State::Track);
    EXPECT_EQ(trk.init_calls, before + 1);      // yeniden tohumlandı
}

// SUSPECT → doğrulanamaz + süre dolar → SEARCH (kilit bırakılır).
TEST(Guidance, SuspectGivesUpToSearch) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController::Params p; p.lost_conf = 0.3f; p.suspect_conf = 0.5f;
    p.suspect_frames = 1; p.reacquire_frames = 2;
    GuidanceController ctrl(trk, ver, p);
    std::vector<Detection> cands{make_cand({10, 10, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.4f;
    ctrl.on_frame(kFrame, {}, out);             // → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);
    // Doğrulayacak aday YOK → reacquire_frames dolunca SEARCH.
    ctrl.on_frame(kFrame, {}, out);
    ctrl.on_frame(kFrame, {}, out);
    EXPECT_EQ(out.state, GuidanceController::State::Search);
    EXPECT_FALSE(out.has_target);
}

// LOCK-INTEGRITY: yüksek güven ama kutu ÇEVRESİ gök değil (çimen) → SUSPECT.
// Güveni ezen geometrik bekçi: tracker yere sürüklenince güven yalan söyler.
TEST(Guidance, LowSkyRingTriggersSuspectDespiteHighConfidence) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController::Params p;  // integrity varsayılan açık, sky_ring_min=0.55
    GuidanceController ctrl(trk, ver, p);
    cv::Mat grass(480, 640, CV_8UC3, cv::Scalar(40, 130, 60));  // yeşil-baskın koyu çimen
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(grass, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.95f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(grass, {}, out);  // yüksek güven AMA çimen çevre → integrity FAIL
    EXPECT_EQ(out.state, GuidanceController::State::Suspect);
    EXPECT_STREQ(out.integrity_reason, "sky");
    EXPECT_LT(out.sky, p.sky_ring_min);
}

// LOCK-INTEGRITY yanlış-pozitif kontrolü: gök-çevreli gerçek hedef → TRACK korunur.
TEST(Guidance, HighSkyRingStaysTracking) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);  // varsayılan (integrity açık)
    cv::Mat sky(480, 640, CV_8UC3, cv::Scalar(235, 170, 90));   // mavi-baskın + parlak gök
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(sky, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.95f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);
    EXPECT_EQ(out.state, GuidanceController::State::Track);
    EXPECT_GT(out.sky, 0.9f);
    EXPECT_TRUE(out.roi.area() > 0);  // TRACK'te dar işlem penceresi üretiliyor
}

// release() her durumdan SEARCH'e döner.
TEST(Guidance, ReleaseReturnsToSearch) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    std::vector<Detection> cands{make_cand({100, 100, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(kFrame, cands, out);
    ctrl.select(0);
    ctrl.on_frame(kFrame, {}, out);
    ASSERT_EQ(ctrl.state(), GuidanceController::State::Track);
    ctrl.release();
    EXPECT_EQ(ctrl.state(), GuidanceController::State::Search);
}
