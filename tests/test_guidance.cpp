// ============================================================================
//  test_guidance — GuidanceController durum makinesi geçişleri.
//
//  Gerçek tracker/discriminator yerine KONTROL EDİLEBİLİR sahteler enjekte edilir
//  → geçişleri deterministik test ederiz (görüntü içeriğinden bağımsız).
// ============================================================================
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "dtrack/guidance/guidance_controller.hpp"

using namespace dtrack;

namespace {

// Güveni ve kutuyu testin dışarıdan dayattığı sahte tek-hedef tracker.
struct FakeTracker : ISingleTargetTracker {
    float    next_conf = 1.0f;
    cv::Rect next_box{10, 10, 20, 20};
    int      init_calls = 0;
    cv::Rect init(const cv::Mat&, const cv::Rect& b, bool /*refine*/ = true) override {
        ++init_calls; next_box = b; return b;
    }
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
    ctrl.on_frame(kFrame, {}, out);  // kilit karesi = tohumlama (track koşmaz)

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
    // Reseed kalite kapıları (edge) için aday kutusunda KESKİN bir hedef olmalı —
    // düz gri karede aday "bulut gibi yumuşak" sayılıp elenirdi.
    cv::Mat frame = kFrame.clone();
    cv::rectangle(frame, cv::Rect(14, 14, 12, 12), cv::Scalar(255), cv::FILLED);
    std::vector<Detection> cands{make_cand({10, 10, 20, 20}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(frame, cands, out);
    ctrl.select(0);
    ctrl.on_frame(frame, {}, out);  // kilit karesi = tohumlama (track koşmaz)

    trk.next_conf = 0.4f;  trk.next_box = {10, 10, 20, 20};
    ctrl.on_frame(frame, {}, out);             // → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);

    const int before = trk.init_calls;
    // Çapaya yakın, yüksek skorlu, keskin aday → doğrulanır.
    ctrl.on_frame(frame, cands, out);
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
    ctrl.on_frame(kFrame, {}, out);  // kilit karesi = tohumlama (track koşmaz)

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
    ctrl.on_frame(grass, {}, out);  // kilit karesi = tohumlama (track koşmaz)

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

// ÇAPALI RE-ACQUIRE: tracker ışınlanınca (motion-fail → SUSPECT) doğrulama,
// sürüklenmiş kutuya örtüşen adayı DEĞİL, son-sağlam konuma (çapa) yakın adayı seçer.
TEST(Guidance, ReacquireAnchoredAtLastGoodNotDriftedBox) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    cv::Mat sky(720, 1280, CV_8UC3, cv::Scalar(235, 170, 90));
    // İki aday konumuna da KESKİN hedef çiz (edge kapısı); ayrım UZAKLIKTAN gelsin.
    cv::rectangle(sky, cv::Rect(320, 238, 20, 20), cv::Scalar(35, 35, 35), cv::FILLED);
    cv::rectangle(sky, cv::Rect(610, 510, 20, 20), cv::Scalar(35, 35, 35), cv::FILLED);
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(sky, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.9f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // sağlam TRACK karesi → çapa burada
    ASSERT_EQ(out.state, GuidanceController::State::Track);

    trk.next_box = {600, 500, 40, 40};            // ışınlanma (~410px sıçrama)
    ctrl.on_frame(sky, {}, out);                  // motion-fail → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);
    EXPECT_STREQ(out.integrity_reason, "motion");

    // Aday A: sürüklenmiş kutuyla örtüşür (eski IoU mantığı bunu seçerdi) — çapadan uzak.
    // Aday B: çapaya yakın → DOĞRU seçim.
    const int before = trk.init_calls;
    std::vector<Detection> sus{make_cand({600, 500, 40, 40}, 0.9f),
                               make_cand({310, 228, 40, 40}, 0.9f)};
    ctrl.on_frame(sky, sus, out);
    EXPECT_EQ(out.state, GuidanceController::State::Track);
    EXPECT_EQ(trk.init_calls, before + 1);
    EXPECT_EQ(out.target, cv::Rect(310, 228, 40, 40));  // B'ye tohumlandı, A'ya değil
}

// RE-ACQUIRE BOYUT BANDI: çapaya yakın ama dev (kilit kutusunun >2.5× alanı) aday RED —
// patlamış kutudan dev sahne bloğuna yeniden kilitlenme engellenir.
TEST(Guidance, ReacquireRejectsOutOfSizeBand) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    cv::Mat sky(720, 1280, CV_8UC3, cv::Scalar(235, 170, 90));
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(sky, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.9f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // sağlam TRACK → çapa
    trk.next_box = {600, 500, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // ışınlanma → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);

    // Çapaya yakın merkezli ama 200×200 dev aday (alan 25× kilit) → reseed YOK.
    std::vector<Detection> sus{make_cand({220, 140, 200, 200}, 0.9f)};
    const int before = trk.init_calls;
    ctrl.on_frame(sky, sus, out);
    EXPECT_EQ(out.state, GuidanceController::State::Suspect);
    EXPECT_EQ(trk.init_calls, before);
}

// RE-ACQUIRE KENAR KAPISI: çapaya yakın, boyutu uygun, gök-çevreli ama YUMUŞAK
// (kenarsız = bulut tutamı) aday RED — geometri bulutu ayıramaz, keskinlik ayırır.
TEST(Guidance, ReacquireRejectsSoftCloudCandidate) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    cv::Mat sky(720, 1280, CV_8UC3, cv::Scalar(235, 170, 90));  // düz gök — aday içi kenarsız
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(sky, cands, out);
    ctrl.select(0);

    trk.next_conf = 0.9f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // çapa
    trk.next_box = {600, 500, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // ışınlanma → SUSPECT
    ASSERT_EQ(out.state, GuidanceController::State::Suspect);

    // Çapa dibinde, boyut bandında, gök-çevreli ama DÜZ (edge≈0) aday → reseed YOK.
    std::vector<Detection> sus{make_cand({305, 224, 40, 40}, 0.9f)};
    const int before = trk.init_calls;
    ctrl.on_frame(sky, sus, out);
    EXPECT_EQ(out.state, GuidanceController::State::Suspect);
    EXPECT_EQ(trk.init_calls, before);
}

// EGO-HAREKET TELAFİSİ: kamera sarsıntısı (büyük öteleme) compensate() ile
// bildirilirse, sarsıntıyı izleyen kutu motion-integrity'yi TETİKLEMEZ — telafi
// prev_box_'u da taşıdığı için sıçrama "ışınlanma" sayılmaz.
TEST(Guidance, CompensateAbsorbsEgoMotionJump) {
    FakeTracker trk; FakeVerifier ver;
    GuidanceController ctrl(trk, ver);
    cv::Mat sky(720, 1280, CV_8UC3, cv::Scalar(235, 170, 90));
    std::vector<Detection> cands{make_cand({300, 220, 40, 40}, 0.9f)};
    GuidanceController::Out out;
    ctrl.on_frame(sky, cands, out);
    ctrl.select(0);
    ctrl.on_frame(sky, {}, out);  // kilit karesi = tohumlama

    trk.next_conf = 0.9f; trk.next_box = {300, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);                  // sağlam TRACK
    ASSERT_EQ(out.state, GuidanceController::State::Track);

    // Kamera 100px sağa sıçradı: hedef görüntüde 100px sola kayar. Telafi bildir,
    // tracker da kaymış konumu raporlasın → TRACK sürmeli (motion FAIL yok).
    const cv::Matx33f shift(1, 0, -100, 0, 1, 0, 0, 0, 1);
    ctrl.compensate(shift);
    trk.next_box = {200, 220, 40, 40};
    ctrl.on_frame(sky, {}, out);
    EXPECT_EQ(out.state, GuidanceController::State::Track);

    // Karşılaştırma: telafi BİLDİRİLMEDEN aynı sıçrama → motion FAIL → SUSPECT.
    trk.next_box = {340, 220, 40, 40};            // 140px ani sıçrama, telafisiz
    ctrl.on_frame(sky, {}, out);
    EXPECT_EQ(out.state, GuidanceController::State::Suspect);
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
