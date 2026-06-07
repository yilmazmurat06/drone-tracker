#pragma once
//
// TargetCue + CueBoard: TAKİP -> TESPİT geri besleme kanalı (kapalı-döngü "cued"
// tracking). Pipeline normalde tek yönlüdür (detect -> track); ama hedefe
// KİLİTLENDİKTEN sonra tracker, hedefin bir sonraki kare için NEREDE olacağını
// (Kalman tahmini) bilir. Bu bilgiyi tespite geri verirsek:
//   - Global tespit o karede hedefi kaçırırsa (düşük SNR / hover), tahmin
//     etrafındaki küçük bir ROI'de DÜŞÜK eşikle (lokal "track-before-detect")
//     hedef kurtarılır -> kilit kesilmez (KESİNTİSİZ TAKİP).
//   - Clutter tahminden uzakta olduğu için kapı dışı kalır -> precision artar.
// Bu, IRST (Infrared Search & Track) literatüründeki klasik "Kalman ile ROI
// tahmini + kayıp telafisi" kapalı-döngü yaklaşımıdır.
//
// IMU gibi cue de pipeline kuyruğuna girmez; küçük bir YAN KANALdır. Tracker
// (üretici, tek thread) yazar, detektör (tüketici, tek thread) okur. Bir kare
// bayat olması sorun değil: cue zaten bir TAHMİN'dir ve kapı yarıçapı pay bırakır.

#include <algorithm>
#include <atomic>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"

namespace dtrack::common {

// Kilitli hedefin bir sonraki kare için tahmini konumu + arama yarıçapı.
struct TargetCue {
    cv::Point2f predicted{0.0f, 0.0f};  // tahmini hedef konumu (GÜNCEL görüntü koord.)
    float gate_radius{0.0f};            // ROI arama yarıçapı (px); coast'ta büyür
    bool valid{false};                  // geçerli (onaylı/coasting) iz var mı?
};

// İz listesinden cue üret: en yüksek güvenli Confirmed/Coasting izi seç, onun
// ileri TAHMİNİNİ ve coast süresiyle BÜYÜYEN kapı yarıçapını kullan. Kapı coast'ta
// genişler çünkü tespitsiz geçen her karede hedefin gerçek konumu tahminden daha
// çok sapabilir (Kalman P kovaryansının büyümesinin pratik karşılığı).
inline TargetCue make_cue(const std::vector<Track>& tracks, float base_radius = 10.0f,
                          float per_coast_px = 2.0f, float max_radius = 40.0f) {
    TargetCue c;
    const Track* best = nullptr;
    for (const auto& t : tracks) {
        if (t.status != TrackStatus::Confirmed && t.status != TrackStatus::Coasting)
            continue;
        if (!best || t.confidence > best->confidence) best = &t;
    }
    if (best) {
        c.valid = true;
        c.predicted = best->predicted;
        c.gate_radius = std::min(
            max_radius, base_radius + per_coast_px * static_cast<float>(best->time_since_update));
    }
    return c;
}

// Tek-yazar / çok-okur lock-free anlık görüntü (seqlock deseni).
//
// seqlock: yazar yazmadan ÖNCE ve SONRA bir sayaç artırır. Sayaç TEK iken yazma
// sürüyordur. Okur, okumadan önce ve sonra sayacı okur; ikisi eşit ve ÇİFT ise
// "yırtılmamış" (tutarlı) bir kopya almıştır, değilse tekrar dener. Mutex yok ->
// gerçek-zaman thread'i kilitte beklemez (projenin SpscRingBuffer ile aynı ruh).
class CueBoard {
public:
    // Tracker thread'i çağırır (TEK yazar).
    void publish(const TargetCue& c) {
        seq_.fetch_add(1, std::memory_order_release);  // tek -> "yazma başladı"
        std::atomic_thread_fence(std::memory_order_release);
        cue_ = c;
        std::atomic_thread_fence(std::memory_order_release);
        seq_.fetch_add(1, std::memory_order_release);  // çift -> "yazma bitti"
    }

    // Detektör thread'i çağırır. Tutarlı (yırtılmamış) bir kopya döndürür.
    TargetCue read() const {
        for (;;) {
            const unsigned s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1u) continue;  // yazma sürüyor -> bekle/yeniden dene
            std::atomic_thread_fence(std::memory_order_acquire);
            TargetCue c = cue_;
            std::atomic_thread_fence(std::memory_order_acquire);
            const unsigned s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) return c;  // yazma araya girmedi -> kopya tutarlı
        }
    }

private:
    std::atomic<unsigned> seq_{0};
    TargetCue cue_{};
};

}  // namespace dtrack::common
