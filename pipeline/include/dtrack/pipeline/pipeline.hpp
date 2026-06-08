#pragma once
// ============================================================================
//  Pipeline — tek-thread'li, deterministik runner (İSKELET).
//
//  Veri akışı (initial.txt mimarisi):
//      FrameSource → [Stabilize] → [Detect] → [Track] → Track'ler
//
//  TASARIM: bağımlılıklar dışarıdan ENJEKTE edilir (dependency injection).
//  Pipeline yalnızca ARAYÜZLERİ (IFrameSource, IStabilizer, ...) bilir; somut
//  sınıfları bilmez. Böylece:
//    - her parçayı ayrı test edebiliriz,
//    - sahte (mock) parçalar takabiliriz,
//    - bir aşamayı (örn. stabilizer) nullptr bırakıp atlayabiliriz.
//
//  Çok-thread'li, lock-free ring buffer'lı sürüm Adım 7'de gelecek; o zaman bu
//  sıralı çağrılar, aşamalar-arası kuyruklarla değiştirilecek.
//
//  DURUM: Aşama implementasyonları henüz yok (Adım 3+). Bu runner, arayüzler
//  üzerinden derlenir ve veri akışını belgeler; gerçek parçalar takılınca çalışır.
// ============================================================================
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/core/types.hpp"
#include "dtrack/io/frame_source.hpp"
#include "dtrack/stabilization/stabilizer.hpp"
#include "dtrack/detection/detector.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack {

class Pipeline {
public:
    // Aşama bağımlılıkları. nullptr olanlar atlanır.
    struct Stages {
        IFrameSource* frame_source = nullptr;  // zorunlu
        IStabilizer*  stabilizer   = nullptr;  // ops. (nullptr → stabilizasyon yok)
        IDetector*    detector     = nullptr;  // ops.
        ITracker*     tracker      = nullptr;  // ops.
    };

    explicit Pipeline(Stages stages) : stages_(stages) {}

    // Tek bir kareyi uçtan uca işler.
    //   out_tracks : bu kare sonunda aktif track'ler
    // return false → kaynak bitti (oynatılacak kare kalmadı)
    bool step(std::vector<Track>& out_tracks) {
        Frame frame;
        if (stages_.frame_source == nullptr || !stages_.frame_source->next(frame)) {
            return false;
        }

        // 1) Stabilizasyon (P1) — opsiyonel.
        Frame stabilized = frame;
        if (stages_.stabilizer != nullptr) {
            cv::Matx33f homography;
            stages_.stabilizer->stabilize(frame, telemetry_window_, stabilized, homography);
        }

        // 2) Tespit (P2).
        std::vector<Detection> detections;
        if (stages_.detector != nullptr) {
            stages_.detector->detect(stabilized, detections);
        }

        // 3) Takip (P4). (Ayraç/P3, detector veya tracker içinde uygulanacak.)
        out_tracks.clear();
        if (stages_.tracker != nullptr) {
            stages_.tracker->update(detections, out_tracks);
        }

        return true;
    }

    // Kaynak bitene kadar tüm kareleri işler; toplam kare sayısını döndürür.
    std::size_t run() {
        std::vector<Track> tracks;
        std::size_t frames = 0;
        while (step(tracks)) {
            ++frames;
        }
        return frames;
    }

    // Bu karenin t_rel'i etrafındaki telemetri penceresi. Adım 2'de io/ doldurur.
    std::vector<Telemetry>& telemetry_window() { return telemetry_window_; }

private:
    Stages                 stages_;
    std::vector<Telemetry> telemetry_window_;  // stabilizer'a verilen pencere
};

} // namespace dtrack
