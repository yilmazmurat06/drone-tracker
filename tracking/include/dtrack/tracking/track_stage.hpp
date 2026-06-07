#pragma once
//
// TrackStage: takibi pipeline'a sarar.
//   girdi:  FrameDetections (kare + tespit listesi)
//   çıktı:  FrameTracks (kare + aktif track listesi)
//
// Kareyi de aşağı taşırız: görselleştirme/çıktı, track kutusunu orijinal kare
// üzerine çizmek için görüntüye ihtiyaç duyar.
//
// Not: Tracker imge koordinatında çalışır (stabilizasyon ego'yu ~0.10 px'e
// indirdiği için koordinat sandviçine gerek yoktur; referans sıfırlaması
// sorunu da böylece ortadan kalkar).

#include <memory>
#include <optional>
#include <vector>

#include "dtrack/common/cue.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/pipeline/stage.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

class TrackStage
    : public pipeline::Stage<common::FrameDetections, common::FrameTracks> {
public:
    // cue_board verilirse (kapalı-döngü): her karede iz tahmini board'a yazılır;
    // detektör (DetectStage) bunu okuyup tahmin ROI'sinde kurtarma yapar.
    explicit TrackStage(std::shared_ptr<ITracker> tracker,
                        std::shared_ptr<common::CueBoard> cue_board = nullptr)
        : Stage("track"), tracker_(std::move(tracker)), cue_board_(std::move(cue_board)) {}

protected:
    std::optional<common::FrameTracks> process(common::FrameDetections&& fd) override {
        if (!fd.frame) return std::nullopt;
        auto tracks = tracker_->update(fd.detections, fd.frame->stamp);
        // Kapalı-döngü geri besleme: iz tahminini detektöre ulaştır (yan kanal).
        if (cue_board_) cue_board_->publish(common::make_cue(tracks));
        return common::FrameTracks{fd.frame, std::move(tracks)};
    }

private:
    std::shared_ptr<ITracker> tracker_;
    std::shared_ptr<common::CueBoard> cue_board_;
};

}  // namespace dtrack::tracking
