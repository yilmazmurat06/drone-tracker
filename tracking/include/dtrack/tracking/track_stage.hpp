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

#include "dtrack/common/types.hpp"
#include "dtrack/pipeline/stage.hpp"
#include "dtrack/tracking/tracker.hpp"

namespace dtrack::tracking {

class TrackStage
    : public pipeline::Stage<common::FrameDetections, common::FrameTracks> {
public:
    explicit TrackStage(std::shared_ptr<ITracker> tracker)
        : Stage("track"), tracker_(std::move(tracker)) {}

protected:
    std::optional<common::FrameTracks> process(common::FrameDetections&& fd) override {
        if (!fd.frame) return std::nullopt;
        auto tracks = tracker_->update(fd.detections, fd.frame->stamp);
        return common::FrameTracks{fd.frame, std::move(tracks)};
    }

private:
    std::shared_ptr<ITracker> tracker_;
};

}  // namespace dtrack::tracking
