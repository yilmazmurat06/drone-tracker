#pragma once
//
// DetectStage: tespiti pipeline'a sarar.
//   girdi:  StabilizedFrame (stabilizasyondan)
//   çıktı:  std::vector<Detection> (her kare için, boş olsa bile)
//
// Boş kareyi de emit ederiz: tracker her karede predict adımını çalıştırmalı
// (kısa tespit boşluklarında track'i tahminle taşımak için), bu yüzden tespit
// olmasa da aşağı akışa "bu karede tespit yok" sinyali gitmeli.

#include <memory>
#include <optional>
#include <vector>

#include "dtrack/common/cue.hpp"
#include "dtrack/common/types.hpp"
#include "dtrack/detection/detector.hpp"
#include "dtrack/detection/discriminator.hpp"
#include "dtrack/pipeline/stage.hpp"

namespace dtrack::detection {

class DetectStage
    : public pipeline::Stage<common::StabilizedFrame, common::FrameDetections> {
public:
    // cue_board verilirse (kapalı-döngü): her karede tracker'ın güncel iz tahmini
    // okunup detektöre verilir -> global tespit kaçırırsa tahmin ROI'sinde kurtarma.
    explicit DetectStage(std::shared_ptr<IDetector> det,
                          std::shared_ptr<IDiscriminator> disc = nullptr,
                          std::shared_ptr<common::CueBoard> cue_board = nullptr)
        : Stage("detect"), det_(std::move(det)), disc_(std::move(disc)),
          cue_board_(std::move(cue_board)) {}

protected:
    std::optional<common::FrameDetections> process(
        common::StabilizedFrame&& sf) override {
        if (!sf.frame) return std::nullopt;
        // Kareyi de taşı: aşağı akış (tracking, görselleştirme) görüntüye ihtiyaç duyar.
        // Boş tespit listesi de geçerli çıktıdır -> emit edilir (tracker her kare tıklar).
        // Kapalı-döngü: detect()'ten ÖNCE tracker'ın son iz tahminini detektöre ver.
        if (cue_board_) det_->set_cue(cue_board_->read());
        // detect() önce çağrılmalı (referans dönüşümünü o kareye göre günceller).
        auto dets = det_->detect(sf);
        if (disc_) disc_->score(dets);
        return common::FrameDetections{sf.frame, std::move(dets),
                                        det_->image_to_reference()};
    }

private:
    std::shared_ptr<IDetector> det_;
    std::shared_ptr<IDiscriminator> disc_;
    std::shared_ptr<common::CueBoard> cue_board_;
};

}  // namespace dtrack::detection
