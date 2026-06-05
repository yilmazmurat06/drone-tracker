#pragma once
//
// CameraStage: bir ICameraSource'u pipeline'ın ilk (kaynak) stage'ine sarar.
// Girdisi yok (Tick), çıktısı FramePtr. Her tetiklemede kaynaktan bir kare çeker
// ve bir sonraki stage'e (stabilizasyon) yollar.
//
// Kaynağın somut türü (sentetik / video / donanım) burada önemsiz: yalnızca
// ICameraSource arayüzü görülür -> kaynağı değiştirmek pipeline'ı etkilemez.

#include <memory>
#include <optional>

#include "dtrack/common/types.hpp"
#include "dtrack/io/camera_source.hpp"
#include "dtrack/pipeline/stage.hpp"

namespace dtrack::io {

class CameraStage : public pipeline::Stage<common::Tick, common::FramePtr> {
public:
    explicit CameraStage(std::shared_ptr<ICameraSource> source)
        : Stage("camera"), source_(std::move(source)) {
        if (source_ && !source_->is_open()) source_->open();
    }

protected:
    std::optional<common::FramePtr> process(common::Tick&& /*tick*/) override {
        if (!source_) return std::nullopt;
        return source_->next_frame();  // kare yoksa nullopt -> emit edilmez
    }

private:
    std::shared_ptr<ICameraSource> source_;
};

}  // namespace dtrack::io
