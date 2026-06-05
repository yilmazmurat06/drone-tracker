#pragma once
//
// ICameraSource: bir kamera kaynağının soyut arayüzü.
// Somut implementasyonlar: gerçek kamera (V4L2/GenICam), dosya/video, sentetik
// test üreteci. Pipeline yalnızca bu arayüzü tanır -> kaynağı takmak-çıkarmak kolay.

#include <optional>

#include "dtrack/common/types.hpp"

namespace dtrack::io {

class ICameraSource {
public:
    virtual ~ICameraSource() = default;

    virtual bool open() = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;

    // Bir sonraki kareyi alır. Kare yoksa (henüz hazır değil / akış bitti) nullopt.
    // Zaman damgası poz anına olabildiğince yakın atanmalı (latency bütçesi için).
    virtual std::optional<common::FramePtr> next_frame() = 0;

    virtual common::Modality modality() const = 0;
};

}  // namespace dtrack::io
