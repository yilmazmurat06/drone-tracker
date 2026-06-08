#pragma once
// ============================================================================
//  SyntheticFrameSource — testler için yapay kare üretir.
//  Gürültülü gökyüzü arka planına bilinen konumda küçük (2-6 piksel) bir hedef
//  çizer. Hedefin GERÇEK konumunu bildiğimiz için tespit/takip doğruluğunu
//  birim testlerinde ölçebiliriz (ground-truth).
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM. Gerektiğinde (test yazarken) implemente
//         edilecek. Birincil veri yolu kayıtlı Liftoff dosyalarıdır.
// ============================================================================
#include <memory>
#include "dtrack/io/frame_source.hpp"

namespace dtrack {

class SyntheticFrameSource : public IFrameSource {
public:
    struct Params {
        int   width  = 1280;
        int   height = 720;
        int   num_frames = 300;
        float noise_sigma = 5.0f;   // arka plan gürültüsü
        int   target_size = 4;      // hedef piksel çapı
    };

    explicit SyntheticFrameSource(Params params = {});
    ~SyntheticFrameSource() override;

    bool next(Frame& out) override;  // impl: test ihtiyacı doğunca

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dtrack
