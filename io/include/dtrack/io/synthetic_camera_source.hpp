#pragma once
//
// SyntheticCameraSource: SceneModel'i gerçek bir cv::Mat kareye render eden,
// ICameraSource arayüzünü uygulayan sentetik kamera.
//
// Üretilen her kare:
//   - kayan dokulu arka plan (optical flow'un takip edeceği köşeler içerir),
//   - ego-motion'a göre kaymış,
//   - 2-6 piksellik alt-piksel hedef leke (Gaussian) eklenmiş,
//   - sensör gürültüsü katılmış.
//
// Zaman: kaynak, oluşturulurken verilen t0'a göre gerçek (wall-clock) zamanı
// kullanır -> kareler ~fps hızında, IMU kaynağıyla zaman damgası üzerinden
// otomatik hizalı. realtime=false ise sanal saatle olabildiğince hızlı üretir
// (offline/deterministik koşu için).

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/io/camera_source.hpp"
#include "dtrack/io/synthetic_scene.hpp"

namespace dtrack::io {

class SyntheticCameraSource : public ICameraSource {
public:
    SyntheticCameraSource(SceneConfig cfg, common::Modality modality, double fps,
                          common::Timestamp t0, bool realtime = true);

    bool open() override;
    void close() override;
    bool is_open() const override { return open_; }

    std::optional<common::FramePtr> next_frame() override;
    common::Modality modality() const override { return modality_; }

    // Yer-gerçeği erişimi (doğruluk ölçümü / test için): son üretilen karedeki
    // hedefin GÖZLEMLENEN konumu.
    Vec2 last_target_observed_px() const { return last_target_; }
    const SceneModel& scene() const { return scene_; }

private:
    void render(double t, cv::Mat& out);

    SceneModel scene_;
    common::Modality modality_;
    double fps_;
    common::Timestamp t0_;
    bool realtime_;

    bool open_{false};
    std::uint64_t frame_index_{0};
    cv::Mat canvas_;        // önceden üretilmiş büyük doku tuvali (arka plan)
    cv::RNG rng_;           // gürültü için
    Vec2 last_target_{};
};

}  // namespace dtrack::io
