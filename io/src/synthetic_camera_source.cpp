#include "dtrack/io/synthetic_camera_source.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "dtrack/common/time.hpp"

namespace dtrack::io {

namespace {
// Tuval, kare boyutundan ego kaymasını karşılayacak kadar büyük olmalı ki
// kare daima tuvalin içinden kırpılsın (kenar taşması olmasın).
constexpr int kCanvasMargin = 64;
}  // namespace

SyntheticCameraSource::SyntheticCameraSource(SceneConfig cfg, common::Modality modality,
                                             double fps, common::Timestamp t0, bool realtime)
    : scene_(cfg),
      modality_(modality),
      fps_(fps),
      t0_(t0),
      realtime_(realtime),
      rng_(cfg.seed) {}

bool SyntheticCameraSource::open() {
    const SceneConfig& cfg = scene_.config();

    // Arka plan tuvalini bir kez üret: taban seviye + rastgele doku.
    // Termalde arka plan daha düz/soğuk, görünürde daha kontrastlı.
    const int cw = cfg.width + 2 * kCanvasMargin;
    const int ch = cfg.height + 2 * kCanvasMargin;
    canvas_ = cv::Mat(ch, cw, CV_8UC1);

    const float texture = (modality_ == common::Modality::Thermal)
                              ? cfg.bg_texture * 0.4f   // termalde daha az doku
                              : cfg.bg_texture;

    // Düşük frekanslı doku: gürültüyü blur'layarak "bulut/arazi" benzeri leke üret.
    cv::Mat noise(ch, cw, CV_32FC1);
    rng_.fill(noise, cv::RNG::NORMAL, 0.0, 1.0);
    cv::GaussianBlur(noise, noise, cv::Size(0, 0), 3.0);
    cv::normalize(noise, noise, -1.0, 1.0, cv::NORM_MINMAX);

    for (int y = 0; y < ch; ++y) {
        auto* row = canvas_.ptr<uchar>(y);
        const auto* nrow = noise.ptr<float>(y);
        for (int x = 0; x < cw; ++x) {
            float v = cfg.bg_level + texture * nrow[x];
            row[x] = static_cast<uchar>(std::clamp(v, 0.0f, 255.0f));
        }
    }

    // Optical flow'un kilitleneceği birkaç keskin köşe/işaret ekle.
    for (int i = 0; i < 40; ++i) {
        const int x = rng_.uniform(kCanvasMargin, cw - kCanvasMargin);
        const int y = rng_.uniform(kCanvasMargin, ch - kCanvasMargin);
        const int b = rng_.uniform(60, 110);
        cv::circle(canvas_, {x, y}, rng_.uniform(1, 3), b, cv::FILLED, cv::LINE_AA);
    }

    open_ = true;
    frame_index_ = 0;
    return true;
}

void SyntheticCameraSource::close() { open_ = false; }

void SyntheticCameraSource::render(double t, cv::Mat& out) {
    const SceneConfig& cfg = scene_.config();

    // Ego kayması kadar ötelenmiş pencereyi tuvalden kırp.
    const Vec2 shift = scene_.ego_shift_px(t);
    const float ox = kCanvasMargin + shift.x;
    const float oy = kCanvasMargin + shift.y;

    // Alt-piksel kayma için warpAffine (saf öteleme matrisi).
    cv::Matx23f M(1, 0, -ox, 0, 1, -oy);
    cv::warpAffine(canvas_, out, M, cv::Size(cfg.width, cfg.height),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT);

    // Hedefi GÖZLEMLENEN konuma çiz (gerçek hareket + ego). Alt-piksel Gaussian.
    const Vec2 tp = scene_.target_observed_px(t);
    last_target_ = tp;
    const int r = std::max(2, static_cast<int>(std::ceil(cfg.target_sigma_px * 3)));
    const float s2 = 2.0f * cfg.target_sigma_px * cfg.target_sigma_px;
    for (int dy = -r; dy <= r; ++dy) {
        const int py = static_cast<int>(std::lround(tp.y)) + dy;
        if (py < 0 || py >= cfg.height) continue;
        auto* row = out.ptr<uchar>(py);
        for (int dx = -r; dx <= r; ++dx) {
            const int px = static_cast<int>(std::lround(tp.x)) + dx;
            if (px < 0 || px >= cfg.width) continue;
            // Alt-piksel merkeze göre gerçek uzaklık.
            const float fx = px - tp.x;
            const float fy = py - tp.y;
            const float g = std::exp(-(fx * fx + fy * fy) / s2);
            const float v = row[px] + cfg.target_intensity * g;
            row[px] = static_cast<uchar>(std::clamp(v, 0.0f, 255.0f));
        }
    }

    // Sensör gürültüsü (her karede bağımsız).
    cv::Mat sensor_noise(out.size(), CV_8UC1);
    rng_.fill(sensor_noise, cv::RNG::NORMAL, 0.0, 3.0);
    out += sensor_noise;
}

std::optional<common::FramePtr> SyntheticCameraSource::next_frame() {
    if (!open_) return std::nullopt;

    common::Timestamp stamp;
    double t;
    if (realtime_) {
        stamp = common::now();
        t = common::millis_between(t0_, stamp) / 1000.0;
        // Bir sonraki kare zamanına kadar bekleyerek ~fps hızını koru.
        const double next_t = (frame_index_ + 1) / fps_;
        const double now_t = common::millis_between(t0_, common::now()) / 1000.0;
        if (now_t < next_t) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(next_t - now_t));
        }
    } else {
        // Sanal saat: deterministik, gerçek zamandan bağımsız.
        t = frame_index_ / fps_;
        stamp = t0_ + std::chrono::duration_cast<common::Duration>(
                          std::chrono::duration<double>(t));
    }

    auto frame = std::make_shared<common::Frame>();
    frame->index = frame_index_++;
    frame->stamp = stamp;
    frame->modality = modality_;
    render(t, frame->image);

    return common::FramePtr(frame);
}

}  // namespace dtrack::io
